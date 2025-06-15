"""
Smart Fallback Strategy: Unified claude-3-5-haiku-20241022 with intelligent error handling
Function: Academic-optimized fallback system for consistent model usage with budget protection
Usage: Automatically handles rate limits, budget constraints, and quality thresholds for research reliability
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class FallbackReason(Enum):
    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMITED = "rate_limited" 
    QUALITY_THRESHOLD = "quality_threshold"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    MODEL_UNAVAILABLE = "model_unavailable"

@dataclass
class FallbackResult:
    success: bool
    model_used: str
    fallback_reason: Optional[FallbackReason]
    cost_estimate: float
    execution_time: float
    quality_score: Optional[float]
    retry_count: int

class SmartFallbackStrategy:
    """
    Academic-optimized fallback strategy using claude-3-5-haiku-20241022 for all tasks
    
    Features:
    - Single model consistency for research reproducibility
    - Intelligent rate limit handling with exponential backoff
    - Budget protection with academic thresholds
    - Quality validation for research standards
    - Portuguese optimization for Brazilian discourse analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.primary_model = "claude-3-5-haiku-20241022"
        self.emergency_fallback = "claude-3-5-sonnet-20241022"
        
        # Academic budget tracking
        self.monthly_budget = config.get('monthly_budget_usd', 50.0)
        self.current_usage = 0.0
        self.budget_alerts = config.get('budget_alert_thresholds', {
            'warning': 0.8,
            'critical': 0.9, 
            'emergency': 0.95
        })
        
        # Smart fallback settings
        fallback_config = config.get('fallback_strategy', {})
        self.enable_smart_fallback = fallback_config.get('enable_smart_fallback', True)
        self.budget_threshold = fallback_config.get('budget_exceeded_threshold', 0.8)
        self.quality_threshold = fallback_config.get('quality_threshold', 0.7)
        self.rate_limit_backoff = fallback_config.get('rate_limit_backoff', True)
        
        # Rate limiting
        self.requests_per_minute = config.get('requests_per_minute', 100)
        self.max_concurrent = config.get('max_concurrent_requests', 5)
        self.request_history = []
        self.active_requests = 0
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2.0)
        self.exponential_backoff = config.get('exponential_backoff', True)
        
        # Stage-specific configurations
        self.stage_configs = config.get('stage_configs', {})
        
        # Portuguese optimization
        self.portuguese_optimization = config.get('portuguese_optimization', True)
        self.preserve_categories = config.get('preserve_original_categories', True)
        
        logger.info(f"🎓 Smart Fallback Strategy initialized with {self.primary_model}")
        logger.info(f"📊 Academic budget: ${self.monthly_budget}/month")
        
    def get_stage_config(self, stage_name: str) -> Dict[str, Any]:
        """Get optimized configuration for specific pipeline stage"""
        
        # Default configuration for claude-3-5-haiku-20241022
        default_config = {
            'model': self.primary_model,
            'temperature': 0.3,
            'max_tokens': 2000,
            'top_p': 0.9
        }
        
        # Stage-specific optimizations
        stage_specific = self.stage_configs.get(stage_name, {})
        config = {**default_config, **stage_specific}
        
        # Ensure we're using the primary model for all stages
        config['model'] = self.primary_model
        
        logger.debug(f"📋 Stage config for {stage_name}: {config}")
        return config
    
    async def execute_with_fallback(self, 
                                  stage_name: str,
                                  api_call_func,
                                  *args, 
                                  **kwargs) -> FallbackResult:
        """
        Execute API call with smart fallback strategy
        
        Args:
            stage_name: Pipeline stage identifier
            api_call_func: Function to execute API call
            *args, **kwargs: Arguments for API call
            
        Returns:
            FallbackResult with execution details
        """
        
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        # Check budget before proceeding
        if not self._check_budget_available():
            logger.warning(f"🚨 Budget exceeded, skipping {stage_name}")
            return FallbackResult(
                success=False,
                model_used=self.primary_model,
                fallback_reason=FallbackReason.BUDGET_EXCEEDED,
                cost_estimate=0.0,
                execution_time=0.0,
                quality_score=None,
                retry_count=0
            )
        
        # Get stage-specific configuration
        stage_config = self.get_stage_config(stage_name)
        
        while retry_count <= self.max_retries:
            try:
                # Wait for rate limiting
                await self._wait_for_rate_limit()
                
                # Track active request
                self.active_requests += 1
                
                # Execute API call with stage configuration
                logger.debug(f"🔄 Executing {stage_name} with {stage_config['model']} (attempt {retry_count + 1})")
                
                result = await api_call_func(
                    model=stage_config['model'],
                    temperature=stage_config['temperature'],
                    max_tokens=stage_config['max_tokens'],
                    *args,
                    **kwargs
                )
                
                # Calculate execution metrics
                execution_time = time.time() - start_time
                cost_estimate = self._estimate_cost(stage_config['model'], stage_config['max_tokens'])
                quality_score = self._evaluate_quality(result)
                
                # Update budget tracking
                self.current_usage += cost_estimate
                self._log_budget_status()
                
                # Validate quality for research standards
                if quality_score and quality_score < self.quality_threshold:
                    logger.warning(f"⚠️ Quality below threshold for {stage_name}: {quality_score:.2f}")
                    
                    # For research, we accept lower quality rather than fail
                    if retry_count < self.max_retries:
                        retry_count += 1
                        await asyncio.sleep(self.retry_delay)
                        continue
                
                logger.info(f"✅ {stage_name} completed with {stage_config['model']} in {execution_time:.2f}s")
                
                return FallbackResult(
                    success=True,
                    model_used=stage_config['model'],
                    fallback_reason=None,
                    cost_estimate=cost_estimate,
                    execution_time=execution_time,
                    quality_score=quality_score,
                    retry_count=retry_count
                )
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Determine fallback reason
                fallback_reason = self._classify_error(e)
                
                logger.warning(f"❌ {stage_name} attempt {retry_count} failed: {e}")
                
                # Handle specific error types
                if fallback_reason == FallbackReason.RATE_LIMITED and self.rate_limit_backoff:
                    delay = self._calculate_backoff_delay(retry_count)
                    logger.info(f"⏱️ Rate limited, waiting {delay:.1f}s before retry")
                    await asyncio.sleep(delay)
                    
                elif fallback_reason == FallbackReason.MODEL_UNAVAILABLE:
                    # Try emergency fallback model
                    logger.info(f"🔄 Trying emergency fallback: {self.emergency_fallback}")
                    stage_config['model'] = self.emergency_fallback
                    
                else:
                    # Standard retry delay
                    if retry_count <= self.max_retries:
                        delay = self.retry_delay * (2 ** retry_count if self.exponential_backoff else 1)
                        await asyncio.sleep(delay)
                        
            finally:
                self.active_requests = max(0, self.active_requests - 1)
        
        # All retries exhausted
        execution_time = time.time() - start_time
        
        logger.error(f"💥 {stage_name} failed after {retry_count} attempts: {last_error}")
        
        return FallbackResult(
            success=False,
            model_used=stage_config['model'],
            fallback_reason=self._classify_error(last_error),
            cost_estimate=0.0,
            execution_time=execution_time,
            quality_score=None,
            retry_count=retry_count
        )
    
    def _check_budget_available(self) -> bool:
        """Check if budget allows for more API calls"""
        usage_percent = self.current_usage / self.monthly_budget
        
        if usage_percent >= self.budget_alerts['emergency']:
            return False
        elif usage_percent >= self.budget_alerts['critical']:
            logger.warning(f"🚨 Critical budget usage: {usage_percent:.1%}")
        elif usage_percent >= self.budget_alerts['warning']:
            logger.warning(f"⚠️ Budget warning: {usage_percent:.1%}")
            
        return True
    
    async def _wait_for_rate_limit(self):
        """Wait if we're approaching rate limits"""
        
        # Clean old requests from history
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        self.request_history = [req_time for req_time in self.request_history if req_time > minute_ago]
        
        # Check requests per minute
        if len(self.request_history) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_history[0]).total_seconds()
            if wait_time > 0:
                logger.debug(f"⏱️ Rate limit wait: {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Check concurrent requests
        while self.active_requests >= self.max_concurrent:
            logger.debug(f"⏱️ Concurrent limit wait: {self.active_requests}/{self.max_concurrent}")
            await asyncio.sleep(0.1)
        
        # Record this request
        self.request_history.append(now)
    
    def _calculate_backoff_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay for rate limiting"""
        base_delay = 1.0
        if self.exponential_backoff:
            return base_delay * (2 ** retry_count) + (retry_count * 0.5)  # Add jitter
        return base_delay
    
    def _classify_error(self, error: Exception) -> FallbackReason:
        """Classify error type for appropriate fallback"""
        error_str = str(error).lower()
        
        if 'rate limit' in error_str or '429' in error_str:
            return FallbackReason.RATE_LIMITED
        elif 'budget' in error_str or 'quota' in error_str:
            return FallbackReason.BUDGET_EXCEEDED
        elif 'timeout' in error_str:
            return FallbackReason.TIMEOUT
        elif 'model not available' in error_str or '404' in error_str:
            return FallbackReason.MODEL_UNAVAILABLE
        else:
            return FallbackReason.API_ERROR
    
    def _estimate_cost(self, model: str, max_tokens: int) -> float:
        """Estimate API call cost for budget tracking"""
        
        # Claude 3.5 Haiku pricing (per 1M tokens)
        if model == "claude-3-5-haiku-20241022":
            input_cost_per_1m = 0.25  # $0.25 per 1M input tokens
            output_cost_per_1m = 1.25  # $1.25 per 1M output tokens
        else:
            # Emergency fallback pricing (Sonnet)
            input_cost_per_1m = 3.0
            output_cost_per_1m = 15.0
        
        # Estimate input tokens (rough approximation)
        estimated_input_tokens = max_tokens * 0.7  # Assume 70% input, 30% output
        estimated_output_tokens = max_tokens * 0.3
        
        total_cost = (
            (estimated_input_tokens / 1_000_000) * input_cost_per_1m +
            (estimated_output_tokens / 1_000_000) * output_cost_per_1m
        )
        
        return total_cost
    
    def _evaluate_quality(self, result: Any) -> Optional[float]:
        """Evaluate response quality for academic standards"""
        
        if not result:
            return 0.0
        
        # Basic quality checks
        quality_score = 1.0
        
        try:
            # Check if result has expected structure
            if hasattr(result, 'content') and result.content:
                content = result.content
                
                # Check content length (not too short)
                if len(str(content)) < 50:
                    quality_score *= 0.5
                
                # Check for Portuguese political categories preservation
                if self.preserve_categories and self.portuguese_optimization:
                    political_terms = ['direita', 'esquerda', 'centro', 'bolsonarismo', 'petismo', 'neutro']
                    content_str = str(content).lower()
                    if any(term in content_str for term in political_terms):
                        quality_score *= 1.1  # Bonus for preserving Portuguese terms
                
                # Check for confidence indicators
                if 'confidence' in str(content).lower() or 'certeza' in str(content).lower():
                    quality_score *= 1.05
                    
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.debug(f"Quality evaluation error: {e}")
            return 0.8  # Default reasonable quality score
    
    def _log_budget_status(self):
        """Log current budget status for academic tracking"""
        usage_percent = (self.current_usage / self.monthly_budget) * 100
        remaining = self.monthly_budget - self.current_usage
        
        logger.debug(f"💰 Budget: ${self.current_usage:.3f}/${self.monthly_budget} ({usage_percent:.1f}%) - ${remaining:.3f} remaining")
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget summary for academic reporting"""
        usage_percent = (self.current_usage / self.monthly_budget) * 100
        
        return {
            'monthly_budget_usd': self.monthly_budget,
            'current_usage_usd': self.current_usage,
            'remaining_budget_usd': self.monthly_budget - self.current_usage,
            'usage_percent': usage_percent,
            'budget_status': self._get_budget_status(usage_percent),
            'primary_model': self.primary_model,
            'emergency_fallback': self.emergency_fallback,
            'requests_this_session': len(self.request_history),
            'active_requests': self.active_requests
        }
    
    def _get_budget_status(self, usage_percent: float) -> str:
        """Get human-readable budget status"""
        if usage_percent >= 95:
            return "EMERGENCY"
        elif usage_percent >= 90:
            return "CRITICAL" 
        elif usage_percent >= 80:
            return "WARNING"
        else:
            return "HEALTHY"
    
    def reset_monthly_budget(self):
        """Reset budget tracking for new month"""
        self.current_usage = 0.0
        self.request_history = []
        logger.info(f"🔄 Monthly budget reset: ${self.monthly_budget}")