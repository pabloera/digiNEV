"""
Academic Configuration Loader for Social Science Research
=========================================================

Simplified configuration management for academic researchers studying 
authoritarianism and violence in Brazilian society.

Features:
- Week 1-2 optimizations pre-configured
- Academic budget defaults
- Portuguese research settings
- Simplified interface for researchers
- Research-focused validation

Author: Social Science Research Optimization
Date: 2025-06-15
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)

class AcademicConfigLoader:
    """
    Simplified configuration loader for academic research
    
    Provides research-focused defaults and automatic optimization
    configuration for Week 1-2 integrations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize academic configuration loader
        
        Args:
            config_path: Path to academic configuration file
                        (defaults to config/academic_settings.yaml)
        """
        self.project_root = Path(__file__).parent.parent
        
        if config_path is None:
            config_path = self.project_root / "config" / "academic_settings.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_academic_config()
        
        logger.info(f"🎓 Academic configuration loaded from {self.config_path}")
    
    def _load_academic_config(self) -> Dict[str, Any]:
        """Load academic configuration with research defaults"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info("✅ Academic configuration file loaded successfully")
                return self._apply_academic_defaults(config)
            else:
                logger.warning(f"Academic config file not found: {self.config_path}")
                return self._get_academic_defaults()
                
        except Exception as e:
            logger.error(f"Error loading academic config: {e}")
            logger.info("🔄 Using academic defaults")
            return self._get_academic_defaults()
    
    def _apply_academic_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply academic defaults to loaded configuration"""
        defaults = self._get_academic_defaults()
        
        # Merge with priority to loaded config
        merged_config = {}
        for key, default_value in defaults.items():
            if key in config:
                if isinstance(default_value, dict) and isinstance(config[key], dict):
                    merged_config[key] = {**default_value, **config[key]}
                else:
                    merged_config[key] = config[key]
            else:
                merged_config[key] = default_value
        
        # Add any additional keys from loaded config
        for key, value in config.items():
            if key not in merged_config:
                merged_config[key] = value
        
        return merged_config
    
    def _get_academic_defaults(self) -> Dict[str, Any]:
        """Get research-focused default configuration"""
        return {
            'academic': {
                'enabled': True,
                'monthly_budget': 50.0,
                'research_focus': 'brazilian_politics',
                'portuguese_optimization': True,
                'cache_optimization': True
            },
            'anthropic': {
                'enable_api_integration': True,
                'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
                'model': 'claude-3-5-haiku-20241022',
                'rate_limit': 20,
                'max_tokens': 1000,
                'temperature': 0.3
            },
            'emergency_cache': {
                'enabled': True,
                'cache_dir': 'cache/academic_embeddings',
                'ttl_hours': 48,
                'max_memory_mb': 256
            },
            'smart_cache': {
                'enabled': True,
                'cache_dir': 'cache/academic_claude',
                'ttl_hours': 72,
                'semantic_similarity_threshold': 0.85,
                'portuguese_normalization': True
            },
            'voyage_embeddings': {
                'model': 'voyage-3.5-lite',
                'cache_enabled': True,
                'batch_size': 128,
                'sampling_rate': 0.04,
                'input_type': 'document'
            },
            'pipeline': {
                'stages_enabled': [
                    '01_chunk_processing',
                    '02_encoding_validation', 
                    '03_deduplication',
                    '04_feature_validation',
                    '05_political_analysis',
                    '06_text_cleaning',
                    '07_linguistic_processing',
                    '08_sentiment_analysis',
                    '09_topic_modeling',
                    '10_tfidf_extraction',
                    '11_clustering',
                    '19_semantic_search',
                    '20_pipeline_validation'
                ],
                'chunk_size': 1000,
                'deduplication_enabled': True,
                'political_analysis_depth': 'full'
            },
            'budget_alerts': {
                'monthly_threshold': 0.8,
                'weekly_threshold': 0.2,
                'enable_auto_downgrade': True
            },
            'research_quality': {
                'reproducibility': True,
                'data_integrity_checks': True,
                'bias_monitoring': True,
                'validation_sampling': 0.1
            },
            'performance': {
                'max_workers': 4,
                'memory_limit_gb': 4.0,
                'timeout_minutes': 30
            },
            'portuguese': {
                'enabled': True,
                'political_entity_recognition': True,
                'brazilian_variants': True,
                'social_media_normalization': True
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete academic configuration"""
        return self.config.copy()
    
    def get_academic_settings(self) -> Dict[str, Any]:
        """Get academic-specific settings"""
        return self.config.get('academic', {})
    
    def get_anthropic_config(self) -> Dict[str, Any]:
        """Get Anthropic API configuration for academic use"""
        anthropic_config = self.config.get('anthropic', {})
        
        # Ensure academic budget awareness
        academic_settings = self.get_academic_settings()
        if academic_settings.get('enabled', False):
            anthropic_config['academic'] = academic_settings
        
        return anthropic_config
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get Week 1-2 cache configuration"""
        return {
            'emergency_cache': self.config.get('emergency_cache', {}),
            'smart_cache': self.config.get('smart_cache', {}),
            'academic_enabled': self.config.get('academic', {}).get('cache_optimization', True)
        }
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get academic pipeline configuration"""
        pipeline_config = self.config.get('pipeline', {})
        
        # Add academic optimizations
        pipeline_config['academic_mode'] = True
        pipeline_config['portuguese_optimization'] = self.config.get('portuguese', {}).get('enabled', True)
        
        return pipeline_config
    
    def is_academic_mode(self) -> bool:
        """Check if academic mode is enabled"""
        return self.config.get('academic', {}).get('enabled', False)
    
    def get_monthly_budget(self) -> float:
        """Get academic monthly budget"""
        return self.config.get('academic', {}).get('monthly_budget', 50.0)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate academic configuration for research use"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'optimizations_enabled': []
        }
        
        # Check academic mode
        if not self.is_academic_mode():
            validation_results['warnings'].append("Academic mode not enabled - missing research optimizations")
        else:
            validation_results['optimizations_enabled'].append("Academic mode")
        
        # Check API key
        api_key = self.config.get('anthropic', {}).get('api_key', '')
        if not api_key or api_key == '${ANTHROPIC_API_KEY}':
            validation_results['errors'].append("ANTHROPIC_API_KEY not configured")
            validation_results['valid'] = False
        
        # Check cache optimizations
        if self.config.get('emergency_cache', {}).get('enabled', False):
            validation_results['optimizations_enabled'].append("Week 1: Emergency Cache")
        
        if self.config.get('smart_cache', {}).get('enabled', False):
            validation_results['optimizations_enabled'].append("Week 2: Smart Cache")
        
        # Check Portuguese optimization
        if self.config.get('portuguese', {}).get('enabled', False):
            validation_results['optimizations_enabled'].append("Portuguese Language Optimization")
        
        # Check budget configuration
        budget = self.get_monthly_budget()
        if budget > 100:
            validation_results['warnings'].append(f"Monthly budget (${budget}) may be high for academic use")
        elif budget < 10:
            validation_results['warnings'].append(f"Monthly budget (${budget}) may be too low for research")
        
        return validation_results
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary optimized for academic researchers"""
        return {
            'configuration_type': 'academic_research',
            'research_focus': self.config.get('academic', {}).get('research_focus', 'general'),
            'optimizations_enabled': {
                'academic_mode': self.is_academic_mode(),
                'week1_emergency_cache': self.config.get('emergency_cache', {}).get('enabled', False),
                'week2_smart_cache': self.config.get('smart_cache', {}).get('enabled', False),
                'portuguese_optimization': self.config.get('portuguese', {}).get('enabled', False),
                'cost_optimization': True
            },
            'budget_configuration': {
                'monthly_budget': self.get_monthly_budget(),
                'auto_downgrade': self.config.get('budget_alerts', {}).get('enable_auto_downgrade', True),
                'cost_monitoring': True
            },
            'research_quality': {
                'reproducibility': self.config.get('research_quality', {}).get('reproducibility', True),
                'data_integrity': self.config.get('research_quality', {}).get('data_integrity_checks', True),
                'bias_monitoring': self.config.get('research_quality', {}).get('bias_monitoring', True)
            },
            'computational_limits': {
                'max_workers': self.config.get('performance', {}).get('max_workers', 4),
                'memory_limit_gb': self.config.get('performance', {}).get('memory_limit_gb', 4.0),
                'suitable_for_academic_computing': True
            }
        }

# Factory function for easy use
def load_academic_config(config_path: Optional[str] = None) -> AcademicConfigLoader:
    """
    Load academic configuration for research
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured academic loader
    """
    return AcademicConfigLoader(config_path)

# Global instance for easy access
_global_academic_config = None

def get_academic_config() -> AcademicConfigLoader:
    """Get global academic configuration instance"""
    global _global_academic_config
    if _global_academic_config is None:
        _global_academic_config = load_academic_config()
    return _global_academic_config