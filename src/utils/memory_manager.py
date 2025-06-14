#!/usr/bin/env python3
"""
Explicit Memory Manager - TASK-016 Implementation
=================================================

Provides explicit memory management and garbage collection for the pipeline
to achieve 50% reduction in memory usage.

Features:
- Explicit memory cleanup after each stage
- Garbage collection optimization
- Memory monitoring and alerts
- DataFrame memory optimization
- Cache cleanup and management

Author: Pablo Emanuel Romero Almada, Ph.D.
Date: 2025-06-14
Version: 5.0.0
"""

import gc
import logging
import psutil
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Explicit memory management for pipeline operations
    """
    
    def __init__(self, 
                 target_memory_limit_gb: float = 4.0,
                 gc_threshold: float = 0.8,
                 enable_monitoring: bool = True):
        
        self.target_memory_limit = target_memory_limit_gb * 1024 * 1024 * 1024
        self.gc_threshold = gc_threshold
        self.enable_monitoring = enable_monitoring
        
        # Memory tracking
        self.initial_memory = None
        self.stage_memory_usage = {}
        self.peak_memory_usage = 0
        
        # Configure garbage collection for better performance
        self._configure_garbage_collection()
        
        logger.info(f"MemoryManager initialized - Target limit: {target_memory_limit_gb:.1f}GB")
    
    def _configure_garbage_collection(self):
        """Optimize garbage collection settings"""
        # Set more aggressive GC thresholds for memory optimization
        gc.set_threshold(700, 10, 10)  # More frequent GC
        
        # Enable GC debug if in development
        if logger.getEffectiveLevel() <= logging.DEBUG:
            gc.set_debug(gc.DEBUG_STATS)
        
        logger.debug("Garbage collection configured for memory optimization")
    
    @contextmanager
    def stage_memory_context(self, stage_name: str):
        """
        Context manager for stage-level memory management
        
        Usage:
            with memory_manager.stage_memory_context("05_political_analysis"):
                # Stage processing code here
                result = process_stage(data)
        """
        initial_memory = self.get_memory_usage()
        self.initial_memory = initial_memory
        
        logger.debug(f"Stage {stage_name} started - Memory: {initial_memory['used_gb']:.2f}GB")
        
        try:
            yield
        finally:
            # Force cleanup after stage
            self.cleanup_stage_memory(stage_name)
            
            final_memory = self.get_memory_usage()
            memory_diff = final_memory['used_gb'] - initial_memory['used_gb']
            
            self.stage_memory_usage[stage_name] = {
                'initial_gb': initial_memory['used_gb'],
                'final_gb': final_memory['used_gb'],
                'peak_gb': max(self.peak_memory_usage, final_memory['used_gb']),
                'difference_gb': memory_diff
            }
            
            logger.info(
                f"Stage {stage_name} completed - Memory change: {memory_diff:+.2f}GB "
                f"(final: {final_memory['used_gb']:.2f}GB)"
            )
    
    def cleanup_stage_memory(self, stage_name: str):
        """Explicit cleanup after stage completion"""
        logger.debug(f"Cleaning up memory for stage: {stage_name}")
        
        # Force garbage collection
        collected = gc.collect()
        
        # Additional cleanup for specific object types
        self._cleanup_pandas_memory()
        self._cleanup_numpy_memory()
        
        # Force another GC pass
        collected += gc.collect()
        
        if collected > 0:
            logger.debug(f"Garbage collected {collected} objects")
        
        # Check if we're approaching memory limit
        current_memory = self.get_memory_usage()
        if current_memory['percent'] > self.gc_threshold * 100:
            logger.warning(
                f"High memory usage detected: {current_memory['percent']:.1f}% "
                f"- Forcing aggressive cleanup"
            )
            self._aggressive_cleanup()
    
    def _cleanup_pandas_memory(self):
        """Cleanup pandas-specific memory"""
        # Clear any cached pandas operations
        if hasattr(pd, '_cache'):
            pd._cache.clear()
        
        # Force pandas internal cleanup
        try:
            # Clear pandas internal caches
            from pandas._libs import lib
            if hasattr(lib, 'cache_clear'):
                lib.cache_clear()
        except (ImportError, AttributeError):
            pass
    
    def _cleanup_numpy_memory(self):
        """Cleanup numpy-specific memory"""
        # Clear numpy internal caches
        try:
            np.core._methods._cleanup()
        except AttributeError:
            pass
        
        # Force numpy cleanup
        try:
            if hasattr(np, '_NoValue'):
                del np._NoValue
        except AttributeError:
            pass
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup when approaching limits"""
        logger.info("Performing aggressive memory cleanup...")
        
        # Multiple GC passes
        for i in range(3):
            collected = gc.collect()
            if collected == 0:
                break
            logger.debug(f"GC pass {i+1}: collected {collected} objects")
        
        # Clear all possible caches
        self._cleanup_pandas_memory()
        self._cleanup_numpy_memory()
        
        # Clear module-level caches if possible
        try:
            import sys
            for module_name, module in sys.modules.items():
                if hasattr(module, '__dict__') and hasattr(module, '_cache'):
                    try:
                        module._cache.clear()
                    except (AttributeError, TypeError):
                        pass
        except Exception:
            pass
    
    def optimize_dataframe_memory(self, df: pd.DataFrame, 
                                 aggressive: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        
        Args:
            df: DataFrame to optimize
            aggressive: Enable aggressive optimization (may lose precision)
            
        Returns:
            Memory-optimized DataFrame
        """
        if df.empty:
            return df
        
        initial_memory = df.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                # Unsigned integers
                if col_max <= 255:
                    df[col] = df[col].astype('uint8')
                elif col_max <= 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max <= 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                # Signed integers
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            if aggressive:
                # Check if we can use float32 without significant precision loss
                try:
                    df_test = df[col].astype('float32')
                    if np.allclose(df[col].dropna(), df_test.dropna(), rtol=1e-6):
                        df[col] = df_test
                except:
                    pass
        
        # Optimize object columns (strings)
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to category if it has few unique values
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                try:
                    df[col] = df[col].astype('category')
                except:
                    pass
        
        final_memory = df.memory_usage(deep=True).sum()
        memory_reduction = (initial_memory - final_memory) / initial_memory * 100
        
        if memory_reduction > 1:  # Only log if significant reduction
            logger.debug(
                f"DataFrame memory optimized: {memory_reduction:.1f}% reduction "
                f"({initial_memory/1024/1024:.1f}MB -> {final_memory/1024/1024:.1f}MB)"
            )
        
        return df
    
    def check_memory_usage(self, raise_on_limit: bool = False) -> Dict[str, Any]:
        """
        Check current memory usage against limits
        
        Args:
            raise_on_limit: Raise exception if over limit
            
        Returns:
            Memory usage information
        """
        memory_info = self.get_memory_usage()
        
        if memory_info['used'] > self.target_memory_limit:
            message = (
                f"Memory usage ({memory_info['used_gb']:.2f}GB) exceeds "
                f"target limit ({self.target_memory_limit/1024**3:.2f}GB)"
            )
            
            if raise_on_limit:
                raise MemoryError(message)
            else:
                logger.warning(message)
                
                # Trigger aggressive cleanup
                self._aggressive_cleanup()
        
        return memory_info
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3)
        }
    
    def get_stage_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report by stage"""
        total_memory_change = 0
        peak_usage = 0
        
        for stage, usage in self.stage_memory_usage.items():
            total_memory_change += usage['difference_gb']
            peak_usage = max(peak_usage, usage['peak_gb'])
        
        return {
            'stage_usage': self.stage_memory_usage,
            'total_memory_change_gb': total_memory_change,
            'peak_memory_usage_gb': peak_usage,
            'target_limit_gb': self.target_memory_limit / (1024**3),
            'current_usage': self.get_memory_usage(),
            'stages_processed': len(self.stage_memory_usage)
        }
    
    def clear_stage_data(self, variables_to_clear: List[str], local_vars: Dict[str, Any]):
        """
        Explicitly clear variables from memory
        
        Args:
            variables_to_clear: List of variable names to clear
            local_vars: Local variables dictionary (usually locals())
        """
        cleared_count = 0
        
        for var_name in variables_to_clear:
            if var_name in local_vars:
                try:
                    # For DataFrames, explicitly delete
                    if isinstance(local_vars[var_name], pd.DataFrame):
                        local_vars[var_name] = pd.DataFrame()
                    
                    del local_vars[var_name]
                    cleared_count += 1
                except:
                    pass
        
        if cleared_count > 0:
            logger.debug(f"Cleared {cleared_count} variables from memory")
            gc.collect()
    
    @staticmethod
    def reduce_dataframe_memory_usage(df: pd.DataFrame, 
                                    columns_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Reduce DataFrame memory by keeping only necessary columns
        
        Args:
            df: DataFrame to reduce
            columns_to_keep: List of columns to keep (if None, keeps all)
            
        Returns:
            Reduced DataFrame
        """
        if columns_to_keep:
            # Keep only specified columns
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns].copy()
        
        # Reset index to free up memory
        if df.index.name or not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=True)
        
        return df


# Global instance
_global_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


@contextmanager
def stage_memory_management(stage_name: str):
    """
    Convenience context manager for stage memory management
    
    Usage:
        with stage_memory_management("05_political_analysis"):
            # Processing code here
            pass
    """
    manager = get_memory_manager()
    with manager.stage_memory_context(stage_name):
        yield manager


def optimize_dataframe_memory(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
    """Convenience function for DataFrame memory optimization"""
    manager = get_memory_manager()
    return manager.optimize_dataframe_memory(df, aggressive)


def cleanup_memory():
    """Convenience function for explicit memory cleanup"""
    manager = get_memory_manager()
    manager._aggressive_cleanup()


def get_memory_report() -> Dict[str, Any]:
    """Convenience function to get memory usage report"""
    manager = get_memory_manager()
    return manager.get_stage_memory_report()