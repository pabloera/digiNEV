"""
Common utilities package for the Monitor do Discurso Digital project
Consolidates shared functionality to eliminate code duplication
"""

from .data_processor import DataProcessingUtils, clean_memory, validate_df_structure
from .api_constants import (
    TOKEN_PRICES, ANTHROPIC_MODELS, VOYAGE_MODELS, STAGE_MODEL_MAPPING,
    COST_LIMITS, RATE_LIMITS, FALLBACK_STRATEGIES, QUALITY_PROFILES,
    get_model_price, calculate_request_cost, get_recommended_model, is_within_cost_limits
)
from .logging_mixin import LoggingMixin, get_standard_logger, log_system_info
from .compression_utils import CompressionUtils, save_csv_optimized, load_csv_auto
from .config_loader import (
    ConfigurationLoader, get_config_loader, get_config, get_config_value,
    get_model_setting, get_path_config
)

__all__ = [
    # Data processing utilities
    'DataProcessingUtils',
    'clean_memory', 
    'validate_df_structure',
    
    # API constants and utilities
    'TOKEN_PRICES',
    'ANTHROPIC_MODELS', 
    'VOYAGE_MODELS',
    'STAGE_MODEL_MAPPING',
    'COST_LIMITS',
    'RATE_LIMITS', 
    'FALLBACK_STRATEGIES',
    'QUALITY_PROFILES',
    'get_model_price',
    'calculate_request_cost',
    'get_recommended_model',
    'is_within_cost_limits',
    
    # Logging utilities
    'LoggingMixin',
    'get_standard_logger',
    'log_system_info',
    
    # Compression utilities
    'CompressionUtils',
    'save_csv_optimized',
    'load_csv_auto',
    
    # Configuration utilities
    'ConfigurationLoader',
    'get_config_loader',
    'get_config',
    'get_config_value',
    'get_model_setting',
    'get_path_config'
]