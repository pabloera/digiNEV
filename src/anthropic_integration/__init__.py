"""
Módulo Anthropic para análise contextualizada - Pipeline Unificado
TDD Phase 3 - Minimal implementations for test passing
"""

# Import only what we've implemented for TDD
try:
    from .base import AnthropicBase
    from .unified_pipeline import UnifiedAnthropicPipeline
    from .voyage_embeddings import VoyageEmbeddings
except ImportError as e:
    # For TDD - provide minimal fallbacks
    AnthropicBase = None
    UnifiedAnthropicPipeline = None
    VoyageEmbeddings = None

# For future implementation
ClusterValidator = None
IntelligentTextCleaner = None
QualitativeClassifier = None
AnthropicSentimentAnalyzer = None
TopicInterpreter = None

__all__ = [
    'AnthropicBase',
    'UnifiedAnthropicPipeline', 
    'VoyageEmbeddings',
    'IntelligentTextCleaner',
    'AnthropicSentimentAnalyzer',
    'TopicInterpreter',
    'ClusterValidator',
    'QualitativeClassifier'
]
