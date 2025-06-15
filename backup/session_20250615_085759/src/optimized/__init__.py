"""
Optimized Components for Monitor do Discurso Digital Pipeline
==============================================

Critical performance optimizations to resolve pipeline bottlenecks:

- EmergencyEmbeddingsCache: Eliminates 4x redundancy in Voyage.ai embeddings
- File dependency fixes
- Import error corrections
- Performance monitoring

Status: Week 1 Critical Fixes Implementation
"""

from .emergency_embeddings import (
    EmergencyEmbeddingsCache,
    VoyageEmbeddingsCacheIntegration,
    create_emergency_embeddings_cache,
    get_global_embeddings_cache
)

__all__ = [
    'EmergencyEmbeddingsCache',
    'VoyageEmbeddingsCacheIntegration', 
    'create_emergency_embeddings_cache',
    'get_global_embeddings_cache'
]