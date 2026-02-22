"""
stages — Modular extraction of TelegramAnalyzer pipeline stages.

Each stage_XX.py contains the logic previously embedded as methods
in analyzer.py. The main function of each stage module is named
_stage_XX_<description>(df) -> pd.DataFrame.

helpers.py holds shared utility functions used across stages
(emoji ratio, caps ratio, political keyword helpers, etc.).

Stage catalogue:
    stage_01  — Feature extraction & data ingestion
    stage_02  — Text pre-processing & feature validation
    stage_03  — Cross-dataset deduplication
    stage_04  — Statistical analysis
    stage_05  — Content quality filter
    stage_06  — Affordances classification (incl. OpenAI batch API)
    stage_07  — Linguistic processing (spaCy)
    stage_08  — Political classification + TCW integration
    stage_09  — TF-IDF vectorization
    stage_10  — Clustering analysis
    stage_11  — Topic modeling (LDA)
    stage_12  — Semantic analysis (sentiment, emotion)
    stage_13  — Temporal analysis
    stage_14  — Network analysis
    stage_15  — Domain analysis
    stage_16  — Event context detection
    stage_17  — Channel analysis
    helpers   — Shared utility functions
"""

# Stage main functions — importable via: from stages import STAGE_REGISTRY
from .stage_01 import _stage_01_feature_extraction
from .stage_02 import _stage_02_text_preprocessing
from .stage_03 import _stage_03_cross_dataset_deduplication
from .stage_04 import _stage_04_statistical_analysis
from .stage_05 import _stage_05_content_quality_filter
from .stage_06 import _stage_06_affordances_classification
from .stage_07 import _stage_07_linguistic_processing
from .stage_08 import _stage_08_political_classification
from .stage_09 import _stage_09_tfidf_vectorization
from .stage_10 import _stage_10_clustering_analysis
from .stage_11 import _stage_11_topic_modeling
from .stage_12 import _stage_12_semantic_analysis
from .stage_13 import _stage_13_temporal_analysis
from .stage_14 import _stage_14_network_analysis
from .stage_15 import _stage_15_domain_analysis
from .stage_16 import _stage_16_event_context
from .stage_17 import _stage_17_channel_analysis

# Ordered registry for pipeline orchestration
STAGE_REGISTRY = [
    ('01', 'feature_extraction', _stage_01_feature_extraction),
    ('02', 'text_preprocessing', _stage_02_text_preprocessing),
    ('03', 'cross_dataset_dedup', _stage_03_cross_dataset_deduplication),
    ('04', 'statistical_analysis', _stage_04_statistical_analysis),
    ('05', 'content_quality_filter', _stage_05_content_quality_filter),
    ('06', 'affordances_classification', _stage_06_affordances_classification),
    ('07', 'linguistic_processing', _stage_07_linguistic_processing),
    ('08', 'political_classification', _stage_08_political_classification),
    ('09', 'tfidf_vectorization', _stage_09_tfidf_vectorization),
    ('10', 'clustering_analysis', _stage_10_clustering_analysis),
    ('11', 'topic_modeling', _stage_11_topic_modeling),
    ('12', 'semantic_analysis', _stage_12_semantic_analysis),
    ('13', 'temporal_analysis', _stage_13_temporal_analysis),
    ('14', 'network_analysis', _stage_14_network_analysis),
    ('15', 'domain_analysis', _stage_15_domain_analysis),
    ('16', 'event_context', _stage_16_event_context),
    ('17', 'channel_analysis', _stage_17_channel_analysis),
]

# Helper functions
from .helpers import (
    _calculate_emoji_ratio,
    _calculate_caps_ratio,
    _calculate_repetition_ratio,
    _detect_portuguese,
    _classify_political_orientation,
    _extract_political_keywords,
    _calculate_political_intensity,
    _classify_domain_type,
    _calculate_domain_trust_score,
    _calculate_sentiment_polarity,
    _calculate_emotion_intensity,
    _detect_aggressive_language,
    _detect_political_context,
    _mentions_government,
    _mentions_opposition,
    _detect_election_context,
    _detect_protest_context,
    _classify_channel_type,
    _analyze_political_frames,
    _mann_kendall_trend_test,
    _detect_information_cascades,
)

__all__ = [
    'STAGE_REGISTRY',
    '_stage_01_feature_extraction',
    '_stage_02_text_preprocessing',
    '_stage_03_cross_dataset_deduplication',
    '_stage_04_statistical_analysis',
    '_stage_05_content_quality_filter',
    '_stage_06_affordances_classification',
    '_stage_07_linguistic_processing',
    '_stage_08_political_classification',
    '_stage_09_tfidf_vectorization',
    '_stage_10_clustering_analysis',
    '_stage_11_topic_modeling',
    '_stage_12_semantic_analysis',
    '_stage_13_temporal_analysis',
    '_stage_14_network_analysis',
    '_stage_15_domain_analysis',
    '_stage_16_event_context',
    '_stage_17_channel_analysis',
]
