#!/usr/bin/env python3
"""
Data Processing Utils - TASK-018 Implementation
===============================================

Common data processing utilities to eliminate 80% algorithm duplication
across multiple processing files.

Consolidates common patterns from:
- Various preprocessing modules
- Data cleaning utilities
- Feature extraction functions
- Statistical analysis helpers

Author: Pablo Emanuel Romero Almada, Ph.D.
Date: 2025-06-14
Version: 5.0.0
"""

import logging
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataProcessingUtils:
    """
    Common data processing utilities for the pipeline
    """
    
    @staticmethod
    def safe_divide(numerator: Union[int, float], 
                   denominator: Union[int, float], 
                   default: float = 0.0) -> float:
        """
        Safe division with default value for zero denominator
        
        Args:
            numerator: Numerator value
            denominator: Denominator value  
            default: Default value when denominator is zero
            
        Returns:
            Division result or default value
        """
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except (TypeError, ZeroDivisionError):
            return default
    
    @staticmethod
    def normalize_text(text: str, 
                      preserve_case: bool = False,
                      remove_accents: bool = False,
                      normalize_unicode: bool = True) -> str:
        """
        Normalize text with various options
        
        Args:
            text: Text to normalize
            preserve_case: Keep original case
            remove_accents: Remove accent marks
            normalize_unicode: Apply Unicode normalization
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Unicode normalization
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove accents
        if remove_accents:
            text = ''.join(
                char for char in unicodedata.normalize('NFD', text)
                if unicodedata.category(char) != 'Mn'
            )
        
        # Case normalization
        if not preserve_case:
            text = text.lower()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_text_features(text: str) -> Dict[str, Any]:
        """
        Extract common text features
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text features
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Basic features
        char_count = len(text)
        word_count = len(text.split()) if text.strip() else 0
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        # Character type counts
        uppercase_count = sum(1 for c in text if c.isupper())
        lowercase_count = sum(1 for c in text if c.islower())
        digit_count = sum(1 for c in text if c.isdigit())
        punctuation_count = sum(1 for c in text if c in '.,!?;:"\'()[]{}')
        
        # Special content
        url_count = len(re.findall(r'https?://\S+', text))
        mention_count = len(re.findall(r'@\w+', text))
        hashtag_count = len(re.findall(r'#\w+', text))
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', text))
        
        # Calculated features
        avg_word_length = DataProcessingUtils.safe_divide(
            sum(len(word) for word in text.split()), word_count
        )
        
        uppercase_ratio = DataProcessingUtils.safe_divide(uppercase_count, char_count)
        punctuation_ratio = DataProcessingUtils.safe_divide(punctuation_count, char_count)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'uppercase_count': uppercase_count,
            'lowercase_count': lowercase_count,
            'digit_count': digit_count,
            'punctuation_count': punctuation_count,
            'uppercase_ratio': uppercase_ratio,
            'punctuation_ratio': punctuation_ratio,
            'url_count': url_count,
            'mention_count': mention_count,
            'hashtag_count': hashtag_count,
            'emoji_count': emoji_count
        }
    
    @staticmethod
    def calculate_text_quality_score(text: str) -> float:
        """
        Calculate text quality score (0-1)
        
        Args:
            text: Text to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0
        
        features = DataProcessingUtils.extract_text_features(text)
        
        # Quality indicators
        has_reasonable_length = 10 <= features['char_count'] <= 5000
        has_words = features['word_count'] >= 3
        reasonable_avg_word_length = 2 <= features['avg_word_length'] <= 15
        not_too_much_uppercase = features['uppercase_ratio'] <= 0.5
        has_some_punctuation = features['punctuation_count'] > 0
        
        # Calculate score
        quality_indicators = [
            has_reasonable_length,
            has_words,
            reasonable_avg_word_length,
            not_too_much_uppercase,
            has_some_punctuation
        ]
        
        base_score = sum(quality_indicators) / len(quality_indicators)
        
        # Bonus for good characteristics
        if features['sentence_count'] > 0:
            base_score += 0.1
        
        if 50 <= features['char_count'] <= 1000:  # Sweet spot
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    @staticmethod
    def detect_language_indicators(text: str) -> Dict[str, int]:
        """
        Detect language indicators in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language indicator counts
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        text_lower = text.lower()
        
        # Portuguese indicators
        portuguese_words = [
            'que', 'não', 'uma', 'com', 'para', 'são', 'dos', 'mais', 'como',
            'mas', 'foi', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre',
            'quando', 'muito', 'depois', 'sem', 'mesmo', 'pode', 'ainda',
            'também', 'só', 'já', 'onde', 'vez', 'todos', 'bem', 'cada'
        ]
        
        portuguese_chars = ['ç', 'ã', 'õ', 'ê', 'â', 'ô', 'á', 'é', 'í', 'ó', 'ú']
        
        # Count Portuguese indicators
        portuguese_word_count = sum(1 for word in portuguese_words if word in text_lower)
        portuguese_char_count = sum(1 for char in portuguese_chars if char in text_lower)
        
        return {
            'portuguese_words': portuguese_word_count,
            'portuguese_chars': portuguese_char_count,
            'likely_portuguese': portuguese_word_count >= 2 or portuguese_char_count >= 1
        }
    
    @staticmethod
    def calculate_similarity_metrics(texts: List[str]) -> Dict[str, float]:
        """
        Calculate similarity metrics for a list of texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with similarity metrics
        """
        if len(texts) < 2:
            return {'avg_jaccard': 0.0, 'max_jaccard': 0.0, 'similarity_score': 0.0}
        
        def jaccard_similarity(text1: str, text2: str) -> float:
            """Calculate Jaccard similarity between two texts"""
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return DataProcessingUtils.safe_divide(intersection, union)
        
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = jaccard_similarity(texts[i], texts[j])
                similarities.append(similarity)
        
        if not similarities:
            return {'avg_jaccard': 0.0, 'max_jaccard': 0.0, 'similarity_score': 0.0}
        
        avg_jaccard = np.mean(similarities)
        max_jaccard = np.max(similarities)
        
        # Calculate overall similarity score
        similarity_score = avg_jaccard
        if max_jaccard > 0.8:  # High similarity detected
            similarity_score += 0.2
        
        return {
            'avg_jaccard': avg_jaccard,
            'max_jaccard': max_jaccard,
            'similarity_score': min(similarity_score, 1.0)
        }
    
    @staticmethod
    def calculate_distribution_stats(values: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate distribution statistics for numerical values
        
        Args:
            values: List of numerical values
            
        Returns:
            Dictionary with statistical measures
        """
        if not values:
            return {
                'count': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0,
                'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0
            }
        
        values = [v for v in values if v is not None and not np.isnan(v)]
        
        if not values:
            return {
                'count': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0,
                'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0
            }
        
        values_array = np.array(values)
        
        try:
            return {
                'count': len(values),
                'mean': float(np.mean(values_array)),
                'median': float(np.median(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'q25': float(np.percentile(values_array, 25)),
                'q75': float(np.percentile(values_array, 75)),
                'skewness': float(stats.skew(values_array)),
                'kurtosis': float(stats.kurtosis(values_array))
            }
        except Exception as e:
            logger.warning(f"Error calculating distribution stats: {e}")
            return {
                'count': len(values), 'mean': 0.0, 'median': 0.0, 'std': 0.0,
                'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0
            }
    
    @staticmethod
    def create_frequency_analysis(items: List[str], top_n: int = 20) -> Dict[str, Any]:
        """
        Create frequency analysis for categorical data
        
        Args:
            items: List of items to analyze
            top_n: Number of top items to return
            
        Returns:
            Dictionary with frequency analysis
        """
        if not items:
            return {'total_count': 0, 'unique_count': 0, 'top_items': [], 'frequency_distribution': {}}
        
        # Clean and normalize items
        clean_items = [str(item).strip().lower() for item in items if item and str(item).strip()]
        
        if not clean_items:
            return {'total_count': 0, 'unique_count': 0, 'top_items': [], 'frequency_distribution': {}}
        
        # Count frequencies
        counter = Counter(clean_items)
        
        # Get top items
        top_items = counter.most_common(top_n)
        
        # Calculate distribution stats
        frequencies = list(counter.values())
        frequency_stats = DataProcessingUtils.calculate_distribution_stats(frequencies)
        
        return {
            'total_count': len(clean_items),
            'unique_count': len(counter),
            'diversity_ratio': DataProcessingUtils.safe_divide(len(counter), len(clean_items)),
            'top_items': top_items,
            'frequency_distribution': frequency_stats
        }
    
    @staticmethod
    def detect_outliers(values: List[Union[int, float]], 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers in numerical data
        
        Args:
            values: List of numerical values
            method: Method to use ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        if not values:
            return {'outliers': [], 'outlier_indices': [], 'outlier_count': 0, 'outlier_ratio': 0.0}
        
        clean_values = [v for v in values if v is not None and not np.isnan(v)]
        
        if len(clean_values) < 3:
            return {'outliers': [], 'outlier_indices': [], 'outlier_count': 0, 'outlier_ratio': 0.0}
        
        values_array = np.array(clean_values)
        outlier_mask = np.zeros(len(clean_values), dtype=bool)
        
        try:
            if method == 'iqr':
                q25, q75 = np.percentile(values_array, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - threshold * iqr
                upper_bound = q75 + threshold * iqr
                outlier_mask = (values_array < lower_bound) | (values_array > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(values_array))
                outlier_mask = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = np.median(values_array)
                mad = np.median(np.abs(values_array - median))
                modified_z_scores = 0.6745 * (values_array - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
            
            outlier_indices = np.where(outlier_mask)[0].tolist()
            outliers = values_array[outlier_mask].tolist()
            
            return {
                'outliers': outliers,
                'outlier_indices': outlier_indices,
                'outlier_count': len(outliers),
                'outlier_ratio': DataProcessingUtils.safe_divide(len(outliers), len(clean_values))
            }
            
        except Exception as e:
            logger.warning(f"Error detecting outliers: {e}")
            return {'outliers': [], 'outlier_indices': [], 'outlier_count': 0, 'outlier_ratio': 0.0}
    
    @staticmethod
    def validate_dataframe_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame quality and completeness
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {
                'total_rows': 0, 'total_columns': 0, 'missing_data_ratio': 0.0,
                'duplicate_ratio': 0.0, 'quality_score': 0.0, 'issues': ['Empty DataFrame']
            }
        
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Calculate ratios
        missing_ratio = DataProcessingUtils.safe_divide(missing_cells, total_cells)
        duplicate_ratio = DataProcessingUtils.safe_divide(duplicate_rows, len(df))
        
        # Identify issues
        issues = []
        if missing_ratio > 0.1:
            issues.append(f"High missing data: {missing_ratio:.1%}")
        if duplicate_ratio > 0.05:
            issues.append(f"High duplicate ratio: {duplicate_ratio:.1%}")
        if len(df) < 100:
            issues.append("Small dataset size")
        
        # Calculate quality score
        quality_score = 1.0
        quality_score -= min(missing_ratio * 2, 0.5)  # Penalize missing data
        quality_score -= min(duplicate_ratio * 3, 0.3)  # Penalize duplicates
        quality_score = max(quality_score, 0.0)
        
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_ratio': missing_ratio,
            'duplicate_ratio': duplicate_ratio,
            'quality_score': quality_score,
            'issues': issues,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }


# Convenience functions for common operations
def safe_divide(num: Union[int, float], den: Union[int, float], default: float = 0.0) -> float:
    """Convenience function for safe division"""
    return DataProcessingUtils.safe_divide(num, den, default)


def normalize_text(text: str, **kwargs) -> str:
    """Convenience function for text normalization"""
    return DataProcessingUtils.normalize_text(text, **kwargs)


def extract_text_features(text: str) -> Dict[str, Any]:
    """Convenience function for text feature extraction"""
    return DataProcessingUtils.extract_text_features(text)


def calculate_text_quality(text: str) -> float:
    """Convenience function for text quality calculation"""
    return DataProcessingUtils.calculate_text_quality_score(text)


def frequency_analysis(items: List[str], top_n: int = 20) -> Dict[str, Any]:
    """Convenience function for frequency analysis"""
    return DataProcessingUtils.create_frequency_analysis(items, top_n)


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function for DataFrame validation"""
    return DataProcessingUtils.validate_dataframe_quality(df)