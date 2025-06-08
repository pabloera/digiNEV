"""
Text Transformation Module

Consolidates text processing operations from scattered preprocessing scripts.
"""

import pandas as pd
import re
import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class TextTransformer:
    """
    Unified text transformation operations.
    
    Consolidates functionality from:
    - replace_linebreaks_chunks.py
    - replace_linebreaks_text_column.py
    - standardize_hashtags_lowercase.py
    - standardize_urls.py
    """
    
    def __init__(self):
        """Initialize text transformer."""
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.hashtag_pattern = re.compile(r'#\w+')
    
    def replace_linebreaks(self, text: Union[str, pd.Series], 
                          replacement: str = ' ') -> Union[str, pd.Series]:
        """
        Replace line breaks with specified replacement.
        
        Args:
            text: Input text or Series
            replacement: String to replace line breaks with
            
        Returns:
            Text with line breaks replaced
        """
        if isinstance(text, pd.Series):
            return text.astype(str).str.replace(r'[\r\n]+', replacement, regex=True)
        else:
            return str(text).replace('\n', replacement).replace('\r', replacement)
    
    def standardize_hashtags(self, text: Union[str, pd.Series]) -> Union[str, pd.Series]:
        """
        Standardize hashtags to lowercase.
        
        Args:
            text: Input text or Series
            
        Returns:
            Text with standardized hashtags
        """
        def _standardize_hashtag(match):
            return match.group().lower()
        
        if isinstance(text, pd.Series):
            return text.astype(str).apply(
                lambda x: self.hashtag_pattern.sub(_standardize_hashtag, x)
            )
        else:
            return self.hashtag_pattern.sub(_standardize_hashtag, str(text))
    
    def standardize_urls(self, text: Union[str, pd.Series]) -> Union[str, pd.Series]:
        """
        Standardize URLs to lowercase.
        
        Args:
            text: Input text or Series
            
        Returns:
            Text with standardized URLs
        """
        def _standardize_url(match):
            return match.group().lower()
        
        if isinstance(text, pd.Series):
            return text.astype(str).apply(
                lambda x: self.url_pattern.sub(_standardize_url, x)
            )
        else:
            return self.url_pattern.sub(_standardize_url, str(text))
    
    def extract_hashtags(self, text: Union[str, pd.Series]) -> Union[List[str], pd.Series]:
        """
        Extract hashtags from text.
        
        Args:
            text: Input text or Series
            
        Returns:
            List of hashtags or Series of hashtag lists
        """
        if isinstance(text, pd.Series):
            return text.astype(str).apply(
                lambda x: self.hashtag_pattern.findall(x.lower())
            )
        else:
            return self.hashtag_pattern.findall(str(text).lower())
    
    def extract_urls(self, text: Union[str, pd.Series]) -> Union[List[str], pd.Series]:
        """
        Extract URLs from text.
        
        Args:
            text: Input text or Series
            
        Returns:
            List of URLs or Series of URL lists
        """
        if isinstance(text, pd.Series):
            return text.astype(str).apply(
                lambda x: self.url_pattern.findall(x)
            )
        else:
            return self.url_pattern.findall(str(text))
    
    def clean_text_column(self, df: pd.DataFrame, 
                         text_column: str = 'text',
                         clean_linebreaks: bool = True,
                         standardize_hashtags: bool = True,
                         standardize_urls: bool = True) -> pd.DataFrame:
        """
        Apply all text cleaning operations to specified column.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to clean
            clean_linebreaks: Whether to replace line breaks
            standardize_hashtags: Whether to standardize hashtags
            standardize_urls: Whether to standardize URLs
            
        Returns:
            DataFrame with cleaned text column
        """
        if text_column not in df.columns:
            logger.warning(f"Text column '{text_column}' not found")
            return df
        
        logger.info(f"Cleaning text column '{text_column}'")
        
        if clean_linebreaks:
            df[text_column] = self.replace_linebreaks(df[text_column])
            logger.debug("Replaced line breaks")
        
        if standardize_hashtags:
            df[text_column] = self.standardize_hashtags(df[text_column])
            logger.debug("Standardized hashtags")
        
        if standardize_urls:
            df[text_column] = self.standardize_urls(df[text_column])
            logger.debug("Standardized URLs")
        
        logger.info(f"Text cleaning completed for '{text_column}'")
        return df
    
    def extract_text_features(self, df: pd.DataFrame,
                            text_column: str = 'text') -> pd.DataFrame:
        """
        Extract hashtags and URLs as separate columns.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            
        Returns:
            DataFrame with extracted features
        """
        if text_column not in df.columns:
            logger.warning(f"Text column '{text_column}' not found")
            return df
        
        logger.info(f"Extracting features from '{text_column}'")
        
        # Extract hashtags
        df['hashtags'] = self.extract_hashtags(df[text_column])
        df['hashtags'] = df['hashtags'].apply(lambda x: ','.join(x) if x else '')
        
        # Extract URLs
        df['urls'] = self.extract_urls(df[text_column])
        df['urls'] = df['urls'].apply(lambda x: ','.join(x) if x else '')
        
        # Add feature flags
        df['has_hashtags'] = (df['hashtags'].str.len() > 0).astype(int)
        df['has_urls'] = (df['urls'].str.len() > 0).astype(int)
        
        logger.info("Feature extraction completed")
        return df


def clean_telegram_text(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Convenience function to apply all text transformations for Telegram data.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        
    Returns:
        DataFrame with cleaned text and extracted features
    """
    transformer = TextTransformer()
    
    # Clean text
    df = transformer.clean_text_column(df, text_column)
    
    # Extract features
    df = transformer.extract_text_features(df, text_column)
    
    return df