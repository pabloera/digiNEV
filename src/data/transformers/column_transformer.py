"""
Column Transformation Module

Consolidates column renaming, creation, and transformation operations
from scattered preprocessing scripts.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ColumnTransformer:
    """
    Unified column transformation operations.
    
    Consolidates functionality from:
    - rename_columns_telegram_text_analysis2.py
    - rename_columns_telegram_text_analysis2_robust.py
    - rename_contem_texto_to_has_txt.py
    - rename_nomes_canais_column.py
    - add_forwarded_column.py
    - add_fwd_from_column.py
    - create_fwd_source_column.py
    - process_binary_columns_classif1.py
    """
    
    def __init__(self):
        """Initialize column transformer."""
        self.column_mappings = {
            'contem_texto': 'has_txt',
            'nomes_canais': 'canal_names',
            'usuarios_canais': 'canal_users',
            'usuarios_canais_agregados': 'aggregated_users'
        }
    
    def rename_columns(self, df: pd.DataFrame, mappings: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Rename columns using provided mappings or default mappings.
        
        Args:
            df: Input DataFrame
            mappings: Custom column mappings (optional)
            
        Returns:
            DataFrame with renamed columns
        """
        if mappings is None:
            mappings = self.column_mappings
            
        # Apply mappings only for columns that exist
        existing_mappings = {old: new for old, new in mappings.items() if old in df.columns}
        
        if existing_mappings:
            logger.info(f"Renaming columns: {existing_mappings}")
            df = df.rename(columns=existing_mappings)
        
        return df
    
    def add_forwarded_column(self, df: pd.DataFrame, 
                           source_col: str = 'fwd_from',
                           target_col: str = 'forwarded') -> pd.DataFrame:
        """
        Add forwarded flag column based on fwd_from content.
        
        Args:
            df: Input DataFrame
            source_col: Column to check for forwarding info
            target_col: Name of new boolean column
            
        Returns:
            DataFrame with forwarded column added
        """
        if source_col not in df.columns:
            logger.warning(f"Source column '{source_col}' not found")
            df[target_col] = 0
            return df
        
        # Create forwarded flag (1 if has fwd_from, 0 otherwise)
        df[target_col] = df[source_col].notna().astype(int)
        
        logger.info(f"Added '{target_col}' column: {df[target_col].sum()} forwarded messages")
        return df
    
    def create_fwd_source_column(self, df: pd.DataFrame,
                               fwd_col: str = 'fwd_from',
                               source_col: str = 'fwd_source') -> pd.DataFrame:
        """
        Create standardized forwarding source column.
        
        Args:
            df: Input DataFrame
            fwd_col: Column with forwarding information
            source_col: Name of new source column
            
        Returns:
            DataFrame with standardized fwd_source column
        """
        if fwd_col not in df.columns:
            logger.warning(f"Forwarding column '{fwd_col}' not found")
            df[source_col] = ''
            return df
        
        # Clean and standardize forwarding sources
        df[source_col] = df[fwd_col].fillna('').astype(str).str.strip().str.lower()
        
        logger.info(f"Created '{source_col}' column with {df[source_col].str.len().gt(0).sum()} sources")
        return df
    
    def process_binary_columns(self, df: pd.DataFrame, 
                             binary_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Standardize binary columns (convert to 0/1).
        
        Args:
            df: Input DataFrame
            binary_columns: List of columns to process as binary
            
        Returns:
            DataFrame with standardized binary columns
        """
        if binary_columns is None:
            # Auto-detect binary-like columns
            binary_columns = [col for col in df.columns 
                            if col.startswith(('has_', 'is_', 'contains_', 'tem_'))]
        
        for col in binary_columns:
            if col in df.columns:
                # Convert various formats to 0/1
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                logger.debug(f"Processed binary column '{col}': {df[col].sum()} positive values")
        
        if binary_columns:
            logger.info(f"Processed {len(binary_columns)} binary columns")
        
        return df
    
    def convert_timestamp(self, df: pd.DataFrame,
                         timestamp_col: str = 'timestamp',
                         format_str: Optional[str] = None) -> pd.DataFrame:
        """
        Convert timestamp column to standardized datetime format.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            format_str: Specific format string (auto-detect if None)
            
        Returns:
            DataFrame with converted timestamp
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return df
        
        try:
            if format_str:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=format_str)
            else:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True)
            
            logger.info(f"Converted timestamp column '{timestamp_col}' to datetime")
            
        except Exception as e:
            logger.error(f"Failed to convert timestamp: {e}")
        
        return df
    
    def apply_all_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all standard transformations in sequence.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Fully transformed DataFrame
        """
        logger.info("Applying all column transformations")
        
        # Apply transformations in order
        df = self.rename_columns(df)
        df = self.add_forwarded_column(df)
        df = self.create_fwd_source_column(df)
        df = self.process_binary_columns(df)
        df = self.convert_timestamp(df)
        
        logger.info("All column transformations completed")
        return df


def transform_telegram_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to apply all Telegram-specific column transformations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Transformed DataFrame
    """
    transformer = ColumnTransformer()
    return transformer.apply_all_transformations(df)