"""
Robust CSV Parsing Utility for Dashboard
Based on the unified pipeline's robust CSV parsing logic
"""

import csv
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

class RobustCSVParser:
    """
    Robust CSV parser that implements the same separator detection and
    parsing logic as the unified pipeline to avoid header concatenation issues.
    """

    def __init__(self):
        # Set CSV field size limit to handle large fields
        csv.field_size_limit(500000)

    def detect_separator(self, file_path: str) -> str:
        """
        Detects CSV separator by analyzing the first line with robust validation
        Based on unified_pipeline.py detect_separator function
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                comma_count = first_line.count(',')
                semicolon_count = first_line.count(';')

                logger.debug(f"First line: {first_line[:100]}...")
                logger.debug(f"Commas: {comma_count}, Semicolons: {semicolon_count}")

                # If only 1 column detected, probably wrong separator
                if comma_count == 0 and semicolon_count == 0:
                    logger.warning("No separator detected in first line")
                    return ';'  # Default fallback for project datasets

                # Prioritize semicolon if equal or greater quantity
                if semicolon_count >= comma_count and semicolon_count > 0:
                    return ';'
                elif comma_count > 0:
                    return ','
                else:
                    logger.warning("Unable to determine separator, using default ';'")
                    return ';'

        except Exception as e:
            logger.error(f"Error detecting separator: {e}")
            return ';'  # Default fallback

    def parse_csv_robust(self, file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Attempts to parse CSV with different configurations and validation
        Uses the same fallback logic as the unified pipeline
        """
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return pd.DataFrame()

        # Detect separator
        separator = self.detect_separator(file_path)
        logger.info(f"Using separator: '{separator}' for file: {file_path}")

        # Attempt to parse with detected separator
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                logger.debug(f"Attempting to read with encoding: {encoding}")
                
                # Configure pandas read parameters
                read_params = {
                    'sep': separator,
                    'encoding': encoding,
                    'on_bad_lines': 'skip',
                    'low_memory': False,
                    'dtype': str,  # Read all as strings initially
                    'na_filter': False,  # Avoid automatic NA conversion
                }
                
                if max_rows:
                    read_params['nrows'] = max_rows
                
                df = pd.read_csv(file_path, **read_params)
                
                # Validation: check if DataFrame has reasonable structure
                if len(df.columns) < 2:
                    logger.warning(f"Only {len(df.columns)} columns found with separator '{separator}', trying comma")
                    read_params['sep'] = ','
                    df = pd.read_csv(file_path, **read_params)
                
                if len(df) == 0:
                    logger.warning("Empty DataFrame loaded")
                    return pd.DataFrame()
                
                logger.info(f"Successfully loaded CSV: {len(df)} rows, {len(df.columns)} columns")
                logger.debug(f"Columns: {list(df.columns[:5])}...")  # Show first 5 columns
                
                return df
                
            except Exception as e:
                logger.warning(f"Failed to read with {encoding}: {e}")
                continue
        
        logger.error(f"Unable to parse CSV file: {file_path}")
        return pd.DataFrame()

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Gets basic information about the CSV file
        """
        try:
            file_size = os.path.getsize(file_path)
            separator = self.detect_separator(file_path)
            
            # Try to get column count and row estimate
            try:
                df_sample = self.parse_csv_robust(file_path, max_rows=100)
                estimated_columns = len(df_sample.columns) if not df_sample.empty else 0
                
                # Estimate total rows based on file size and sample
                if not df_sample.empty and file_size > 0:
                    sample_size = len(df_sample)
                    bytes_per_row = file_size / max(sample_size * 10, 1)  # Rough estimate
                    estimated_rows = int(file_size / bytes_per_row)
                else:
                    estimated_rows = 0
                    
            except Exception:
                estimated_columns = 0
                estimated_rows = 0
            
            return {
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'separator': separator,
                'estimated_rows': estimated_rows,
                'estimated_columns': estimated_columns,
                'exists': True
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {
                'file_size_mb': 0,
                'separator': ';',
                'estimated_rows': 0,
                'estimated_columns': 0,
                'exists': False,
                'error': str(e)
            }

# Global instance for easy access
robust_csv_parser = RobustCSVParser()

def parse_csv_file(file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function to parse CSV files robustly
    
    Args:
        file_path: Path to CSV file
        max_rows: Maximum number of rows to read (optional)
        
    Returns:
        Parsed DataFrame or empty DataFrame if failed
    """
    return robust_csv_parser.parse_csv_robust(file_path, max_rows)

def get_csv_info(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to get CSV file information
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Dictionary with file information
    """
    return robust_csv_parser.get_file_info(file_path)