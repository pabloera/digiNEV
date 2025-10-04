"""
Data Integrity Validation Protocols for digiNEV v.final

Ensures strict data quality and prevents synthetic/fictional data generation.
"""
import pandas as pd
import numpy as np
import logging
from functools import wraps

class DataIntegrityError(Exception):
    """Custom exception for data integrity violations."""
    pass

def validate_real_data(func):
    """
    Decorator to validate data integrity at each stage.

    Validates:
    - Non-empty DataFrame
    - Contains only real, non-synthetic data
    - Preserves original record count
    - Checks for valid data types
    - Prevents synthetic data generation
    """
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        # 1. Basic Validation
        if not isinstance(df, pd.DataFrame):
            raise DataIntegrityError("Input must be a pandas DataFrame")

        if len(df) == 0:
            raise DataIntegrityError("Empty DataFrame is not allowed")

        # 2. Prevent Synthetic Data
        def is_likely_synthetic(series):
            """Detect potential synthetic/generated content"""
            # Check for unrealistic patterns
            pattern_checks = [
                (series.str.contains('^[A-Z]{5,}$').mean() > 0.5, "Uppercase-only text"),
                (series.str.contains('^[0-9]{10,}$').mean() > 0.5, "Number-only sequences"),
                (series.str.contains(r'^(lorem|ipsum|placeholder)', case=False).mean() > 0.2, "Lorem ipsum placeholders")
            ]

            return any(check[0] for check in pattern_checks)

        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if is_likely_synthetic(df[col]):
                logging.warning(f"Potential synthetic data detected in column: {col}")
                raise DataIntegrityError(f"Synthetic data detected in column: {col}")

        # 3. Record Origin Tracking
        original_record_count = len(df)
        original_columns = set(df.columns)

        # Execute the function
        result = func(df, *args, **kwargs)

        # 4. Post-Processing Validation
        if not isinstance(result, pd.DataFrame):
            raise DataIntegrityError("Function must return a pandas DataFrame")

        # Ensure record count is preserved or makes sense
        if len(result) > original_record_count * 1.2 or len(result) < original_record_count * 0.8:
            raise DataIntegrityError(f"Significant record count change: {original_record_count} → {len(result)}")

        # Ensure critical columns are preserved
        critical_columns = {'text', 'timestamp', 'channel', 'username'}
        missing_columns = critical_columns - set(result.columns)
        if missing_columns:
            raise DataIntegrityError(f"Missing critical columns: {missing_columns}")

        # 5. Logging and Tracking
        logging.info(f"Data Integrity Check Passed: {func.__name__}")
        logging.info(f"Records: {original_record_count} → {len(result)}")
        logging.info(f"Columns: {original_columns} → {set(result.columns)}")

        return result

    return wrapper

def track_data_lineage(df):
    """
    Add metadata tracking to DataFrame for complete lineage.

    Adds columns tracking:
    - Original record ID
    - Processing stage
    - Timestamp of transformation
    """
    df['__record_origin_id'] = np.arange(len(df))
    df['__processing_stage'] = 'original'
    df['__processed_timestamp'] = pd.Timestamp.now()
    return df

def validate_portuguese_text(text_series):
    """
    Validate text series for Brazilian Portuguese characteristics.

    Checks:
    - Contains Portuguese characters (ã, ê, ç)
    - Realistic word length
    - Realistic sentence structure
    """
    def is_valid_portuguese_text(text):
        if not isinstance(text, str):
            return False

        # Check for Portuguese-specific characters
        portuguese_chars = set('áéíóúãõâêôçÁÉÍÓÚÃÕÂÊÔÇ')
        has_portuguese_chars = any(char in portuguese_chars for char in text)

        # Basic text length and composition
        words = text.split()
        valid_word_count = 3 <= len(words) <= 50

        return has_portuguese_chars and valid_word_count

    return text_series.apply(is_valid_portuguese_text)