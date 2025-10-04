import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import os
import chardet
import re

class DatasetValidator:
    def __init__(self, log_file='/tmp/dataset_validation.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger()

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        with open(file_path, 'rb') as file:
            result = chardet.detect(file.read(10000))  # Read first 10KB
        return result['encoding']

    def validate_dataset(self, file_path: str) -> Dict[str, Any]:
        """Lightweight dataset validation."""
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Basic file stats
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # Detect encoding
        detected_encoding = self.detect_encoding(file_path)
        self.logger.info(f"Detected encoding for {file_path}: {detected_encoding}")

        # Read first few rows to validate structure with robust parsing
        for encoding in [detected_encoding, 'latin1', 'utf-8']:
            try:
                # Use iterator to process in chunks and handle bad lines
                chunks = []
                for chunk in pd.read_csv(
                    file_path,
                    sep=';',
                    encoding=encoding,
                    nrows=1000,
                    chunksize=200,
                    on_bad_lines='skip'  # New method for handling bad lines
                ):
                    chunks.append(chunk)

                # Combine chunks
                df = pd.concat(chunks, ignore_index=True)

                if len(df) > 0:
                    break  # Successfully read the data
            except Exception as e:
                self.logger.warning(f"Failed to read with {encoding} encoding: {e}")
                continue

        if len(df) == 0:
            raise ValueError("Unable to read dataset with any encoding")

        # Lightweight validation checks
        validation_results = {
            'filename': os.path.basename(file_path),
            'total_records': len(df),
            'columns': list(df.columns),
            'file_size_mb': file_size_mb,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'dtype_summary': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'text_column_checks': self._validate_text_columns(df)
        }

        # Log basic validation results
        self._log_validation_results(validation_results)

        return validation_results

    def _validate_text_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform text column specific lightweight validations."""
        def contains_portuguese_chars(series):
            """Check if any row in the series contains Portuguese characters."""
            if series.dtype != 'object':
                return False  # Not a text series

            pattern = r'[áéíóúãõçÁÉÍÓÚÃÕÇ]'
            # Drop NaN and convert to string to avoid errors
            text_series = series.dropna().astype(str)
            return any(text_series.str.contains(pattern, regex=True, case=False))

        text_columns = df.select_dtypes(include=['object']).columns
        text_checks = {}

        for col in text_columns:
            text_checks[col] = {
                'unique_values': len(df[col].dropna().unique()),
                'avg_length': df[col].dropna().astype(str).str.len().mean(),
                'contains_portuguese_chars': contains_portuguese_chars(df[col]),
                'empty_ratio': (df[col].isna() | (df[col] == '')).mean(),
                'total_records': len(df)
            }

        return text_checks

    def _log_validation_results(self, results: Dict[str, Any]):
        """Log basic validation results."""
        self.logger.info(f"Dataset Validation for {results['filename']}")
        self.logger.info(f"Total Records: {results['total_records']}")
        self.logger.info(f"File Size: {results['file_size_mb']:.2f} MB")
        self.logger.info(f"Memory Usage: {results['memory_usage_mb']:.2f} MB")

def main():
    # Sample usage
    validator = DatasetValidator()
    try:
        result = validator.validate_dataset('/path/to/your/dataset.csv')
        print(result)
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()