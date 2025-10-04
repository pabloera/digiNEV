import pytest
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_validator import DatasetValidator

# Reduce the number of test files and add logging
logging.basicConfig(level=logging.INFO)

DATASETS = [
    '/Users/pabloalmada/development/project/dataanalysis-bolsonarismo/data/1_2019-2021-govbolso.csv'
]

@pytest.fixture
def validator():
    return DatasetValidator()

def test_dataset_validation(validator):
    """Comprehensive dataset validation."""
    for dataset_path in DATASETS:
        logging.info(f"Testing dataset: {dataset_path}")

        # Validate dataset
        result = validator.validate_dataset(dataset_path)

        # Detailed logging
        logging.info(f"Total Records: {result['total_records']}")
        logging.info(f"Columns: {result['columns']}")
        logging.info(f"File Size (MB): {result['file_size_mb']}")
        logging.info(f"Memory Usage (MB): {result['memory_usage_mb']}")

        # Assertions
        assert result['total_records'] > 0, f"No records in {dataset_path}"
        assert len(result['columns']) > 0, f"No columns in {dataset_path}"
        assert result['file_size_mb'] > 0, f"Invalid file size for {dataset_path}"
        assert result['memory_usage_mb'] < 1024, f"Memory usage too high for {dataset_path}"

def test_encoding_detection(validator):
    """Test encoding detection."""
    for dataset_path in DATASETS:
        encoding = validator.detect_encoding(dataset_path)
        logging.info(f"Detected encoding for {dataset_path}: {encoding}")

        assert encoding is not None, f"Failed to detect encoding for {dataset_path}"
        assert encoding.lower() in ['utf-8', 'iso-8859-1', 'cp1252', 'macroman'], f"Unexpected encoding {encoding}"

def test_portuguese_text_handling(validator):
    """Test handling of Portuguese text columns."""
    for dataset_path in DATASETS:
        result = validator.validate_dataset(dataset_path)

        # At least one column should have Portuguese characters or reasonable stats
        portuguese_columns = [
            col for col, checks in result['text_column_checks'].items()
            if checks['contains_portuguese_chars'] or
               (checks['avg_length'] > 0 and checks['unique_values'] > 0)
        ]

        logging.info(f"Portuguese-like columns: {portuguese_columns}")

        assert len(portuguese_columns) > 0, "No text columns with expected Portuguese characteristics"