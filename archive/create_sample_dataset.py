#!/usr/bin/env python3
"""
Create a random sample dataset of 1000 cases from 5 source datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a random sample of 1000 cases from 5 datasets."""

    data_dir = Path("batch_analyzer/data")

    # Define the dataset files
    datasets = [
        "1_2019-2021-govbolso.csv",
        "2_2021-2022-pandemia.csv",
        "3_2022-2023-poseleic.csv",
        "4_2022-2023-elec.csv",
        "5_2022-2023-elec-extra.csv"
    ]

    all_samples = []

    # Calculate samples per dataset (proportional or fixed)
    # Using fixed 200 samples per dataset for balanced representation
    samples_per_dataset = 200

    for dataset_file in datasets:
        filepath = data_dir / dataset_file

        if not filepath.exists():
            logger.warning(f"Dataset not found: {filepath}")
            continue

        logger.info(f"Processing: {dataset_file}")

        try:
            # Read dataset - using comma separator based on file inspection
            # For large files, we'll use chunking to handle memory efficiently

            # First, get a small sample to check the structure
            sample_check = pd.read_csv(filepath, encoding='utf-8', nrows=5)
            logger.info(f"  Columns: {len(sample_check.columns)}")

            # Get total rows for sampling
            with open(filepath, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # -1 for header
            logger.info(f"  Total rows: {total_rows:,}")

            if total_rows <= samples_per_dataset:
                # If dataset is smaller than sample size, take all
                df = pd.read_csv(filepath, encoding='utf-8')
                sample = df
            else:
                # For large datasets, read in chunks and sample
                chunk_size = 50000
                chunks_to_sample = []

                for chunk in pd.read_csv(filepath, encoding='utf-8', chunksize=chunk_size):
                    # Sample proportionally from each chunk
                    chunk_sample_size = int((len(chunk) / total_rows) * samples_per_dataset) + 1
                    if chunk_sample_size > 0 and len(chunk) > 0:
                        chunk_sample = chunk.sample(n=min(chunk_sample_size, len(chunk)),
                                                   random_state=42)
                        chunks_to_sample.append(chunk_sample)

                # Combine all chunk samples
                df = pd.concat(chunks_to_sample, ignore_index=True)

                # Final sampling to get exact number
                if len(df) > samples_per_dataset:
                    sample = df.sample(n=samples_per_dataset, random_state=42)
                else:
                    sample = df

            # Add source dataset column
            sample['source_dataset'] = dataset_file

            all_samples.append(sample)
            logger.info(f"  Sampled {len(sample)} rows")

        except Exception as e:
            logger.error(f"Error processing {dataset_file}: {e}")
            continue

    if not all_samples:
        logger.error("No samples collected!")
        return

    # Combine all samples
    combined_df = pd.concat(all_samples, ignore_index=True)

    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the sampled dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = data_dir / f"sample_1000_cases_{timestamp}.csv"

    combined_df.to_csv(output_file, index=False, encoding='utf-8')

    logger.info(f"\n✅ Sample dataset created successfully!")
    logger.info(f"   Output: {output_file}")
    logger.info(f"   Total samples: {len(combined_df)}")
    logger.info(f"\n   Distribution by source:")
    for source, count in combined_df['source_dataset'].value_counts().items():
        logger.info(f"     {source}: {count}")

    # Save metadata
    metadata = {
        'created_at': timestamp,
        'total_samples': len(combined_df),
        'source_distribution': combined_df['source_dataset'].value_counts().to_dict(),
        'columns': combined_df.columns.tolist(),
        'shape': combined_df.shape
    }

    metadata_file = data_dir / f"sample_1000_cases_{timestamp}_metadata.txt"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("Sample Dataset Metadata\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {metadata['total_samples']}\n")
        f.write(f"Shape: {metadata['shape']}\n\n")
        f.write("Source distribution:\n")
        for source, count in metadata['source_distribution'].items():
            f.write(f"  - {source}: {count}\n")
        f.write(f"\nColumns ({len(metadata['columns'])}):\n")
        for col in metadata['columns']:
            f.write(f"  - {col}\n")

    logger.info(f"   Metadata: {metadata_file}")

    return output_file

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    create_sample_dataset()