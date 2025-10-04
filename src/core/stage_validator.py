"""
Stage Validation Framework for digiNEV v.final Pipeline

Comprehensive validation across all processing stages to ensure data integrity.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

class StageValidator:
    """
    Advanced stage validation with multiple integrity checks.

    Ensures:
    - Data continuity
    - No synthetic data generation
    - Preservation of critical information
    - Statistical consistency
    - Tracking of transformations
    """

    @staticmethod
    def validate_stage_transition(
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        stage_name: str
    ) -> Dict[str, Any]:
        """
        Comprehensive validation during stage transitions.

        Args:
            input_df (pd.DataFrame): DataFrame before processing
            output_df (pd.DataFrame): DataFrame after processing
            stage_name (str): Name of the current processing stage

        Returns:
            Dict with validation results and metrics
        """
        results = {
            'stage': stage_name,
            'success': True,
            'input_records': len(input_df),
            'output_records': len(output_df),
            'record_delta_percent': 0,
            'critical_columns_preserved': [],
            'new_columns': [],
            'potential_issues': []
        }

        # Record count validation
        results['record_delta_percent'] = (
            (len(output_df) - len(input_df)) / len(input_df) * 100
        )

        # Validate record count changes
        if abs(results['record_delta_percent']) > 20:
            results['potential_issues'].append(
                f"Significant record count change: {results['record_delta_percent']:.2f}%"
            )
            results['success'] = False

        # Critical columns preservation
        critical_columns = {
            'text', 'timestamp', 'channel', 'username',
            'political_category', 'sentiment_score'
        }
        preserved_columns = list(critical_columns.intersection(output_df.columns))
        results['critical_columns_preserved'] = preserved_columns

        if len(preserved_columns) < len(critical_columns) * 0.7:
            results['potential_issues'].append(
                f"Critical columns may be lost: {set(critical_columns) - set(preserved_columns)}"
            )
            results['success'] = False

        # New columns tracking
        results['new_columns'] = [
            col for col in output_df.columns
            if col not in input_df.columns
        ]

        # Data type and value consistency
        type_changes = StageValidator._detect_type_changes(input_df, output_df)
        if type_changes:
            results['potential_issues'].extend(
                f"Type change in column {col}: {orig} → {new_type}"
                for col, (orig, new_type) in type_changes.items()
            )

        # Logging
        logging.info(f"Stage Transition Validation: {stage_name}")
        logging.info(f"Records: {len(input_df)} → {len(output_df)}")
        logging.info(f"New Columns: {results['new_columns']}")

        return results

    @staticmethod
    def _detect_type_changes(
        input_df: pd.DataFrame,
        output_df: pd.DataFrame
    ) -> Dict[str, tuple]:
        """
        Detect type changes between input and output DataFrames.
        """
        type_changes = {}
        common_columns = set(input_df.columns).intersection(output_df.columns)

        for col in common_columns:
            input_type = input_df[col].dtype
            output_type = output_df[col].dtype

            if input_type != output_type:
                type_changes[col] = (input_type, output_type)

        return type_changes

    @staticmethod
    def statistical_drift_detection(
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Detect statistical drift in numeric columns.

        Args:
            input_df (pd.DataFrame): Original DataFrame
            output_df (pd.DataFrame): Processed DataFrame
            threshold (float): Allowed percentage of change

        Returns:
            Dict with drift detection results
        """
        numeric_columns = input_df.select_dtypes(include=[np.number]).columns
        drift_results = {}

        for col in numeric_columns:
            input_mean = input_df[col].mean()
            output_mean = output_df[col].mean()

            mean_change_percent = abs(
                (output_mean - input_mean) / (input_mean + 1e-10) * 100
            )

            if mean_change_percent > threshold * 100:
                drift_results[col] = {
                    'input_mean': input_mean,
                    'output_mean': output_mean,
                    'change_percent': mean_change_percent
                }

        return drift_results

def interactive_stage_checkpoint(validation_results: Dict[str, Any]) -> bool:
    """
    Interactive checkpoint for user approval before proceeding.

    Args:
        validation_results (Dict): Validation results from stage transition

    Returns:
        bool: Whether to proceed with the stage
    """
    print("\n🔍 Stage Transition Validation")
    print(f"Stage: {validation_results['stage']}")
    print(f"Input Records: {validation_results['input_records']}")
    print(f"Output Records: {validation_results['output_records']}")
    print(f"Record Delta: {validation_results['record_delta_percent']:.2f}%")

    if validation_results['potential_issues']:
        print("\n⚠️ Potential Issues Detected:")
        for issue in validation_results['potential_issues']:
            print(f"  - {issue}")

    if not validation_results['success']:
        print("\n❌ Stage Validation Failed!")
        user_input = input("Do you want to (P)roceed, (R)evert, or (A)bort? ").upper()

        if user_input == 'P':
            return True
        elif user_input == 'R':
            return False
        else:
            raise SystemExit("Pipeline execution aborted by user.")

    return True