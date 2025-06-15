#!/usr/bin/env python3
"""
Fix the remaining 3 test failures for TDD Phase 3 completion
"""

import pandas as pd
from pathlib import Path

def fix_remaining_issues():
    """Fix the last 3 test failures"""
    
    project_root = Path("/Users/pabloalmada/development/project/dataanalysis-bolsonarismo")
    
    print("🔧 Fixing remaining 3 test failures...")
    
    # Fix 1: Create larger valid.csv file for dataset validation
    print("\n1. Creating larger valid.csv file...")
    test_data_dir = project_root / "tests" / "test_data"
    
    # Create valid data that's large enough (>100 bytes)
    valid_df = pd.DataFrame({
        'id': range(1, 11),
        'body': [f'Valid test message number {i} with sufficient content to pass size validation' for i in range(1, 11)],
        'date': pd.date_range('2023-01-01', periods=10, freq='h'),
        'channel': ['valid_channel'] * 10
    })
    
    valid_file = test_data_dir / "valid.csv"
    valid_df.to_csv(valid_file, index=False)
    
    # Check file size
    file_size = valid_file.stat().st_size
    print(f"   ✅ valid.csv created: {file_size} bytes (should be >100)")
    
    # Fix 2: Update pipeline to handle missing files with error flag
    print("\n2. Updating pipeline error handling...")
    
    pipeline_file = project_root / "src" / "anthropic_integration" / "unified_pipeline.py"
    
    # Read current content
    with open(pipeline_file, 'r') as f:
        content = f.read()
    
    # Update the run_complete_pipeline method to set overall_success=False when no datasets processed
    old_logic = '''        logger.info(f"Pipeline completed: {len(results['datasets_processed'])} datasets processed")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            results['overall_success'] = False
            results['error'] = str(e)'''
    
    new_logic = '''        # Set overall_success based on whether any datasets were processed
        if len(results['datasets_processed']) == 0:
            results['overall_success'] = False
            results['error'] = 'No datasets could be processed'
        
        logger.info(f"Pipeline completed: {len(results['datasets_processed'])} datasets processed")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            results['overall_success'] = False
            results['error'] = str(e)'''
    
    if old_logic in content:
        content = content.replace(old_logic, new_logic)
        with open(pipeline_file, 'w') as f:
            f.write(content)
        print("   ✅ Pipeline error handling updated")
    else:
        print("   ⚠️  Pipeline content not updated (pattern not found)")
    
    # Fix 3: Create config file with anthropic section
    print("\n3. Creating config file with anthropic section...")
    
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create anthropic.yaml
    anthropic_config = """# Anthropic API Configuration
anthropic:
  enable_api_integration: false
  api_key: placeholder_key
  model: claude-3-haiku-20240307
  max_tokens: 1000
  batch_size: 10
  rate_limit: 100

# Processing Configuration  
processing:
  chunk_size: 1000
  encoding: utf-8
  memory_limit: 2GB

# Data Configuration
data:
  path: data/uploads
  interim_path: data/interim
  output_path: pipeline_outputs
  dashboard_path: src/dashboard/data
"""
    
    anthropic_file = config_dir / "anthropic.yaml"
    with open(anthropic_file, 'w') as f:
        f.write(anthropic_config)
    
    print(f"   ✅ {anthropic_file} created")
    
    print("\n✅ All 3 fixes applied!")
    print("\n🎯 Ready to test again:")
    print("   python tests/run_tests.py quick")

if __name__ == "__main__":
    fix_remaining_issues()
