#!/usr/bin/env python3
"""
Model Update Validation: Verify claude-3-5-haiku-20241022 consistency and API key centralization
Function: Validate all model references use correct version and API keys are properly centralized in .env
Usage: poetry run python validate_model_updates.py
"""

import os
import sys
import re
import yaml
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_env_file() -> bool:
    """Validate .env file has correct API keys and model configuration"""
    logger.info("\n🔍 Validating .env File Configuration...")
    
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        logger.error("❌ .env file not found")
        return False
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            env_content = f.read()
        
        # Check for API keys
        if "ANTHROPIC_API_KEY=" not in env_content:
            logger.error("❌ ANTHROPIC_API_KEY not found in .env")
            return False
        logger.info("✅ ANTHROPIC_API_KEY found in .env")
        
        if "VOYAGE_API_KEY=" not in env_content:
            logger.error("❌ VOYAGE_API_KEY not found in .env")
            return False
        logger.info("✅ VOYAGE_API_KEY found in .env")
        
        # Check model configuration
        if "ANTHROPIC_MODEL=claude-3-5-haiku-20241022" not in env_content:
            logger.error("❌ ANTHROPIC_MODEL should be claude-3-5-haiku-20241022 in .env")
            return False
        logger.info("✅ ANTHROPIC_MODEL correctly set to claude-3-5-haiku-20241022")
        
        # Check no 'latest' references
        if "latest" in env_content.lower():
            logger.warning("⚠️ Found 'latest' reference in .env file")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading .env file: {e}")
        return False

def validate_config_files() -> bool:
    """Validate configuration files reference .env properly"""
    logger.info("\n📁 Validating Configuration Files...")
    
    config_dir = Path(__file__).parent / "config"
    success = True
    
    # Check anthropic.yaml.example
    anthropic_config = config_dir / "anthropic.yaml.example"
    if anthropic_config.exists():
        try:
            with open(anthropic_config, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "${ANTHROPIC_API_KEY}" in content:
                logger.info("✅ anthropic.yaml.example references .env for API key")
            else:
                logger.error("❌ anthropic.yaml.example should reference ${ANTHROPIC_API_KEY}")
                success = False
            
            if "claude-3-5-haiku-20241022" in content:
                logger.info("✅ anthropic.yaml.example uses correct model version")
            else:
                logger.error("❌ anthropic.yaml.example should use claude-3-5-haiku-20241022")
                success = False
                
        except Exception as e:
            logger.error(f"❌ Error reading anthropic.yaml.example: {e}")
            success = False
    
    # Check voyage_embeddings.yaml.example
    voyage_config = config_dir / "voyage_embeddings.yaml.example"
    if voyage_config.exists():
        try:
            with open(voyage_config, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "${VOYAGE_API_KEY}" in content:
                logger.info("✅ voyage_embeddings.yaml.example references .env for API key")
            else:
                logger.error("❌ voyage_embeddings.yaml.example should reference ${VOYAGE_API_KEY}")
                success = False
                
        except Exception as e:
            logger.error(f"❌ Error reading voyage_embeddings.yaml.example: {e}")
            success = False
    
    return success

def scan_python_files_for_models() -> bool:
    """Scan Python files for model references"""
    logger.info("\n🐍 Scanning Python Files for Model References...")
    
    project_root = Path(__file__).parent
    python_files = list(project_root.rglob("*.py"))
    
    # Exclude directories and files that shouldn't be validated  
    exclude_patterns = [
        'backup', 'archive', '__pycache__', '.venv', '.vscode', '.github', 
        '.idea', 'node_modules', 'validate_model_updates.py', 'temp',
        'cache', 'logs', 'deployment_backups'
    ]
    python_files = [f for f in python_files if 
                   not any(part in str(f) for part in exclude_patterns)]
    
    issues = []
    correct_references = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for old model references
            old_models = [
                "claude-3-haiku-20240307",
                "claude-3-5-haiku-latest",
                "claude-haiku-latest"
            ]
            
            for old_model in old_models:
                if old_model in content:
                    issues.append(f"❌ {py_file.relative_to(project_root)}: Found {old_model}")
            
            # Count correct references
            if "claude-3-5-haiku-20241022" in content:
                correct_references += 1
        
        except Exception as e:
            logger.debug(f"Skipping {py_file}: {e}")
    
    if issues:
        logger.error("❌ Found old model references:")
        for issue in issues:
            logger.error(f"  {issue}")
        return False
    else:
        logger.info(f"✅ No old model references found in Python files")
        logger.info(f"✅ Found {correct_references} files with claude-3-5-haiku-20241022")
        return True

def scan_yaml_files_for_models() -> bool:
    """Scan YAML files for model references"""
    logger.info("\n📄 Scanning YAML Files for Model References...")
    
    project_root = Path(__file__).parent
    yaml_files = list(project_root.rglob("*.yaml")) + list(project_root.rglob("*.yml"))
    
    # Exclude directories that shouldn't be validated
    exclude_patterns = [
        'backup', 'archive', '.venv', '.vscode', '.github', 
        '.idea', 'node_modules', 'temp', 'cache', 'logs'
    ]
    yaml_files = [f for f in yaml_files if 
                 not any(part in str(f) for part in exclude_patterns)]
    
    issues = []
    correct_references = 0
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for old model references
            old_models = [
                "claude-3-haiku-20240307",
                "claude-3-5-haiku-latest",
                "claude-haiku-latest"
            ]
            
            for old_model in old_models:
                if old_model in content:
                    issues.append(f"❌ {yaml_file.relative_to(project_root)}: Found {old_model}")
            
            # Count correct references
            if "claude-3-5-haiku-20241022" in content:
                correct_references += 1
        
        except Exception as e:
            logger.debug(f"Skipping {yaml_file}: {e}")
    
    if issues:
        logger.error("❌ Found old model references in YAML files:")
        for issue in issues:
            logger.error(f"  {issue}")
        return False
    else:
        logger.info(f"✅ No old model references found in YAML files")
        logger.info(f"✅ Found {correct_references} YAML files with claude-3-5-haiku-20241022")
        return True

def validate_api_key_usage() -> bool:
    """Validate that API keys are properly using environment variables"""
    logger.info("\n🔑 Validating API Key Usage...")
    
    project_root = Path(__file__).parent
    python_files = list(project_root.rglob("*.py"))
    
    # Exclude directories that shouldn't be validated
    exclude_patterns = [
        'backup', 'archive', '__pycache__', '.venv', '.vscode', '.github', 
        '.idea', 'node_modules', 'temp', 'cache', 'logs', 'deployment_backups'
    ]
    python_files = [f for f in python_files if 
                   not any(part in str(f) for part in exclude_patterns)]
    
    hardcoded_keys = []
    env_usage = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for hardcoded API keys (more specific patterns)
            # Anthropic API keys start with sk-ant-api03-
            if re.search(r'sk-ant-api03-[0-9a-zA-Z-_]{50,}', content):
                hardcoded_keys.append(f"❌ {py_file.relative_to(project_root)}: Hardcoded Anthropic API key")
            
            # Voyage API keys start with pa- and are longer (avoid false positives)
            if re.search(r'pa-[0-9a-zA-Z-_]{40,}', content):
                hardcoded_keys.append(f"❌ {py_file.relative_to(project_root)}: Hardcoded Voyage API key")
            
            # Check for proper environment variable usage
            if "ANTHROPIC_API_KEY" in content or "VOYAGE_API_KEY" in content:
                env_usage += 1
        
        except Exception as e:
            logger.debug(f"Skipping {py_file}: {e}")
    
    if hardcoded_keys:
        logger.error("❌ Found hardcoded API keys:")
        for key_issue in hardcoded_keys:
            logger.error(f"  {key_issue}")
        return False
    else:
        logger.info(f"✅ No hardcoded API keys found")
        logger.info(f"✅ Found {env_usage} files properly using environment variables")
        return True

def test_academic_config_loader() -> bool:
    """Test that academic config loader works correctly"""
    logger.info("\n🎓 Testing Academic Config Loader...")
    
    try:
        # Add project root to path for import
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from src.academic_config import get_academic_config
        
        config = get_academic_config()
        academic_config = config.get_anthropic_config()
        
        # Check model configuration
        model = academic_config.get('model')
        if model != 'claude-3-5-haiku-20241022':
            logger.error(f"❌ Academic config model is {model}, should be claude-3-5-haiku-20241022")
            return False
        
        logger.info("✅ Academic config loader uses correct model")
        
        # Check API key usage
        api_key = academic_config.get('api_key')
        if api_key and not api_key.startswith('${'):
            # Check if it's using environment variable
            if 'ANTHROPIC_API_KEY' in str(api_key):
                logger.info("✅ Academic config uses environment variable for API key")
            else:
                logger.warning("⚠️ Academic config API key usage unclear")
        else:
            logger.info("✅ Academic config properly references environment variables")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Academic config loader test failed: {e}")
        return False

def generate_summary_report(results: Dict[str, bool]) -> None:
    """Generate validation summary report"""
    logger.info("\n" + "="*60)
    logger.info("📋 MODEL UPDATE VALIDATION SUMMARY")
    logger.info("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info("-"*60)
    logger.info(f"📊 Results: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL VALIDATIONS PASSED!")
        logger.info("✅ claude-3-5-haiku-20241022 uniformly configured")
        logger.info("✅ API keys properly centralized in .env file")
        logger.info("💰 Cost-optimized configuration ready for deployment")
    else:
        logger.warning("⚠️ Some validations failed - review issues before deployment")
    
    logger.info("="*60)

def main():
    """Main validation function"""
    logger.info("🚀 Starting Model Update Validation...")
    logger.info("🎯 Validating claude-3-5-haiku-20241022 and .env API key centralization")
    
    # Run validation tests
    results = {
        ".env File Configuration": validate_env_file(),
        "Config Files Reference .env": validate_config_files(), 
        "Python Files Model References": scan_python_files_for_models(),
        "YAML Files Model References": scan_yaml_files_for_models(),
        "API Key Usage": validate_api_key_usage(),
        "Academic Config Loader": test_academic_config_loader()
    }
    
    # Generate summary
    generate_summary_report(results)
    
    # Return overall success
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)