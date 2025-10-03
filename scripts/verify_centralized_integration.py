#!/usr/bin/env python3
"""
Verification of centralized system integration
Tests that all configuration files and execution scripts properly reference the Clean Scientific Analyzer
"""

import sys
import os
from pathlib import Path

def verify_clean_analyzer_import():
    """Test that Clean Scientific Analyzer can be imported"""
    try:
        sys.path.insert(0, 'src')
        from analyzer import Analyzer
        print("✅ Analyzer v.final import: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ Analyzer v.final import: FAILED - {e}")
        return False

def verify_config_references():
    """Verify configuration files reference centralized system"""
    config_files = [
        'config/settings.yaml',
        'config/default.yaml',
        'config/research.yaml'
    ]

    results = {}

    for config_file in config_files:
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                content = f.read()

            # Check for centralized system references
            has_scientific = 'Analyzer' in content or 'analyzer' in content
            has_old_refs = any(ref in content for ref in [
                'UnifiedAnthropicPipeline',
                'RefactoredPipeline',
                'PragmaticPipelineBridge'
            ])

            results[config_file] = {
                'exists': True,
                'has_scientific_refs': has_scientific,
                'has_old_refs': has_old_refs,
                'status': '✅' if has_scientific and not has_old_refs else '⚠️'
            }
        else:
            results[config_file] = {
                'exists': False,
                'status': '❌'
            }

    print("\n📋 Configuration Files Analysis:")
    print("-" * 50)
    for config_file, data in results.items():
        print(f"{data['status']} {config_file}")
        if data['exists']:
            if data.get('has_scientific_refs'):
                print(f"   ✅ References centralized system")
            if data.get('has_old_refs'):
                print(f"   ⚠️ Still has old system references")

    return results

def verify_execution_scripts():
    """Verify execution scripts reference centralized system"""
    scripts = [
        'run_pipeline.py',
        'test_clean_analyzer.py'
    ]

    results = {}

    for script in scripts:
        if Path(script).exists():
            with open(script, 'r') as f:
                content = f.read()

            has_clean_analyzer = 'Analyzer' in content and 'from src.analyzer import' in content
            has_old_refs = any(ref in content for ref in [
                'UnifiedAnthropicPipeline',
                'RefactoredPipeline',
                'PragmaticPipelineBridge'
            ])

            results[script] = {
                'exists': True,
                'has_clean_analyzer': has_clean_analyzer,
                'has_old_refs': has_old_refs,
                'status': '✅' if has_clean_analyzer else '❌'
            }
        else:
            results[script] = {
                'exists': False,
                'status': '❌'
            }

    print("\n🚀 Execution Scripts Analysis:")
    print("-" * 50)
    for script, data in results.items():
        print(f"{data['status']} {script}")
        if data['exists']:
            if data.get('has_clean_analyzer'):
                print(f"   ✅ Uses Analyzer v.final")
            if data.get('has_old_refs'):
                print(f"   ⚠️ Still has old system references")

    return results

def verify_removed_structures():
    """Verify that parallel structures were actually removed"""
    removed_dirs = [
        'src/core',
        'src/pipeline_stages',
        'src/anthropic_integration',
        'src/preprocessing'
    ]

    print("\n🗑️ Removed Parallel Structures:")
    print("-" * 50)

    all_removed = True
    for dir_path in removed_dirs:
        if Path(dir_path).exists():
            print(f"❌ {dir_path} still exists")
            all_removed = False
        else:
            print(f"✅ {dir_path} removed")

    return all_removed

def verify_centralized_structure():
    """Verify centralized structure is intact"""
    required_files = [
        'src/analyzer.py',
        'src/lexicon_loader.py',
        'config/settings.yaml'
    ]

    print("\n🏗️ Centralized Structure:")
    print("-" * 50)

    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            all_present = False

    return all_present

def main():
    """Run complete integration verification"""
    print("🔍 CENTRALIZED SYSTEM INTEGRATION VERIFICATION")
    print("=" * 60)

    # Test 1: Import verification
    import_ok = verify_clean_analyzer_import()

    # Test 2: Configuration verification
    config_results = verify_config_references()

    # Test 3: Execution scripts verification
    script_results = verify_execution_scripts()

    # Test 4: Verify old structures removed
    structures_removed = verify_removed_structures()

    # Test 5: Verify centralized structure intact
    structure_intact = verify_centralized_structure()

    print("\n📊 INTEGRATION SUMMARY:")
    print("=" * 60)

    tests = [
        ("Analyzer v.final Import", import_ok),
        ("Configuration Files", all(r.get('has_scientific_refs', False) for r in config_results.values() if r.get('exists', False))),
        ("Execution Scripts", all(r.get('has_clean_analyzer', False) for r in script_results.values() if r.get('exists', False))),
        ("Parallel Structures Removed", structures_removed),
        ("Centralized Structure Intact", structure_intact)
    ]

    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 INTEGRATION VERIFICATION: SUCCESS")
        print("✅ System is properly centralized and integrated")
    else:
        print("⚠️ INTEGRATION VERIFICATION: ISSUES FOUND")
        print("❌ Some components need attention")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)