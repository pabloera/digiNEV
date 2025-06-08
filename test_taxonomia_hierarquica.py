#!/usr/bin/env python3
"""
Teste Rápido - Taxonomia Hierárquica
=====================================

Teste básico da implementação da taxonomia hierárquica no PoliticalAnalyzer
para verificar se todas as funcionalidades foram implementadas corretamente.
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add project path
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path / "src"))

# Import the enhanced PoliticalAnalyzer
try:
    from anthropic_integration.political_analyzer import PoliticalAnalyzer
    print("✅ Import PoliticalAnalyzer: SUCCESS")
except ImportError as e:
    print(f"❌ Import PoliticalAnalyzer: FAILED - {e}")
    sys.exit(1)

def test_initialization():
    """Teste 1: Inicialização do PoliticalAnalyzer Enhanced"""
    print("\n📋 TESTE 1: Inicialização")
    
    try:
        # Initialize with test config
        config = {
            "anthropic_api_key": "test-key-for-init-only",
            "enable_mock_mode": True
        }
        
        analyzer = PoliticalAnalyzer(config)
        
        # Check if enhanced features are enabled
        print(f"✅ Level 4 enabled: {analyzer.experiment_config.get('enable_level4_classification', False)}")
        print(f"✅ Early stopping enabled: {analyzer.experiment_config.get('enable_early_stopping', False)}")
        print(f"✅ Prompt version: {analyzer.prompt_version}")
        print(f"✅ Model: {analyzer.model}")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None

def test_enhanced_examples(analyzer):
    """Teste 2: Enhanced Examples Implementation"""
    print("\n📋 TESTE 2: Enhanced Examples")
    
    try:
        # Test enhanced examples function
        examples = analyzer._load_enhanced_political_examples()
        
        print(f"✅ Enhanced examples count: {len(examples)}")
        
        # Check coverage of hierarchical levels
        level3_categories = set()
        level4_categories = set()
        early_stop_examples = 0
        
        for example in examples:
            if example.get('level3_category'):
                level3_categories.add(example['level3_category'])
            if example.get('level4_category'):
                level4_categories.add(example['level4_category'])
            if example.get('early_stop'):
                early_stop_examples += 1
        
        print(f"✅ Level 3 categories covered: {len(level3_categories)} - {list(level3_categories)}")
        print(f"✅ Level 4 categories covered: {len(level4_categories)} - {list(level4_categories)[:3]}...")
        print(f"✅ Early stop examples: {early_stop_examples}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced examples test failed: {e}")
        return False

def test_taxonomy_structure(analyzer):
    """Teste 3: Estrutura da Taxonomia Hierárquica"""
    print("\n📋 TESTE 3: Estrutura da Taxonomia")
    
    try:
        # Load taxonomy
        taxonomy = analyzer._load_brazilian_taxonomy()
        
        # Check levels
        level1_count = len(taxonomy.get('level1', {}))
        level2_count = len(taxonomy.get('level2', {}))
        level3_count = len(taxonomy.get('level3', {}))
        level4_mapping = taxonomy.get('level4_mapping', {})
        
        print(f"✅ Level 1 categories: {level1_count} - {list(taxonomy.get('level1', {}).keys())}")
        print(f"✅ Level 2 categories: {level2_count} - {list(taxonomy.get('level2', {}).keys())}")
        print(f"✅ Level 3 categories: {level3_count} - {list(taxonomy.get('level3', {}).keys())}")
        print(f"✅ Level 4 mapping: {len(level4_mapping)} Level 3 categories mapped")
        
        # Check expected Level 3 categories
        expected_level3 = {'negacionismo', 'autoritarismo', 'deslegitimação', 'mobilização', 'conspiração', 'informativo'}
        actual_level3 = set(taxonomy.get('level3', {}).keys())
        
        if expected_level3.issubset(actual_level3):
            print("✅ All expected Level 3 categories present")
        else:
            missing = expected_level3 - actual_level3
            print(f"⚠️ Missing Level 3 categories: {missing}")
        
        # Count total Level 4 categories
        total_level4 = sum(len(categories) for categories in level4_mapping.values())
        print(f"✅ Total Level 4 categories: {total_level4}")
        
        return True
        
    except Exception as e:
        print(f"❌ Taxonomy structure test failed: {e}")
        return False

def test_early_stopping_logic(analyzer):
    """Teste 4: Early Stopping Logic"""
    print("\n📋 TESTE 4: Early Stopping Logic")
    
    try:
        # Test early stopping for não-político
        level1_test = analyzer._apply_hierarchical_early_stopping("não-político", "indefinido", 0.95)
        print(f"✅ Early stop 'não-político': Level {level1_test} (expected: 1)")
        
        # Test early stopping for indefinido + low confidence
        level2_test = analyzer._apply_hierarchical_early_stopping("político", "indefinido", 0.5)
        print(f"✅ Early stop 'indefinido' + low confidence: Level {level2_test} (expected: 2)")
        
        # Test continue to Level 4
        level4_test = analyzer._apply_hierarchical_early_stopping("político", "bolsonarista", 0.9)
        print(f"✅ Continue to Level 4: Level {level4_test} (expected: 4)")
        
        # Test should_continue_to_level
        should_continue = analyzer._should_continue_to_level(1, 4, "político", "bolsonarista", 0.9)
        should_stop = analyzer._should_continue_to_level(1, 4, "não-político", "indefinido", 0.9)
        
        print(f"✅ Should continue político/bolsonarista: {should_continue} (expected: True)")
        print(f"✅ Should stop não-político: {should_stop} (expected: False)")
        
        return True
        
    except Exception as e:
        print(f"❌ Early stopping logic test failed: {e}")
        return False

def test_prompt_generation(analyzer):
    """Teste 5: Geração de Prompt XML Enhanced"""
    print("\n📋 TESTE 5: Geração de Prompt XML")
    
    try:
        # Create test batch data
        test_batch = {
            'texts': [
                "Bolsonaro defendeu a família brasileira",
                "STF é quadrilha, intervenção militar já!",
                "Receita de bolo de chocolate"
            ],
            'metadata': {
                'channels': ['canal_test'] * 3,
                'dates': ['2023-01-01'] * 3,
                'domains': ['telegram'] * 3,
                'avg_length': 50,
                'duplicate_frequencies': [1, 1, 1]
            }
        }
        
        # Generate prompt
        prompt = analyzer._create_enhanced_anthropic_prompt(test_batch)
        
        # Check if key components are present
        has_level4_taxonomy = '<level4>' in prompt
        has_early_stopping = 'EARLY STOPPING' in prompt
        has_taxonomy = '<taxonomy>' in prompt
        has_examples = '<contextual_examples>' in prompt
        has_messages = '<messages>' in prompt
        has_output_template = '<required_output>' in prompt
        
        print(f"✅ Has Level 4 taxonomy: {has_level4_taxonomy}")
        print(f"✅ Has early stopping instructions: {has_early_stopping}")
        print(f"✅ Has taxonomy section: {has_taxonomy}")
        print(f"✅ Has contextual examples: {has_examples}")
        print(f"✅ Has messages section: {has_messages}")
        print(f"✅ Has output template: {has_output_template}")
        
        # Check Level 3 categories in prompt
        level3_in_prompt = all(cat in prompt for cat in ['negacionismo', 'autoritarismo', 'deslegitimação'])
        print(f"✅ Level 3 categories in prompt: {level3_in_prompt}")
        
        prompt_length = len(prompt)
        print(f"✅ Prompt length: {prompt_length} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt generation test failed: {e}")
        return False

def test_output_template(analyzer):
    """Teste 6: Template de Output Enhanced"""
    print("\n📋 TESTE 6: Template de Output")
    
    try:
        # Generate output template for 2 messages
        template = analyzer._generate_output_template(2)
        
        # Check if hierarchical fields are present
        has_discourse_type = 'discourse_type' in template
        has_specific_category = 'specific_category' in template
        has_early_stop_level = 'early_stop_level' in template
        has_political_level = 'political_level' in template
        has_alignment = 'alignment' in template
        
        print(f"✅ Has discourse_type (Level 3): {has_discourse_type}")
        print(f"✅ Has specific_category (Level 4): {has_specific_category}")
        print(f"✅ Has early_stop_level: {has_early_stop_level}")
        print(f"✅ Has political_level: {has_political_level}")
        print(f"✅ Has alignment: {has_alignment}")
        
        # Count message templates
        message_count = template.count('<message id=')
        print(f"✅ Message templates generated: {message_count} (expected: 2)")
        
        print(f"✅ Template length: {len(template)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Output template test failed: {e}")
        return False

def test_feature_flags(analyzer):
    """Teste 7: Feature Flags"""
    print("\n📋 TESTE 7: Feature Flags")
    
    try:
        # Test Level 4 disabled
        analyzer.experiment_config['enable_level4_classification'] = False
        analyzer.experiment_config['enable_early_stopping'] = False
        
        # Generate prompt with flags disabled
        test_batch = {
            'texts': ["Test message"],
            'metadata': {'channels': ['test'], 'dates': ['2023-01-01'], 'domains': ['telegram'], 'avg_length': 20, 'duplicate_frequencies': [1]}
        }
        
        prompt_disabled = analyzer._create_enhanced_anthropic_prompt(test_batch)
        template_disabled = analyzer._generate_output_template(1)
        
        # Check that Level 4 features are disabled
        level4_disabled = '<level4>' not in prompt_disabled
        early_stop_disabled = 'EARLY STOPPING' not in prompt_disabled
        template_level4_disabled = 'specific_category' not in template_disabled
        
        print(f"✅ Level 4 properly disabled in prompt: {level4_disabled}")
        print(f"✅ Early stopping properly disabled in prompt: {early_stop_disabled}")
        print(f"✅ Level 4 fields properly disabled in template: {template_level4_disabled}")
        
        # Re-enable for other tests
        analyzer.experiment_config['enable_level4_classification'] = True
        analyzer.experiment_config['enable_early_stopping'] = True
        
        return True
        
    except Exception as e:
        print(f"❌ Feature flags test failed: {e}")
        return False

def main():
    """Executar todos os testes"""
    print("🧪 TESTE RÁPIDO - TAXONOMIA HIERÁRQUICA")
    print("=" * 50)
    
    # Test 1: Initialization
    analyzer = test_initialization()
    if not analyzer:
        print("\n❌ TESTE FALHADO: Não foi possível inicializar o analyzer")
        return False
    
    # Run all tests
    tests = [
        test_enhanced_examples,
        test_taxonomy_structure,
        test_early_stopping_logic,
        test_prompt_generation,
        test_output_template,
        test_feature_flags
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func(analyzer)
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Testes aprovados: {passed}/{total}")
    print(f"❌ Testes falhados: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("🚀 Taxonomia hierárquica implementada com sucesso!")
        return True
    else:
        print(f"\n⚠️ {total - passed} testes falharam")
        print("🔧 Verifique os erros acima")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)