#!/usr/bin/env python3
"""
Script de validação da configuração do Batch Analyzer
Verifica se todas as dependências e configurações estão corretas
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_file(filepath, description):
    """Verifica se arquivo existe"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} não encontrado: {filepath}")
        return False

def check_module(module_name):
    """Verifica se módulo Python pode ser importado"""
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"✅ Módulo '{module_name}' disponível")
        return True
    else:
        print(f"⚠️ Módulo '{module_name}' não instalado")
        return False

def validate_batch_analyzer():
    """Valida configuração completa do Batch Analyzer"""

    print("🔍 Validação do Batch Analyzer Independente")
    print("=" * 50)

    all_ok = True

    # 1. Verifica estrutura de diretórios
    print("\n📁 Estrutura de Diretórios:")
    dirs_to_check = [
        ('batch_analyzer', 'Diretório principal'),
        ('batch_analyzer/config', 'Diretório de configuração'),
        ('batch_analyzer/data', 'Diretório de dados'),
        ('batch_analyzer/outputs', 'Diretório de saída'),
        ('batch_analyzer/docs', 'Diretório de documentação'),
    ]

    for dir_path, desc in dirs_to_check:
        if not check_file(dir_path, desc):
            all_ok = False

    # 2. Verifica arquivos essenciais
    print("\n📄 Arquivos Essenciais:")
    files_to_check = [
        ('batch_analyzer/batch_analysis.py', 'Script principal'),
        ('batch_analyzer/README.md', 'Documentação'),
        ('batch_analyzer/requirements.txt', 'Dependências'),
        ('batch_analyzer/.env.example', 'Exemplo de configuração'),
        ('batch_analyzer/config/default.yaml', 'Config padrão'),
        ('batch_analyzer/config/academic.yaml', 'Config acadêmica'),
    ]

    for file_path, desc in files_to_check:
        if not check_file(file_path, desc):
            all_ok = False

    # 3. Verifica módulos Python básicos
    print("\n🐍 Módulos Python Básicos:")
    basic_modules = ['pandas', 'numpy', 'yaml', 'tqdm']

    for module in basic_modules:
        if not check_module(module):
            all_ok = False

    # 4. Verifica módulos opcionais de IA
    print("\n🤖 Módulos de IA (Opcionais):")
    ai_modules = ['anthropic', 'voyageai', 'spacy']

    ai_available = False
    for module in ai_modules:
        if check_module(module):
            ai_available = True

    if not ai_available:
        print("⚠️ Nenhuma API de IA disponível - sistema usará métodos heurísticos")

    # 5. Verifica arquivo de teste
    print("\n🧪 Arquivos de Teste:")
    test_files = [
        ('batch_analyzer/test_batch.py', 'Script de teste'),
        ('batch_analyzer/data/sample_messages.csv', 'Dados de exemplo'),
    ]

    for file_path, desc in test_files:
        check_file(file_path, desc)

    # 6. Testa importação do módulo principal
    print("\n⚙️ Teste de Importação:")
    try:
        sys.path.insert(0, 'batch_analyzer')
        from batch_analysis import IntegratedBatchAnalyzer, BatchConfig
        print("✅ Módulo principal importado com sucesso")
        print(f"   - IntegratedBatchAnalyzer: OK")
        print(f"   - BatchConfig: OK")
    except ImportError as e:
        print(f"❌ Erro ao importar módulo: {e}")
        all_ok = False

    # 7. Verifica variáveis de ambiente
    print("\n🔐 Variáveis de Ambiente:")
    if os.getenv('ANTHROPIC_API_KEY'):
        print("✅ ANTHROPIC_API_KEY configurada")
    else:
        print("⚠️ ANTHROPIC_API_KEY não configurada (opcional)")

    if os.getenv('VOYAGE_API_KEY'):
        print("✅ VOYAGE_API_KEY configurada")
    else:
        print("⚠️ VOYAGE_API_KEY não configurada (opcional)")

    # Resumo final
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ VALIDAÇÃO COMPLETA: Sistema pronto para uso!")
        print("\n🚀 Para começar:")
        print("   1. cd batch_analyzer")
        print("   2. ./run_analysis.sh --test     # Teste rápido")
        print("   3. ./run_analysis.sh --dev data/sample_messages.csv")
    else:
        print("⚠️ VALIDAÇÃO PARCIAL: Alguns componentes precisam atenção")
        print("\nRecomendações:")
        print("   1. Instale dependências: pip install -r batch_analyzer/requirements.txt")
        print("   2. Configure APIs (opcional): cp batch_analyzer/.env.example batch_analyzer/.env")

    return all_ok

if __name__ == "__main__":
    success = validate_batch_analyzer()
    sys.exit(0 if success else 1)