#!/usr/bin/env python3
"""
DASHBOARD BOLSONARISMO v4.9.5 - INTEGRADO COM PIPELINE + STAGE 07 SPACY
=======================================================================
Script para iniciar dashboard integrado com monitoramento em tempo real
do pipeline e visualização completa dos resultados.

🔤 v4.9.5: Dashboard atualizado para Stage 07 spaCy operacional com 9 features linguísticas.
🛠️ v4.9.5: Pipeline inicializa 35/35 componentes (100% vs 48.6% anterior).
🚨 v4.9.4: Correção de deduplicação - monitora 784K registros processados.
📊 v4.9.5: Separadores CSV padronizados com `;` em todos os 22 stages.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Configurar PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Configurar variáveis de ambiente
os.environ['STREAMLIT_TELEMETRY_OPTOUT'] = '1'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'


def setup_dashboard_environment():
    """Configurar ambiente do dashboard"""
    # Criar diretório .streamlit se não existir
    streamlit_dir = Path.home() / '.streamlit'
    streamlit_dir.mkdir(exist_ok=True)

    # Criar arquivo de configuração para pular email
    config_file = streamlit_dir / 'credentials.toml'
    with open(config_file, 'w') as f:
        f.write('[general]\nemail = ""\n')

    # Garantir que diretórios do dashboard existem
    dashboard_dirs = [
        project_root / 'src' / 'dashboard' / 'data',
        project_root / 'src' / 'dashboard' / 'data' / 'uploads',
        project_root / 'src' / 'dashboard' / 'data' / 'dashboard_results',
        project_root / 'src' / 'dashboard' / 'temp'
    ]

    for directory in dashboard_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    print("✅ Ambiente do dashboard configurado")


def check_pipeline_integration():
    """Verificar integração com pipeline"""
    try:
        # Verificar se arquivos de resultados existem
        results_dir = project_root / 'src' / 'dashboard' / 'data' / 'dashboard_results'
        result_files = list(results_dir.glob('*.json')) + list(results_dir.glob('*.csv'))

        if result_files:
            print(f"📊 {len(result_files)} arquivos de resultados encontrados")
            print("🔗 Dashboard integrado com resultados do pipeline")
        else:
            print("📋 Nenhum resultado do pipeline encontrado ainda")
            print("💡 Execute 'python run_pipeline.py' para gerar dados para análise")

        return len(result_files) > 0

    except Exception as e:
        print(f"⚠️  Erro verificando integração: {e}")
        return False


def monitor_pipeline_results():
    """Monitora novos resultados do pipeline em background"""
    results_dir = project_root / 'src' / 'dashboard' / 'data' / 'dashboard_results'

    last_count = 0
    while True:
        try:
            current_files = list(results_dir.glob('*.json')) + list(results_dir.glob('*.csv'))
            current_count = len(current_files)

            if current_count > last_count:
                print(f"🔄 Novos resultados detectados: {current_count - last_count} arquivos")
                last_count = current_count

            time.sleep(5)  # Verificar a cada 5 segundos

        except Exception:
            break


def main():
    """Função principal para iniciar dashboard integrado"""

    print("🎯 DASHBOARD BOLSONARISMO v4.6 - PIPELINE INTEGRADO")
    print("=" * 60)

    # 1. Configurar ambiente
    print("📋 Configurando ambiente...")
    setup_dashboard_environment()

    # 2. Verificar integração com pipeline
    print("🔗 Verificando integração com pipeline...")
    has_results = check_pipeline_integration()

    # 3. Iniciar monitoramento em background
    if has_results:
        print("🔄 Iniciando monitoramento de resultados...")
        monitor_thread = threading.Thread(target=monitor_pipeline_results, daemon=True)
        monitor_thread.start()

    # 4. Executar dashboard
    print("\n🚀 Iniciando Dashboard Integrado...")
    print("📊 Acesse em: http://localhost:8501")
    print("🎯 Features disponíveis:")
    print("   - Visualização de resultados do pipeline")
    print("   - Monitoramento em tempo real")
    print("   - Análise interativa de dados processados")
    print("   - Upload e processamento de novos datasets")
    print("\nPressione Ctrl+C para parar\n")

    try:
        # Executar o dashboard principal com configurações otimizadas
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(project_root / 'src' / 'dashboard' / 'app.py'),
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false',
            '--server.headless', 'true',
            '--theme.primaryColor', '#1f77b4',
            '--theme.backgroundColor', '#ffffff',
            '--theme.secondaryBackgroundColor', '#f0f2f6'
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard encerrado")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("\n💡 Dicas para resolver problemas:")
        print("- Verifique se o streamlit está instalado: pip install streamlit")
        print("- Verifique se as dependências estão instaladas")
        print("- Execute 'python run_pipeline.py' para gerar dados primeiro")
        print("- Verifique se as APIs estão configuradas no arquivo .env")


if __name__ == "__main__":
    main()
