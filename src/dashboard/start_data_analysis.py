#!/usr/bin/env python3
"""
Script de Inicialização - Dashboard de Análise de Dados v4.9.7
==============================================================

Dashboard focado exclusivamente nos RESULTADOS das análises de dados
geradas pelos stages do pipeline. Apresenta insights, visualizações
e descobertas sobre o discurso político brasileiro nos dados do Telegram.

🎯 NOVO FOCO: Análise dos dados processados, não monitoramento do pipeline
📊 OBJETIVO: Insights sobre o conteúdo político brasileiro
🔍 ESCOPO: Dashboards analíticos e visualizações de resultados
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Função principal para iniciar o dashboard de análise de dados"""
    
    print("🏛️ DASHBOARD DE ANÁLISE POLÍTICA - TELEGRAM BRASIL")
    print("=" * 60)
    print("🎯 Foco: Análise dos resultados processados pelo pipeline")
    print("📊 Objetivo: Insights sobre discurso político brasileiro")
    print("🔍 Escopo: Visualizações de dados, não monitoramento técnico")
    print("=" * 60)
    
    # Configurar ambiente
    project_root = Path(__file__).parent.parent.parent
    os.environ['STREAMLIT_TELEMETRY_OPTOUT'] = '1'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Verificar se dados estão disponíveis
    data_path = project_root / "data/interim/sample_dataset_v495_19_pipeline_validated.csv"
    if data_path.exists():
        print(f"Dataset encontrado: {data_path.name}")
        print(f"📊 Dados prontos para análise!")
    else:
        print("⚠️  Dataset não encontrado")
        print("💡 Execute 'poetry run python run_pipeline.py' para gerar dados")
    
    print("\\n🚀 Iniciando Dashboard de Análise de Dados...")
    print("📊 Acesse em: http://localhost:8503")
    print("\\n🎯 Páginas disponíveis:")
    print("   📋 Visão Geral - Resumo executivo das análises")
    print("   🏛️ Análise Política - Categorias e alinhamentos políticos")
    print("   😊 Análise de Sentimento - Distribuição de sentimentos")
    print("   💬 Análise do Discurso - Tipos de discurso identificados")
    print("   📅 Análise Temporal - Padrões ao longo do tempo")
    print("   🔤 Análise Linguística - Características linguísticas")
    print("   🔍 Análise de Agrupamentos - Clusters semânticos")
    print("   🌐 Análise de Redes - Interações e menções")
    print("   ⚖️ Análise Comparativa - Comparações entre dimensões")
    print("\\nPressione Ctrl+C para parar\\n")
    
    try:
        # Executar o dashboard de análise de dados
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(project_root / 'src' / 'dashboard' / 'data_analysis_dashboard.py'),
            '--server.port', '8503',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false',
            '--server.headless', 'true',
            '--theme.primaryColor', '#2E4057',
            '--theme.backgroundColor', '#ffffff',
            '--theme.secondaryBackgroundColor', '#f0f2f6'
        ])
    except KeyboardInterrupt:
        print("\\n\\nDashboard de análise encerrado")
    except Exception as e:
        print(f"\\n❌ Erro: {e}")
        print("\\n💡 Dicas para resolver problemas:")
        print("- Verifique se o Poetry está ativo: poetry shell")
        print("- Verifique se o dataset existe: ls data/interim/")
        print("- Execute o pipeline primeiro: poetry run python run_pipeline.py")

if __name__ == "__main__":
    main()