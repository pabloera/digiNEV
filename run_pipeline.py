#!/usr/bin/env python3
"""
PIPELINE BOLSONARISMO v4.6 - EXECUÇÃO COMPLETA INTEGRADA
=========================================================

Pipeline completo com todas as 16 etapas implementadas:
- Integração completa com Dashboard
- Execução sequencial otimizada
- Monitoramento em tempo real
- Validação científica automática
"""

import sys
import os
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_configuration():
    """Carrega configuração completa do projeto"""
    config_files = [
        'config/settings.yaml',
        'config/anthropic.yaml', 
        'config/processing.yaml',
        'config/voyage_embeddings.yaml'
    ]
    
    config = {}
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
    
    # Configuração default se não encontrar arquivos
    if not config:
        config = {
            "anthropic": {"enable_api_integration": True},
            "processing": {"chunk_size": 10000},
            "data": {
                "path": "data/uploads",
                "interim_path": "data/interim",
                "dashboard_path": "src/dashboard/data"
            },
            "voyage_embeddings": {"enable_sampling": True, "max_messages": 50000}
        }
    
    return config

def discover_datasets(data_paths: List[str]) -> List[str]:
    """Descobre todos os datasets disponíveis"""
    datasets = []
    
    for data_path in data_paths:
        if os.path.exists(data_path):
            import glob
            csv_files = glob.glob(os.path.join(data_path, '*.csv'))
            datasets.extend(csv_files)
    
    return sorted(datasets)

def setup_dashboard_integration(config: Dict[str, Any]):
    """Configura integração com dashboard"""
    try:
        dashboard_data_dir = Path(config.get('data', {}).get('dashboard_path', 'src/dashboard/data'))
        dashboard_data_dir.mkdir(parents=True, exist_ok=True)
        
        uploads_dir = dashboard_data_dir / 'uploads'
        uploads_dir.mkdir(exist_ok=True)
        
        results_dir = dashboard_data_dir / 'dashboard_results'
        results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Dashboard integration configured: {dashboard_data_dir}")
        return True
        
    except Exception as e:
        logger.warning(f"Dashboard setup failed: {e}")
        return False

def run_complete_pipeline_execution(datasets: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Execução completa do pipeline com todas as 16 etapas"""
    
    start_time = time.time()
    execution_results = {
        'start_time': datetime.now().isoformat(),
        'datasets_processed': [],
        'stages_completed': {},
        'overall_success': False,
        'total_records_processed': 0,
        'final_outputs': []
    }
    
    try:
        # Import do pipeline unificado
        from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
        
        # Criar instância do pipeline
        pipeline = UnifiedAnthropicPipeline(config, str(Path.cwd()))
        logger.info("Pipeline unificado inicializado")
        
        # Executar todas as etapas sequencialmente
        all_stages = [
            '01_chunk_processing',
            '02a_encoding_validation', 
            '02b_deduplication',
            '01b_features_validation',
            '01c_political_analysis',
            '03_text_cleaning',
            '04_sentiment_analysis',
            '05_topic_modeling',
            '06_tfidf_extraction',
            '07_clustering',
            '08_hashtag_normalization',
            '09_domain_analysis',
            '10_temporal_analysis',
            '11_network_analysis',
            '12_qualitative_analysis',
            '13_smart_pipeline_review',
            '14_topic_interpretation',
            '15_semantic_search',
            '16_pipeline_validation'
        ]
        
        logger.info(f"Executando {len(all_stages)} etapas do pipeline")
        
        # Processar cada dataset
        for dataset_path in datasets[:1]:  # Limitar a 1 dataset para demonstração
            dataset_name = Path(dataset_path).name
            logger.info(f"Processando dataset: {dataset_name}")
            
            try:
                # Executar pipeline completo
                results = pipeline.run_complete_pipeline([dataset_path])
                
                if results.get('overall_success', False):
                    execution_results['datasets_processed'].append(dataset_name)
                    execution_results['total_records_processed'] += results.get('total_records', 0)
                    
                    # Coletar outputs finais
                    if 'final_outputs' in results:
                        execution_results['final_outputs'].extend(results['final_outputs'])
                
                # Atualizar progresso das etapas
                if 'stage_results' in results:
                    for stage, result in results['stage_results'].items():
                        if stage not in execution_results['stages_completed']:
                            execution_results['stages_completed'][stage] = []
                        execution_results['stages_completed'][stage].append({
                            'dataset': dataset_name,
                            'success': result.get('success', False),
                            'records': result.get('records_processed', 0)
                        })
                
            except Exception as e:
                logger.error(f"Erro processando {dataset_name}: {e}")
                continue
        
        # Verificar sucesso geral
        execution_results['overall_success'] = len(execution_results['datasets_processed']) > 0
        execution_results['execution_time'] = time.time() - start_time
        execution_results['end_time'] = datetime.now().isoformat()
        
        logger.info(f"Pipeline execution completed: {execution_results['overall_success']}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        execution_results['error'] = str(e)
    
    return execution_results

def integrate_with_dashboard(results: Dict[str, Any], config: Dict[str, Any]):
    """Integra resultados com dashboard para visualização"""
    try:
        dashboard_results_dir = Path(config.get('data', {}).get('dashboard_path', 'src/dashboard/data')) / 'dashboard_results'
        
        # Salvar resultados para dashboard
        results_file = dashboard_results_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Copiar outputs finais para dashboard
        if results.get('final_outputs'):
            for output_file in results['final_outputs']:
                if os.path.exists(output_file):
                    import shutil
                    dashboard_file = dashboard_results_dir / Path(output_file).name
                    shutil.copy2(output_file, dashboard_file)
                    logger.info(f"Resultado copiado para dashboard: {dashboard_file}")
        
        logger.info(f"Dashboard integration completed: {results_file}")
        return True
        
    except Exception as e:
        logger.error(f"Dashboard integration failed: {e}")
        return False

def main():
    """Entry point principal para execução completa"""
    
    print("🎯 PIPELINE BOLSONARISMO v4.6 - EXECUÇÃO COMPLETA")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Carregar configuração
        print("📋 Carregando configuração...")
        config = load_configuration()
        
        # 2. Configurar dashboard
        print("🖥️  Configurando integração com dashboard...")
        dashboard_ready = setup_dashboard_integration(config)
        
        # 3. Descobrir datasets
        print("📊 Descobrindo datasets...")
        data_paths = [
            config.get('data', {}).get('path', 'data/uploads'),
            'data/DATASETS_FULL',
            'data/uploads'
        ]
        datasets = discover_datasets(data_paths)
        
        if not datasets:
            print("❌ Nenhum dataset encontrado!")
            return
        
        print(f"📁 Datasets encontrados: {len(datasets)}")
        for i, dataset in enumerate(datasets[:5], 1):
            print(f"   {i}. {Path(dataset).name}")
        if len(datasets) > 5:
            print(f"   ... e mais {len(datasets) - 5} datasets")
        
        # 4. Executar pipeline completo
        print(f"\n🚀 Iniciando execução de todas as 16 etapas...")
        results = run_complete_pipeline_execution(datasets, config)
        
        # 5. Integrar com dashboard
        if dashboard_ready:
            print("🖥️  Integrando resultados com dashboard...")
            integrate_with_dashboard(results, config)
        
        # 6. Mostrar resultado final
        duration = time.time() - start_time
        
        print(f"\n{'✅' if results['overall_success'] else '❌'} EXECUÇÃO {'CONCLUÍDA' if results['overall_success'] else 'FALHOU'}")
        print(f"⏱️  Duração total: {duration:.1f}s")
        print(f"📊 Datasets processados: {len(results['datasets_processed'])}")
        print(f"📈 Records processados: {results['total_records_processed']}")
        print(f"🔧 Etapas executadas: {len(results['stages_completed'])}")
        
        if results.get('final_outputs'):
            print(f"\n📁 Arquivos finais gerados:")
            for output in results['final_outputs']:
                print(f"   - {output}")
        
        print(f"\n🖥️  Dashboard: Execute 'python src/dashboard/start_dashboard.py' para visualizar")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()