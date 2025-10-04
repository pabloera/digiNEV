#!/usr/bin/env python3
"""
Pipeline de Processamento de Datasets - digiNEV v.final
======================================================

Processa os 5 datasets principais com execução stage-by-stage.
"""

import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime

# Adicionar src ao path
sys.path.append('src')
from analyzer import Analyzer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_dataset_with_stages(dataset_path: str, analyzer: Analyzer) -> dict:
    """
    Processa um dataset com pipeline stage-by-stage.

    Args:
        dataset_path: Caminho para o dataset
        analyzer: Instância do Analyzer

    Returns:
        Dict com resultados do processamento
    """
    logger.info(f"📄 PROCESSANDO: {Path(dataset_path).name}")
    logger.info("=" * 70)

    # Verificar se arquivo existe
    if not Path(dataset_path).exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return {"error": "Dataset não encontrado"}

    # Obter informações do arquivo
    file_size_mb = Path(dataset_path).stat().st_size / (1024 * 1024)
    logger.info(f"📊 Tamanho do arquivo: {file_size_mb:.1f} MB")

    # Carregar amostra para análise inicial
    try:
        logger.info("🔍 Carregando amostra inicial...")
        # Detectar separador automaticamente
        with open(dataset_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line.count(',') > first_line.count(';'):
                separator = ','
            else:
                separator = ';'

        logger.info(f"🔧 Separador detectado: '{separator}'")

        df_sample = pd.read_csv(dataset_path, sep=separator, nrows=10, encoding='utf-8')
        logger.info(f"✅ Estrutura detectada: {len(df_sample.columns)} colunas")
        logger.info(f"📋 Colunas: {df_sample.columns.tolist()}")

        # Mostrar amostra de dados
        if 'body' in df_sample.columns:
            sample_text = df_sample['body'].dropna().iloc[0] if len(df_sample['body'].dropna()) > 0 else "N/A"
            logger.info(f"📝 Amostra de texto: '{sample_text[:60]}...'")

    except Exception as e:
        logger.error(f"❌ Erro ao carregar amostra: {e}")
        return {"error": f"Erro ao carregar: {e}"}

    # Verificar se deve usar chunking
    logger.info("🔧 Determinando estratégia de processamento...")

    if file_size_mb > 100:  # > 100MB usa chunking
        logger.info(f"📦 Arquivo grande ({file_size_mb:.1f} MB) - usando auto-chunking")
        chunk_size = 5000
    else:
        logger.info(f"📄 Arquivo pequeno ({file_size_mb:.1f} MB) - processamento completo")
        chunk_size = None

    # Processar dataset
    try:
        logger.info("🚀 INICIANDO PROCESSAMENTO STAGE-BY-STAGE")
        logger.info("=" * 70)

        if chunk_size:
            # Processamento por chunks
            logger.info(f"📦 Processando em chunks de {chunk_size} registros...")

            # Contar total de registros
            total_lines = sum(1 for _ in open(dataset_path, 'r', encoding='utf-8')) - 1
            logger.info(f"📊 Total de registros estimado: {total_lines:,}")

            # Processar primeiro chunk para teste
            df_chunk = pd.read_csv(dataset_path, sep=separator, nrows=chunk_size, encoding='utf-8')
            logger.info(f"🔬 Processando chunk teste: {len(df_chunk)} registros")

            result = analyzer.analyze_dataset(df_chunk)

        else:
            # Processamento completo
            df_full = pd.read_csv(dataset_path, sep=separator, encoding='utf-8')
            logger.info(f"🔬 Processando dataset completo: {len(df_full)} registros")

            result = analyzer.analyze_dataset(df_full)

        # Verificar resultado
        if result and 'data' in result:
            df_processed = result['data']
            stats = result['stats']

            logger.info("✅ PROCESSAMENTO CONCLUÍDO")
            logger.info("=" * 50)
            logger.info(f"📊 Registros processados: {len(df_processed):,}")
            logger.info(f"📈 Colunas geradas: {len(df_processed.columns)}")
            logger.info(f"🎯 Stages completados: {stats['stages_completed']}")
            logger.info(f"🔧 Features extraídas: {stats['features_extracted']}")
            logger.info(f"❌ Erros de processamento: {stats['processing_errors']}")

            # Salvar resultado
            output_filename = f"processed_{Path(dataset_path).stem}.csv"
            output_path = Path("data") / "processed" / output_filename
            output_path.parent.mkdir(exist_ok=True)

            df_processed.to_csv(output_path, index=False, sep=';')
            logger.info(f"💾 Resultado salvo: {output_path}")

            return {
                "success": True,
                "dataset": Path(dataset_path).name,
                "records_processed": len(df_processed),
                "columns_generated": len(df_processed.columns),
                "stages_completed": stats['stages_completed'],
                "features_extracted": stats['features_extracted'],
                "processing_errors": stats['processing_errors'],
                "output_file": str(output_path)
            }

        else:
            logger.error("❌ Erro no resultado do processamento")
            return {"error": "Erro no processamento"}

    except Exception as e:
        logger.error(f"❌ Erro durante processamento: {e}")
        return {"error": f"Erro durante processamento: {e}"}

def main():
    """Função principal - processa todos os datasets."""
    logger.info("🔬 PIPELINE DE PROCESSAMENTO DIGINEV v.final")
    logger.info("=" * 70)
    logger.info(f"🕒 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Datasets para processar
    datasets = [
        "data/1_2019-2021-govbolso.csv",
        "data/2_2021-2022-pandemia.csv",
        "data/3_2022-2023-poseleic.csv",
        "data/4_2022-2023-elec.csv",
        "data/5_2022-2023-elec-extra.csv"
    ]

    # Inicializar analyzer
    logger.info("🔧 Inicializando Analyzer v.final...")
    analyzer = Analyzer()

    # Processar cada dataset
    results = []

    for i, dataset_path in enumerate(datasets, 1):
        logger.info(f"\n🎯 DATASET {i}/5: {Path(dataset_path).name}")
        logger.info("=" * 70)

        result = process_dataset_with_stages(dataset_path, analyzer)
        results.append(result)

        if result.get("success"):
            logger.info("✅ Dataset processado com sucesso!")
        else:
            logger.error(f"❌ Erro no dataset: {result.get('error', 'Erro desconhecido')}")

        logger.info("-" * 70)

    # Relatório final
    logger.info("\n📊 RELATÓRIO FINAL")
    logger.info("=" * 50)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    logger.info(f"✅ Datasets processados com sucesso: {len(successful)}")
    logger.info(f"❌ Datasets com erro: {len(failed)}")

    if successful:
        total_records = sum(r["records_processed"] for r in successful)
        total_columns = sum(r["columns_generated"] for r in successful)

        logger.info(f"📊 Total de registros processados: {total_records:,}")
        logger.info(f"📈 Total de colunas geradas: {total_columns:,}")

    logger.info(f"🕒 Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("🎉 PIPELINE CONCLUÍDO!")

if __name__ == "__main__":
    main()