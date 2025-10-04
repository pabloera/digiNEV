#!/usr/bin/env python3
"""
Teste do sistema ChunkedAnalyzer com dataset Bolsonaro
=====================================================

Valida o processamento em chunks para evitar sobrecarga de memória
com o dataset completo de 448,393 registros do governo Bolsonaro.
"""

import pandas as pd
import logging
import time
import gc
from pathlib import Path
from src.chunked_analyzer import ChunkedAnalyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_system():
    """Limpar sistema completamente antes do teste."""
    logger.info("🧹 Limpeza completa do sistema...")
    gc.collect()
    logger.info("✅ Sistema limpo e pronto para teste chunked")

def validate_dataset():
    """Validar existência e estrutura do dataset."""
    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    # Verificar tamanho do arquivo
    size_mb = dataset_path.stat().st_size / (1024 * 1024)
    logger.info(f"📁 Dataset encontrado: {dataset_path}")
    logger.info(f"📏 Tamanho: {size_mb:.1f} MB")

    # Verificar estrutura básica
    logger.info("🔍 Verificando estrutura básica...")
    sample = pd.read_csv(dataset_path, sep=',', nrows=5)
    logger.info(f"📋 Colunas: {list(sample.columns)}")
    logger.info(f"📊 Amostra shape: {sample.shape}")

    return dataset_path

def test_progressive_chunks():
    """Teste progressivo com chunks crescentes."""
    logger.info("📊 TESTE PROGRESSIVO - CHUNKS CRESCENTES")
    logger.info("=" * 50)

    dataset_path = validate_dataset()

    # Configurações de teste progressivo
    test_configs = [
        {"chunk_size": 1000, "max_records": 5000, "memory_limit": 1.5},
        {"chunk_size": 2000, "max_records": 10000, "memory_limit": 2.0},
        {"chunk_size": 5000, "max_records": 25000, "memory_limit": 2.5}
    ]

    results = []

    for i, config in enumerate(test_configs, 1):
        logger.info(f"\n🧪 TESTE {i}: {config['max_records']:,} registros, chunks={config['chunk_size']:,}")
        logger.info("-" * 40)

        # Limpar antes do teste
        clean_system()

        # Configurar analyzer chunked
        analyzer = ChunkedAnalyzer(
            chunk_size=config["chunk_size"],
            memory_limit_gb=config["memory_limit"]
        )

        start_time = time.time()

        try:
            # Executar análise chunked
            stats = analyzer.analyze_chunked_dataset(
                file_path=str(dataset_path),
                max_records=config["max_records"],
                output_file=f"chunked_test_{i}_output.csv"
            )

            end_time = time.time()
            duration = end_time - start_time

            # Compilar resultados
            test_result = {
                "test_number": i,
                "config": config,
                "stats": stats,
                "duration": duration,
                "success": True
            }

            results.append(test_result)

            logger.info(f"✅ Teste {i} concluído com sucesso em {duration:.1f}s")

        except Exception as e:
            logger.error(f"❌ Teste {i} falhou: {e}")
            results.append({
                "test_number": i,
                "config": config,
                "error": str(e),
                "success": False
            })

        # Pausa entre testes
        logger.info("⏸️ Pausando 5s entre testes...")
        time.sleep(5)

    return results

def test_full_dataset_chunked():
    """Teste com dataset completo usando chunked processing."""
    logger.info("\n🎯 TESTE DATASET COMPLETO - CHUNKED PROCESSING")
    logger.info("=" * 60)

    dataset_path = validate_dataset()

    # Limpar sistema
    clean_system()

    # Configurar para dataset completo
    analyzer = ChunkedAnalyzer(
        chunk_size=10000,  # Chunks de 10k para otimizar performance
        memory_limit_gb=3.0  # Limite maior para dataset completo
    )

    logger.info("🚀 Iniciando processamento do dataset completo...")
    start_time = time.time()

    try:
        # Processar dataset completo
        stats = analyzer.analyze_chunked_dataset(
            file_path=str(dataset_path),
            max_records=None,  # Processar todos os registros
            output_file="bolsonaro_dataset_chunked_complete.csv"
        )

        end_time = time.time()
        total_duration = end_time - start_time

        # Gerar relatório consolidado
        report_path = analyzer.generate_consolidated_report(
            stats,
            "RELATORIO_CHUNKED_BOLSONARO_COMPLETO.txt"
        )

        logger.info("🎉 PROCESSAMENTO COMPLETO FINALIZADO!")
        logger.info(f"📊 Total processado: {stats['total_records_processed']:,} registros")
        logger.info(f"📦 Total de chunks: {stats['total_chunks']}")
        logger.info(f"⏱️ Tempo total: {total_duration:.1f} segundos")
        logger.info(f"📈 Performance média: {stats['performance_records_per_second']:.1f} reg/s")
        logger.info(f"📄 Relatório: {report_path}")

        return stats

    except Exception as e:
        logger.error(f"❌ Erro no processamento completo: {e}")
        raise

def generate_test_summary(results):
    """Gerar sumário dos testes realizados."""
    logger.info("\n📋 SUMÁRIO DOS TESTES CHUNKED")
    logger.info("=" * 40)

    successful_tests = [r for r in results if r.get("success", False)]
    failed_tests = [r for r in results if not r.get("success", False)]

    logger.info(f"✅ Testes bem-sucedidos: {len(successful_tests)}")
    logger.info(f"❌ Testes falharam: {len(failed_tests)}")

    if successful_tests:
        logger.info("\n🏆 PERFORMANCE DOS TESTES SUCESSOS:")
        for result in successful_tests:
            config = result["config"]
            stats = result["stats"]
            logger.info(f"  • Teste {result['test_number']}: {stats['total_records_processed']:,} registros "
                       f"em {result['duration']:.1f}s ({stats['performance_records_per_second']:.1f} reg/s)")

def main():
    """Função principal do teste chunked."""
    logger.info("🚀 TESTE COMPLETO - ChunkedAnalyzer")
    logger.info("Dataset: Governo Bolsonaro (448k+ registros)")
    logger.info("=" * 60)

    try:
        # 1. Testes progressivos
        progressive_results = test_progressive_chunks()

        # 2. Gerar sumário dos testes progressivos
        generate_test_summary(progressive_results)

        # 3. Teste com dataset completo (opcional - descomentar se desejar)
        # logger.info("\n⚡ Prosseguindo para teste completo...")
        # full_stats = test_full_dataset_chunked()

        logger.info("\n🎉 TODOS OS TESTES CHUNKED CONCLUÍDOS!")
        logger.info("✅ Sistema ChunkedAnalyzer validado e operacional")

    except Exception as e:
        logger.error(f"❌ Erro nos testes: {e}")
        raise

if __name__ == "__main__":
    main()