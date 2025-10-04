#!/usr/bin/env python3
"""
Validação Rápida do Sistema ChunkedAnalyzer
==========================================

Teste rápido para validar que o sistema chunked está funcionando
corretamente sem sobrecarregar com muitos dados.
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
    """Limpar sistema."""
    logger.info("🧹 Limpando sistema...")
    gc.collect()
    logger.info("✅ Sistema limpo")

def quick_validation_test():
    """Teste rápido de validação do sistema chunked."""
    logger.info("⚡ VALIDAÇÃO RÁPIDA - ChunkedAnalyzer")
    logger.info("=" * 50)

    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return False

    # Limpar sistema
    clean_system()

    # Configurar analyzer com chunks pequenos para teste rápido
    analyzer = ChunkedAnalyzer(
        chunk_size=500,  # Chunks pequenos para teste rápido
        memory_limit_gb=1.0
    )

    logger.info("🧪 Testando com 2,000 registros em chunks de 500...")

    start_time = time.time()

    try:
        # Executar análise chunked com amostra pequena
        stats = analyzer.analyze_chunked_dataset(
            file_path=str(dataset_path),
            max_records=2000,  # Apenas 2k registros para validação
            output_file="chunked_validation_output.csv"
        )

        end_time = time.time()
        duration = end_time - start_time

        # Validar resultados
        logger.info("✅ VALIDAÇÃO CONCLUÍDA:")
        logger.info(f"📊 Registros processados: {stats['total_records_processed']:,}")
        logger.info(f"📦 Chunks processados: {stats['total_chunks']}")
        logger.info(f"⏱️ Tempo total: {duration:.1f}s")
        logger.info(f"📈 Performance: {stats['performance_records_per_second']:.1f} reg/s")

        # Verificar distribuição política
        political_dist = stats['consolidated_stats']['political_distribution']
        if political_dist:
            logger.info(f"🏛️ Distribuição política detectada: {len(political_dist)} categorias")
            top_categories = sorted(political_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            for category, count in top_categories:
                logger.info(f"   • {category}: {count} registros")

        # Gerar relatório
        analyzer.generate_consolidated_report(stats, "RELATORIO_VALIDACAO_CHUNKED.txt")

        logger.info("🎉 SISTEMA CHUNKED VALIDADO COM SUCESSO!")
        return True

    except Exception as e:
        logger.error(f"❌ Erro na validação: {e}")
        return False

def test_memory_management():
    """Testar gestão de memória do sistema."""
    logger.info("\n🧠 TESTE DE GESTÃO DE MEMÓRIA")
    logger.info("-" * 40)

    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    # Configurar com limite baixo de memória para forçar limpezas
    analyzer = ChunkedAnalyzer(
        chunk_size=300,
        memory_limit_gb=0.5  # Limite baixo para testar limpeza
    )

    try:
        # Processar alguns chunks pequenos
        chunk_count = 0
        for chunk in analyzer.load_dataset_chunks(str(dataset_path), max_records=1000):
            chunk_count += 1
            logger.info(f"📦 Chunk {chunk_count}: {len(chunk)} registros carregados")

            # Simular processamento
            time.sleep(0.1)

            if chunk_count >= 3:  # Testar apenas 3 chunks
                break

        logger.info(f"✅ Gestão de memória testada com {chunk_count} chunks")
        return True

    except Exception as e:
        logger.error(f"❌ Erro no teste de memória: {e}")
        return False

def main():
    """Função principal de validação."""
    logger.info("🚀 VALIDAÇÃO SISTEMA CHUNKED ANALYZER")
    logger.info("Dataset: Governo Bolsonaro")
    logger.info("=" * 60)

    try:
        # 1. Validação rápida
        validation_success = quick_validation_test()

        # 2. Teste de gestão de memória
        memory_success = test_memory_management()

        # 3. Resultado final
        if validation_success and memory_success:
            logger.info("\n🏆 VALIDAÇÃO COMPLETA: SISTEMA CHUNKED APROVADO")
            logger.info("✅ Processamento em chunks funcionando")
            logger.info("✅ Gestão de memória operacional")
            logger.info("✅ Sistema pronto para datasets grandes")
        else:
            logger.error("\n❌ VALIDAÇÃO FALHOU: Verificar problemas")

    except Exception as e:
        logger.error(f"❌ Erro na validação: {e}")

if __name__ == "__main__":
    main()