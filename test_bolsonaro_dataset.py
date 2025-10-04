#!/usr/bin/env python3
"""
Teste do digiNEV v.final com Dataset Governo Bolsonaro (2019-2021)
================================================================

Dataset: 1_2019-2021-govbolso.csv (448,393 registros)
Teste com amostra progressiva para validar performance e classificação política.
"""

import pandas as pd
import logging
import time
import gc
from pathlib import Path
from src.analyzer import Analyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_system():
    """Limpar sistema de memória residual."""
    logger.info("🧹 Limpando sistema...")
    gc.collect()
    logger.info("✅ Sistema limpo")

def load_bolsonaro_dataset(sample_size=1000):
    """Carregar dataset do governo Bolsonaro."""
    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    logger.info(f"📂 Carregando dataset: {dataset_path}")
    logger.info(f"📏 Tamanho da amostra: {sample_size:,} registros")

    # Carregar com pandas usando vírgula como separador
    df = pd.read_csv(dataset_path, sep=',', encoding='utf-8', nrows=sample_size)

    logger.info(f"✅ Dataset carregado: {len(df):,} registros, {len(df.columns)} colunas")
    logger.info(f"📋 Colunas: {list(df.columns)}")

    # Mostrar amostra do conteúdo político
    if 'body' in df.columns:
        sample_texts = df['body'].dropna().head(3).tolist()
        logger.info("📄 Amostra de textos políticos:")
        for i, text in enumerate(sample_texts, 1):
            logger.info(f"   {i}. {text[:100]}...")

    return df

def test_political_classification(df, analyzer):
    """Testar classificação política específica."""
    logger.info("🏛️ TESTANDO CLASSIFICAÇÃO POLÍTICA:")

    # Extrair amostra com termos políticos específicos
    political_keywords = ['bolsonaro', 'lula', 'pt', 'direita', 'esquerda', 'governo']

    if 'body' in df.columns:
        political_sample = df[
            df['body'].str.contains('|'.join(political_keywords), case=False, na=False)
        ].head(20)

        if len(political_sample) > 0:
            logger.info(f"🎯 Encontrados {len(political_sample)} textos com termos políticos")

            # Processar apenas a amostra política
            result = analyzer.analyze_dataset(political_sample.copy())

            if 'political_spectrum' in result['data'].columns:
                political_dist = result['data']['political_spectrum'].value_counts()
                logger.info(f"📊 Distribuição política: {political_dist.to_dict()}")
                return political_dist
            else:
                logger.warning("⚠️ Coluna political_spectrum não encontrada")
        else:
            logger.warning("⚠️ Nenhum texto político encontrado na amostra")

    return None

def run_full_analysis(df):
    """Executar análise completa do dataset."""
    logger.info("🔬 INICIANDO ANÁLISE COMPLETA:")

    # Limpar sistema antes da análise
    clean_system()

    # Inicializar analyzer
    analyzer = Analyzer()

    # Cronometrar análise
    start_time = time.time()

    try:
        # Executar análise
        result = analyzer.analyze_dataset(df.copy())

        end_time = time.time()
        duration = end_time - start_time

        # Estatísticas finais
        data = result['data']
        stats = result['stats']

        logger.info("✅ ANÁLISE CONCLUÍDA:")
        logger.info(f"⏱️ Tempo total: {duration:.2f} segundos")
        logger.info(f"📊 Registros processados: {len(data):,}")
        logger.info(f"📈 Performance: {len(data)/duration:.1f} registros/segundo")
        logger.info(f"🎯 Stages completados: {stats['stages_completed']}")
        logger.info(f"🔧 Features extraídas: {stats['features_extracted']}")
        logger.info(f"📋 Colunas geradas: {len(data.columns)}")

        # Análise política específica
        if 'political_spectrum' in data.columns:
            political_dist = data['political_spectrum'].value_counts()
            logger.info(f"🏛️ Distribuição política: {political_dist.head().to_dict()}")

        # Análise temporal
        if 'has_temporal_data' in data.columns:
            temporal_valid = data['has_temporal_data'].sum()
            logger.info(f"📅 Dados temporais válidos: {temporal_valid:,}/{len(data):,}")

        # Análise de coordenação
        if 'potential_coordination' in data.columns:
            coordination = data['potential_coordination'].sum()
            logger.info(f"🔗 Potencial coordenação: {coordination:,}/{len(data):,}")

        return result

    except Exception as e:
        logger.error(f"❌ Erro na análise: {e}")
        raise

def main():
    """Função principal do teste."""
    logger.info("🚀 TESTE digiNEV v.final - Dataset Governo Bolsonaro")
    logger.info("=" * 60)

    try:
        # Teste progressivo com amostras crescentes
        sample_sizes = [1000, 5000, 10000]

        for sample_size in sample_sizes:
            logger.info(f"\n📊 TESTE COM {sample_size:,} REGISTROS:")
            logger.info("-" * 40)

            # Carregar dataset
            df = load_bolsonaro_dataset(sample_size)

            # Executar análise
            result = run_full_analysis(df)

            # Limpar memória entre testes
            del df, result
            clean_system()

            logger.info(f"✅ Teste com {sample_size:,} registros concluído")

        logger.info("\n🎉 TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")

    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        raise

if __name__ == "__main__":
    main()