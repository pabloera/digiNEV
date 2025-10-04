#!/usr/bin/env python3
"""
Teste do Stage 02 Corrigido
===========================

Testa o Stage 02 com estrutura CSV correta e validação de features.
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append('src')
from analyzer import Analyzer

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stage02_corrected():
    """Testar Stage 02 com dataset estruturado corretamente."""
    logger.info("🔬 TESTE STAGE 02 CORRIGIDO - DATASET ESTRUTURADO")
    logger.info("=" * 60)

    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return

    # === CARREGAR AMOSTRA COM SEPARADOR CORRETO ===
    logger.info("📂 Carregando amostra com separador vírgula...")

    df_sample = pd.read_csv(dataset_path, sep=',', nrows=5, encoding='utf-8')

    logger.info(f"✅ Amostra carregada: {len(df_sample)} registros")
    logger.info(f"📊 Colunas disponíveis: {list(df_sample.columns)}")

    # === MOSTRAR DADOS ORIGINAIS ===
    logger.info("\n📋 DADOS ORIGINAIS:")
    print("=" * 80)

    for i in range(min(3, len(df_sample))):
        print(f"\n🔹 REGISTRO {i+1}:")
        row = df_sample.iloc[i]

        print(f"   datetime: {row['datetime']}")
        print(f"   body: {str(row['body'])[:100]}...")
        print(f"   url: {row['url']}")
        print(f"   hashtag: {row['hashtag']}")
        print(f"   channel: {row['channel']}")
        print(f"   mentions: {row['mentions']}")
        print(f"   body_cleaned: {str(row['body_cleaned'])[:80]}...")
        print("-" * 40)

    # === EXECUTAR STAGE 02 CORRIGIDO ===
    logger.info("\n🧹 EXECUTANDO STAGE 02 CORRIGIDO...")

    # Inicializar analyzer
    analyzer = Analyzer()

    # Executar Stage 01 primeiro
    df_processed = analyzer._stage_01_feature_extraction(df_sample.copy())

    # Executar Stage 02 corrigido
    df_final = analyzer._stage_02_text_preprocessing(df_processed)

    # === MOSTRAR RESULTADOS ===
    logger.info("\n✅ RESULTADOS STAGE 02 CORRIGIDO:")
    print("=" * 80)

    # Verificar se features foram validadas
    feature_columns = ['url', 'hashtag', 'mentions', 'channel']

    logger.info("🔍 VALIDAÇÃO DE FEATURES:")
    for feature in feature_columns:
        if feature in df_final.columns:
            non_empty_original = df_sample[feature].notna().sum()
            non_empty_final = df_final[feature].notna().sum()
            logger.info(f"   • {feature}: {non_empty_original} → {non_empty_final} registros válidos")

    # Verificar body_cleaned
    if 'body_cleaned' in df_final.columns:
        logger.info(f"\n🧹 BODY_CLEANED ATUALIZADO:")
        for i in range(min(2, len(df_final))):
            original_body = str(df_sample['body'].iloc[i])[:100]
            cleaned_body = str(df_final['body_cleaned'].iloc[i])[:100]
            print(f"   Original: {original_body}...")
            print(f"   Limpo:    {cleaned_body}...")
            print("-" * 50)

    # Verificar texto normalizado
    if 'normalized_text' in df_final.columns:
        avg_length = df_final['normalized_text'].str.len().mean()
        logger.info(f"\n📏 TEXTO NORMALIZADO: {avg_length:.1f} caracteres média")

    # === SALVAR RESULTADOS ===
    output_file = "stage02_corrected_results.csv"
    df_final.to_csv(output_file, index=False, sep=';')
    logger.info(f"\n💾 Resultados salvos em: {output_file}")

    logger.info("\n🎉 TESTE DO STAGE 02 CORRIGIDO CONCLUÍDO!")
    return df_final

if __name__ == "__main__":
    test_stage02_corrected()