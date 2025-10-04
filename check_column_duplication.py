#!/usr/bin/env python3
"""
Verificar Duplicação de Colunas
===============================

Analisa se há colunas duplicadas com mesmo conteúdo após processamento.
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append('src')
from analyzer import Analyzer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_column_duplication():
    """Verificar duplicação de colunas após processamento."""
    logger.info("🔍 VERIFICAÇÃO DE DUPLICAÇÃO DE COLUNAS")
    logger.info("=" * 50)

    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return

    # Carregar amostra
    df_sample = pd.read_csv(dataset_path, sep=',', nrows=10, encoding='utf-8')
    logger.info(f"✅ Amostra carregada: {len(df_sample)} registros")

    # Mostrar colunas originais
    logger.info(f"\n📊 COLUNAS ORIGINAIS ({len(df_sample.columns)}):")
    for col in df_sample.columns:
        print(f"   • {col}")

    # Executar processamento completo
    analyzer = Analyzer()
    df_stage01 = analyzer._stage_01_feature_extraction(df_sample.copy())
    df_final = analyzer._stage_02_text_preprocessing(df_stage01)

    logger.info(f"\n📊 COLUNAS FINAIS ({len(df_final.columns)}):")
    for col in df_final.columns:
        print(f"   • {col}")

    # === ANÁLISE DE DUPLICAÇÃO ===
    logger.info("\n🔍 ANÁLISE DE POSSÍVEIS DUPLICAÇÕES:")
    print("=" * 60)

    potential_duplicates = [
        ('datetime', 'datetime_standardized'),
        ('hashtag', 'hashtags_extracted'),
        ('url', 'urls_extracted'),
        ('mentions', 'mentions_extracted'),
        ('body', 'normalized_text'),
        ('body', 'body_cleaned')
    ]

    duplications_found = []

    for col1, col2 in potential_duplicates:
        if col1 in df_final.columns and col2 in df_final.columns:
            logger.info(f"\n🔹 COMPARANDO: {col1} vs {col2}")

            # Amostras dos dados
            sample1 = df_final[col1].dropna().head(3).tolist()
            sample2 = df_final[col2].dropna().head(3).tolist()

            print(f"   {col1}:")
            for i, val in enumerate(sample1):
                print(f"      {i+1}. {str(val)[:80]}...")

            print(f"   {col2}:")
            for i, val in enumerate(sample2):
                print(f"      {i+1}. {str(val)[:80]}...")

            # Verificar se são funcionalmente duplicadas
            if len(sample1) > 0 and len(sample2) > 0:
                if str(sample1[0]).lower() == str(sample2[0]).lower():
                    duplications_found.append((col1, col2))
                    logger.warning(f"⚠️ POSSÍVEL DUPLICAÇÃO: {col1} ≈ {col2}")
                else:
                    logger.info(f"✅ DIFERENTES: {col1} ≠ {col2}")

    # === ANÁLISE DE FEATURES ESPECÍFICAS ===
    logger.info(f"\n🔖 ANÁLISE DE FEATURES ESPECÍFICAS:")
    print("=" * 60)

    feature_analysis = {
        'hashtags': [],
        'urls': [],
        'mentions': [],
        'datetime': []
    }

    # Coletar todas as colunas relacionadas a cada feature
    for col in df_final.columns:
        if 'hashtag' in col.lower():
            feature_analysis['hashtags'].append(col)
        elif 'url' in col.lower():
            feature_analysis['urls'].append(col)
        elif 'mention' in col.lower():
            feature_analysis['mentions'].append(col)
        elif 'datetime' in col.lower() or 'time' in col.lower():
            feature_analysis['datetime'].append(col)

    for feature_type, related_columns in feature_analysis.items():
        if len(related_columns) > 1:
            logger.info(f"\n🔹 {feature_type.upper()} - {len(related_columns)} colunas:")
            for col in related_columns:
                sample_data = df_final[col].dropna().head(2).tolist()
                print(f"   • {col}: {sample_data}")

    # === RECOMENDAÇÕES ===
    logger.info(f"\n💡 RECOMENDAÇÕES:")
    print("=" * 40)

    if duplications_found:
        logger.warning(f"⚠️ {len(duplications_found)} possíveis duplicações encontradas")
        for col1, col2 in duplications_found:
            logger.info(f"   • Considerar remover: {col1} ou {col2}")
    else:
        logger.info("✅ Nenhuma duplicação real encontrada")

    # Verificar se colunas originais ainda são necessárias
    original_features = ['hashtag', 'url', 'mentions', 'datetime']
    extracted_features = ['hashtags_extracted', 'urls_extracted', 'mentions_extracted', 'datetime_standardized']

    logger.info(f"\n📋 ANÁLISE DE NECESSIDADE:")
    for orig, extr in zip(original_features, extracted_features):
        if orig in df_final.columns and extr in df_final.columns:
            orig_valid = df_final[orig].notna().sum()
            extr_valid = df_final[extr].notna().sum() if extr != 'datetime_standardized' else df_final[extr].notna().sum()

            if orig_valid == 0 and extr_valid > 0:
                logger.info(f"   • {orig}: PODE SER REMOVIDA (vazia, {extr} tem dados)")
            elif orig_valid > 0 and extr_valid > 0:
                logger.info(f"   • {orig}: MANTER (dados únicos complementares a {extr})")

    logger.info("\n🎉 ANÁLISE DE DUPLICAÇÃO CONCLUÍDA!")

if __name__ == "__main__":
    check_column_duplication()