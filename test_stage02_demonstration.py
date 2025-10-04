#!/usr/bin/env python3
"""
Demonstração do Stage 02 com Dados Reais
========================================

Mostra o procedimento completo de extração de features e limpeza
aplicado a uma amostra real do dataset.
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

def load_sample_data():
    """Carregar amostra de 10 registros do dataset."""
    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return None

    logger.info("📂 Carregando amostra de 10 registros...")

    # Carregar apenas 10 registros para demonstração
    df_sample = pd.read_csv(dataset_path, sep=';', nrows=10, encoding='utf-8')

    logger.info(f"✅ Amostra carregada: {len(df_sample)} registros")
    logger.info(f"📊 Colunas disponíveis: {list(df_sample.columns)}")

    return df_sample

def demonstrate_stage02():
    """Demonstrar Stage 02 completo com dados reais."""
    logger.info("🔬 DEMONSTRAÇÃO STAGE 02 - DADOS REAIS")
    logger.info("=" * 60)

    # === CARREGAR DADOS ===
    df_sample = load_sample_data()
    if df_sample is None:
        return

    # === MOSTRAR DADOS ORIGINAIS ===
    logger.info("\n📋 DADOS ORIGINAIS (Primeiros 3 registros):")
    print("=" * 80)

    for i in range(min(3, len(df_sample))):
        print(f"\n🔹 REGISTRO {i+1}:")
        row = df_sample.iloc[i]

        # Mostrar colunas principais
        for col in df_sample.columns:
            value = str(row[col])[:100]  # Limitar para visualização
            if len(str(row[col])) > 100:
                value += "..."
            print(f"   {col}: {value}")
        print("-" * 40)

    # === EXECUTAR STAGE 02 ===
    logger.info("\n🧹 EXECUTANDO STAGE 02 COMPLETO...")

    # Inicializar analyzer
    analyzer = Analyzer()

    # Executar Stage 01 primeiro (identificar coluna principal)
    df_processed = analyzer._stage_01_feature_extraction(df_sample.copy())

    # Executar Stage 02 (foco da demonstração)
    df_final = analyzer._stage_02_text_preprocessing(df_processed)

    # === MOSTRAR RESULTADOS ===
    logger.info("\n✅ RESULTADOS STAGE 02:")
    print("=" * 80)

    # Mostrar novas colunas criadas
    new_columns = [col for col in df_final.columns if col not in df_sample.columns]
    logger.info(f"🆕 Novas colunas criadas: {len(new_columns)}")
    for col in new_columns:
        print(f"   • {col}")

    # === DEMONSTRAR EXTRAÇÃO DE FEATURES ===
    logger.info("\n🔍 FEATURES EXTRAÍDAS (3 primeiros registros):")
    print("=" * 80)

    feature_columns = ['hashtags', 'urls', 'mentions', 'channel_name']

    for i in range(min(3, len(df_final))):
        print(f"\n🔹 REGISTRO {i+1} - FEATURES:")

        # Texto original
        main_col = df_final['main_text_column'].iloc[0]
        original_text = str(df_final[main_col].iloc[i])[:200]
        print(f"   📝 Texto original: {original_text}...")

        # Features extraídas
        for feature in feature_columns:
            if feature in df_final.columns:
                value = df_final[feature].iloc[i]
                print(f"   🔖 {feature}: {value}")

        # Texto limpo
        clean_text = str(df_final['normalized_text'].iloc[i])[:150]
        print(f"   🧹 Texto limpo: {clean_text}...")
        print("-" * 50)

    # === ESTATÍSTICAS DE LIMPEZA ===
    logger.info("\n📊 ESTATÍSTICAS DE LIMPEZA:")
    print("=" * 50)

    main_col = df_final['main_text_column'].iloc[0]

    # Comprimento médio antes e depois
    original_lengths = df_final[main_col].str.len()
    clean_lengths = df_final['normalized_text'].str.len()

    print(f"📏 Comprimento médio original: {original_lengths.mean():.1f} caracteres")
    print(f"📏 Comprimento médio limpo: {clean_lengths.mean():.1f} caracteres")
    print(f"📉 Redução média: {((original_lengths.mean() - clean_lengths.mean()) / original_lengths.mean() * 100):.1f}%")

    # Contagem de features
    for feature in feature_columns:
        if feature in df_final.columns:
            if feature == 'channel_name':
                non_empty = df_final[feature].notna().sum()
            else:
                non_empty = df_final[feature].apply(lambda x: len(x) > 0 if isinstance(x, list) else bool(x)).sum()
            print(f"🔖 {feature}: {non_empty}/{len(df_final)} registros ({non_empty/len(df_final)*100:.1f}%)")

    # === SALVAR RESULTADOS ===
    output_file = "stage02_demonstration_results.csv"
    df_final.to_csv(output_file, index=False, sep=';')
    logger.info(f"\n💾 Resultados salvos em: {output_file}")

    logger.info("\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    return df_final

if __name__ == "__main__":
    demonstrate_stage02()