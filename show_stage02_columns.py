#!/usr/bin/env python3
"""
Visualização das Colunas Geradas pelo Stage 02
==============================================

Mostra todas as colunas criadas e seus tipos de dados.
"""

import pandas as pd
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_stage02_columns():
    """Analisar colunas geradas pelo Stage 02."""

    logger.info("📊 ANÁLISE DAS COLUNAS GERADAS - STAGE 02")
    logger.info("=" * 60)

    # Carregar resultados do Stage 02
    try:
        df = pd.read_csv("stage02_demonstration_results.csv", sep=';')
        logger.info(f"✅ Arquivo carregado: {len(df)} registros")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar arquivo: {e}")
        return

    # === ANÁLISE COMPLETA DAS COLUNAS ===
    logger.info(f"\n📋 TOTAL DE COLUNAS: {len(df.columns)}")
    print("=" * 80)

    # Agrupar colunas por categoria
    column_categories = {
        'Originais': [],
        'Extração Features (Stage 01)': [],
        'Features Texto (Stage 02)': [],
        'Limpeza Texto (Stage 02)': [],
        'Metadados Sistema': []
    }

    # Categorizar colunas
    for col in df.columns:
        if col.startswith('datetime,') or 'body' in col or 'url' in col:
            column_categories['Originais'].append(col)
        elif col in ['emojis_extracted', 'has_interrogation', 'has_exclamation', 'has_caps_words', 'has_portuguese_words']:
            column_categories['Extração Features (Stage 01)'].append(col)
        elif col in ['hashtags', 'urls', 'mentions', 'channel_name']:
            column_categories['Features Texto (Stage 02)'].append(col)
        elif col in ['normalized_text']:
            column_categories['Limpeza Texto (Stage 02)'].append(col)
        else:
            column_categories['Metadados Sistema'].append(col)

    # Mostrar por categoria
    for category, columns in column_categories.items():
        if columns:
            print(f"\n🔹 {category.upper()} ({len(columns)} colunas):")
            for i, col in enumerate(columns, 1):
                col_type = str(df[col].dtype)
                sample_value = str(df[col].iloc[0])[:50]
                if len(str(df[col].iloc[0])) > 50:
                    sample_value += "..."
                print(f"   {i:2d}. {col:<25} [{col_type:<10}] → {sample_value}")

    # === ANÁLISE DE DADOS ===
    print(f"\n📊 ANÁLISE DE DADOS:")
    print("=" * 50)

    # Features extraídas
    feature_stats = {}

    if 'hashtags' in df.columns:
        hashtag_count = df['hashtags'].apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0).sum()
        feature_stats['Hashtags total'] = hashtag_count

    if 'urls' in df.columns:
        url_count = df['urls'].apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0).sum()
        feature_stats['URLs total'] = url_count

    if 'mentions' in df.columns:
        mention_count = df['mentions'].apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0).sum()
        feature_stats['Mentions total'] = mention_count

    if 'emojis_extracted' in df.columns:
        emoji_total = df['emojis_extracted'].apply(lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else 0).sum()
        feature_stats['Emojis total'] = emoji_total

    # Mostrar estatísticas
    for stat_name, value in feature_stats.items():
        print(f"🔖 {stat_name}: {value}")

    # Comprimentos de texto
    if 'normalized_text' in df.columns:
        avg_clean_length = df['normalized_text'].str.len().mean()
        print(f"📏 Comprimento médio texto limpo: {avg_clean_length:.1f} caracteres")

    # === CRIAR ARQUIVO DE VISUALIZAÇÃO ===
    output_data = []

    for i, row in df.iterrows():
        record = {
            'ID': i + 1,
            'Texto_Original': str(row['datetime,body,url,hashtag,channel,is_fwrd,mentions,sender,media_type,domain,body_cleaned'])[:100] + "...",
            'Texto_Limpo': str(row['normalized_text'])[:100] + "..." if 'normalized_text' in row else "N/A",
            'Hashtags': str(row['hashtags']) if 'hashtags' in row else "N/A",
            'URLs': str(row['urls']) if 'urls' in row else "N/A",
            'Mentions': str(row['mentions']) if 'mentions' in row else "N/A",
            'Emojis': str(row['emojis_extracted']) if 'emojis_extracted' in row else "N/A",
            'Has_Portuguese': str(row['has_portuguese_words']) if 'has_portuguese_words' in row else "N/A",
            'Has_Exclamation': str(row['has_exclamation']) if 'has_exclamation' in row else "N/A"
        }
        output_data.append(record)

    # Salvar visualização
    visualization_df = pd.DataFrame(output_data)
    output_file = "stage02_columns_visualization.csv"
    visualization_df.to_csv(output_file, index=False, sep=';')

    logger.info(f"\n💾 Arquivo de visualização criado: {output_file}")

    # === RESUMO FINAL ===
    print(f"\n🎯 RESUMO STAGE 02:")
    print("=" * 40)
    print(f"📊 Total colunas processadas: {len(df.columns)}")
    print(f"📝 Registros analisados: {len(df)}")
    print(f"🔖 Features extraídas: hashtags, urls, mentions, channel_name")
    print(f"🧹 Texto normalizado: normalized_text")
    print(f"✅ Sistema pronto para Stage 03 (spaCy)")

    return df

if __name__ == "__main__":
    analyze_stage02_columns()