#!/usr/bin/env python3
"""
Teste Stage 02 com 100 Casos Reais
==================================

Processamento de 100 registros reais para validar melhorias do Stage 02.
"""

import pandas as pd
import logging
import sys
import time
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

def test_stage02_100_cases():
    """Testar Stage 02 com 100 casos reais."""
    logger.info("🔬 TESTE STAGE 02 - 100 CASOS REAIS")
    logger.info("=" * 60)

    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return

    # === CARREGAR 100 REGISTROS ALEATÓRIOS ===
    logger.info("📂 Carregando 100 registros aleatórios do dataset...")

    start_time = time.time()

    # Primeiro, contar total de linhas para amostragem aleatória
    total_lines = sum(1 for line in open(dataset_path, encoding='utf-8')) - 1  # -1 para header
    logger.info(f"📊 Dataset total: {total_lines:,} registros")

    # Gerar índices aleatórios
    import random
    random.seed(42)  # Para reprodutibilidade
    random_indices = sorted(random.sample(range(total_lines), min(100, total_lines)))

    # Carregar dataset completo e selecionar índices aleatórios
    df_full = pd.read_csv(dataset_path, sep=',', encoding='utf-8')
    df_sample = df_full.iloc[random_indices].reset_index(drop=True)

    load_time = time.time() - start_time

    logger.info(f"✅ Amostra aleatória: {len(df_sample)} registros de {total_lines:,} totais em {load_time:.2f}s")
    logger.info(f"📊 Colunas: {list(df_sample.columns)}")
    logger.info(f"🎲 Índices aleatórios: {random_indices[:10]}... (seed=42)")

    # === ANÁLISE INICIAL DOS DADOS ===
    logger.info("\n📋 ANÁLISE INICIAL DOS DADOS:")
    print("=" * 60)

    # Estatísticas de features existentes
    feature_stats = {}

    if 'url' in df_sample.columns:
        url_count = df_sample['url'].notna().sum()
        feature_stats['URLs'] = f"{url_count}/100 ({url_count}%)"

    if 'hashtag' in df_sample.columns:
        hashtag_count = df_sample['hashtag'].notna().sum()
        feature_stats['Hashtags'] = f"{hashtag_count}/100 ({hashtag_count}%)"

    if 'mentions' in df_sample.columns:
        mention_count = df_sample['mentions'].notna().sum()
        feature_stats['Mentions'] = f"{mention_count}/100 ({mention_count}%)"

    if 'channel' in df_sample.columns:
        channel_count = df_sample['channel'].notna().sum()
        feature_stats['Channels'] = f"{channel_count}/100 ({channel_count}%)"

    print("🔖 FEATURES ORIGINAIS:")
    for feature, stat in feature_stats.items():
        print(f"   • {feature}: {stat}")

    # Análise de body
    if 'body' in df_sample.columns:
        body_lengths = df_sample['body'].str.len()
        body_non_empty = df_sample['body'].notna().sum()
        print(f"\n📝 ANÁLISE DO BODY:")
        print(f"   • Registros com body: {body_non_empty}/100 ({body_non_empty}%)")
        print(f"   • Comprimento médio: {body_lengths.mean():.1f} chars")
        print(f"   • Comprimento máximo: {body_lengths.max():.0f} chars")

    # === EXECUTAR STAGE 02 COMPLETO ===
    logger.info("\n🧹 EXECUTANDO STAGE 02 COM MELHORIAS...")

    # Inicializar analyzer
    analyzer = Analyzer()

    # Executar pipeline completo (Stage 01 + 02)
    process_start = time.time()

    # Stage 01
    df_stage01 = analyzer._stage_01_feature_extraction(df_sample.copy())

    # Stage 02 (com melhorias)
    df_final = analyzer._stage_02_text_preprocessing(df_stage01)

    process_time = time.time() - process_start

    # === ANÁLISE DOS RESULTADOS ===
    logger.info("\n✅ ANÁLISE DOS RESULTADOS:")
    print("=" * 60)

    # Performance
    records_per_second = 100 / process_time
    print(f"⚡ PERFORMANCE:")
    print(f"   • Tempo processamento: {process_time:.2f}s")
    print(f"   • Velocidade: {records_per_second:.1f} registros/segundo")

    # Novas colunas criadas
    original_cols = set(df_sample.columns)
    final_cols = set(df_final.columns)
    new_cols = final_cols - original_cols

    print(f"\n🆕 NOVAS COLUNAS ({len(new_cols)}):")
    for col in sorted(new_cols):
        sample_value = str(df_final[col].iloc[0])[:40]
        if len(str(df_final[col].iloc[0])) > 40:
            sample_value += "..."
        print(f"   • {col}: {sample_value}")

    # Validação de features
    print(f"\n🔍 VALIDAÇÃO DE FEATURES:")
    validation_results = {}

    for feature in ['url', 'hashtag', 'mentions', 'channel']:
        if feature in df_final.columns:
            original_valid = df_sample[feature].notna().sum()
            final_valid = df_final[feature].notna().sum()

            if final_valid >= original_valid:
                status = "✅"
            else:
                status = "⚠️"

            validation_results[feature] = {
                'original': original_valid,
                'final': final_valid,
                'status': status
            }

            print(f"   {status} {feature}: {original_valid} → {final_valid}")

    # Limpeza de texto
    if 'normalized_text' in df_final.columns:
        normalized_lengths = df_final['normalized_text'].str.len()
        print(f"\n🧹 LIMPEZA DE TEXTO:")
        print(f"   • Texto normalizado: {normalized_lengths.mean():.1f} chars média")

        if 'body' in df_sample.columns:
            original_lengths = df_sample['body'].str.len()
            reduction = ((original_lengths.mean() - normalized_lengths.mean()) / original_lengths.mean() * 100)
            print(f"   • Redução tamanho: {reduction:.1f}%")

    # === EXEMPLOS DE TRANSFORMAÇÃO ===
    logger.info("\n📋 EXEMPLOS DE TRANSFORMAÇÃO (5 casos):")
    print("=" * 80)

    for i in range(min(5, len(df_final))):
        print(f"\n🔹 CASO {i+1}:")

        # Texto original
        if 'body' in df_sample.columns and pd.notna(df_sample['body'].iloc[i]):
            original = str(df_sample['body'].iloc[i])[:100]
            print(f"   📝 Original: {original}...")

        # Features detectadas
        features_found = []
        if 'url' in df_final.columns and pd.notna(df_final['url'].iloc[i]):
            features_found.append(f"URL: {df_final['url'].iloc[i]}")
        if 'mentions' in df_final.columns and pd.notna(df_final['mentions'].iloc[i]):
            features_found.append(f"Mention: {df_final['mentions'].iloc[i]}")

        if features_found:
            print(f"   🔖 Features: {'; '.join(features_found)}")

        # Texto limpo
        if 'normalized_text' in df_final.columns:
            normalized = str(df_final['normalized_text'].iloc[i])[:100]
            print(f"   🧹 Limpo: {normalized}...")

        print("-" * 50)

    # === SALVAR RESULTADOS ===
    output_file = "stage02_100_cases_results.csv"
    df_final.to_csv(output_file, index=False, sep=';')

    # Salvar relatório
    report_file = "stage02_100_cases_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO STAGE 02 - 100 CASOS REAIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"📊 Dataset: 100 registros processados\n")
        f.write(f"⚡ Performance: {records_per_second:.1f} registros/segundo\n")
        f.write(f"🆕 Colunas criadas: {len(new_cols)}\n\n")

        f.write("🔍 VALIDAÇÃO DE FEATURES:\n")
        for feature, result in validation_results.items():
            f.write(f"   • {feature}: {result['original']} → {result['final']} {result['status']}\n")

        f.write(f"\n🧹 LIMPEZA DE TEXTO:\n")
        if 'normalized_text' in df_final.columns:
            f.write(f"   • Texto normalizado: {normalized_lengths.mean():.1f} chars média\n")

    logger.info(f"\n💾 RESULTADOS SALVOS:")
    logger.info(f"   • Dados: {output_file}")
    logger.info(f"   • Relatório: {report_file}")

    logger.info("\n🎉 TESTE 100 CASOS CONCLUÍDO COM SUCESSO!")

    return df_final

if __name__ == "__main__":
    test_stage02_100_cases()