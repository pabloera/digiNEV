#!/usr/bin/env python3
"""
Teste Colunas Otimizadas
========================

Validar que body_cleaned foi removido e datetime foi otimizado.
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

def test_optimized_columns():
    """Testar otimizações de colunas."""
    logger.info("🔬 TESTE COLUNAS OTIMIZADAS")
    logger.info("=" * 50)

    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return

    # Carregar amostra
    df_sample = pd.read_csv(dataset_path, sep=',', nrows=10, encoding='utf-8')
    logger.info(f"✅ Amostra carregada: {len(df_sample)} registros")

    # Colunas originais
    original_columns = set(df_sample.columns)
    logger.info(f"📊 Colunas originais: {len(original_columns)}")

    # Executar processamento completo
    analyzer = Analyzer()
    df_stage01 = analyzer._stage_01_feature_extraction(df_sample.copy())
    df_final = analyzer._stage_02_text_preprocessing(df_stage01)

    # Colunas finais
    final_columns = set(df_final.columns)
    logger.info(f"📊 Colunas finais: {len(final_columns)}")

    # === VERIFICAÇÕES DE OTIMIZAÇÃO ===
    logger.info("\n✅ VERIFICAÇÕES DE OTIMIZAÇÃO:")
    print("=" * 60)

    # 1. Verificar se body_cleaned foi removido
    if 'body_cleaned' in final_columns:
        logger.error("❌ body_cleaned ainda presente (deveria ter sido removido)")
    else:
        logger.info("✅ body_cleaned removido com sucesso")

    # 2. Verificar datetime otimizado
    if 'datetime_standardized' in final_columns:
        logger.error("❌ datetime_standardized presente (deveria usar nome 'datetime')")
    elif 'datetime' in final_columns:
        logger.info("✅ datetime otimizado (nome único)")

        # Verificar formato
        datetime_sample = df_final['datetime'].dropna().head(3).tolist()
        logger.info("📅 Amostras datetime:")
        for i, dt in enumerate(datetime_sample, 1):
            print(f"   {i}. {dt}")

        # Verificar se está no formato brasileiro
        format_check = all('/' in str(dt) and len(str(dt)) == 19 for dt in datetime_sample if pd.notna(dt))
        if format_check:
            logger.info("✅ Formato brasileiro DD/MM/AAAA HH:MM:SS confirmado")
        else:
            logger.warning("⚠️ Formato pode não estar correto")
    else:
        logger.warning("⚠️ Nenhuma coluna datetime encontrada")

    # 3. Verificar se colunas essenciais estão presentes
    essential_columns = ['body', 'normalized_text', 'datetime', 'emojis_extracted']
    missing_essential = [col for col in essential_columns if col not in final_columns]

    if missing_essential:
        logger.error(f"❌ Colunas essenciais ausentes: {missing_essential}")
    else:
        logger.info("✅ Todas as colunas essenciais presentes")

    # === COMPARAÇÃO ANTES/DEPOIS ===
    logger.info("\n📊 COMPARAÇÃO ANTES/DEPOIS:")
    print("=" * 50)

    print(f"📈 Colunas originais: {len(original_columns)}")
    print(f"📈 Colunas finais: {len(final_columns)}")
    print(f"📈 Colunas adicionadas: {len(final_columns) - len(original_columns)}")

    # Colunas adicionadas
    new_columns = final_columns - original_columns
    logger.info(f"\n🆕 COLUNAS ADICIONADAS ({len(new_columns)}):")
    for col in sorted(new_columns):
        print(f"   • {col}")

    # Colunas removidas (se houver)
    removed_columns = original_columns - final_columns
    if removed_columns:
        logger.info(f"\n🗑️ COLUNAS REMOVIDAS ({len(removed_columns)}):")
        for col in sorted(removed_columns):
            print(f"   • {col}")

    # === ANÁLISE DE DUPLICAÇÃO ===
    logger.info("\n🔍 VERIFICAÇÃO FINAL DE DUPLICAÇÃO:")
    print("=" * 50)

    # Verificar se ainda há duplicações
    potential_duplicates = [
        ('body', 'body_cleaned'),
        ('datetime', 'datetime_standardized')
    ]

    duplications_found = 0
    for col1, col2 in potential_duplicates:
        if col1 in final_columns and col2 in final_columns:
            logger.warning(f"⚠️ DUPLICAÇÃO ENCONTRADA: {col1} e {col2}")
            duplications_found += 1

    if duplications_found == 0:
        logger.info("✅ Nenhuma duplicação encontrada")

    # === VALIDAÇÃO DE FUNCIONALIDADE ===
    logger.info("\n🧪 VALIDAÇÃO DE FUNCIONALIDADE:")
    print("=" * 50)

    # Verificar se datetime pode ser usado para análise temporal
    if 'datetime' in df_final.columns:
        try:
            test_conversion = pd.to_datetime(df_final['datetime'].dropna(),
                                           format='%d/%m/%Y %H:%M:%S', errors='coerce')
            temporal_ready = test_conversion.notna().sum()
            logger.info(f"📅 Datetime funcional para análise temporal: {temporal_ready} registros")
        except Exception as e:
            logger.error(f"❌ Erro na validação temporal: {e}")

    # Verificar se normalized_text está funcionando
    if 'normalized_text' in df_final.columns:
        text_stats = df_final['normalized_text'].str.len()
        avg_length = text_stats.mean()
        logger.info(f"📝 Texto normalizado funcional: {avg_length:.1f} chars média")

    # Salvar resultado
    output_file = "optimized_columns_results.csv"
    df_final.to_csv(output_file, index=False, sep=';')

    logger.info(f"\n💾 Resultado salvo: {output_file}")
    logger.info("🎉 TESTE DE OTIMIZAÇÃO CONCLUÍDO!")

    return df_final

if __name__ == "__main__":
    test_optimized_columns()