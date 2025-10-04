#!/usr/bin/env python3
"""
Teste Stage 01 Simplificado
============================

Validar que as colunas desnecessárias foram removidas.
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

def test_stage01_simplified():
    """Testar Stage 01 sem colunas desnecessárias."""
    logger.info("🔬 TESTE STAGE 01 SIMPLIFICADO")
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

    # Executar Stage 01 simplificado
    analyzer = Analyzer()
    df_processed = analyzer._stage_01_feature_extraction(df_sample.copy())

    # Analisar colunas criadas
    final_columns = set(df_processed.columns)
    new_columns = final_columns - original_columns

    logger.info(f"\n🆕 COLUNAS CRIADAS ({len(new_columns)}):")
    for col in sorted(new_columns):
        print(f"   • {col}")

    # Verificar que colunas desnecessárias foram removidas
    unwanted_columns = ['has_interrogation', 'has_exclamation', 'has_caps_words', 'has_portuguese_words']
    found_unwanted = [col for col in unwanted_columns if col in final_columns]

    if found_unwanted:
        logger.error(f"❌ COLUNAS DESNECESSÁRIAS ENCONTRADAS: {found_unwanted}")
    else:
        logger.info("✅ COLUNAS DESNECESSÁRIAS REMOVIDAS COM SUCESSO")

    # Verificar colunas essenciais
    essential_columns = ['emojis_extracted', 'emojis_count', 'main_text_column', 'timestamp_column']
    missing_essential = [col for col in essential_columns if col not in final_columns]

    if missing_essential:
        logger.error(f"❌ COLUNAS ESSENCIAIS AUSENTES: {missing_essential}")
    else:
        logger.info("✅ COLUNAS ESSENCIAIS PRESENTES")

    # Salvar resultado
    output_file = "stage01_simplified_results.csv"
    df_processed.to_csv(output_file, index=False, sep=';')

    logger.info(f"\n💾 Resultado salvo: {output_file}")
    logger.info(f"📊 Total colunas finais: {len(final_columns)}")
    logger.info("🎉 TESTE CONCLUÍDO!")

    return df_processed

if __name__ == "__main__":
    test_stage01_simplified()