#!/usr/bin/env python3
"""
Teste de Padronização de Datetime
=================================

Testa a padronização de datetime para formato brasileiro DD/MM/AAAA HH:MM:SS.
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

def test_datetime_standardization():
    """Testar padronização de datetime."""
    logger.info("🔬 TESTE PADRONIZAÇÃO DATETIME")
    logger.info("=" * 50)

    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return

    # Carregar amostra
    df_sample = pd.read_csv(dataset_path, sep=',', nrows=20, encoding='utf-8')
    logger.info(f"✅ Amostra carregada: {len(df_sample)} registros")

    # Mostrar formatos originais de datetime
    logger.info("\n📅 FORMATOS ORIGINAIS DE DATETIME:")
    print("=" * 60)

    if 'datetime' in df_sample.columns:
        datetime_samples = df_sample['datetime'].dropna().head(10)
        for i, dt in enumerate(datetime_samples, 1):
            print(f"   {i:2d}. {dt}")

    # Executar Stage 01 com padronização
    logger.info("\n🔄 EXECUTANDO PADRONIZAÇÃO...")
    analyzer = Analyzer()
    df_processed = analyzer._stage_01_feature_extraction(df_sample.copy())

    # Verificar resultado da padronização
    if 'datetime_standardized' in df_processed.columns:
        logger.info("\n✅ RESULTADOS DA PADRONIZAÇÃO:")
        print("=" * 70)

        # Estatísticas
        valid_count = df_processed['datetime_standardized'].notna().sum()
        total_count = len(df_processed)
        success_rate = (valid_count / total_count) * 100

        print(f"📊 ESTATÍSTICAS:")
        print(f"   • Total registros: {total_count}")
        print(f"   • Conversões válidas: {valid_count}")
        print(f"   • Taxa de sucesso: {success_rate:.1f}%")

        # Comparação lado a lado
        print(f"\n📋 COMPARAÇÃO ORIGINAL → PADRONIZADO:")
        print("=" * 70)

        comparison_data = []
        for i in range(min(10, len(df_processed))):
            original = df_processed['datetime'].iloc[i] if 'datetime' in df_processed.columns else 'N/A'
            standardized = df_processed['datetime_standardized'].iloc[i]

            comparison_data.append({
                'ID': i+1,
                'Original': str(original),
                'Padronizado': str(standardized)
            })

            print(f"   {i+1:2d}. {original} → {standardized}")

        # Verificar se formato está correto (DD/MM/AAAA HH:MM:SS)
        valid_format_count = 0
        for standardized in df_processed['datetime_standardized'].dropna():
            if len(str(standardized)) == 19 and '/' in str(standardized) and ':' in str(standardized):
                valid_format_count += 1

        format_success_rate = (valid_format_count / valid_count) * 100 if valid_count > 0 else 0

        print(f"\n✅ VALIDAÇÃO DO FORMATO:")
        print(f"   • Formato DD/MM/AAAA HH:MM:SS: {valid_format_count}/{valid_count} ({format_success_rate:.1f}%)")

        # Verificar se coluna pode ser usada para análises temporais
        try:
            # Tentar converter de volta para datetime para análises
            test_conversion = pd.to_datetime(df_processed['datetime_standardized'].dropna(),
                                           format='%d/%m/%Y %H:%M:%S', errors='coerce')
            temporal_ready = test_conversion.notna().sum()
            temporal_rate = (temporal_ready / valid_count) * 100 if valid_count > 0 else 0

            print(f"   • Pronto para análise temporal: {temporal_ready}/{valid_count} ({temporal_rate:.1f}%)")

        except Exception as e:
            logger.warning(f"⚠️ Erro na validação temporal: {e}")

    else:
        logger.error("❌ Coluna datetime_standardized não foi criada")

    # Verificar metadata_columns_count
    if 'metadata_columns_count' in df_processed.columns:
        metadata_count = df_processed['metadata_columns_count'].iloc[0]
        logger.info(f"\n📊 CONTAGEM DE METADADOS: {metadata_count} colunas")

        # Listar quais são consideradas metadados
        all_columns = set(df_processed.columns)
        non_metadata = {'body', 'datetime_standardized', 'emojis_extracted', 'emojis_count',
                       'hashtags_extracted', 'urls_extracted', 'mentions_extracted'}
        metadata_columns = all_columns - non_metadata

        logger.info(f"📋 Colunas de metadados identificadas:")
        for col in sorted(metadata_columns):
            if not col.startswith(('main_', 'timestamp_', 'metadata_', 'has_')):
                logger.info(f"   • {col}")

    # Salvar resultado
    output_file = "datetime_standardization_results.csv"
    df_processed.to_csv(output_file, index=False, sep=';')

    logger.info(f"\n💾 Resultado salvo: {output_file}")
    logger.info("🎉 TESTE DE PADRONIZAÇÃO CONCLUÍDO!")

    return df_processed

if __name__ == "__main__":
    test_datetime_standardization()