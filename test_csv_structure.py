#!/usr/bin/env python3
"""
Verificar Estrutura Real do CSV
==============================

Analisar a estrutura do dataset para entender as colunas existentes.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_csv_structure():
    """Analisar estrutura real do CSV."""

    dataset_path = Path("data/1_2019-2021-govbolso.csv")

    if not dataset_path.exists():
        logger.error(f"❌ Dataset não encontrado: {dataset_path}")
        return

    logger.info("🔍 ANALISANDO ESTRUTURA DO CSV")
    logger.info("=" * 50)

    # Tentar diferentes separadores
    separators = [';', ',', '\t']

    for sep in separators:
        try:
            logger.info(f"\n🧪 Testando separador: '{sep}'")

            # Ler apenas primeira linha para ver estrutura
            df_sample = pd.read_csv(dataset_path, sep=sep, nrows=5, encoding='utf-8')

            logger.info(f"📊 Colunas encontradas ({len(df_sample.columns)}): {list(df_sample.columns)}")

            # Mostrar sample da primeira linha
            if len(df_sample) > 0:
                logger.info(f"📝 Primeira linha de dados:")
                for col in df_sample.columns:
                    value = str(df_sample[col].iloc[0])[:100]
                    if len(str(df_sample[col].iloc[0])) > 100:
                        value += "..."
                    logger.info(f"   {col}: {value}")

            # Se encontrou múltiplas colunas, este é o separador correto
            if len(df_sample.columns) > 5:
                logger.info(f"✅ Separador correto encontrado: '{sep}'")
                return df_sample, sep

        except Exception as e:
            logger.warning(f"❌ Erro com separador '{sep}': {e}")

    logger.error("❌ Nenhum separador válido encontrado")
    return None, None

if __name__ == "__main__":
    analyze_csv_structure()