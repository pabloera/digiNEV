"""
Simple Text Cleaner - Versão Simplificada e Robusta
====================================================

Este módulo fornece limpeza de texto básica mas eficiente,
usando métodos tradicionais Python combinados com validação opcional via API.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SimpleTextCleaner:
    """Limpeza de texto simples e robusta"""

    def __init__(self):
        """Inicializa limpador simples"""
        self.cleaning_patterns = {
            # Múltiplos espaços
            'multiple_spaces': (r'\s+', ' '),
            # Quebras de linha múltiplas
            'multiple_newlines': (r'\n+', ' '),
            # Caracteres de controle
            'control_chars': (r'[\x00-\x1f\x7f-\x9f]', ''),
            # Espaços no início/fim
            'strip_spaces': (r'^\s+|\s+$', ''),
        }

    def clean_text_simple(
        self,
        df: pd.DataFrame,
        text_column: str = "body_cleaned",
        output_column: str = "text_cleaned",
        backup: bool = True
    ) -> pd.DataFrame:
        """
        Limpa textos usando métodos Python simples e robustos

        Args:
            df: DataFrame com textos
            text_column: Coluna de texto a limpar
            output_column: Coluna para texto limpo
            backup: Se deve fazer backup

        Returns:
            DataFrame com textos limpos
        """
        logger.info(f"Iniciando limpeza simples de {len(df)} textos")

        # Backup se solicitado
        if backup:
            backup_file = f"data/interim/simple_cleaning_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(backup_file, index=False, sep=';', encoding='utf-8')
            logger.info(f"Backup criado: {backup_file}")

        # Criar DataFrame resultado
        result_df = df.copy()

        # Verificar se coluna existe
        if text_column not in df.columns:
            logger.warning(f"Coluna '{text_column}' não encontrada. Usando 'body_cleaned'")
            text_column = 'body_cleaned' if 'body_cleaned' in df.columns else 'body'

        # Inicializar coluna de saída
        result_df[output_column] = ""

        # Processar textos
        cleaned_count = 0
        for idx, row in result_df.iterrows():
            try:
                original_text = str(row.get(text_column, "")).strip()

                if not original_text or original_text == 'nan':
                    result_df.at[idx, output_column] = ""
                    continue

                # Aplicar limpezas básicas
                cleaned_text = self._clean_single_text(original_text)
                result_df.at[idx, output_column] = cleaned_text
                cleaned_count += 1

            except Exception as e:
                logger.warning(f"Erro limpando linha {idx}: {e}")
                result_df.at[idx, output_column] = str(row.get(text_column, ""))

        logger.info(f"✅ Limpeza concluída: {cleaned_count}/{len(df)} textos processados")
        return result_df

    def _clean_single_text(self, text: str) -> str:
        """Limpa um texto individual"""
        if not text or not text.strip():
            return ""

        cleaned = text

        try:
            # Aplicar padrões de limpeza
            for pattern_name, (pattern, replacement) in self.cleaning_patterns.items():
                cleaned = re.sub(pattern, replacement, cleaned)

            # Limpezas específicas para Telegram político
            cleaned = self._clean_telegram_specific(cleaned)

            # Final cleanup
            cleaned = cleaned.strip()

            return cleaned

        except Exception as e:
            logger.warning(f"Erro na limpeza de texto: {e}")
            return text

    def _clean_telegram_specific(self, text: str) -> str:
        """Limpezas específicas para mensagens do Telegram"""

        # Preservar estruturas importantes
        cleaned = text

        # Remover apenas spam óbvio, preservar conteúdo político
        spam_patterns = [
            r'\*{5,}',  # Múltiplos asteriscos
            r'={5,}',   # Múltiplos iguais
            r'-{5,}',   # Múltiplos hífens
        ]

        for pattern in spam_patterns:
            cleaned = re.sub(pattern, '', cleaned)

        # Normalizar espaços finais
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned


def create_text_cleaner_fallback():
    """Cria instância do limpador de texto para fallback"""
    return SimpleTextCleaner()
