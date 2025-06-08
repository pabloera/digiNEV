#!/usr/bin/env python3
"""
Auto Column Detector - Fallback simples para detectar colunas automaticamente
"""

import pandas as pd
from typing import Dict, List, Any
import logging


class AutoColumnDetectorAI:
    """
    Detector automático de colunas - versão fallback simples
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detecta automaticamente as colunas principais do dataset
        
        Args:
            df: DataFrame para analisar
            
        Returns:
            Mapeamento de tipos de coluna para nomes de coluna
        """
        column_mapping = {}
        
        # Detectar coluna de texto principal
        text_candidates = ['texto', 'text', 'message', 'content', 'text_cleaned']
        for col in text_candidates:
            if col in df.columns:
                column_mapping['text'] = col
                break
        
        # Detectar coluna de timestamp
        time_candidates = ['timestamp', 'date', 'datetime', 'data', 'time']
        for col in time_candidates:
            if col in df.columns:
                column_mapping['timestamp'] = col
                break
        
        # Detectar coluna de canal
        channel_candidates = ['canal', 'canais', 'channel', 'source']
        for col in channel_candidates:
            if col in df.columns:
                column_mapping['channel'] = col
                break
        
        # Detectar coluna de usuário
        user_candidates = ['usuario', 'user', 'author', 'sender']
        for col in user_candidates:
            if col in df.columns:
                column_mapping['user'] = col
                break
        
        # Log das detecções
        self.logger.info(f"Colunas detectadas: {column_mapping}")
        
        return column_mapping
    
    def validate_required_columns(self, df: pd.DataFrame, required: List[str]) -> Dict[str, bool]:
        """
        Valida se as colunas obrigatórias estão presentes
        
        Args:
            df: DataFrame para validar
            required: Lista de colunas obrigatórias
            
        Returns:
            Dicionário indicando quais colunas estão presentes
        """
        validation = {}
        for col in required:
            validation[col] = col in df.columns
        
        return validation
    
    def suggest_missing_columns(self, df: pd.DataFrame, missing: List[str]) -> Dict[str, str]:
        """
        Sugere colunas alternativas para as que estão faltando
        
        Args:
            df: DataFrame para analisar
            missing: Lista de colunas faltando
            
        Returns:
            Sugestões de colunas alternativas
        """
        suggestions = {}
        
        # Mapeamento de sugestões
        suggestion_map = {
            'texto': ['text', 'message', 'content'],
            'text': ['texto', 'message', 'content'],
            'timestamp': ['date', 'datetime', 'data'],
            'canal': ['canais', 'channel', 'source'],
            'usuario': ['user', 'author', 'sender']
        }
        
        for missing_col in missing:
            if missing_col in suggestion_map:
                for suggestion in suggestion_map[missing_col]:
                    if suggestion in df.columns:
                        suggestions[missing_col] = suggestion
                        break
        
        return suggestions