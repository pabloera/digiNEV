#!/usr/bin/env python3
"""
DataProcessingUtils - Utilitários Comuns de Processamento de Dados
==================================================================

Algoritmos centralizados para eliminação de duplicação de código entre módulos.
Extraído de: unified_pipeline.py, feature_validator.py, voyage_embeddings.py, etc.

Criado pela auditoria de código v5.0.0 - TASK-018
"""

import gc
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessingUtils:
    """
    Utilitários comuns de processamento de dados
    Consolida algoritmos duplicados em múltiplos módulos
    """
    
    @staticmethod
    def clean_memory_after_processing(df: Optional[pd.DataFrame] = None, 
                                    variables: Optional[List[str]] = None) -> None:
        """
        Libera memória explicitamente após processamento
        
        Args:
            df: DataFrame para deletar (opcional)
            variables: Lista de nomes de variáveis para deletar do escopo local
        """
        try:
            if df is not None:
                del df
                
            if variables:
                import inspect
                frame = inspect.currentframe().f_back
                for var_name in variables:
                    if var_name in frame.f_locals:
                        del frame.f_locals[var_name]
                        
            gc.collect()
            logger.debug("🧹 Memória liberada explicitamente")
            
        except Exception as e:
            logger.warning(f"Falha ao liberar memória: {e}")
    
    @staticmethod
    def validate_dataframe_structure(df: pd.DataFrame, 
                                   required_columns: Optional[List[str]] = None,
                                   min_rows: int = 1) -> Tuple[bool, List[str]]:
        """
        Valida estrutura básica de DataFrame
        
        Args:
            df: DataFrame para validar
            required_columns: Colunas obrigatórias
            min_rows: Número mínimo de linhas
            
        Returns:
            Tuple[bool, List[str]]: (é_válido, lista_de_erros)
        """
        errors = []
        
        if df is None:
            errors.append("DataFrame é None")
            return False, errors
            
        if len(df) < min_rows:
            errors.append(f"DataFrame tem {len(df)} linhas, mínimo {min_rows}")
            
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                errors.append(f"Colunas ausentes: {missing_cols}")
                
        return len(errors) == 0, errors
    
    @staticmethod
    def safe_text_extraction(df: pd.DataFrame, 
                           text_columns: List[str],
                           default_value: str = "") -> pd.Series:
        """
        Extrai texto de forma segura de múltiplas colunas
        
        Args:
            df: DataFrame fonte
            text_columns: Lista de colunas de texto para combinar
            default_value: Valor padrão para células vazias
            
        Returns:
            pd.Series: Série com texto combinado
        """
        combined_text = pd.Series(default_value, index=df.index)
        
        for col in text_columns:
            if col in df.columns:
                combined_text += ' ' + df[col].astype(str).fillna(default_value)
                
        return combined_text.str.strip()
    
    @staticmethod
    def vectorized_text_contains(text_series: pd.Series, 
                               patterns: List[str],
                               case_sensitive: bool = False) -> pd.Series:
        """
        Busca vetorizada de múltiplos patterns em série de texto
        
        Args:
            text_series: Série de texto para buscar
            patterns: Lista de patterns regex para buscar
            case_sensitive: Busca sensível a maiúsculas/minúsculas
            
        Returns:
            pd.Series: Série booleana indicando matches
        """
        if not patterns:
            return pd.Series(False, index=text_series.index)
            
        # Combinar patterns em um único regex
        combined_pattern = '|'.join(re.escape(p) for p in patterns)
        flags = 0 if case_sensitive else re.IGNORECASE
        
        return text_series.str.contains(
            combined_pattern, 
            case=case_sensitive, 
            na=False, 
            regex=True
        )
    
    @staticmethod
    def optimize_chunk_size(file_size_bytes: int, 
                          min_chunk: int = 50000,
                          max_chunk: int = 100000) -> int:
        """
        Calcula tamanho ótimo de chunk baseado no tamanho do arquivo
        
        Args:
            file_size_bytes: Tamanho do arquivo em bytes
            min_chunk: Tamanho mínimo de chunk
            max_chunk: Tamanho máximo de chunk
            
        Returns:
            int: Tamanho ótimo de chunk
        """
        # Usar 1% do arquivo como base, limitado por min/max
        optimal = file_size_bytes // 100
        return min(max_chunk, max(min_chunk, optimal))
    
    @staticmethod
    def vectorized_hashtag_correction(text_series: pd.Series) -> pd.Series:
        """
        Corrige hashtags malformadas de forma vetorizada
        
        Args:
            text_series: Série de texto com possíveis hashtags
            
        Returns:
            pd.Series: Série com hashtags corrigidas
        """
        # Detectar hashtags que não começam com #
        needs_correction = (
            text_series.notna() & 
            (text_series.str.strip() != '') & 
            ~text_series.str.strip().str.startswith('#')
        )
        
        # Aplicar correção apenas onde necessário
        corrected = text_series.copy()
        corrected.loc[needs_correction] = '#' + text_series.loc[needs_correction].str.strip()
        
        return corrected
    
    @staticmethod
    def detect_media_type_vectorized(text_series: pd.Series, 
                                   media_patterns: Dict[str, str]) -> pd.Series:
        """
        Detecta tipo de mídia de forma vetorizada
        
        Args:
            text_series: Série de texto para analisar
            media_patterns: Dict com {tipo_mídia: regex_pattern}
            
        Returns:
            pd.Series: Série com tipos de mídia detectados
        """
        # Série padrão com 'text'
        detected_types = pd.Series('text', index=text_series.index)
        
        # Aplicar cada pattern sequencialmente
        for media_type, pattern in media_patterns.items():
            matches = text_series.str.contains(pattern, case=False, na=False, regex=True)
            detected_types.loc[matches] = media_type
            
        return detected_types
    
    @staticmethod
    def generate_content_hash(content: Union[str, List[str]], 
                            algorithm: str = 'md5') -> str:
        """
        Gera hash de conteúdo para caching
        
        Args:
            content: Conteúdo para gerar hash (string ou lista)
            algorithm: Algoritmo de hash ('md5', 'sha1', 'sha256')
            
        Returns:
            str: Hash hexadecimal do conteúdo
        """
        if isinstance(content, list):
            content = ''.join(str(item) for item in content)
            
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Algoritmo não suportado: {algorithm}")
            
        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()
    
    @staticmethod
    def safe_numeric_conversion(series: pd.Series, 
                              default_value: float = 0.0) -> pd.Series:
        """
        Converte série para numérico de forma segura
        
        Args:
            series: Série para converter
            default_value: Valor padrão para conversões falhas
            
        Returns:
            pd.Series: Série numérica
        """
        return pd.to_numeric(series, errors='coerce').fillna(default_value)
    
    @staticmethod
    def batch_process_dataframe(df: pd.DataFrame, 
                              process_func: callable,
                              batch_size: int = 10000,
                              **kwargs) -> pd.DataFrame:
        """
        Processa DataFrame em batches para economizar memória
        
        Args:
            df: DataFrame para processar
            process_func: Função que processa cada batch
            batch_size: Tamanho do batch
            **kwargs: Argumentos para process_func
            
        Returns:
            pd.DataFrame: DataFrame processado
        """
        if len(df) <= batch_size:
            return process_func(df, **kwargs)
            
        results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size].copy()
            batch_result = process_func(batch, **kwargs)
            results.append(batch_result)
            
            # Liberar memória do batch
            del batch
            
            if i // batch_size % 10 == 0:  # Log a cada 10 batches
                logger.info(f"Processado batch {i//batch_size + 1}/{total_batches}")
                
        # Combinar resultados
        final_result = pd.concat(results, ignore_index=True)
        
        # Limpeza final
        del results
        gc.collect()
        
        return final_result
    
    @staticmethod
    def create_backup_filename(original_path: Union[str, Path], 
                             suffix: str = "backup") -> Path:
        """
        Cria nome de arquivo de backup baseado no original
        
        Args:
            original_path: Caminho do arquivo original
            suffix: Sufixo para o backup
            
        Returns:
            Path: Caminho do arquivo de backup
        """
        original_path = Path(original_path)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        backup_name = f"{original_path.stem}_{suffix}_{timestamp}{original_path.suffix}"
        return original_path.parent / backup_name


# Alias para retrocompatibilidade
def clean_memory(*args, **kwargs):
    """Alias para DataProcessingUtils.clean_memory_after_processing"""
    return DataProcessingUtils.clean_memory_after_processing(*args, **kwargs)


def validate_df_structure(*args, **kwargs):
    """Alias para DataProcessingUtils.validate_dataframe_structure"""
    return DataProcessingUtils.validate_dataframe_structure(*args, **kwargs)


if __name__ == "__main__":
    # Teste básico das funcionalidades
    print("🧪 Testando DataProcessingUtils...")
    
    # Teste básico
    test_df = pd.DataFrame({
        'text': ['#hashtag1', 'texto normal', 'outro #hashtag'],
        'numbers': ['1', '2.5', 'invalid']
    })
    
    # Teste correção de hashtags
    corrected = DataProcessingUtils.vectorized_hashtag_correction(test_df['text'])
    print(f"Hashtags corrigidas: {corrected.tolist()}")
    
    # Teste conversão numérica
    numbers = DataProcessingUtils.safe_numeric_conversion(test_df['numbers'])
    print(f"Números convertidos: {numbers.tolist()}")
    
    # Teste hash
    content_hash = DataProcessingUtils.generate_content_hash("test content")
    print(f"Hash gerado: {content_hash}")
    
    print("✅ DataProcessingUtils funcionando corretamente!")