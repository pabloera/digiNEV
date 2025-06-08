"""
Robust CSV Parsing Utility for Dashboard
Based on the unified pipeline's robust CSV parsing logic
"""

import pandas as pd
import os
import logging
import csv
from typing import Optional, Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)

class RobustCSVParser:
    """
    Robust CSV parser that implements the same separator detection and 
    parsing logic as the unified pipeline to avoid header concatenation issues.
    """
    
    def __init__(self):
        # Set CSV field size limit to handle large fields
        csv.field_size_limit(500000)
    
    def detect_separator(self, file_path: str) -> str:
        """
        Detecta o separador do CSV analisando a primeira linha com validação robusta
        Based on unified_pipeline.py detect_separator function
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                comma_count = first_line.count(',')
                semicolon_count = first_line.count(';')
                
                logger.debug(f"Primeira linha: {first_line[:100]}...")
                logger.debug(f"Vírgulas: {comma_count}, Ponto-e-vírgulas: {semicolon_count}")
                
                # Se há apenas 1 coluna detectada, provavelmente separador errado
                if comma_count == 0 and semicolon_count == 0:
                    logger.warning("Nenhum separador detectado na primeira linha")
                    return ';'  # Fallback padrão para datasets do projeto
                
                # Priorizar ponto-e-vírgula se há mais ou igual quantidade
                if semicolon_count >= comma_count and semicolon_count > 0:
                    return ';'
                elif comma_count > 0:
                    return ','
                else:
                    return ';'  # Fallback padrão
        except Exception as e:
            logger.error(f"Erro na detecção de separador: {e}")
            return ';'  # Default para ponto-e-vírgula (padrão do projeto)
    
    def _try_parse_csv(self, file_path: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Tenta parsear CSV com diferentes configurações e validação
        Based on unified_pipeline.py try_parse_csv function
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            if df is not None:
                # VALIDAÇÃO CRÍTICA: Verificar se CSV foi parseado corretamente
                if len(df.columns) == 1 and ',' in df.columns[0]:
                    logger.error(f"❌ CSV mal parseado: header concatenado detectado")
                    logger.error(f"Header problemático: {df.columns[0][:100]}...")
                    return None  # Forçar nova tentativa
                
                # Verificar se temos colunas esperadas (incluindo novas colunas políticas)
                expected_cols = [
                    'message_id', 'datetime', 'body', 'body_cleaned', 'channel', 'text', 'content',
                    # Novas colunas de feature validation
                    'text_length', 'word_count', 'emoji_count', 'media_type', 'is_forwarded',
                    # Novas colunas de análise política
                    'political_alignment', 'conspiracy_score', 'negacionism_score', 
                    'emotional_tone', 'misinformation_risk'
                ]
                if not any(col in df.columns for col in expected_cols):
                    logger.warning(f"⚠️  Colunas esperadas não encontradas: {list(df.columns)[:5]}")
                    # Não é erro crítico para o dashboard, pode ser dataset diferente
                
                logger.debug(f"✅ CSV parseado: {len(df.columns)} colunas, {len(df)} linhas")
            return df
        except Exception as e:
            logger.warning(f"Tentativa de parsing falhou: {e}")
            return None
    
    def _get_parse_configurations(self, separators_to_try: List[str]) -> List[Dict[str, Any]]:
        """
        Gera configurações de parsing para diferentes separadores
        Based on unified_pipeline.py parse_configs logic
        """
        parse_configs = []
        
        for sep in separators_to_try:
            parse_configs.extend([
                # Configuração padrão com separador detectado
                {
                    'sep': sep,
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'engine': 'python',
                    'quoting': 1,  # QUOTE_ALL
                    'skipinitialspace': True
                },
                # Configuração com escape e field size
                {
                    'sep': sep,
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'engine': 'python',
                    'quoting': 3,  # QUOTE_NONE
                    'escapechar': '\\'
                },
                # Configuração básica robusta
                {
                    'sep': sep,
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'engine': 'python',
                    'quoting': 0,  # QUOTE_MINIMAL
                    'doublequote': True
                },
                # Configuração de fallback sem pandas C engine
                {
                    'sep': sep,
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'engine': 'python',
                    'quoting': 2,  # QUOTE_NONNUMERIC
                    'doublequote': True
                },
                # Configuração ultra-robusta para arquivos problemáticos
                {
                    'sep': sep,
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'engine': 'python',
                    'quoting': 3,  # QUOTE_NONE
                    'escapechar': None,
                    'doublequote': False,
                    'skipinitialspace': True
                }
            ])
        
        return parse_configs
    
    def load_csv_robust(self, file_path: str, nrows: Optional[int] = None, 
                       chunksize: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Carrega CSV com detecção automática de separador e parsing robusto
        
        Args:
            file_path: Caminho para o arquivo CSV
            nrows: Número máximo de linhas a carregar (para dashboard performance)
            chunksize: Se especificado, carrega em chunks (para arquivos grandes)
            
        Returns:
            DataFrame carregado ou None se falhou
        """
        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado: {file_path}")
            return None
        
        # Verificar tamanho do arquivo
        file_size = os.path.getsize(file_path)
        logger.info(f"📁 Carregando arquivo: {file_path} ({file_size/1024:.1f}KB)")
        
        # Detectar separador automaticamente
        detected_sep = self.detect_separator(file_path)
        logger.info(f"📊 Separador detectado: '{detected_sep}'")
        
        # Preparar configurações com ambos separadores para garantir parsing correto
        separators_to_try = [detected_sep]
        if detected_sep == ';':
            separators_to_try.append(',')  # Tentar vírgula como fallback
        else:
            separators_to_try.append(';')  # Tentar ponto-e-vírgula como fallback
        
        parse_configs = self._get_parse_configurations(separators_to_try)
        logger.info(f"📋 Configurações de parsing preparadas: {len(parse_configs)} tentativas")
        
        # Se arquivo muito grande ou chunksize especificado, usar chunks
        if chunksize or file_size > 200 * 1024 * 1024:  # >200MB
            return self._load_with_chunks(file_path, parse_configs, chunksize, nrows)
        else:
            return self._load_complete(file_path, parse_configs, nrows)
    
    def _load_with_chunks(self, file_path: str, parse_configs: List[Dict], 
                         chunksize: Optional[int], nrows: Optional[int]) -> Optional[pd.DataFrame]:
        """Carrega arquivo usando chunks"""
        effective_chunksize = chunksize or 10000
        logger.info(f"Arquivo grande detectado, usando processamento em chunks (size: {effective_chunksize})")
        
        for i, config in enumerate(parse_configs):
            try:
                chunk_iterator = pd.read_csv(file_path, chunksize=effective_chunksize, **config)
                chunks = []
                total_rows = 0
                
                for chunk in chunk_iterator:
                    # Validar cada chunk
                    if len(chunk.columns) == 1 and ',' in chunk.columns[0]:
                        logger.error(f"❌ Chunk mal parseado com config {i+1}, tentando próxima")
                        chunks = []  # Limpar chunks inválidos
                        break
                    
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    # Parar se atingiu nrows (para dashboard performance)
                    if nrows and total_rows >= nrows:
                        break
                
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    
                    # Limitar a nrows se especificado
                    if nrows and len(df) > nrows:
                        df = df.head(nrows)
                    
                    logger.info(f"✅ Parsing bem-sucedido em chunks (config {i+1}): {len(df)} linhas, {len(df.columns)} colunas")
                    return df
                    
            except Exception as e:
                logger.warning(f"Configuração de chunk {i+1} falhou: {e}")
                continue
        
        logger.error("❌ Todas as configurações de parsing em chunks falharam")
        return None
    
    def _load_complete(self, file_path: str, parse_configs: List[Dict], 
                      nrows: Optional[int]) -> Optional[pd.DataFrame]:
        """Carrega arquivo completo"""
        logger.info(f"Carregando arquivo completo")
        
        for i, config in enumerate(parse_configs):
            # Adicionar nrows se especificado
            if nrows:
                config['nrows'] = nrows
            
            df = self._try_parse_csv(file_path, **config)
            if df is not None:
                logger.info(f"✅ Parsing bem-sucedido com configuração {i+1}: {len(df)} linhas, {len(df.columns)} colunas")
                logger.debug(f"Colunas detectadas: {list(df.columns)[:10]}")
                return df
        
        logger.error("❌ Todas as configurações de parsing falharam")
        return None
    
    def validate_csv_detailed(self, file_path: str) -> Dict[str, Any]:
        """
        Valida estrutura do arquivo CSV com feedback detalhado
        Compatível com a função validate_csv_detailed existente no dashboard
        """
        try:
            # Detectar separador
            detected_sep = self.detect_separator(file_path)
            
            # Tentar carregar uma amostra pequena
            df = self.load_csv_robust(file_path, nrows=10)
            
            if df is not None and len(df.columns) > 1:
                return {
                    'valid': True,
                    'message': f'CSV válido - {len(df.columns)} colunas, separador: "{detected_sep}"',
                    'separator': detected_sep,
                    'columns': len(df.columns),
                    'rows_sample': len(df),
                    'column_names': list(df.columns)
                }
            else:
                return {
                    'valid': False,
                    'message': 'Arquivo CSV inválido ou mal formado',
                    'separator': detected_sep,
                    'columns': 0,
                    'rows_sample': 0
                }
                
        except Exception as e:
            return {
                'valid': False,
                'message': f'Erro ao validar CSV: {str(e)}',
                'separator': ';',
                'columns': 0,
                'rows_sample': 0
            }


# Singleton instance for easy usage
robust_csv_parser = RobustCSVParser()

# Convenience functions for backward compatibility
def load_csv_robust(file_path: str, nrows: Optional[int] = None, 
                   chunksize: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Carrega CSV com parsing robusto"""
    return robust_csv_parser.load_csv_robust(file_path, nrows, chunksize)

def validate_csv_detailed(file_path: str) -> Dict[str, Any]:
    """Valida CSV com detalhes"""
    return robust_csv_parser.validate_csv_detailed(file_path)

def detect_separator(file_path: str) -> str:
    """Detecta separador do CSV"""
    return robust_csv_parser.detect_separator(file_path)