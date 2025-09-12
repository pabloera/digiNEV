"""
Módulo Data Validator - Gatekeeper de Validação de Dados (Fase 2)

Este módulo implementa o primeiro estágio do pipeline para validar,
limpar e padronizar os dados de entrada, garantindo que os estágios
subsequentes sempre recebam dados em formato conhecido e confiável.
"""

import csv
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import chardet
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

class DataSchema(BaseModel):
    """Schema esperado para os dados de entrada."""
    required_columns: List[str] = Field(
        default=["message", "date", "channel"],
        description="Colunas obrigatórias no dataset"
    )
    optional_columns: List[str] = Field(
        default=["user", "views", "forwards"],
        description="Colunas opcionais que podem estar presentes"
    )
    expected_encodings: List[str] = Field(
        default=["utf-8", "utf-8-sig", "latin-1", "cp1252"],
        description="Encodings aceitos para os arquivos"
    )
    max_file_size_mb: int = Field(
        default=500,
        description="Tamanho máximo do arquivo em MB"
    )
    supported_delimiters: List[str] = Field(
        default=[",", ";", "\t"],
        description="Delimitadores suportados para CSV"
    )

class ValidationResult(BaseModel):
    """Resultado da validação de um arquivo."""
    is_valid: bool
    file_path: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    detected_encoding: Optional[str] = None
    detected_delimiter: Optional[str] = None
    row_count: int = 0
    column_count: int = 0
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)

class DataValidator:
    """
    Gatekeeper de validação de dados para o pipeline digiNEV.
    
    Funcionalidades:
    - Validação de encoding de arquivos
    - Detecção automática de delimitadores
    - Verificação de integridade estrutural
    - Validação de esquema de dados
    - Sistema de quarentena para arquivos inválidos
    """
    
    def __init__(self, config: Dict[str, Any], quarantine_dir: Optional[Path] = None):
        """
        Inicializa o validador com configuração.
        
        Args:
            config: Configuração do sistema
            quarantine_dir: Diretório para arquivos em quarentena
        """
        self.config = config
        self.schema = DataSchema()
        self.quarantine_dir = quarantine_dir or Path("data/quarantine")
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Logs detalhados
        self.validation_log = self.quarantine_dir / "validation_log.txt"
        
    def detect_encoding(self, file_path: Path) -> Tuple[str, float]:
        """
        Detecta o encoding de um arquivo usando chardet.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Tuple com (encoding, confidence)
        """
        try:
            with open(file_path, 'rb') as f:
                # Lê uma amostra do arquivo para detecção
                sample = f.read(10000)
                
            result = chardet.detect(sample)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0.0)
            
            # Verifica se o encoding detectado está na lista de aceitos
            if encoding not in self.schema.expected_encodings:
                logger.warning(f"Encoding detectado '{encoding}' não está na lista de aceitos")
                
            return encoding, confidence
            
        except Exception as e:
            logger.error(f"Erro ao detectar encoding: {e}")
            return 'utf-8', 0.0
    
    def detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """
        Detecta o delimitador de um arquivo CSV.
        
        Args:
            file_path: Caminho para o arquivo
            encoding: Encoding a ser usado
            
        Returns:
            Delimitador detectado
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Lê algumas linhas para análise
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 5:  # Analisa apenas as primeiras 5 linhas
                        break
                    sample_lines.append(line)
                
                sample_text = ''.join(sample_lines)
                
            # Usa o csv.Sniffer para detectar delimitador
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample_text, delimiters=';,\t').delimiter
            
            if delimiter not in self.schema.supported_delimiters:
                logger.warning(f"Delimitador detectado '{delimiter}' não está na lista de suportados")
                
            return delimiter
            
        except Exception as e:
            logger.error(f"Erro ao detectar delimitador: {e}")
            return ','  # Padrão
    
    def validate_file_structure(self, file_path: Path) -> ValidationResult:
        """
        Valida a estrutura de um arquivo de dados.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Resultado da validação
        """
        result = ValidationResult(
            is_valid=True,
            file_path=str(file_path)
        )
        
        try:
            # Verifica se o arquivo existe
            if not file_path.exists():
                result.is_valid = False
                result.errors.append(f"Arquivo não encontrado: {file_path}")
                return result
            
            # Verifica tamanho do arquivo
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.schema.max_file_size_mb:
                result.is_valid = False
                result.errors.append(f"Arquivo muito grande: {file_size_mb:.1f}MB > {self.schema.max_file_size_mb}MB")
                return result
            
            # Detecta encoding
            encoding, confidence = self.detect_encoding(file_path)
            result.detected_encoding = encoding
            
            if confidence < 0.7:
                result.warnings.append(f"Baixa confiança na detecção de encoding: {confidence:.2f}")
            
            # Detecta delimitador
            delimiter = self.detect_delimiter(file_path, encoding)
            result.detected_delimiter = delimiter
            
            # Tenta carregar o arquivo como DataFrame
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    nrows=100  # Lê apenas uma amostra para validação
                )
                
                result.row_count = len(df)
                result.column_count = len(df.columns)
                
                # Valida colunas obrigatórias
                missing_cols = []
                for required_col in self.schema.required_columns:
                    if required_col not in df.columns:
                        missing_cols.append(required_col)
                        
                result.missing_columns = missing_cols
                
                # Identifica colunas extras
                expected_cols = set(self.schema.required_columns + self.schema.optional_columns)
                actual_cols = set(df.columns)
                extra_cols = list(actual_cols - expected_cols)
                result.extra_columns = extra_cols
                
                # Aplica validações
                if missing_cols:
                    result.is_valid = False
                    result.errors.append(f"Colunas obrigatórias ausentes: {missing_cols}")
                
                if result.row_count == 0:
                    result.is_valid = False
                    result.errors.append("Arquivo vazio ou sem dados válidos")
                
                # Validações adicionais
                if 'message' in df.columns:
                    empty_messages = df['message'].isna().sum()
                    if empty_messages > len(df) * 0.5:  # Mais de 50% de mensagens vazias
                        result.warnings.append(f"Muitas mensagens vazias: {empty_messages}/{len(df)}")
                
            except pd.errors.EmptyDataError:
                result.is_valid = False
                result.errors.append("Arquivo CSV vazio")
            except pd.errors.ParserError as e:
                result.is_valid = False
                result.errors.append(f"Erro ao analisar CSV: {e}")
            except UnicodeDecodeError as e:
                result.is_valid = False
                result.errors.append(f"Erro de encoding: {e}")
                
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Erro inesperado na validação: {e}")
            logger.error(f"Erro na validação de {file_path}: {e}")
        
        return result
    
    def quarantine_file(self, file_path: Path, validation_result: ValidationResult) -> Path:
        """
        Move um arquivo inválido para quarentena.
        
        Args:
            file_path: Arquivo original
            validation_result: Resultado da validação
            
        Returns:
            Caminho do arquivo em quarentena
        """
        try:
            # Cria nome único para o arquivo em quarentena
            quarantine_file = self.quarantine_dir / f"{file_path.stem}_quarantine{file_path.suffix}"
            
            # Move o arquivo
            import shutil
            shutil.move(str(file_path), str(quarantine_file))
            
            # Salva log detalhado
            log_file = self.quarantine_dir / f"{file_path.stem}_validation_error.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(validation_result.dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Arquivo movido para quarentena: {quarantine_file}")
            return quarantine_file
            
        except Exception as e:
            logger.error(f"Erro ao mover arquivo para quarentena: {e}")
            raise
    
    def validate_dataset(self, data_path: Path) -> Dict[str, ValidationResult]:
        """
        Valida todos os arquivos CSV em um diretório.
        
        Args:
            data_path: Diretório com os dados
            
        Returns:
            Dicionário com resultados da validação por arquivo
        """
        results = {}
        
        if not data_path.exists():
            logger.error(f"Diretório de dados não encontrado: {data_path}")
            return results
        
        # Encontra todos os arquivos CSV
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"Nenhum arquivo CSV encontrado em: {data_path}")
            return results
        
        logger.info(f"Validando {len(csv_files)} arquivos CSV...")
        
        valid_files = 0
        quarantined_files = 0
        
        for csv_file in csv_files:
            logger.info(f"Validando: {csv_file.name}")
            
            validation_result = self.validate_file_structure(csv_file)
            results[str(csv_file)] = validation_result
            
            if validation_result.is_valid:
                valid_files += 1
                logger.info(f"✅ {csv_file.name}: VÁLIDO")
            else:
                quarantined_files += 1
                logger.warning(f"❌ {csv_file.name}: INVÁLIDO")
                
                # Move para quarentena
                try:
                    self.quarantine_file(csv_file, validation_result)
                except Exception as e:
                    logger.error(f"Falha ao mover {csv_file.name} para quarentena: {e}")
        
        # Log final
        logger.info(f"Validação concluída: {valid_files} válidos, {quarantined_files} em quarentena")
        
        return results
    
    def get_valid_files(self, data_path: Path) -> List[Path]:
        """
        Retorna lista de arquivos válidos após validação.
        
        Args:
            data_path: Diretório com os dados
            
        Returns:
            Lista de caminhos para arquivos válidos
        """
        results = self.validate_dataset(data_path)
        
        valid_files = []
        for file_path, result in results.items():
            if result.is_valid:
                valid_files.append(Path(file_path))
        
        return valid_files