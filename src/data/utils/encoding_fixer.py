#!/usr/bin/env python3
"""
Módulo para correção de problemas de encoding em arquivos CSV.

Este módulo fornece funcionalidades para detectar e corrigir problemas de encoding
em arquivos de texto, especialmente útil para dados em português que podem ter
sido corrompidos durante processamento ou transferência.
"""

import pandas as pd
import ftfy
from pathlib import Path
import logging
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
from typing import Optional, Tuple, Dict, Any
import yaml
from datetime import datetime
import sys
from tqdm import tqdm


class EncodingFixer:
    """
    Classe para corrigir problemas de encoding em arquivos CSV.
    
    Attributes:
        config (dict): Configurações carregadas do arquivo YAML
        logger (logging.Logger): Logger para registrar operações
        stats (dict): Estatísticas do processamento
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Inicializa o EncodingFixer.
        
        Args:
            config_path: Caminho para o arquivo de configuração YAML
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.stats = {
            'total_rows': 0,
            'modified_rows': 0,
            'errors': 0,
            'samples': []
        }
        
    def _load_config(self, config_path: str) -> dict:
        """Carrega configurações do arquivo YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Arquivo de configuração não encontrado: {config_path}")
            print("Usando configurações padrão...")
            return {
                'encoding': {
                    'source_encoding': 'utf-8',
                    'target_encoding': 'utf-8'
                },
                'processing': {
                    'chunk_size': 10000,
                    'sample_size': 1000
                },
                'paths': {
                    'raw_data': 'data/processed/textanalysis/telegram_text_analysis2.csv',
                    'fixed_data': 'data/interim/telegram_text_analysis_fixed.csv',
                    'sample_data': 'data/interim/samples/'
                },
                'logging': {
                    'level': 'INFO',
                    'file': 'logs/processing.log'
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Configura o sistema de logging."""
        log_file = self.config.get('logging', {}).get('file', 'logs/encoding_fixer.log')
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        
        # Criar diretório de logs se não existir
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configurar logger
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level))
        
        # Remover handlers existentes
        logger.handlers.clear()
        
        # Handler para arquivo
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def detect_encoding(self, file_path: str, sample_size: int = 10000) -> Dict[str, Any]:
        """
        Detecta o encoding de um arquivo.
        
        Args:
            file_path: Caminho para o arquivo
            sample_size: Número de bytes para analisar
            
        Returns:
            Dicionário com informações sobre o encoding detectado
        """
        self.logger.info(f"Detectando encoding do arquivo: {file_path}")
        
        try:
            if CHARDET_AVAILABLE:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(sample_size)
                    result = chardet.detect(raw_data)
                    
                self.logger.info(f"Encoding detectado: {result['encoding']} "
                               f"(confiança: {result['confidence']:.2%})")
                return result
            else:
                # Fallback quando chardet não está disponível
                self.logger.warning("chardet não disponível, assumindo UTF-8")
                return {'encoding': 'utf-8', 'confidence': 0.9}
                
        except Exception as e:
            self.logger.error(f"Erro ao detectar encoding: {str(e)}")
            return {'encoding': 'utf-8', 'confidence': 0.0}
    
    def fix_text_column(self, text: str) -> Tuple[str, bool]:
        """
        Corrige texto se necessário.
        
        Args:
            text: Texto a ser corrigido
            
        Returns:
            Tupla (texto_corrigido, foi_modificado)
        """
        if pd.isna(text) or not isinstance(text, str):
            return text, False
            
        try:
            # Aplicar correção com ftfy
            fixed_text = ftfy.fix_text(text)
            
            # Verificar se houve mudança
            was_modified = fixed_text != text
            
            if was_modified and len(self.stats['samples']) < 10:
                self.stats['samples'].append({
                    'original': text[:100] + '...' if len(text) > 100 else text,
                    'fixed': fixed_text[:100] + '...' if len(fixed_text) > 100 else fixed_text
                })
                
            return fixed_text, was_modified
            
        except Exception as e:
            self.logger.warning(f"Erro ao corrigir texto: {str(e)}")
            self.stats['errors'] += 1
            return text, False
    
    def process_chunk(self, chunk: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Processa um chunk do dataframe.
        
        Args:
            chunk: DataFrame chunk para processar
            columns: Lista de colunas para processar (None = todas as colunas de texto)
            
        Returns:
            DataFrame processado
        """
        if columns is None:
            # Processar todas as colunas de texto
            columns = chunk.select_dtypes(include=['object']).columns.tolist()
        
        chunk_modified = 0
        
        for col in columns:
            if col in chunk.columns:
                self.logger.debug(f"Processando coluna: {col}")
                
                for idx in chunk.index:
                    value = chunk.loc[idx, col]
                    fixed_value, was_modified = self.fix_text_column(value)
                    
                    if was_modified:
                        chunk.loc[idx, col] = fixed_value
                        chunk_modified += 1
        
        self.stats['modified_rows'] += chunk_modified
        return chunk
    
    def fix_csv_file(self, 
                     input_path: Optional[str] = None,
                     output_path: Optional[str] = None,
                     columns: Optional[list] = None,
                     dry_run: bool = False,
                     sample_only: bool = False) -> Dict[str, Any]:
        """
        Método principal para processar arquivo CSV completo.
        
        Args:
            input_path: Caminho do arquivo de entrada
            output_path: Caminho do arquivo de saída
            columns: Lista de colunas para processar
            dry_run: Se True, apenas mostra o que seria corrigido
            sample_only: Se True, processa apenas uma amostra
            
        Returns:
            Dicionário com estatísticas do processamento
        """
        # Usar caminhos padrão se não especificados
        if input_path is None:
            input_path = self.config['paths']['raw_data']
        if output_path is None:
            output_path = self.config['paths']['fixed_data']
            
        self.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Iniciando processamento")
        self.logger.info(f"Arquivo de entrada: {input_path}")
        self.logger.info(f"Arquivo de saída: {output_path}")
        
        # Detectar encoding
        encoding_info = self.detect_encoding(input_path)
        source_encoding = encoding_info['encoding'] or self.config['encoding']['source_encoding']
        
        # Resetar estatísticas
        self.stats = {
            'total_rows': 0,
            'modified_rows': 0,
            'errors': 0,
            'samples': [],
            'start_time': datetime.now()
        }
        
        try:
            # Determinar tamanho do chunk
            chunk_size = self.config['processing']['chunk_size']
            if sample_only:
                chunk_size = self.config['processing']['sample_size']
            
            # Processar arquivo em chunks
            chunks_processed = 0
            
            with tqdm(desc="Processando chunks", unit="chunk") as pbar:
                for chunk in pd.read_csv(input_path, 
                                       encoding=source_encoding,
                                       chunksize=chunk_size,
                                       on_bad_lines='warn'):
                    
                    self.stats['total_rows'] += len(chunk)
                    
                    if dry_run:
                        # Modo dry-run: apenas analisar
                        self._analyze_chunk(chunk, columns)
                    else:
                        # Processar chunk
                        chunk = self.process_chunk(chunk, columns)
                        
                        # Salvar chunk processado
                        if chunks_processed == 0:
                            # Primeiro chunk: criar novo arquivo
                            chunk.to_csv(output_path, index=False, encoding='utf-8')
                        else:
                            # Chunks subsequentes: anexar ao arquivo
                            chunk.to_csv(output_path, mode='a', header=False, 
                                       index=False, encoding='utf-8')
                    
                    chunks_processed += 1
                    pbar.update(1)
                    
                    if sample_only and chunks_processed >= 1:
                        break
            
            # Finalizar processamento
            self.stats['end_time'] = datetime.now()
            self.stats['duration'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            # Log de estatísticas
            self._log_statistics()
            
            # Salvar relatório se não for dry-run
            if not dry_run and not sample_only:
                self._save_report(output_path)
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"Erro durante processamento: {str(e)}")
            raise
    
    def _analyze_chunk(self, chunk: pd.DataFrame, columns: Optional[list] = None):
        """Analisa chunk no modo dry-run."""
        if columns is None:
            columns = chunk.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in chunk.columns:
                for value in chunk[col]:
                    _, was_modified = self.fix_text_column(value)
                    if was_modified:
                        self.stats['modified_rows'] += 1
    
    def _log_statistics(self):
        """Registra estatísticas do processamento."""
        self.logger.info("\n" + "="*50)
        self.logger.info("ESTATÍSTICAS DO PROCESSAMENTO")
        self.logger.info("="*50)
        self.logger.info(f"Total de linhas processadas: {self.stats['total_rows']:,}")
        self.logger.info(f"Linhas modificadas: {self.stats['modified_rows']:,}")
        self.logger.info(f"Taxa de modificação: {self.stats['modified_rows']/max(1, self.stats['total_rows']):.2%}")
        self.logger.info(f"Erros encontrados: {self.stats['errors']}")
        
        if 'duration' in self.stats:
            self.logger.info(f"Tempo de processamento: {self.stats['duration']:.2f} segundos")
        
        if self.stats['samples']:
            self.logger.info("\nEXEMPLOS DE CORREÇÕES:")
            for i, sample in enumerate(self.stats['samples'], 1):
                self.logger.info(f"\nExemplo {i}:")
                self.logger.info(f"  Original: {sample['original']}")
                self.logger.info(f"  Corrigido: {sample['fixed']}")
    
    def _save_report(self, output_path: str):
        """Salva relatório do processamento."""
        report_path = Path(output_path).with_suffix('.report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE CORREÇÃO DE ENCODING\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Arquivo processado: {output_path}\n\n")
            
            f.write("ESTATÍSTICAS:\n")
            f.write(f"- Total de linhas: {self.stats['total_rows']:,}\n")
            f.write(f"- Linhas modificadas: {self.stats['modified_rows']:,}\n")
            f.write(f"- Taxa de modificação: {self.stats['modified_rows']/max(1, self.stats['total_rows']):.2%}\n")
            f.write(f"- Erros: {self.stats['errors']}\n")
            
            if 'duration' in self.stats:
                f.write(f"- Tempo de processamento: {self.stats['duration']:.2f} segundos\n")
            
            if self.stats['samples']:
                f.write("\n\nEXEMPLOS DE CORREÇÕES:\n")
                for i, sample in enumerate(self.stats['samples'], 1):
                    f.write(f"\n{i}. Original: {sample['original']}\n")
                    f.write(f"   Corrigido: {sample['fixed']}\n")


def main():
    """Função principal para teste do módulo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Corrigir problemas de encoding em arquivos CSV')
    parser.add_argument('--input', '-i', help='Arquivo de entrada')
    parser.add_argument('--output', '-o', help='Arquivo de saída')
    parser.add_argument('--columns', '-c', nargs='+', help='Colunas para processar')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Modo dry-run (apenas análise)')
    parser.add_argument('--sample', '-s', action='store_true', help='Processar apenas amostra')
    parser.add_argument('--config', default='config/settings.yaml', help='Arquivo de configuração')
    
    args = parser.parse_args()
    
    # Criar instância do EncodingFixer
    fixer = EncodingFixer(config_path=args.config)
    
    # Processar arquivo
    stats = fixer.fix_csv_file(
        input_path=args.input,
        output_path=args.output,
        columns=args.columns,
        dry_run=args.dry_run,
        sample_only=args.sample
    )
    
    print(f"\nProcessamento {'(DRY RUN) ' if args.dry_run else ''}concluído!")
    print(f"Linhas modificadas: {stats['modified_rows']:,} de {stats['total_rows']:,}")


if __name__ == "__main__":
    main()