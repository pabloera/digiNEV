#!/usr/bin/env python3
"""
Chunk Processor - Processamento Genérico de Arquivos em Chunks

Este módulo fornece uma classe genérica para processar arquivos grandes em chunks,
com suporte a checkpointing, barra de progresso e consolidação automática de resultados.

Classes:
    ChunkProcessor: Processador genérico de chunks configurável
    ChunkConfig: Configuração para processamento em chunks

Autor: Pablo Almada
Data: 2025-01-26
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuração para processamento em chunks"""
    chunk_size: int = 10000
    encoding: str = 'utf-8'
    delimiter: str = ';'
    checkpoint_interval: int = 5  # Salvar checkpoint a cada N chunks
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    save_intermediate: bool = False
    intermediate_format: str = 'parquet'  # parquet, csv, pickle
    compression: Optional[str] = 'gzip'  # Para parquet/csv
    memory_limit_mb: int = 1024
    n_jobs: int = 1  # Para processamento paralelo futuro
    on_bad_lines: str = 'skip'
    low_memory: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkConfig':
        """Cria instância a partir de dicionário"""
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ChunkConfig':
        """Carrega configuração de arquivo YAML"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get('chunk_processing', {}))


class ChunkProcessor:
    """
    Processador genérico de chunks para arquivos grandes

    Esta classe fornece uma interface genérica para processar arquivos CSV grandes
    em chunks, com suporte a checkpointing e consolidação automática de resultados.

    Attributes:
        config: Configuração do processamento
        checkpoint_dir: Diretório para salvar checkpoints
        temp_dir: Diretório temporário para arquivos intermediários
    """

    def __init__(self,
                 config: Optional[ChunkConfig] = None,
                 checkpoint_dir: Optional[Union[str, Path]] = None,
                 temp_dir: Optional[Union[str, Path]] = None):
        """
        Inicializa o ChunkProcessor

        Args:
            config: Configuração do processamento
            checkpoint_dir: Diretório para checkpoints (padrão: ./checkpoints)
            temp_dir: Diretório temporário (padrão: ./temp)
        """
        self.config = config or ChunkConfig()
        self.checkpoint_dir = Path(checkpoint_dir or './checkpoints')
        self.temp_dir = Path(temp_dir or './temp')

        # Criar diretórios se necessário
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Estado do processamento
        self.current_checkpoint = None
        self.processing_stats = {
            'chunks_processed': 0,
            'rows_processed': 0,
            'errors': [],
            'start_time': None,
            'end_time': None,
            'checkpoints_saved': 0
        }

        logger.info(f"ChunkProcessor inicializado - chunk_size: {self.config.chunk_size}")

    def _generate_checkpoint_id(self, input_file: Path, process_func: Callable) -> str:
        """
        Gera ID único para checkpoint baseado no arquivo e função

        Args:
            input_file: Arquivo de entrada
            process_func: Função de processamento

        Returns:
            ID do checkpoint
        """
        # Combinar informações para gerar hash único
        info = f"{input_file.absolute()}_{process_func.__name__}_{self.config.chunk_size}"
        return hashlib.md5(info.encode()).hexdigest()[:8]

    def _save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_id: str):
        """
        Salva checkpoint do processamento

        Args:
            checkpoint_data: Dados do checkpoint
            checkpoint_id: ID do checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"

        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            self.processing_stats['checkpoints_saved'] += 1
            logger.debug(f"Checkpoint salvo: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Carrega checkpoint se existir

        Args:
            checkpoint_id: ID do checkpoint

        Returns:
            Dados do checkpoint ou None
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            logger.info(f"Checkpoint carregado: {checkpoint_data['chunks_processed']} chunks já processados")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {e}")
            return None

    def _cleanup_checkpoint(self, checkpoint_id: str):
        """Remove checkpoint após conclusão"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.debug(f"Checkpoint removido: {checkpoint_path}")

    @contextmanager
    def _progress_bar(self, total: Optional[int] = None, desc: str = "Processando"):
        """
        Context manager para barra de progresso

        Args:
            total: Total de itens
            desc: Descrição
        """
        if self.config.enable_progress_bar:
            pbar = tqdm(total=total, desc=desc, unit='chunks')
            try:
                yield pbar
            finally:
                pbar.close()
        else:
            # Dummy progress bar
            class DummyPbar:
                def update(self, n=1): pass
                def set_postfix(self, **kwargs): pass
            yield DummyPbar()

    def _estimate_total_chunks(self, file_path: Path) -> int:
        """
        Estima número total de chunks em um arquivo

        Args:
            file_path: Caminho do arquivo

        Returns:
            Número estimado de chunks
        """
        try:
            # Contar linhas de forma eficiente
            with open(file_path, 'rb') as f:
                lines = sum(1 for _ in f)

            # Descontar cabeçalho
            total_rows = max(0, lines - 1)
            total_chunks = (total_rows + self.config.chunk_size - 1) // self.config.chunk_size

            logger.info(f"Arquivo tem aproximadamente {total_rows:,} linhas em {total_chunks} chunks")
            return total_chunks

        except Exception as e:
            logger.warning(f"Não foi possível estimar chunks: {e}")
            return None

    def _save_intermediate_result(self,
                                 result: Union[pd.DataFrame, Dict, List],
                                 chunk_idx: int,
                                 output_dir: Path) -> Path:
        """
        Salva resultado intermediário

        Args:
            result: Resultado do processamento
            chunk_idx: Índice do chunk
            output_dir: Diretório de saída

        Returns:
            Caminho do arquivo salvo
        """
        if self.config.intermediate_format == 'parquet':
            output_path = output_dir / f"chunk_{chunk_idx:06d}.parquet"
            if isinstance(result, pd.DataFrame):
                result.to_parquet(output_path, compression=self.config.compression)
            else:
                # Converter para DataFrame se necessário
                pd.DataFrame(result).to_parquet(output_path, compression=self.config.compression)

        elif self.config.intermediate_format == 'csv':
            output_path = output_dir / f"chunk_{chunk_idx:06d}.csv"
            if isinstance(result, pd.DataFrame):
                result.to_csv(output_path, index=False, compression=self.config.compression)
            else:
                pd.DataFrame(result).to_csv(output_path, index=False, compression=self.config.compression)

        elif self.config.intermediate_format == 'pickle':
            output_path = output_dir / f"chunk_{chunk_idx:06d}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)

        else:
            raise ValueError(f"Formato não suportado: {self.config.intermediate_format}")

        return output_path

    def _consolidate_results(self,
                           intermediate_dir: Path,
                           output_file: Path,
                           consolidate_func: Optional[Callable] = None) -> Path:
        """
        Consolida resultados intermediários

        Args:
            intermediate_dir: Diretório com resultados intermediários
            output_file: Arquivo de saída final
            consolidate_func: Função customizada de consolidação

        Returns:
            Caminho do arquivo consolidado
        """
        logger.info("Consolidando resultados...")

        # Listar arquivos intermediários
        pattern = f"chunk_*.{self.config.intermediate_format}"
        if self.config.intermediate_format == 'pickle':
            pattern = "chunk_*.pkl"

        chunk_files = sorted(intermediate_dir.glob(pattern))

        if not chunk_files:
            logger.warning("Nenhum arquivo intermediário encontrado")
            return None

        # Se houver função customizada de consolidação
        if consolidate_func:
            result = consolidate_func(chunk_files)
            if isinstance(result, pd.DataFrame):
                result.to_csv(output_file, index=False, sep=self.config.delimiter)
            else:
                with open(output_file, 'wb') as f:
                    pickle.dump(result, f)
            return output_file

        # Consolidação padrão para DataFrames
        if self.config.intermediate_format in ['parquet', 'csv']:
            chunks = []

            for chunk_file in tqdm(chunk_files, desc="Consolidando"):
                if self.config.intermediate_format == 'parquet':
                    chunk = pd.read_parquet(chunk_file)
                else:
                    chunk = pd.read_csv(chunk_file, compression=self.config.compression)
                chunks.append(chunk)

            # Concatenar todos os chunks
            logger.info(f"Concatenando {len(chunks)} chunks...")
            result_df = pd.concat(chunks, ignore_index=True)

            # Salvar resultado final
            logger.info(f"Salvando resultado consolidado: {output_file}")
            result_df.to_csv(
                output_file,
                index=False,
                sep=self.config.delimiter,
                encoding=self.config.encoding
            )

        else:
            # Para pickle, apenas copiar o último resultado
            # (assumindo que cada chunk contém o estado acumulado)
            shutil.copy(chunk_files[-1], output_file)

        return output_file

    def process_file(self,
                    input_file: Union[str, Path],
                    process_func: Callable[[pd.DataFrame, int], Any],
                    output_file: Optional[Union[str, Path]] = None,
                    consolidate_func: Optional[Callable] = None,
                    columns: Optional[List[str]] = None,
                    dtype: Optional[Dict[str, type]] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Processa arquivo em chunks

        Args:
            input_file: Arquivo de entrada
            process_func: Função que processa cada chunk
                         Assinatura: func(chunk_df, chunk_idx) -> result
            output_file: Arquivo de saída (opcional)
            consolidate_func: Função customizada para consolidar resultados
            columns: Colunas específicas para ler
            dtype: Tipos de dados das colunas
            **kwargs: Argumentos adicionais para process_func

        Returns:
            Dict com estatísticas e resultado do processamento
        """
        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

        # Configurar saída
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None

        # ID único para este processamento
        checkpoint_id = self._generate_checkpoint_id(input_path, process_func)

        # Diretório para resultados intermediários
        intermediate_dir = self.temp_dir / f"intermediate_{checkpoint_id}"
        intermediate_dir.mkdir(exist_ok=True)

        # Carregar checkpoint se existir
        checkpoint_data = None
        start_chunk = 0

        if self.config.enable_checkpointing:
            checkpoint_data = self._load_checkpoint(checkpoint_id)
            if checkpoint_data:
                start_chunk = checkpoint_data['chunks_processed']
                self.processing_stats = checkpoint_data['stats']
                logger.info(f"Retomando do chunk {start_chunk}")

        # Iniciar timer se for novo processamento
        if not checkpoint_data:
            self.processing_stats['start_time'] = time.time()

        # Estimar total de chunks
        total_chunks = self._estimate_total_chunks(input_path)

        # Criar iterador de chunks
        chunk_iterator = pd.read_csv(
            input_path,
            chunksize=self.config.chunk_size,
            encoding=self.config.encoding,
            delimiter=self.config.delimiter,
            usecols=columns,
            dtype=dtype,
            on_bad_lines=self.config.on_bad_lines,
            low_memory=self.config.low_memory,
            engine='python'
        )

        # Processar chunks
        results = []

        try:
            with self._progress_bar(total=total_chunks, desc="Processando chunks") as pbar:
                # Pular chunks já processados
                for i in range(start_chunk):
                    next(chunk_iterator, None)
                    pbar.update(1)

                # Processar chunks restantes
                for chunk_idx, chunk in enumerate(chunk_iterator, start=start_chunk):
                    try:
                        # Processar chunk
                        logger.debug(f"Processando chunk {chunk_idx} ({len(chunk)} linhas)")
                        result = process_func(chunk, chunk_idx, **kwargs)

                        # Salvar resultado intermediário se configurado
                        if self.config.save_intermediate and result is not None:
                            self._save_intermediate_result(result, chunk_idx, intermediate_dir)

                        # Acumular resultado
                        results.append(result)

                        # Atualizar estatísticas
                        self.processing_stats['chunks_processed'] += 1
                        self.processing_stats['rows_processed'] += len(chunk)

                        # Atualizar barra de progresso
                        pbar.update(1)
                        pbar.set_postfix({
                            'rows': f"{self.processing_stats['rows_processed']:,}",
                            'errors': len(self.processing_stats['errors'])
                        })

                        # Salvar checkpoint periodicamente
                        if (self.config.enable_checkpointing and
                            self.processing_stats['chunks_processed'] % self.config.checkpoint_interval == 0):

                            checkpoint_data = {
                                'chunks_processed': self.processing_stats['chunks_processed'],
                                'stats': self.processing_stats,
                                'intermediate_dir': str(intermediate_dir)
                            }
                            self._save_checkpoint(checkpoint_data, checkpoint_id)

                    except Exception as e:
                        logger.error(f"Erro no chunk {chunk_idx}: {e}")
                        self.processing_stats['errors'].append({
                            'chunk': chunk_idx,
                            'error': str(e)
                        })

                        # Continuar ou parar dependendo da política
                        if len(self.processing_stats['errors']) > 10:
                            logger.error("Muitos erros, interrompendo processamento")
                            raise

        except KeyboardInterrupt:
            logger.warning("Processamento interrompido pelo usuário")
            if self.config.enable_checkpointing:
                logger.info("Salvando checkpoint antes de sair...")
                checkpoint_data = {
                    'chunks_processed': self.processing_stats['chunks_processed'],
                    'stats': self.processing_stats,
                    'intermediate_dir': str(intermediate_dir)
                }
                self._save_checkpoint(checkpoint_data, checkpoint_id)
            raise

        finally:
            # Finalizar estatísticas
            self.processing_stats['end_time'] = time.time()
            if self.processing_stats['start_time']:
                elapsed = self.processing_stats['end_time'] - self.processing_stats['start_time']
                self.processing_stats['elapsed_time'] = elapsed
                logger.info(f"Processamento levou {elapsed:.2f} segundos")

        # Consolidar resultados se houver arquivo de saída
        consolidated_file = None
        if output_path and self.config.save_intermediate:
            consolidated_file = self._consolidate_results(
                intermediate_dir,
                output_path,
                consolidate_func
            )

        # Limpar checkpoint e arquivos temporários
        if self.config.enable_checkpointing:
            self._cleanup_checkpoint(checkpoint_id)

        if intermediate_dir.exists() and not self.config.save_intermediate:
            shutil.rmtree(intermediate_dir)

        # Retornar estatísticas e resultados
        return {
            'stats': self.processing_stats,
            'results': results if not self.config.save_intermediate else None,
            'output_file': str(consolidated_file) if consolidated_file else None,
            'config': self.config.to_dict()
        }

    def parallel_process_file(self,
                            input_file: Union[str, Path],
                            process_func: Callable,
                            n_jobs: Optional[int] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Processa arquivo em paralelo (implementação futura)

        Args:
            input_file: Arquivo de entrada
            process_func: Função de processamento
            n_jobs: Número de jobs paralelos
            **kwargs: Argumentos adicionais

        Returns:
            Resultados do processamento
        """
        # TODO: Implementar processamento paralelo com multiprocessing/joblib
        logger.warning("Processamento paralelo ainda não implementado, usando processamento serial")
        return self.process_file(input_file, process_func, **kwargs)


def example_process_function(chunk: pd.DataFrame, chunk_idx: int) -> pd.DataFrame:
    """
    Exemplo de função de processamento de chunk

    Args:
        chunk: DataFrame chunk
        chunk_idx: Índice do chunk

    Returns:
        DataFrame processado
    """
    # Exemplo: converter texto para minúsculas
    for col in chunk.select_dtypes(include=['object']).columns:
        chunk[col] = chunk[col].str.lower()

    return chunk


def main():
    """
    Exemplo de uso do ChunkProcessor
    """
    from pathlib import Path

    # Configurar caminhos
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / 'data' / 'raw' / 'telegram_combined_full.csv'
    output_file = base_dir / 'data' / 'interim' / 'telegram_processed_chunks.csv'

    print("🔄 ChunkProcessor - Exemplo de Uso")
    print("=" * 50)

    # Verificar se arquivo existe
    if not input_file.exists():
        print(f"❌ Arquivo não encontrado: {input_file}")
        return

    # Criar configuração
    config = ChunkConfig(
        chunk_size=5000,
        checkpoint_interval=3,
        save_intermediate=True,
        intermediate_format='parquet'
    )

    # Criar processor
    processor = ChunkProcessor(
        config=config,
        checkpoint_dir=base_dir / 'checkpoints',
        temp_dir=base_dir / 'temp'
    )

    # Função de processamento personalizada
    def custom_process(chunk: pd.DataFrame, chunk_idx: int) -> pd.DataFrame:
        """Processa chunk - exemplo com análise básica"""
        # Contar tipos de conteúdo
        stats = {
            'chunk_idx': chunk_idx,
            'total_rows': len(chunk),
            'has_text': chunk['has_txt'].sum() if 'has_txt' in chunk.columns else 0,
            'has_image': chunk['has_img'].sum() if 'has_img' in chunk.columns else 0,
            'has_video': chunk['has_vid'].sum() if 'has_vid' in chunk.columns else 0,
        }

        # Retornar estatísticas como DataFrame
        return pd.DataFrame([stats])

    try:
        # Processar arquivo
        print(f"\n📂 Processando: {input_file.name}")
        print(f"📊 Configuração: chunks de {config.chunk_size} linhas")
        print()

        result = processor.process_file(
            input_file=input_file,
            process_func=custom_process,
            output_file=output_file,
            columns=['has_txt', 'has_img', 'has_vid', 'has_aud']  # Colunas específicas
        )

        # Mostrar estatísticas
        print(f"\n✅ Processamento concluído!")
        print(f"📊 Estatísticas:")
        print(f"   - Chunks processados: {result['stats']['chunks_processed']}")
        print(f"   - Linhas processadas: {result['stats']['rows_processed']:,}")
        print(f"   - Tempo total: {result['stats'].get('elapsed_time', 0):.2f}s")
        print(f"   - Checkpoints salvos: {result['stats']['checkpoints_saved']}")

        if result['stats']['errors']:
            print(f"   - ⚠️  Erros: {len(result['stats']['errors'])}")

        if result['output_file']:
            print(f"\n📁 Resultado salvo em: {result['output_file']}")

    except KeyboardInterrupt:
        print("\n⚠️  Processamento interrompido! Use o mesmo comando para retomar.")
    except Exception as e:
        print(f"\n❌ Erro: {e}")


if __name__ == "__main__":
    main()
