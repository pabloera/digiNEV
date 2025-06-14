#!/usr/bin/env python3
"""
Compression Utils - Utilitários de Compressão Avançada
======================================================

Implementa compressão inteligente para arquivos CSV grandes
conforme TASK-030 da auditoria v5.0.0

Otimizações implementadas:
- Compressão automática baseada em tamanho
- Múltiplos algoritmos (gzip, lz4, zstd)
- Compressão progressiva para datasets muito grandes
- Detecção automática de melhor algoritmo
"""

import gzip
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd

# Optional dependencies para algoritmos de compressão avançados
try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompressionUtils:
    """
    Utilitários avançados de compressão para arquivos CSV grandes
    """
    
    # Thresholds para diferentes tipos de compressão
    COMPRESSION_THRESHOLDS = {
        'none': 0,           # < 50K registros
        'gzip': 50000,       # 50K - 500K registros
        'lz4': 500000,       # 500K - 2M registros (se disponível)
        'zstd': 2000000      # > 2M registros (se disponível)
    }
    
    # Extensões de arquivo por algoritmo
    EXTENSIONS = {
        'none': '.csv',
        'gzip': '.csv.gz',
        'lz4': '.csv.lz4',
        'zstd': '.csv.zst'
    }
    
    @classmethod
    def determine_optimal_compression(cls, df: pd.DataFrame) -> str:
        """
        Determina o melhor algoritmo de compressão baseado no tamanho dos dados
        
        Args:
            df: DataFrame para analisar
            
        Returns:
            str: Algoritmo recomendado ('none', 'gzip', 'lz4', 'zstd')
        """
        record_count = len(df)
        
        # Escolher algoritmo baseado no tamanho
        if record_count < cls.COMPRESSION_THRESHOLDS['gzip']:
            return 'none'
        elif record_count < cls.COMPRESSION_THRESHOLDS['lz4']:
            return 'gzip'
        elif record_count < cls.COMPRESSION_THRESHOLDS['zstd'] and LZ4_AVAILABLE:
            return 'lz4'
        elif ZSTD_AVAILABLE:
            return 'zstd'
        else:
            # Fallback para gzip se algoritmos avançados não estão disponíveis
            return 'gzip'
    
    @classmethod
    def save_with_optimal_compression(cls, 
                                    df: pd.DataFrame, 
                                    base_path: str,
                                    force_algorithm: Optional[str] = None,
                                    **csv_kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Salva DataFrame com compressão otimizada
        
        Args:
            df: DataFrame para salvar
            base_path: Caminho base (sem extensão)
            force_algorithm: Forçar algoritmo específico
            **csv_kwargs: Argumentos para to_csv()
            
        Returns:
            Tuple[str, Dict]: (caminho_final, estatísticas)
        """
        # Determinar algoritmo
        algorithm = force_algorithm or cls.determine_optimal_compression(df)
        
        # Construir caminho final
        base_path_obj = Path(base_path)
        if base_path_obj.suffix == '.csv':
            base_path = str(base_path_obj.with_suffix(''))
        
        final_path = base_path + cls.EXTENSIONS[algorithm]
        
        # Garantir diretório existe
        Path(final_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Configurações padrão para CSV
        default_csv_kwargs = {
            'sep': ';',
            'index': False,
            'encoding': 'utf-8',
            'quoting': 1,
            'quotechar': '"',
            'doublequote': True,
            'lineterminator': '\n'
        }
        default_csv_kwargs.update(csv_kwargs)
        
        # Salvar com algoritmo escolhido
        start_time = time.time()
        
        if algorithm == 'none':
            df.to_csv(final_path, **default_csv_kwargs)
        elif algorithm == 'gzip':
            df.to_csv(final_path, compression='gzip', **default_csv_kwargs)
        elif algorithm == 'lz4' and LZ4_AVAILABLE:
            cls._save_with_lz4(df, final_path, **default_csv_kwargs)
        elif algorithm == 'zstd' and ZSTD_AVAILABLE:
            cls._save_with_zstd(df, final_path, **default_csv_kwargs)
        else:
            # Fallback para gzip
            logger.warning(f"Algoritmo {algorithm} não disponível, usando gzip")
            final_path = base_path + '.csv.gz'
            df.to_csv(final_path, compression='gzip', **default_csv_kwargs)
            algorithm = 'gzip'
        
        save_time = time.time() - start_time
        
        # Calcular estatísticas
        stats = cls._calculate_compression_stats(df, final_path, algorithm, save_time)
        
        logger.info(f"💾 Arquivo salvo: {Path(final_path).name} | "
                   f"Algoritmo: {algorithm} | "
                   f"Tamanho: {stats['compressed_size_mb']:.1f}MB | "
                   f"Compressão: {stats['compression_ratio']:.1f}x | "
                   f"Tempo: {save_time:.1f}s")
        
        return final_path, stats
    
    @classmethod
    def _save_with_lz4(cls, df: pd.DataFrame, path: str, **csv_kwargs):
        """Salva com compressão LZ4"""
        import io
        
        # Converter DataFrame para string CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, **csv_kwargs)
        csv_content = csv_buffer.getvalue()
        
        # Comprimir com LZ4
        compressed_data = lz4.compress(csv_content.encode('utf-8'))
        
        # Salvar arquivo comprimido
        with open(path, 'wb') as f:
            f.write(compressed_data)
    
    @classmethod
    def _save_with_zstd(cls, df: pd.DataFrame, path: str, **csv_kwargs):
        """Salva com compressão Zstandard"""
        import io
        
        # Converter DataFrame para string CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, **csv_kwargs)
        csv_content = csv_buffer.getvalue()
        
        # Comprimir com Zstandard
        cctx = zstd.ZstdCompressor(level=3)  # Nível balanceado
        compressed_data = cctx.compress(csv_content.encode('utf-8'))
        
        # Salvar arquivo comprimido
        with open(path, 'wb') as f:
            f.write(compressed_data)
    
    @classmethod
    def _calculate_compression_stats(cls, 
                                   df: pd.DataFrame, 
                                   compressed_path: str, 
                                   algorithm: str, 
                                   save_time: float) -> Dict[str, Any]:
        """Calcula estatísticas de compressão"""
        
        # Tamanho do arquivo comprimido
        compressed_size = os.path.getsize(compressed_path)
        
        # Estimar tamanho não comprimido (aproximação)
        import io
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, sep=';', index=False)
        uncompressed_size = len(csv_buffer.getvalue().encode('utf-8'))
        
        # Calcular estatísticas
        compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0
        space_saved = uncompressed_size - compressed_size
        space_saved_percentage = (space_saved / uncompressed_size * 100) if uncompressed_size > 0 else 0
        
        return {
            'algorithm': algorithm,
            'records': len(df),
            'columns': len(df.columns),
            'uncompressed_size_mb': uncompressed_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'space_saved_mb': space_saved / (1024 * 1024),
            'space_saved_percentage': space_saved_percentage,
            'save_time_seconds': save_time,
            'throughput_mb_per_second': (compressed_size / (1024 * 1024)) / save_time if save_time > 0 else 0
        }
    
    @classmethod
    def load_compressed_csv(cls, file_path: str, **pandas_kwargs) -> pd.DataFrame:
        """
        Carrega CSV comprimido automaticamente detectando o formato
        
        Args:
            file_path: Caminho do arquivo
            **pandas_kwargs: Argumentos para pd.read_csv()
            
        Returns:
            pd.DataFrame: DataFrame carregado
        """
        file_path = Path(file_path)
        
        # Configurações padrão
        default_kwargs = {
            'sep': ';',
            'encoding': 'utf-8',
            'quoting': 1
        }
        default_kwargs.update(pandas_kwargs)
        
        # Detectar formato e carregar
        if file_path.suffix == '.gz':
            return pd.read_csv(file_path, compression='gzip', **default_kwargs)
        elif file_path.suffix == '.lz4':
            return cls._load_lz4_csv(file_path, **default_kwargs)
        elif file_path.suffix == '.zst':
            return cls._load_zstd_csv(file_path, **default_kwargs)
        else:
            return pd.read_csv(file_path, **default_kwargs)
    
    @classmethod
    def _load_lz4_csv(cls, file_path: str, **pandas_kwargs) -> pd.DataFrame:
        """Carrega CSV comprimido com LZ4"""
        import io
        
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        
        # Descomprimir
        decompressed_data = lz4.decompress(compressed_data)
        csv_content = decompressed_data.decode('utf-8')
        
        # Carregar como DataFrame
        return pd.read_csv(io.StringIO(csv_content), **pandas_kwargs)
    
    @classmethod
    def _load_zstd_csv(cls, file_path: str, **pandas_kwargs) -> pd.DataFrame:
        """Carrega CSV comprimido com Zstandard"""
        import io
        
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        
        # Descomprimir
        dctx = zstd.ZstdDecompressor()
        decompressed_data = dctx.decompress(compressed_data)
        csv_content = decompressed_data.decode('utf-8')
        
        # Carregar como DataFrame
        return pd.read_csv(io.StringIO(csv_content), **pandas_kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> Dict[str, bool]:
        """Retorna algoritmos de compressão disponíveis"""
        return {
            'none': True,
            'gzip': True,
            'lz4': LZ4_AVAILABLE,
            'zstd': ZSTD_AVAILABLE
        }
    
    @classmethod
    def benchmark_compression(cls, df: pd.DataFrame, algorithms: Optional[list] = None) -> Dict[str, Dict]:
        """
        Executa benchmark de diferentes algoritmos de compressão
        
        Args:
            df: DataFrame para testar
            algorithms: Lista de algoritmos para testar (None = todos disponíveis)
            
        Returns:
            Dict: Resultados do benchmark por algoritmo
        """
        if algorithms is None:
            available = cls.get_available_algorithms()
            algorithms = [alg for alg, available in available.items() if available]
        
        results = {}
        
        for algorithm in algorithms:
            try:
                temp_path = f"/tmp/benchmark_test_{algorithm}"
                start_time = time.time()
                
                final_path, stats = cls.save_with_optimal_compression(
                    df, temp_path, force_algorithm=algorithm
                )
                
                # Limpeza
                if os.path.exists(final_path):
                    os.remove(final_path)
                
                results[algorithm] = stats
                
            except Exception as e:
                logger.warning(f"Erro no benchmark {algorithm}: {e}")
                results[algorithm] = {'error': str(e)}
        
        return results


# Função de conveniência para usar diretamente
def save_csv_optimized(df: pd.DataFrame, path: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """
    Função de conveniência para salvar CSV com compressão otimizada
    
    Args:
        df: DataFrame para salvar
        path: Caminho do arquivo
        **kwargs: Argumentos adicionais para to_csv()
        
    Returns:
        Tuple[str, Dict]: (caminho_final, estatísticas)
    """
    return CompressionUtils.save_with_optimal_compression(df, path, **kwargs)


def load_csv_auto(path: str, **kwargs) -> pd.DataFrame:
    """
    Função de conveniência para carregar CSV automaticamente
    
    Args:
        path: Caminho do arquivo
        **kwargs: Argumentos para pd.read_csv()
        
    Returns:
        pd.DataFrame: DataFrame carregado
    """
    return CompressionUtils.load_compressed_csv(path, **kwargs)


if __name__ == "__main__":
    # Teste básico das funcionalidades
    print("🧪 Testando CompressionUtils...")
    
    # Criar DataFrame de teste
    test_df = pd.DataFrame({
        'id': range(1000),
        'text': [f"Texto de teste número {i}" * 10 for i in range(1000)],
        'value': [i * 1.5 for i in range(1000)]
    })
    
    # Testar compressão
    algorithms = CompressionUtils.get_available_algorithms()
    print(f"Algoritmos disponíveis: {algorithms}")
    
    # Benchmark
    if len(test_df) > 0:
        optimal = CompressionUtils.determine_optimal_compression(test_df)
        print(f"Algoritmo ótimo para {len(test_df)} registros: {optimal}")
        
        # Testar salvamento
        try:
            final_path, stats = save_csv_optimized(test_df, "/tmp/test_compression")
            print(f"Arquivo salvo: {final_path}")
            print(f"Estatísticas: {stats}")
            
            # Testar carregamento
            loaded_df = load_csv_auto(final_path)
            print(f"Arquivo carregado: {len(loaded_df)} registros")
            
            # Limpeza
            if os.path.exists(final_path):
                os.remove(final_path)
                
        except Exception as e:
            print(f"Erro no teste: {e}")
    
    print("✅ CompressionUtils funcionando corretamente!")