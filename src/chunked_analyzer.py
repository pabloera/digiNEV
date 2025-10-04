#!/usr/bin/env python3
"""
Chunked Analyzer - Sistema de processamento em chunks para datasets grandes
=========================================================================

Processa datasets grandes em chunks menores para evitar sobrecarga de memória
e permite análise de datasets com centenas de milhares de registros.
"""

import pandas as pd
import logging
import time
import gc
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from src.analyzer import Analyzer

class ChunkedAnalyzer:
    """Analyzer com processamento em chunks para datasets grandes."""

    def __init__(self, chunk_size: int = 5000, memory_limit_gb: float = 2.0):
        """
        Inicializar chunked analyzer.

        Args:
            chunk_size: Tamanho de cada chunk para processamento
            memory_limit_gb: Limite de memória em GB antes de forçar limpeza
        """
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.logger = logging.getLogger(self.__class__.__name__)
        self.analyzer = Analyzer()

        self.logger.info(f"✅ ChunkedAnalyzer inicializado: chunks={chunk_size:,}, limite_memoria={memory_limit_gb}GB")

    def _check_memory(self) -> bool:
        """Verificar se memória está próxima do limite."""
        try:
            import psutil
            memory_gb = psutil.Process().memory_info().rss / 1024**3
            if memory_gb > self.memory_limit_gb:
                self.logger.warning(f"⚠️ Memória alta: {memory_gb:.1f}GB > {self.memory_limit_gb}GB")
                return True
            return False
        except ImportError:
            return False

    def _clean_memory(self):
        """Limpar memória forçadamente."""
        gc.collect()
        self.logger.info("🧹 Memória limpa")

    def load_dataset_chunks(self, file_path: str, max_records: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Carregar dataset em chunks.

        Args:
            file_path: Caminho para o dataset
            max_records: Número máximo de registros (None = todos)

        Yields:
            DataFrame chunks
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {file_path}")

        self.logger.info(f"📂 Carregando dataset em chunks: {file_path}")
        self.logger.info(f"📏 Chunk size: {self.chunk_size:,}, Max records: {max_records or 'Todos'}")

        records_processed = 0
        chunk_number = 1

        # Determinar separador baseado na extensão
        separator = ';' if '4_2022-2023-elec' in str(file_path) else ','

        try:
            # Usar chunksize do pandas para leitura eficiente
            for chunk in pd.read_csv(file_path, sep=separator, chunksize=self.chunk_size, encoding='utf-8'):
                if max_records and records_processed >= max_records:
                    break

                # Ajustar chunk se exceder max_records
                if max_records and records_processed + len(chunk) > max_records:
                    remaining = max_records - records_processed
                    chunk = chunk.head(remaining)

                records_processed += len(chunk)

                self.logger.info(f"📦 Chunk {chunk_number}: {len(chunk):,} registros, Total: {records_processed:,}")

                yield chunk
                chunk_number += 1

                # Verificar memória após cada chunk
                if self._check_memory():
                    self._clean_memory()

        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dataset: {e}")
            raise

    def analyze_chunked_dataset(self, file_path: str, max_records: Optional[int] = None,
                               output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analisar dataset grande em chunks e consolidar resultados.

        Args:
            file_path: Caminho para o dataset
            max_records: Número máximo de registros para processar
            output_file: Arquivo para salvar resultados consolidados

        Returns:
            Estatísticas consolidadas da análise
        """
        start_time = time.time()
        self.logger.info("🔬 Iniciando análise chunked")

        consolidated_results = []
        total_records = 0
        total_chunks = 0

        # Estatísticas consolidadas
        consolidated_stats = {
            'political_distribution': {},
            'temporal_stats': {'valid_timestamps': 0, 'total_records': 0},
            'coordination_stats': {'coordinated': 0, 'total_records': 0},
            'domain_stats': {'with_links': 0, 'total_records': 0},
            'stages_completed': 0,
            'features_extracted': 0,
            'processing_errors': 0
        }

        try:
            # Processar cada chunk
            for chunk in self.load_dataset_chunks(file_path, max_records):
                chunk_start = time.time()

                # Analisar chunk
                result = self.analyzer.analyze_dataset(chunk.copy())

                # Extrair dados do resultado
                chunk_data = result['data']
                chunk_stats = result['stats']

                # Consolidar estatísticas
                total_records += len(chunk_data)
                total_chunks += 1

                # Consolidar distribuição política
                if 'political_spectrum' in chunk_data.columns:
                    political_dist = chunk_data['political_spectrum'].value_counts()
                    for category, count in political_dist.items():
                        consolidated_stats['political_distribution'][category] = \
                            consolidated_stats['political_distribution'].get(category, 0) + count

                # Consolidar estatísticas temporais
                if 'has_temporal_data' in chunk_data.columns:
                    valid_temporal = chunk_data['has_temporal_data'].sum()
                    consolidated_stats['temporal_stats']['valid_timestamps'] += valid_temporal
                    consolidated_stats['temporal_stats']['total_records'] += len(chunk_data)

                # Consolidar coordenação
                if 'potential_coordination' in chunk_data.columns:
                    coordinated = chunk_data['potential_coordination'].sum()
                    consolidated_stats['coordination_stats']['coordinated'] += coordinated
                    consolidated_stats['coordination_stats']['total_records'] += len(chunk_data)

                # Consolidar domínios
                if 'has_external_links' in chunk_data.columns:
                    with_links = chunk_data['has_external_links'].sum()
                    consolidated_stats['domain_stats']['with_links'] += with_links
                    consolidated_stats['domain_stats']['total_records'] += len(chunk_data)

                # Manter stats do último chunk (assumindo que são consistentes)
                consolidated_stats['stages_completed'] = chunk_stats.get('stages_completed', 0)
                consolidated_stats['features_extracted'] = chunk_stats.get('features_extracted', 0)

                # Salvar chunk processado se solicitado
                if output_file:
                    chunk_output = f"{output_file.replace('.csv', '')}_chunk_{total_chunks}.csv"
                    chunk_data.to_csv(chunk_output, index=False, sep=';')
                    self.logger.info(f"💾 Chunk salvo: {chunk_output}")

                chunk_time = time.time() - chunk_start
                chunk_performance = len(chunk_data) / chunk_time if chunk_time > 0 else 0

                self.logger.info(f"✅ Chunk {total_chunks} processado: {len(chunk_data):,} registros em {chunk_time:.1f}s ({chunk_performance:.1f} reg/s)")

                # Limpar memória entre chunks
                del chunk_data, result, chunk
                self._clean_memory()

        except Exception as e:
            self.logger.error(f"❌ Erro na análise chunked: {e}")
            consolidated_stats['processing_errors'] += 1
            raise

        # Estatísticas finais
        end_time = time.time()
        total_time = end_time - start_time
        overall_performance = total_records / total_time if total_time > 0 else 0

        final_stats = {
            'total_records_processed': total_records,
            'total_chunks': total_chunks,
            'total_time_seconds': total_time,
            'performance_records_per_second': overall_performance,
            'consolidated_stats': consolidated_stats
        }

        self.logger.info("🎉 ANÁLISE CHUNKED CONCLUÍDA:")
        self.logger.info(f"📊 Total de registros: {total_records:,}")
        self.logger.info(f"📦 Total de chunks: {total_chunks}")
        self.logger.info(f"⏱️ Tempo total: {total_time:.1f}s")
        self.logger.info(f"📈 Performance: {overall_performance:.1f} registros/segundo")
        self.logger.info(f"🏛️ Distribuição política: {dict(list(consolidated_stats['political_distribution'].items())[:5])}")

        return final_stats

    def generate_consolidated_report(self, stats: Dict[str, Any], output_file: str = "consolidated_analysis_report.txt"):
        """Gerar relatório consolidado da análise."""
        report_path = Path(output_file)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("📋 RELATÓRIO CONSOLIDADO - digiNEV v.final\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"📊 ESTATÍSTICAS GERAIS:\n")
            f.write(f"• Total de registros processados: {stats['total_records_processed']:,}\n")
            f.write(f"• Total de chunks: {stats['total_chunks']}\n")
            f.write(f"• Tempo total: {stats['total_time_seconds']:.1f} segundos\n")
            f.write(f"• Performance: {stats['performance_records_per_second']:.1f} registros/segundo\n\n")

            consolidated = stats['consolidated_stats']

            f.write(f"🏛️ DISTRIBUIÇÃO POLÍTICA:\n")
            for category, count in consolidated['political_distribution'].items():
                percentage = (count / stats['total_records_processed']) * 100
                f.write(f"• {category}: {count:,} ({percentage:.1f}%)\n")
            f.write("\n")

            f.write(f"📅 ANÁLISE TEMPORAL:\n")
            temporal = consolidated['temporal_stats']
            if temporal['total_records'] > 0:
                temporal_pct = (temporal['valid_timestamps'] / temporal['total_records']) * 100
                f.write(f"• Timestamps válidos: {temporal['valid_timestamps']:,}/{temporal['total_records']:,} ({temporal_pct:.1f}%)\n")
            f.write("\n")

            f.write(f"🔗 ANÁLISE DE COORDENAÇÃO:\n")
            coordination = consolidated['coordination_stats']
            if coordination['total_records'] > 0:
                coord_pct = (coordination['coordinated'] / coordination['total_records']) * 100
                f.write(f"• Potencial coordenação: {coordination['coordinated']:,}/{coordination['total_records']:,} ({coord_pct:.1f}%)\n")
            f.write("\n")

            f.write(f"🌐 ANÁLISE DE DOMÍNIOS:\n")
            domain = consolidated['domain_stats']
            if domain['total_records'] > 0:
                domain_pct = (domain['with_links'] / domain['total_records']) * 100
                f.write(f"• Mensagens com links: {domain['with_links']:,}/{domain['total_records']:,} ({domain_pct:.1f}%)\n")
            f.write("\n")

            f.write(f"🔧 PIPELINE:\n")
            f.write(f"• Stages completados: {consolidated['stages_completed']}\n")
            f.write(f"• Features extraídas: {consolidated['features_extracted']}\n")
            f.write(f"• Erros de processamento: {consolidated['processing_errors']}\n")

        self.logger.info(f"📄 Relatório salvo: {report_path}")
        return report_path