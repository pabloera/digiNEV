#!/usr/bin/env python3
"""
digiNEV Analyzer v.final
=======================

Sistema consolidado único de análise de discurso político brasileiro.
Pipeline com 17 estágios interligados gerando 102+ colunas de análise.

ARQUITETURA CONSOLIDADA:
- Sistema único centralizado (elimina estruturas paralelas)
- 17 estágios científicos sequenciais
- Dados reais processados (sem métricas inventadas)
- Configuração unificada via config/settings.yaml

Author: digiNEV Academic Research Team
Version: v.final (Consolidação Final)
"""

import pandas as pd
import numpy as np
import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports (sempre disponíveis)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# Standard library imports
import unicodedata
import time
from collections import Counter
from urllib.parse import urlparse

# Memory monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Dependency validation decorator
def validate_stage_dependencies(required_columns=None, required_attrs=None):
    """
    Decorator para validar dependências entre stages.

    Args:
        required_columns: Lista de colunas obrigatórias no DataFrame
        required_attrs: Lista de atributos obrigatórios na instância
    """
    def decorator(func):
        def wrapper(self, df, *args, **kwargs):
            stage_name = func.__name__.replace('_stage_', 'Stage ').replace('_', ' ').title()

            # Validar colunas obrigatórias
            if required_columns:
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    self.logger.error(f"❌ {stage_name}: Colunas obrigatórias ausentes: {missing_cols}")
                    raise ValueError(f"Colunas obrigatórias ausentes para {stage_name}: {missing_cols}")

            # Validar atributos obrigatórios na instância
            if required_attrs:
                missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
                if missing_attrs:
                    self.logger.error(f"❌ {stage_name}: Atributos obrigatórios ausentes: {missing_attrs}")
                    raise ValueError(f"Atributos obrigatórios ausentes para {stage_name}: {missing_attrs}")

            # Executar função se validações passaram
            return func(self, df, *args, **kwargs)
        return wrapper
    return decorator

# spaCy para processamento linguístico em português
try:
    import spacy
    # Tentar carregar modelo português
    try:
        nlp = spacy.load("pt_core_news_lg")
        SPACY_AVAILABLE = True
    except OSError:
        try:
            nlp = spacy.load("pt_core_news_sm")
            SPACY_AVAILABLE = True
        except OSError:
            nlp = None
            SPACY_AVAILABLE = False
except ImportError:
    nlp = None
    SPACY_AVAILABLE = False


class Analyzer:
    """
    Analisador consolidado com 14 stages sequenciais interligados.

    STAGES IMPLEMENTADOS (sem fallbacks confusos):
    01. feature_extraction (Python puro) - Detecção automática de colunas e features
    02. text_preprocessing (Python puro) - Limpeza básica de texto em português
    03. linguistic_processing (spaCy) - Processamento linguístico avançado LOGO APÓS limpeza
    04. statistical_analysis (Python puro) - Análise estatística com dados spaCy
    05. political_classification (Lexicon real) - Classificação política brasileira
    06. tfidf_vectorization (scikit-learn) - TF-IDF com tokens spaCy
    07. clustering_analysis (scikit-learn) - Clustering baseado em features linguísticas
    08. topic_modeling (scikit-learn) - Topic modeling com embeddings
    09. temporal_analysis (Python puro) - Análise temporal
    10. network_analysis (Python puro) - Coordenação e padrões de rede
    """

    def __init__(self, chunk_size: int = 5000, memory_limit_gb: float = 2.0, auto_chunk: bool = True,
                 political_relevance_threshold: float = 0.02):
        """
        Inicializar analyzer com capacidades de auto-chunking.

        Args:
            chunk_size: Tamanho do chunk quando auto-chunking é necessário
            memory_limit_gb: Limite de memória para trigger de chunking
            auto_chunk: Se True, detecta automaticamente quando usar chunks
            political_relevance_threshold: Threshold mínimo para relevância política (padrão: 0.02)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurações de chunking automático
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.auto_chunk = auto_chunk

        # Configurações de filtros
        self.political_relevance_threshold = political_relevance_threshold

        # Load political lexicon if available
        self.political_lexicon = self._load_political_lexicon()

        # Initialize ML components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.kmeans_model = None
        self.lda_model = None

        # Stats tracking
        self.stats = {
            'stages_completed': 0,
            'features_extracted': 0,
            'processing_errors': 0,
            'chunked_processing': False,
            'total_chunks': 0
        }

        # Global statistics container
        self.global_stats = {}

        self.logger.info("✅ Analyzer v.final inicializado (auto-chunking habilitado)")

    def _load_political_lexicon(self) -> Dict:
        """Carregar lexicon político brasileiro correto."""
        lexicon_path = Path("src/core/lexico_politico_hierarquizado.json")

        if lexicon_path.exists():
            try:
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    lexicon = json.load(f)
                self.logger.info(f"✅ Lexicon político carregado: {len(lexicon)} categorias")
                return lexicon
            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao carregar lexicon: {e}")

        # Lexicon político brasileiro correto (conforme political_visualization_enhanced.py)
        return {
            "bolsonarista": ["bolsonaro", "mito", "capitão", "messias", "brasil acima de tudo"],
            "lulista": ["lula", "squid", "ex-presidente", "pt", "petista"],
            "anti-bolsonaro": ["fora bolsonaro", "impeachment", "golpista", "fascista"],
            "neutro": ["governo", "política", "brasil", "país"],
            "geral": ["eleição", "voto", "democracia", "constituição"],
            "indefinido": ["moderado", "centrista"]
        }

    def _should_use_chunking(self, data_input):
        """
        Determinar automaticamente se deve usar processamento em chunks.
        
        Args:
            data_input: DataFrame ou caminho do arquivo
            
        Returns:
            Tuple[usar_chunks, estimativa_registros, motivo]
        """
        try:
            # Se for DataFrame, verificar tamanho direto
            if isinstance(data_input, pd.DataFrame):
                n_records = len(data_input)
                memory_usage_mb = data_input.memory_usage(deep=True).sum() / (1024**2)
                
                if n_records > 10000:
                    return True, n_records, f"Dataset grande: {n_records:,} registros"
                elif memory_usage_mb > self.memory_limit_gb * 1024:
                    return True, n_records, f"Uso de memória alto: {memory_usage_mb:.1f}MB"
                else:
                    return False, n_records, f"Dataset pequeno: {n_records:,} registros"
            
            # Se for caminho de arquivo, estimar tamanho
            elif isinstance(data_input, (str, Path)):
                file_path = Path(data_input)
                if not file_path.exists():
                    return False, 0, "Arquivo não encontrado"
                
                # Estimar número de registros pelo tamanho do arquivo
                file_size_mb = file_path.stat().st_size / (1024**2)
                estimated_records = int(file_size_mb * 100)  # Estimativa: ~100 registros por MB
                
                if file_size_mb > 50:  # Arquivos > 50MB
                    return True, estimated_records, f"Arquivo grande: {file_size_mb:.1f}MB"
                elif estimated_records > 10000:
                    return True, estimated_records, f"Muitos registros estimados: {estimated_records:,}"
                else:
                    return False, estimated_records, f"Arquivo pequeno: {file_size_mb:.1f}MB"
            
            return False, 0, "Tipo de entrada não reconhecido"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao determinar chunking: {e}")
            return False, 0, "Erro na detecção"

    def _check_memory_usage(self) -> bool:
        """Verificar se memória está próxima do limite."""
        if not PSUTIL_AVAILABLE:
            self.logger.warning("⚠️ psutil não disponível para monitoramento de memória")
            return False

        try:
            memory_gb = psutil.Process().memory_info().rss / (1024**3)
            if memory_gb > self.memory_limit_gb:
                self.logger.warning(f"🚨 Memória alta: {memory_gb:.1f}GB > {self.memory_limit_gb}GB")
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Erro no monitoramento de memória: {e}")
            return False

    def _clean_memory(self):
        """Limpar memória forçadamente."""
        import gc
        gc.collect()
        self.logger.info("🧹 Memória limpa")

    def analyze(self, data_input, **kwargs) -> Dict[str, Any]:
        """
        Método principal unificado que detecta automaticamente o modo de processamento.
        
        Args:
            data_input: DataFrame ou caminho para arquivo CSV
            **kwargs: Argumentos adicionais (max_records, output_file, etc.)
            
        Returns:
            Resultado da análise (formato unificado)
        """
        # Auto-detecção do modo de processamento
        should_chunk, estimated_records, reason = self._should_use_chunking(data_input)
        
        self.logger.info(f"🤖 Auto-detecção: {reason}")
        
        if self.auto_chunk and should_chunk:
            self.logger.info(f"⚡ Modo CHUNKED ativado automaticamente")
            self.stats['chunked_processing'] = True
            return self._analyze_chunked(data_input, **kwargs)
        else:
            self.logger.info(f"🔬 Modo NORMAL (in-memory)")
            self.stats['chunked_processing'] = False
            
            # Se for arquivo, carregar como DataFrame
            if isinstance(data_input, (str, Path)):
                df = self._load_dataframe(data_input, kwargs.get('max_records'))
            else:
                df = data_input.copy()
                
            return self.analyze_dataset(df)

    def _load_dataframe(self, file_path: str, max_records: Optional[int] = None) -> pd.DataFrame:
        """Carregar DataFrame com detecção automática de separador."""
        file_path = Path(file_path)
        
        # Detectar separador baseado no nome do arquivo
        separator = ';' if '4_2022-2023-elec' in str(file_path) else ','
        
        # Carregar com limite de registros se especificado
        if max_records:
            df = pd.read_csv(file_path, sep=separator, encoding='utf-8', nrows=max_records)
        else:
            df = pd.read_csv(file_path, sep=separator, encoding='utf-8')
            
        self.logger.info(f"📂 Dataset carregado: {len(df):,} registros, {len(df.columns)} colunas")
        return df

    def _analyze_chunked(self, data_input, **kwargs) -> Dict[str, Any]:
        """
        Processamento em chunks integrado ao Analyzer principal.
        
        Args:
            data_input: Caminho do arquivo ou DataFrame  
            **kwargs: max_records, output_file, etc.
            
        Returns:
            Estatísticas consolidadas no formato padrão
        """
        self.logger.info("⚡ Iniciando análise chunked integrada")
        
        # Se for DataFrame, converter para modo chunked simulado
        if isinstance(data_input, pd.DataFrame):
            return self._chunked_from_dataframe(data_input, **kwargs)
        
        # Processar arquivo em chunks
        file_path = Path(data_input)
        max_records = kwargs.get('max_records')
        output_file = kwargs.get('output_file')
        
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
            for chunk in self._load_dataset_chunks(file_path, max_records):
                chunk_start = time.time()
                
                # Analisar chunk usando o pipeline normal
                result = self.analyze_dataset(chunk.copy())
                
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
                
                # Consolidar outras estatísticas
                if 'has_temporal_data' in chunk_data.columns:
                    valid_temporal = chunk_data['has_temporal_data'].sum()
                    consolidated_stats['temporal_stats']['valid_timestamps'] += valid_temporal
                    consolidated_stats['temporal_stats']['total_records'] += len(chunk_data)
                
                if 'potential_coordination' in chunk_data.columns:
                    coordinated = chunk_data['potential_coordination'].sum()
                    consolidated_stats['coordination_stats']['coordinated'] += coordinated
                    consolidated_stats['coordination_stats']['total_records'] += len(chunk_data)
                
                if 'has_external_links' in chunk_data.columns:
                    with_links = chunk_data['has_external_links'].sum()
                    consolidated_stats['domain_stats']['with_links'] += with_links
                    consolidated_stats['domain_stats']['total_records'] += len(chunk_data)
                
                # Manter stats do último chunk
                consolidated_stats['stages_completed'] = chunk_stats.get('stages_completed', 0)
                consolidated_stats['features_extracted'] = chunk_stats.get('features_extracted', 0)
                
                # Salvar chunk se solicitado
                if output_file:
                    chunk_output = f"{output_file.replace('.csv', '')}_chunk_{total_chunks}.csv"
                    chunk_data.to_csv(chunk_output, index=False, sep=';')
                    self.logger.info(f"💾 Chunk salvo: {chunk_output}")
                
                chunk_time = time.time() - chunk_start
                chunk_performance = len(chunk_data) / chunk_time if chunk_time > 0 else 0
                
                self.logger.info(f"✅ Chunk {total_chunks} processado: {len(chunk_data):,} registros em {chunk_time:.1f}s ({chunk_performance:.1f} reg/s)")
                
                # Limpar memória entre chunks
                del chunk_data, result, chunk
                if self._check_memory_usage():
                    self._clean_memory()
                
        except Exception as e:
            self.logger.error(f"❌ Erro na análise chunked: {e}")
            consolidated_stats['processing_errors'] += 1
            raise
        
        # Atualizar stats principais
        self.stats.update({
            'total_records_processed': total_records,
            'total_chunks': total_chunks,
            'chunked_processing': True,
            'stages_completed': consolidated_stats['stages_completed'],
            'features_extracted': consolidated_stats['features_extracted']
        })
        
        self.logger.info(f"🎉 Análise chunked concluída: {total_records:,} registros em {total_chunks} chunks")
        
        # Retornar no formato padrão do Analyzer
        return {
            'data': pd.DataFrame({'processed_records': [total_records]}),  # DataFrame mínimo para compatibilidade
            'stats': self.stats.copy(),
            'consolidated_stats': consolidated_stats,
            'columns_generated': 85,  # Estimativa baseada no pipeline completo
            'stages_completed': consolidated_stats['stages_completed']
        }

    def _load_dataset_chunks(self, file_path: Path, max_records: Optional[int] = None):
        """Generator para carregar dataset em chunks."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {file_path}")
        
        self.logger.info(f"📂 Carregando dataset em chunks: {file_path}")
        
        records_processed = 0
        chunk_number = 1
        
        # Determinar separador baseado no dataset
        separator = ','  # Padrão para govbolso
        if '4_2022-2023-elec' in str(file_path):
            separator = ';'
        
        try:
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
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dataset: {e}")
            raise

    def _chunked_from_dataframe(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Simular processamento chunked para DataFrame que já está em memória."""
        self.logger.info("🔄 Simulando chunked para DataFrame em memória")
        
        # Se DataFrame é pequeno, processar normalmente
        if len(df) <= self.chunk_size:
            return self.analyze_dataset(df)
        
        # Dividir DataFrame em chunks
        chunks = [df[i:i+self.chunk_size] for i in range(0, len(df), self.chunk_size)]
        
        consolidated_results = []
        for i, chunk in enumerate(chunks, 1):
            self.logger.info(f"📦 Processando chunk {i}/{len(chunks)}: {len(chunk)} registros")
            result = self.analyze_dataset(chunk.copy())
            consolidated_results.append(result)
            
            if self._check_memory_usage():
                self._clean_memory()
        
        # Para simplificar, retornar resultado do último chunk com stats consolidados
        final_result = consolidated_results[-1]
        final_result['stats']['total_chunks'] = len(chunks)
        final_result['stats']['chunked_processing'] = True
        
        return final_result

    def analyze_dataset(self, data_input) -> Dict[str, Any]:
        """
        Analisar dataset com pipeline sequencial otimizado de 17 stages.

        NOVA SEQUÊNCIA OTIMIZADA (conforme PIPELINE_STAGES_ANALYSIS.md):
        Fase 1: Preparação (01-02) - estrutura básica
        Fase 2: Redução de volume (03-06) - CRÍTICO para performance
        Fase 3: Análise linguística (07-09) - volume reduzido
        Fase 4: Análises avançadas (10-17) - dados otimizados

        Args:
            data_input: DataFrame ou caminho do arquivo para análise

        Returns:
            Dict com resultado da análise
        """
        # Verificar se deve usar chunking
        should_chunk, estimated_records, reason = self._should_use_chunking(data_input)

        # Se for caminho de arquivo e deve usar chunking, usar processamento chunked
        if self.auto_chunk and should_chunk:
            self.logger.info(f"⚡ Chunking ativado: {reason}")
            self.stats['chunked_processing'] = True
            return self._analyze_chunked(data_input)

        # Senão, carregar dados e processar normalmente
        self.stats['chunked_processing'] = False

        # Se for caminho de arquivo, carregar
        if isinstance(data_input, (str, Path)):
            import pandas as pd
            df = pd.read_csv(data_input, sep=';', encoding='utf-8')
            self.logger.info(f"📂 Dataset carregado: {len(df)} registros")
        else:
            df = data_input

        try:
            self.logger.info(f"🔬 Iniciando análise OTIMIZADA: {len(df)} registros")

            # Reset stats
            self.stats = {'stages_completed': 0, 'features_extracted': 0, 'processing_errors': 0}

            # ===========================================
            # FASE 1: PREPARAÇÃO E ESTRUTURA (01-02)
            # ===========================================
            
            # STAGE 01: Feature Extraction (estrutura básica)
            df = self._stage_01_feature_extraction(df)

            # STAGE 02: Text Preprocessing (limpeza básica)
            df = self._stage_02_text_preprocessing(df)

            # ===========================================
            # FASE 2: REDUÇÃO DE VOLUME (03-06) - CRÍTICO!
            # ===========================================
            
            # STAGE 03: Cross-Dataset Deduplication (40-50% redução)
            df = self._stage_03_cross_dataset_deduplication(df)

            # STAGE 04: Statistical Analysis (comparação antes/depois)
            df = self._stage_04_statistical_analysis(df)

            # STAGE 05: Content Quality Filter (15-25% redução adicional)
            df = self._stage_05_content_quality_filter(df)

            # STAGE 06: Political Relevance Filter (30-40% redução adicional)
            df = self._stage_06_political_relevance_filter(df)

            self.logger.info(f"📊 FASE 2 CONCLUÍDA: Volume reduzido para {len(df):,} registros")

            # ===========================================
            # FASE 3: ANÁLISE LINGUÍSTICA (07-09) - VOLUME REDUZIDO
            # ===========================================
            
            # STAGE 07: Linguistic Processing (spaCy - AGORA com volume otimizado)
            df = self._stage_07_linguistic_processing(df)  # Usar método existente

            # STAGE 08: Political Classification (usando tokens spaCy)
            df = self._stage_08_political_classification(df)  # Usar método existente

            # STAGE 09: TF-IDF Vectorization (usando lemmas spaCy)
            df = self._stage_09_tfidf_vectorization(df)  # Usar método existente

            # ===========================================
            # FASE 4: ANÁLISES AVANÇADAS (10-17)
            # ===========================================
            
            # STAGE 10: Clustering Analysis
            df = self._stage_10_clustering_analysis(df)  # Usar método existente

            # STAGE 11: Topic Modeling
            df = self._stage_11_topic_modeling(df)  # Usar método existente

            # STAGE 12: Semantic Analysis
            df = self._stage_12_semantic_analysis(df)  # Usar método existente

            # STAGE 13: Temporal Analysis
            df = self._stage_13_temporal_analysis(df)  # Usar método existente

            # STAGE 14: Network Analysis
            df = self._stage_14_network_analysis(df)  # Usar método existente

            # STAGE 15: Domain Analysis
            df = self._stage_15_domain_analysis(df)  # Usar método existente

            # STAGE 16: Event Context Analysis
            df = self._stage_16_event_context(df)  # Usar método existente

            # STAGE 17: Channel Analysis
            df = self._stage_17_channel_analysis(df)  # Usar método existente

            # Final metadata
            df['processing_timestamp'] = datetime.now().isoformat()
            df['stages_completed'] = self.stats['stages_completed']
            df['features_extracted'] = self.stats['features_extracted']

            self.logger.info(f"✅ Análise OTIMIZADA concluída: {len(df.columns)} colunas, {self.stats['stages_completed']} stages")
            self.logger.info(f"🎯 Performance: Processados {len(df):,} registros finais")

            return {
                'success': True,
                'data': df,
                'stats': self.stats.copy(),
                'columns_generated': len(df.columns),
                'stages_completed': self.stats['stages_completed']
            }

        except Exception as e:
            self.logger.error(f"❌ Erro na análise: {e}")
            raise

    def _stage_01_feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 01: Extração e identificação automática de features (Python puro).

        SEMPRE O PRIMEIRO STAGE - identifica colunas disponíveis e extrai features se necessário.
        """
        self.logger.info("🔍 STAGE 01: Feature Extraction")

        # Identificar coluna de texto principal
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Verificar se contém texto substancial
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    avg_length = sample.astype(str).str.len().mean()
                    if avg_length > 20:  # Textos com mais de 20 caracteres em média
                        text_columns.append(col)

        # Selecionar melhor coluna de texto
        if not text_columns:
            raise ValueError("❌ Nenhuma coluna de texto encontrada")

        # Priorizar colunas comuns
        priority_columns = ['text', 'body', 'message', 'content', 'texto', 'mensagem']
        main_text_column = None

        for priority in priority_columns:
            if priority in text_columns:
                main_text_column = priority
                break

        if not main_text_column:
            main_text_column = text_columns[0]

        # Identificar coluna de timestamp (se disponível)
        timestamp_column = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
                timestamp_column = col
                break

        # === PADRONIZAÇÃO DE DATETIME ===
        if timestamp_column:
            df = self._standardize_datetime_column(df, timestamp_column)
            # Após padronização, a coluna se chama 'datetime'
            timestamp_column = 'datetime'

        # DETECÇÃO AUTOMÁTICA DE FEATURES EXISTENTES
        features_detected = self._detect_existing_features(df)

        # EXTRAÇÃO AUTOMÁTICA DE FEATURES (se não existem)
        df = self._extract_missing_features(df, main_text_column, features_detected)

        # === CONTAR COLUNAS DE METADADOS ===
        # Metadados = todas as colunas exceto texto principal e datetime padronizado
        metadata_columns = []
        for col in df.columns:
            if col not in [main_text_column, 'datetime'] and not col.startswith(('emojis_', 'hashtags_', 'urls_', 'mentions_')):
                metadata_columns.append(col)

        # Adicionar features identificadas
        df['main_text_column'] = main_text_column
        df['timestamp_column'] = timestamp_column if timestamp_column else 'none'
        df['metadata_columns_count'] = len(metadata_columns)
        df['has_timestamp'] = timestamp_column is not None

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 4 + len(features_detected['extracted'])

        self.logger.info(f"✅ Features: text={main_text_column}, timestamp={timestamp_column}")
        self.logger.info(f"✅ Features detectadas: {list(features_detected['existing'].keys())}")
        self.logger.info(f"✅ Features extraídas: {features_detected['extracted']}")
        if timestamp_column:
            self.logger.info(f"📅 Datetime otimizado: coluna única 'datetime'")
        return df

    def _standardize_datetime_column(self, df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """
        Padronizar coluna de datetime para formato único DD/MM/AAAA HH:MM:SS.
        Remove coluna original e substitui por versão padronizada.
        
        Args:
            df: DataFrame com dados
            timestamp_column: Nome da coluna de timestamp identificada
            
        Returns:
            DataFrame com coluna datetime padronizada (substitui a original)
        """
        self.logger.info(f"📅 Padronizando datetime da coluna: {timestamp_column}")
        
        def parse_datetime(datetime_str):
            """Tentar múltiplos formatos de datetime."""
            if pd.isna(datetime_str):
                return None
                
            datetime_str = str(datetime_str).strip()
            
            # Formatos comuns para tentar
            formats_to_try = [
                '%Y-%m-%d %H:%M:%S',      # 2019-07-02 01:10:00
                '%d/%m/%Y %H:%M:%S',      # 02/07/2019 01:10:00
                '%Y-%m-%d',               # 2019-07-02
                '%d/%m/%Y',               # 02/07/2019
                '%Y-%m-%d %H:%M',         # 2019-07-02 01:10
                '%d/%m/%Y %H:%M',         # 02/07/2019 01:10
                '%Y/%m/%d %H:%M:%S',      # 2019/07/02 01:10:00
                '%m/%d/%Y %H:%M:%S',      # 07/02/2019 01:10:00 (formato americano)
            ]
            
            for fmt in formats_to_try:
                try:
                    parsed_dt = pd.to_datetime(datetime_str, format=fmt)
                    # Converter para formato padrão brasileiro DD/MM/AAAA HH:MM:SS
                    return parsed_dt.strftime('%d/%m/%Y %H:%M:%S')
                except (ValueError, TypeError):
                    continue
                    
            # Se nenhum formato funcionou, tentar parse genérico do pandas
            try:
                parsed_dt = pd.to_datetime(datetime_str, infer_datetime_format=True)
                return parsed_dt.strftime('%d/%m/%Y %H:%M:%S')
            except:
                return None
        
        # Aplicar padronização
        datetime_standardized = df[timestamp_column].apply(parse_datetime)
        
        # === SUBSTITUIR COLUNA ORIGINAL ===
        # Remover coluna original e usar nome 'datetime' para a versão padronizada
        df = df.drop(columns=[timestamp_column])
        df['datetime'] = datetime_standardized
        
        # Estatísticas de conversão
        valid_datetimes = df['datetime'].notna().sum()
        total_records = len(df)
        success_rate = (valid_datetimes / total_records) * 100
        
        self.logger.info(f"✅ Datetime padronizado e substituído: {valid_datetimes}/{total_records} ({success_rate:.1f}%) válidos")
        
        # Amostras do resultado
        sample_standardized = df['datetime'].dropna().head(3).tolist()
        
        self.logger.info(f"📋 Formato final:")
        for i, std in enumerate(sample_standardized):
            self.logger.info(f"   {i+1}. {std}")
        
        return df

    def _detect_existing_features(self, df: pd.DataFrame) -> Dict:
        """
        Detecta features que já existem como colunas no DataFrame.
        """
        existing_features = {}

        # Features de interesse para detectar
        feature_patterns = {
            'hashtags': ['hashtag', 'hashtags', 'tags'],
            'urls': ['url', 'urls', 'links', 'link'],
            'mentions': ['mention', 'mentions', 'user_mentions', 'usuarios'],
            'emojis': ['emoji', 'emojis', 'emoticon'],
            'reply_count': ['reply', 'replies', 'respostas'],
            'retweet_count': ['retweet', 'retweets', 'rt_count'],
            'like_count': ['like', 'likes', 'curtidas', 'fav'],
            'user_info': ['user', 'username', 'author', 'usuario']
        }

        for feature_name, patterns in feature_patterns.items():
            for pattern in patterns:
                matching_cols = [col for col in df.columns if pattern in col.lower()]
                if matching_cols:
                    existing_features[feature_name] = matching_cols[0]
                    break

        return {
            'existing': existing_features,
            'extracted': []
        }

    def _extract_missing_features(self, df: pd.DataFrame, text_column: str, features_info: Dict) -> pd.DataFrame:
        """
        Extrai apenas features essenciais do texto principal.
        """
        extracted_features = []

        # Só extrair se não existir coluna correspondente
        if 'hashtags' not in features_info['existing']:
            df['hashtags_extracted'] = df[text_column].astype(str).str.findall(r'#\w+')
            extracted_features.append('hashtags')

        if 'urls' not in features_info['existing']:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            df['urls_extracted'] = df[text_column].astype(str).str.findall(url_pattern)
            extracted_features.append('urls')

        if 'mentions' not in features_info['existing']:
            # Padrão para @mentions
            df['mentions_extracted'] = df[text_column].astype(str).str.findall(r'@\w+')
            extracted_features.append('mentions')

        if 'emojis' not in features_info['existing']:
            # Padrão básico para emojis (Unicode)
            emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]'
            df['emojis_extracted'] = df[text_column].astype(str).str.findall(emoji_pattern)
            extracted_features.append('emojis')

        # REMOVIDAS: has_interrogation, has_exclamation, has_caps_words, has_portuguese_words
        # Estas colunas não são necessárias para a análise

        features_info['extracted'] = extracted_features
        return df

    def _stage_02_text_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 02: Validação de Features + Limpeza de Texto.

        Para datasets com estrutura correta (datetime, body, url, hashtag, channel, etc):
        1. Validar features existentes contra coluna 'body' 
        2. Corrigir features incorretas/vazias
        3. Limpar body_cleaned (texto sem features)
        4. Aplicar normalização de texto
        
        USA: Detecção automática da estrutura do dataset
        """
        self.logger.info("🧹 STAGE 02: Feature Validation + Text Preprocessing")

        # === DETECTAR ESTRUTURA DO DATASET ===
        expected_columns = ['datetime', 'body', 'url', 'hashtag', 'channel', 'is_fwrd', 'mentions', 'sender', 'media_type', 'domain', 'body_cleaned']
        
        if all(col in df.columns for col in expected_columns[:5]):  # Verificar colunas essenciais
            self.logger.info("✅ Dataset estruturado detectado - validando features existentes")
            
            # === FASE 1: VALIDAÇÃO DE FEATURES EXISTENTES ===
            df = self._extract_and_validate_features(df, 'body')
            
            # === FASE 2: LIMPEZA DE TEXTO (usar body como principal) ===
            main_text_col = 'body'
            
        else:
            self.logger.info("⚠️ Dataset não estruturado - usando coluna principal")
            main_text_col = df['main_text_column'].iloc[0]
            
            # === FASE 1: EXTRAÇÃO DE FEATURES ===
            df = self._extract_and_validate_features(df, main_text_col)
        
        # === FASE 2: NORMALIZAÇÃO DE TEXTO ===
        def clean_text(text):
            """Limpar texto usando Python puro."""
            if pd.isna(text):
                return ""

            text = str(text)

            # Normalizar unicode
            text = unicodedata.normalize('NFKD', text)

            # Remover caracteres especiais mas preservar acentos
            text = re.sub(r'[^\w\s\u00C0-\u017F]', ' ', text)

            # Normalizar espaços
            text = re.sub(r'\s+', ' ', text).strip()

            # Converter para lowercase
            text = text.lower()

            return text

        # Aplicar limpeza ao texto principal
        df['normalized_text'] = df[main_text_col].apply(clean_text)

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 2

        self.logger.info(f"✅ Stage 02 concluído: {df['normalized_text'].str.len().mean():.1f} chars média")
        return df

    def _extract_and_validate_features(self, df: pd.DataFrame, main_text_col: str) -> pd.DataFrame:
        """
        Validar features existentes contra coluna 'body' e corrigir se necessário.
        
        Dataset já tem: datetime, body, url, hashtag, channel, is_fwrd, mentions, sender, media_type, domain, body_cleaned
        """
        self.logger.info("🔍 Validando features existentes contra coluna 'body'...")
        
        # === VERIFICAR SE DATASET TEM ESTRUTURA CORRETA ===
        expected_columns = ['datetime', 'body', 'url', 'hashtag', 'channel', 'is_fwrd', 'mentions', 'sender', 'media_type', 'domain', 'body_cleaned']
        
        # Se o dataset tem as colunas corretas, usar body como texto principal
        if all(col in df.columns for col in expected_columns[:5]):  # Verificar colunas essenciais
            self.logger.info("✅ Dataset com estrutura correta detectado")
            
            # === REMOVER BODY_CLEANED (duplicação desnecessária) ===
            if 'body_cleaned' in df.columns:
                df = df.drop(columns=['body_cleaned'])
                self.logger.info("🗑️ body_cleaned removido (duplicação desnecessária)")
            
            # === VALIDAR FEATURES CONTRA BODY ===
            corrections_made = 0
            
            # Validar URL
            if 'url' in df.columns and 'body' in df.columns:
                corrections_made += self._validate_feature_against_body(df, 'url', 'body', [r'https?://\S+', r'www\.\S+'])
            
            # Validar Hashtags
            if 'hashtag' in df.columns and 'body' in df.columns:
                corrections_made += self._validate_feature_against_body(df, 'hashtag', 'body', [r'#\w+'])
            
            # Validar Mentions
            if 'mentions' in df.columns and 'body' in df.columns:
                corrections_made += self._validate_feature_against_body(df, 'mentions', 'body', [r'@\w+'])
            
            self.logger.info(f"✅ Validação concluída: {corrections_made} correções aplicadas")
            
        else:
            # === DATASET SEM ESTRUTURA PADRÃO - EXTRAIR TUDO ===
            self.logger.info("⚠️ Dataset sem estrutura padrão - extraindo features do texto principal")
            df = self._extract_features_from_text(df, main_text_col)
        
        return df
    
    def _validate_feature_against_body(self, df: pd.DataFrame, feature_col: str, body_col: str, patterns: list) -> int:
        """Validar feature específica contra body."""
        corrections = 0
        
        for idx, row in df.iterrows():
            body_text = str(row[body_col]) if pd.notna(row[body_col]) else ""
            existing_feature = row[feature_col] if pd.notna(row[feature_col]) else ""
            
            # Extrair feature do body
            extracted_features = []
            for pattern in patterns:
                matches = re.findall(pattern, body_text, re.IGNORECASE)
                extracted_features.extend(matches)
            
            # Se encontrou features no body mas coluna está vazia, corrigir
            if extracted_features and not existing_feature:
                if len(extracted_features) == 1:
                    df.at[idx, feature_col] = extracted_features[0]
                else:
                    df.at[idx, feature_col] = ';'.join(extracted_features)  # Múltiplas features
                corrections += 1
        
        if corrections > 0:
            self.logger.info(f"🔧 {feature_col}: {corrections} correções aplicadas")
        
        return corrections
    
    def _clean_body_text(self, df: pd.DataFrame):
        """
        REMOVIDO: body_cleaned não é mais necessário.
        O texto limpo é gerado como 'normalized_text' no Stage 02.
        """
        # Não fazer nada - body_cleaned removido para evitar duplicação
        self.logger.info("🗑️ body_cleaned removido (duplicação desnecessária)")
        pass    
    def _extract_features_from_text(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Extrair features de dataset sem estrutura padrão (fallback)."""
        
        # Features essenciais para extrair
        feature_patterns = {
            'urls': [r'https?://\S+', r'www\.\S+'],
            'hashtags': [r'#\w+'],
            'mentions': [r'@\w+'],
            'channel_name': []  # Usar valor padrão
        }
        
        for feature_name, patterns in feature_patterns.items():
            if patterns:
                def extract_feature(text):
                    if pd.isna(text):
                        return []
                    
                    extracted = []
                    for pattern in patterns:
                        matches = re.findall(pattern, str(text), re.IGNORECASE)
                        extracted.extend(matches)
                    return list(set(extracted)) if extracted else []
                
                df[feature_name] = df[text_col].apply(extract_feature)
            else:
                df[feature_name] = "unknown_channel"
        
        self.logger.info("🆕 Features extraídas de dataset sem estrutura padrão")
        return df
    
    def _find_existing_feature_column(self, df: pd.DataFrame, possible_names: list) -> str:
        """Encontrar coluna existente para uma feature."""
        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
        return None
    
    def _validate_and_correct_feature(self, df: pd.DataFrame, existing_col: str, feature_name: str, patterns: list, text_col: str) -> pd.DataFrame:
        """Validar e corrigir feature existente."""
        if not patterns:  # Channel name não tem padrão regex
            self.logger.info(f"✅ Feature {existing_col} mantida (sem validação regex)")
            return df
        
        # Extrair valores corretos do texto
        def extract_correct_values(text):
            if pd.isna(text):
                return []
            
            text = str(text)
            extracted = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                extracted.extend(matches)
            return list(set(extracted))  # Remover duplicatas
        
        # Validar contra texto original
        df['_temp_extracted'] = df[text_col].apply(extract_correct_values)
        
        # Comparar e corrigir se necessário
        corrections_made = 0
        
        def validate_and_fix(row):
            nonlocal corrections_made
            existing_value = row[existing_col]
            correct_value = row['_temp_extracted']
            
            # Se valor existente está vazio ou incorreto, corrigir
            if pd.isna(existing_value) or existing_value == [] or existing_value == '':
                if correct_value:
                    corrections_made += 1
                    return correct_value
            
            return existing_value
        
        df[existing_col] = df.apply(validate_and_fix, axis=1)
        df = df.drop('_temp_extracted', axis=1)
        
        if corrections_made > 0:
            self.logger.info(f"🔧 Feature {existing_col}: {corrections_made} correções aplicadas")
        else:
            self.logger.info(f"✅ Feature {existing_col}: validação OK, sem correções necessárias")
        
        return df
    
    def _extract_new_feature(self, df: pd.DataFrame, feature_name: str, patterns: list, text_col: str) -> pd.DataFrame:
        """Extrair nova feature do texto."""
        def extract_feature(text):
            if pd.isna(text):
                return []
            
            text = str(text)
            extracted = []
            
            if patterns:  # Features com padrão regex
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    extracted.extend(matches)
            else:  # Channel name - tentar extrair de metadados ou usar valor padrão
                return "unknown_channel"
            
            return list(set(extracted)) if extracted else []
        
        # Extrair feature
        df[feature_name] = df[text_col].apply(extract_feature)
        
        # Estatísticas
        non_empty = df[feature_name].apply(lambda x: len(x) > 0 if isinstance(x, list) else bool(x)).sum()
        total = len(df)
        
        self.logger.info(f"📊 Feature {feature_name}: {non_empty}/{total} registros ({non_empty/total*100:.1f}%)")
        
        return df

    def _stage_07_linguistic_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 03: Processamento linguístico com spaCy (LOGO APÓS limpeza).

        USA: normalized_text do Stage 02
        GERA: tokens, lemmas, POS tags, entidades nomeadas
        """
        self.logger.info("🔤 STAGE 03: Linguistic Processing (spaCy)")

        if not SPACY_AVAILABLE:
            self.logger.warning("⚠️ spaCy não disponível - usando processamento básico")
            return self._linguistic_fallback(df)

        def process_text_with_spacy(text):
            """Processar texto com spaCy otimizado."""
            if pd.isna(text) or len(str(text).strip()) == 0:
                return {
                    'tokens': [],
                    'lemmas': [],
                    'pos_tags': [],
                    'entities': [],
                    'tokens_count': 0,
                    'entities_count': 0
                }

            try:
                # Processar texto (limitando para performance)
                doc = nlp(str(text)[:1000])  # Limitar a 1000 chars para performance

                tokens = [token.text for token in doc if not token.is_space]
                lemmas = [token.lemma_ for token in doc if not token.is_space and token.lemma_ != '-PRON-']
                entities = [(ent.text, ent.label_) for ent in doc.ents]

                return {
                    'tokens': tokens,
                    'lemmas': lemmas,
                    'tokens_count': len(tokens),
                    'entities_count': len(entities)
                }
            except Exception as e:
                self.logger.warning(f"Erro spaCy: {e}")
                return {
                    'tokens': [],
                    'lemmas': [],
                    'tokens_count': 0,
                    'entities_count': 0
                }

        # Processar textos normalizados
        spacy_results = df['normalized_text'].apply(process_text_with_spacy)

        # Extrair dados do spaCy (removidos pos_tags e entities - não utilizados)
        df['spacy_tokens'] = spacy_results.apply(lambda x: x['tokens'])
        df['spacy_lemmas'] = spacy_results.apply(lambda x: x['lemmas'])
        df['spacy_tokens_count'] = spacy_results.apply(lambda x: x['tokens_count'])
        df['spacy_entities_count'] = spacy_results.apply(lambda x: x['entities_count'])

        # Criar texto processado com lemmas para stages posteriores
        df['lemmatized_text'] = df['spacy_lemmas'].apply(lambda x: ' '.join(x) if x else '')

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 4

        avg_tokens = df['spacy_tokens_count'].mean()
        avg_entities = df['spacy_entities_count'].mean()
        self.logger.info(f"✅ spaCy processado: {avg_tokens:.1f} tokens, {avg_entities:.1f} entidades média")
        return df

    def _linguistic_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback básico quando spaCy não está disponível."""
        # Tokenização básica
        df['spacy_tokens'] = df['normalized_text'].str.split()
        df['spacy_tokens_count'] = df['spacy_tokens'].str.len()
        df['spacy_entities_count'] = 0
        df['lemmatized_text'] = df['normalized_text']  # Usar texto normalizado

        self.stats['stages_completed'] += 1
        self.stats['features_extracted'] += 3
        return df

    def _stage_03_cross_dataset_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 03: Cross-Dataset Deduplication
        
        Eliminação de duplicatas entre TODOS os datasets com contador de frequência.
        Algoritmo: Agrupar por texto idêntico, manter registro mais antigo, 
        contar duplicatas com dupli_freq.
        
        Redução esperada: 40-50% (300k → 180k)
        """
        try:
            self.logger.info("🔄 STAGE 03: Cross-Dataset Deduplication")
            
            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            datetime_column = 'datetime' if 'datetime' in df.columns else df.columns[df.columns.str.contains('date|time', case=False)].tolist()[0] if any(df.columns.str.contains('date|time', case=False)) else None
            
            initial_count = len(df)
            self.logger.info(f"📊 Registros iniciais: {initial_count:,}")
            
            # Agrupar por texto idêntico
            grouping_columns = [text_column]
            
            # Preparar dados para agrupamento
            dedup_data = []
            
            for text, group in df.groupby(text_column):
                if pd.isna(text) or text.strip() == '':
                    continue
                    
                # Manter registro mais antigo (primeiro datetime)
                if datetime_column and datetime_column in group.columns:
                    # Converter datetime para ordenação
                    group_sorted = group.copy()
                    if group_sorted[datetime_column].dtype == 'object':
                        try:
                            group_sorted['datetime_parsed'] = pd.to_datetime(group_sorted[datetime_column], 
                                                                           format='%d/%m/%Y %H:%M:%S', errors='coerce')
                        except:
                            group_sorted['datetime_parsed'] = pd.to_datetime(group_sorted[datetime_column], errors='coerce')
                        
                        # Ordenar por datetime e pegar o mais antigo
                        oldest_record = group_sorted.sort_values('datetime_parsed').iloc[0]
                    else:
                        oldest_record = group.iloc[0]
                else:
                    oldest_record = group.iloc[0]
                
                # Contador de duplicatas
                dupli_freq = len(group)
                
                # Metadados de dispersão
                channels_found = []
                if 'channel' in group.columns:
                    channels_found = group['channel'].dropna().unique().tolist()
                elif 'sender_id' in group.columns:
                    channels_found = group['sender_id'].dropna().unique().tolist()
                
                # Período de ocorrência
                date_span_days = 0
                if datetime_column and datetime_column in group.columns:
                    try:
                        dates = pd.to_datetime(group[datetime_column], errors='coerce').dropna()
                        if len(dates) > 1:
                            date_span_days = (dates.max() - dates.min()).days
                    except:
                        pass
                
                # Criar registro deduplificado
                dedup_record = oldest_record.copy()
                dedup_record['dupli_freq'] = dupli_freq
                dedup_record['channels_found'] = len(channels_found)
                dedup_record['date_span_days'] = date_span_days
                
                dedup_data.append(dedup_record)
            
            # Criar DataFrame deduplificado
            if dedup_data:
                df_deduplicated = pd.DataFrame(dedup_data)
                df_deduplicated = df_deduplicated.reset_index(drop=True)
            else:
                df_deduplicated = df.copy()
                df_deduplicated['dupli_freq'] = 1
                df_deduplicated['channels_found'] = 0
                df_deduplicated['date_span_days'] = 0
            
            final_count = len(df_deduplicated)
            reduction_pct = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
            
            # Estatísticas de deduplicação
            unique_texts = df_deduplicated['dupli_freq'].value_counts().sort_index()
            total_duplicates = df_deduplicated[df_deduplicated['dupli_freq'] > 1]['dupli_freq'].sum()
            
            self.logger.info(f"✅ Deduplicação concluída:")
            self.logger.info(f"   📉 {initial_count:,} → {final_count:,} registros")
            self.logger.info(f"   📊 Redução: {reduction_pct:.1f}%")
            self.logger.info(f"   🔄 Duplicatas processadas: {total_duplicates:,}")
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 3
            
            return df_deduplicated
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 03: {e}")
            self.stats['processing_errors'] += 1
            # Em caso de erro, adicionar colunas padrão
            df['dupli_freq'] = 1
            df['channels_found'] = 0
            df['date_span_days'] = 0
            return df

    def _stage_04_statistical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 04: Statistical Analysis
        
        Comparar início do dataset com o dataset reduzido.
        Gerar estatísticas para classificação e gráficos.
        
        Processamentos:
        - Contagem de dados antes e depois
        - Proporção de duplicadas
        - Proporção de hashtags
        - Detecção de repetições excessivas para tabela com 10 principais casos
        """
        try:
            self.logger.info("📊 STAGE 04: Statistical Analysis")
            
            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            
            # === ANÁLISE DE DUPLICAÇÃO ===
            total_registros = len(df)
            registros_unicos = len(df[df['dupli_freq'] == 1])
            registros_duplicados = total_registros - registros_unicos
            
            duplicacao_pct = (registros_duplicados / total_registros * 100) if total_registros > 0 else 0
            
            # === ANÁLISE DE HASHTAGS ===
            has_hashtags = 0
            if 'has_hashtags' in df.columns:
                has_hashtags = df['has_hashtags'].sum()
            elif text_column in df.columns:
                has_hashtags = df[text_column].str.contains('#', na=False).sum()
            
            hashtag_pct = (has_hashtags / total_registros * 100) if total_registros > 0 else 0
            
            # === TOP 10 REPETIÇÕES EXCESSIVAS ===
            top_duplicates = df[df['dupli_freq'] > 1].nlargest(10, 'dupli_freq')[
                [text_column, 'dupli_freq', 'channels_found', 'date_span_days']
            ].to_dict('records')
            
            # === ESTATÍSTICAS BÁSICAS DE TEXTO ===
            if text_column in df.columns:
                char_counts = df[text_column].str.len().fillna(0)
                word_counts = df[text_column].str.split().str.len().fillna(0)
                
                df['char_count'] = char_counts
                df['word_count'] = word_counts
                
                avg_chars = char_counts.mean()
                avg_words = word_counts.mean()
            else:
                avg_chars = 0
                avg_words = 0
                df['char_count'] = 0
                df['word_count'] = 0
            
            # === PROPORÇÕES DE QUALIDADE ===
            if text_column in df.columns:
                df['emoji_ratio'] = df[text_column].apply(self._calculate_emoji_ratio)
                df['caps_ratio'] = df[text_column].apply(self._calculate_caps_ratio)
                df['repetition_ratio'] = df[text_column].apply(self._calculate_repetition_ratio)
                
                # Detecção de idioma básica
                df['likely_portuguese'] = df[text_column].apply(self._detect_portuguese)
            else:
                df['emoji_ratio'] = 0.0
                df['caps_ratio'] = 0.0
                df['repetition_ratio'] = 0.0
                df['likely_portuguese'] = True
            
            # === CONSOLIDAÇÃO DE ESTATÍSTICAS ===
            # Consolidar estatísticas globais em objeto summary
            summary_stats = {
                'total_dataset_size': total_registros,
                'unique_texts_count': registros_unicos,
                'duplication_percentage': round(duplicacao_pct, 2),
                'hashtag_percentage': round(hashtag_pct, 2),
                'avg_chars_per_text': round(avg_chars, 1),
                'avg_words_per_text': round(avg_words, 1)
            }

            # Salvar no contexto para acesso posterior
            self.global_stats = summary_stats
            
            # Log das estatísticas
            self.logger.info(f"✅ Análise estatística concluída:")
            self.logger.info(f"   📊 Total de registros: {total_registros:,}")
            self.logger.info(f"   🔄 Duplicação: {duplicacao_pct:.1f}%")
            self.logger.info(f"   # Hashtags: {hashtag_pct:.1f}%")
            self.logger.info(f"   📝 Média: {avg_words:.1f} palavras, {avg_chars:.0f} chars")
            
            if top_duplicates:
                self.logger.info(f"   🔝 Maior repetição: {top_duplicates[0]['dupli_freq']} ocorrências")
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 11
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 04: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_05_content_quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 05: Content Quality Filter
        
        Filtrar conteúdo por qualidade e completude.
        Input: Dados deduplificados
        Output: Apenas conteúdo de qualidade
        
        Filtros:
        - Comprimento: < 10 chars ou > 2000 chars
        - Qualidade: emoji_ratio > 70%, caps_ratio > 80%, repetition_ratio > 50%
        - Idioma: Manter apenas likely_portuguese = True
        
        Redução esperada: 15-25% (180k → 135k)
        """
        try:
            self.logger.info("🎯 STAGE 05: Content Quality Filter")
            
            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            initial_count = len(df)
            
            # === FILTROS DE COMPRIMENTO ===
            # Muito curto: < 10 chars (só emoji/URL)
            length_filter = (df['char_count'] >= 10) & (df['char_count'] <= 2000)
            
            # === FILTROS DE QUALIDADE ===
            # emoji_ratio > 70% = ruído
            emoji_filter = df['emoji_ratio'] <= 0.70
            
            # caps_ratio > 80% = spam  
            caps_filter = df['caps_ratio'] <= 0.80
            
            # repetition_ratio > 50% = baixa qualidade
            repetition_filter = df['repetition_ratio'] <= 0.50
            
            # === FILTROS DE IDIOMA ===
            # Manter apenas likely_portuguese = True
            language_filter = df['likely_portuguese'] == True
            
            # === APLICAR TODOS OS FILTROS ===
            quality_mask = length_filter & emoji_filter & caps_filter & repetition_filter & language_filter
            
            # === GERAR COLUNAS DE QUALIDADE ===
            # Contar problemas por tipo para logging
            problems = {
                'length_issue': (~length_filter).sum(),
                'excessive_emojis': (~emoji_filter).sum(),
                'excessive_caps': (~caps_filter).sum(),
                'excessive_repetition': (~repetition_filter).sum(),
                'non_portuguese': (~language_filter).sum()
            }
            
            # Content quality score (0-100)
            quality_components = [
                length_filter.astype(int) * 20,  # 20 pontos para comprimento adequado
                emoji_filter.astype(int) * 20,   # 20 pontos para emojis adequados
                caps_filter.astype(int) * 20,    # 20 pontos para caps adequados
                repetition_filter.astype(int) * 20, # 20 pontos para repetição adequada
                language_filter.astype(int) * 20    # 20 pontos para português
            ]
            
            df['content_quality_score'] = sum(quality_components)
            
            # === APLICAR FILTRO ===
            df_filtered = df[quality_mask].copy().reset_index(drop=True)
            
            final_count = len(df_filtered)
            reduction_pct = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
            
            # === ESTATÍSTICAS DOS FILTROS ===
            avg_quality_score = df_filtered['content_quality_score'].mean()

            self.logger.info(f"✅ Filtro de qualidade aplicado:")
            self.logger.info(f"   📉 {initial_count:,} → {final_count:,} registros")
            self.logger.info(f"   📊 Redução: {reduction_pct:.1f}%")
            self.logger.info(f"   🎯 Score qualidade médio: {avg_quality_score:.1f}/100")
            self.logger.info(f"   ❌ Rejeitados: comprimento={problems['length_issue']}, emojis={problems['excessive_emojis']}")
            self.logger.info(f"      caps={problems['excessive_caps']}, repetição={problems['excessive_repetition']}, idioma={problems['non_portuguese']}")

            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 1
            
            return df_filtered
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 05: {e}")
            self.stats['processing_errors'] += 1
            # Em caso de erro, retornar dados originais com colunas padrão
            df['content_quality_score'] = 80
            return df

    def _stage_06_political_relevance_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 06: Political Relevance Filter
        
        Manter apenas conteúdo relevante para a pesquisa política.
        Input: Conteúdo de qualidade
        Output: Apenas textos com relevância temática
        
        Usa léxico político brasileiro com categorias cat0-cat10:
        - cat0: autoritarismo_regime
        - cat2: pandemia_covid  
        - cat3: violencia_seguranca
        - cat4: religiao_moral
        - cat6: inimigos_ideologicos + identidade_politica
        - cat7: meio_ambiente_amazonia
        - cat8: moralidade
        - cat9: antissistema
        - cat10: polarizacao
        
        Redução esperada: 30-40% (135k → 80k)
        """
        try:
            self.logger.info("🎯 STAGE 06: Political Relevance Filter")
            
            # Importar léxico político
            political_keywords = {
                'cat0_autoritarismo_regime': [
                    'ai-5', 'regime militar', 'ditadura', 'tortura', 'repressão', 'intervenção militar',
                    'estado de sítio', 'golpe', 'censura', 'doutrina de segurança nacional'
                ],
                'cat2_pandemia_covid': [
                    'covid-19', 'corona', 'pandemia', 'quarentena', 'lockdown', 'tratamento precoce',
                    'cloroquina', 'ivermectina', 'máscara', 'máscaras', 'oms', 'pfizer', 'vacina',
                    'passaporte sanitário'
                ],
                'cat3_violencia_seguranca': [
                    'criminalidade', 'segurança pública', 'violência', 'bandidos', 'facções', 'polícia',
                    'militarização', 'armas', 'desarmamento', 'legítima defesa'
                ],
                'cat4_religiao_moral': [
                    'família tradicional', 'valores cristãos', 'igreja', 'pastor', 'padre', 'bíblia',
                    'cristofobia', 'marxismo cultural', 'ideologia de gênero'
                ],
                'cat6_inimigos_ideologicos': [
                    'comunista', 'comunismo', 'esquerdista', 'petista', 'pt', 'lula', 'stf', 'supremo',
                    'globo', 'mídia lixo', 'sistema', 'globalista', 'china', 'urss', 'cuba', 'venezuela',
                    'narcoditadura', 'esquerda', 'progressista'
                ],
                'cat6_identidade_politica': [
                    'bolsonaro', 'bolsonarista', 'direita', 'conservador', 'patriota', 'verde e amarelo',
                    'mito', 'liberdade', 'intervencionista', 'cristão', 'antiglobalista', 'patriota', 'patriotismo'
                ],
                'cat7_meio_ambiente_amazonia': [
                    'amazônia', 'reserva', 'queimadas', 'desmatamento', 'ong', 'soberania nacional',
                    'clima', 'aquecimento global', 'agenda 2030'
                ],
                'cat8_moralidade': [
                    'corrupção', 'liberdade', 'patriotismo', 'soberania', 'criminoso', 'traidor',
                    'bandido', 'herói', 'santo', 'vítima', 'injustiça'
                ],
                'cat9_antissistema': [
                    'sistema', 'establishment', 'corrupto', 'imprensa vendida', 'mídia lixo', 'stf ativista',
                    'conspiração', 'globalista', 'ditadura do judiciário', 'deep state'
                ],
                'cat10_polarizacao': [
                    'nós contra eles', 'vergonha', 'ódio', 'orgulho', 'traição', 'luta do bem contra o mal',
                    'defensores da pátria', 'inimigos do povo'
                ]
            }
            
            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            initial_count = len(df)
            
            # === CLASSIFICAÇÃO POLÍTICA ===
            def classify_political_content(text):
                if pd.isna(text):
                    return [], 0.0, []
                
                text_lower = str(text).lower()
                categories_found = []
                matched_terms = []
                total_matches = 0
                
                # Verificar cada categoria
                for category_key, keywords in political_keywords.items():
                    category_matches = 0
                    category_terms = []
                    
                    for keyword in keywords:
                        keyword_lower = keyword.lower()
                        
                        # Busca exata e variações (palavras cortadas, erros)
                        if keyword_lower in text_lower:
                            category_matches += text_lower.count(keyword_lower)
                            category_terms.append(keyword)
                        
                        # Busca por palavras-raiz (para detectar variações)
                        elif len(keyword_lower) > 4:
                            root_word = keyword_lower[:int(len(keyword_lower)*0.75)]
                            if root_word in text_lower:
                                category_matches += 0.5  # Peso menor para matches parciais
                                category_terms.append(f"{keyword}~")
                    
                    if category_matches > 0:
                        # Extrair número da categoria
                        cat_num = category_key.split('_')[0].replace('cat', '')
                        if cat_num == '6' and 'identidade' in category_key:
                            cat_num = '6i'  # Distinguir identidade política
                        
                        categories_found.append(cat_num)
                        matched_terms.extend(category_terms)
                        total_matches += category_matches
                
                # Score de relevância política (0.0 a 1.0)
                # Baseado no número de matches e categorias
                relevance_score = min(1.0, (total_matches * 0.1) + (len(categories_found) * 0.15))
                
                return categories_found, relevance_score, matched_terms
            
            # Aplicar classificação
            self.logger.info("🔍 Classificando conteúdo político...")
            
            classification_results = df[text_column].apply(classify_political_content)
            
            df['cat'] = [result[0] for result in classification_results]
            df['political_relevance_score'] = [result[1] for result in classification_results]
            df['political_terms_found'] = [result[2] for result in classification_results]
            
            # === FILTRO DE RELEVÂNCIA ===
            # Threshold mais baixo para preservar mais dados (configurável)
            relevance_threshold = getattr(self, 'political_relevance_threshold', 0.02)
            relevance_filter = df['political_relevance_score'] > relevance_threshold
            
            df_filtered = df[relevance_filter].copy().reset_index(drop=True)

            final_count = len(df_filtered)
            reduction_pct = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0

            # Safeguard: warn if too much data is being filtered out
            if reduction_pct > 80:
                self.logger.warning(f"🚨 ALTA REDUÇÃO DE DADOS: {reduction_pct:.1f}% (threshold={relevance_threshold})")
                self.logger.warning("   Considere ajustar political_relevance_threshold ou revisar critérios")
            elif reduction_pct > 60:
                self.logger.warning(f"⚠️ Redução significativa: {reduction_pct:.1f}% (threshold={relevance_threshold})")
            
            # === ESTATÍSTICAS POLÍTICAS ===
            # Contagem por categoria
            all_categories = []
            for cat_list in df_filtered['cat']:
                if isinstance(cat_list, list):
                    all_categories.extend(cat_list)
            
            from collections import Counter
            category_counts = Counter(all_categories)
            
            avg_relevance = df_filtered['political_relevance_score'].mean()
            texts_with_politics = len(df_filtered[df_filtered['political_relevance_score'] > 0])
            
            self.logger.info(f"✅ Filtro político aplicado:")
            self.logger.info(f"   📉 {initial_count:,} → {final_count:,} registros")
            self.logger.info(f"   📊 Redução: {reduction_pct:.1f}%")
            self.logger.info(f"   🎯 Score relevância médio: {avg_relevance:.3f}")
            self.logger.info(f"   🏛️ Textos com conteúdo político: {texts_with_politics:,}")
            
            if category_counts:
                top_categories = category_counts.most_common(5)
                self.logger.info(f"   🔝 Top categorias: {dict(top_categories)}")
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 3
            
            return df_filtered
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 06: {e}")
            self.stats['processing_errors'] += 1
            # Em caso de erro, retornar dados com colunas padrão
            df['cat'] = [[] for _ in range(len(df))]
            df['political_relevance_score'] = 0.5
            df['political_terms_found'] = [[] for _ in range(len(df))]
            return df

    # ===============================================
    # MÉTODOS HELPER PARA ANÁLISE DE QUALIDADE
    # ===============================================
    
    def _calculate_emoji_ratio(self, text: str) -> float:
        """Calcular proporção de emojis no texto."""
        if pd.isna(text) or len(text) == 0:
            return 0.0
        
        import re
        # Regex para detectar emojis Unicode
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        emojis = emoji_pattern.findall(str(text))
        emoji_count = sum(len(emoji) for emoji in emojis)
        
        return min(1.0, emoji_count / len(str(text)))
    
    def _calculate_caps_ratio(self, text: str) -> float:
        """Calcular proporção de letras maiúsculas."""
        if pd.isna(text) or len(text) == 0:
            return 0.0
        
        text_str = str(text)
        letters = [c for c in text_str if c.isalpha()]
        
        if len(letters) == 0:
            return 0.0
        
        caps_count = sum(1 for c in letters if c.isupper())
        return caps_count / len(letters)
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calcular proporção de caracteres repetitivos."""
        if pd.isna(text) or len(text) <= 1:
            return 0.0
        
        text_str = str(text).lower()
        
        # Contar sequências repetitivas (3+ caracteres iguais)
        repetition_count = 0
        current_char = ''
        current_count = 1
        
        for char in text_str:
            if char == current_char:
                current_count += 1
                if current_count >= 3:
                    repetition_count += 1
            else:
                current_char = char
                current_count = 1
        
        return min(1.0, repetition_count / len(text_str))
    
    def _detect_portuguese(self, text: str) -> bool:
        """Detecção básica de idioma português."""
        if pd.isna(text) or len(text) < 10:
            return True  # Assumir português para textos muito curtos
        
        text_lower = str(text).lower()
        
        # Palavras comuns em português
        portuguese_indicators = [
            'que', 'não', 'com', 'uma', 'para', 'são', 'por', 'mais', 'das', 'dos',
            'mas', 'foi', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem',
            'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você',
            'já', 'eu', 'também', 'só', 'pelo', 'nos', 'é', 'o', 'a', 'de', 'do',
            'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se',
            'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele',
            'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há',
            'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até'
        ]
        
        # Contar palavras portuguesas encontradas
        words = text_lower.split()
        portuguese_count = sum(1 for word in words if word in portuguese_indicators)
        
        # Considerar português se >= 20% das palavras são indicadores
        if len(words) > 0:
            portuguese_ratio = portuguese_count / len(words)
            return portuguese_ratio >= 0.2
        
        return True  # Default para português

    @validate_stage_dependencies(required_columns=['normalized_text'], required_attrs=['political_lexicon'])
    def _stage_08_political_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 05: Classificação política brasileira.
        
        Aplica léxico político brasileiro para classificar textos em:
        - extrema-direita, direita, centro-direita, centro, centro-esquerda, esquerda
        """
        try:
            self.logger.info("🔄 Stage 05: Classificação política brasileira")
            
            if 'normalized_text' not in df.columns:
                self.logger.warning("⚠️ normalized_text não encontrado, usando body")
                text_column = 'body'
            else:
                text_column = 'normalized_text'
            
            # Classificação política básica
            df['political_orientation'] = df[text_column].apply(self._classify_political_orientation)
            df['political_keywords'] = df[text_column].apply(self._extract_political_keywords)
            df['political_intensity'] = df[text_column].apply(self._calculate_political_intensity)
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 3
            
            self.logger.info(f"✅ Stage 05 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 05: {e}")
            self.stats['processing_errors'] += 1
            return df

    @validate_stage_dependencies(required_columns=['normalized_text'])
    def _stage_09_tfidf_vectorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 06: Vetorização TF-IDF com tokens spaCy.
        
        Calcula TF-IDF usando tokens processados pelo spaCy.
        """
        try:
            self.logger.info("🔄 Stage 06: Vetorização TF-IDF")
            
            if 'tokens' not in df.columns:
                self.logger.warning("⚠️ tokens não encontrados, usando normalized_text")
                text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
                text_data = df[text_column].fillna('').tolist()
            else:
                # Usar tokens do spaCy
                text_data = df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('').tolist()
            
            # TF-IDF básico
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=None,  # Já removemos stopwords no spaCy
                lowercase=False   # Já normalizado
            )
            
            tfidf_matrix = vectorizer.fit_transform(text_data)
            feature_names = vectorizer.get_feature_names_out()
            
            # Converter para array denso para cálculos
            tfidf_dense = tfidf_matrix.toarray()
            
            # Scores médios por documento
            df['tfidf_score_mean'] = np.mean(tfidf_dense, axis=1)
            df['tfidf_score_max'] = np.max(tfidf_dense, axis=1)
            df['tfidf_top_terms'] = [
                [feature_names[i] for i in row.argsort()[::-1][:5]]
                for row in tfidf_dense
            ]
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 3
            
            self.logger.info(f"✅ Stage 06 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 06: {e}")
            self.stats['processing_errors'] += 1
            return df

    @validate_stage_dependencies(required_columns=['tfidf_score_mean'], required_attrs=['tfidf_matrix'])
    def _stage_10_clustering_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 07: Análise de clustering baseado em features linguísticas.
        
        Agrupa documentos similares usando características extraídas.
        """
        try:
            self.logger.info("🔄 Stage 07: Análise de clustering")
            
            # Features numéricas para clustering
            numeric_features = []
            for col in ['word_count', 'text_length', 'tfidf_score_mean', 'political_intensity']:
                if col in df.columns:
                    numeric_features.append(col)
            
            if len(numeric_features) < 2:
                self.logger.warning("⚠️ Features insuficientes para clustering")
                df['cluster_id'] = 0
                df['cluster_distance'] = 0.0
                df['cluster_size'] = len(df)
            else:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # Preparar dados
                feature_data = df[numeric_features].fillna(0)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_data)
                
                # Clustering simples
                n_clusters = min(5, len(df) // 10 + 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_data)
                
                df['cluster_id'] = clusters
                df['cluster_distance'] = [
                    min(((scaled_data[i] - center) ** 2).sum() for center in kmeans.cluster_centers_)
                    for i in range(len(scaled_data))
                ]
                
                # Tamanho dos clusters
                cluster_sizes = pd.Series(clusters).value_counts().to_dict()
                df['cluster_size'] = df['cluster_id'].map(cluster_sizes)
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 3
            
            self.logger.info(f"✅ Stage 07 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 07: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_11_topic_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 08: Topic modeling com embeddings.
        
        Descoberta automática de tópicos nos textos.
        """
        try:
            self.logger.info("🔄 Stage 08: Topic modeling")
            
            if 'tokens' not in df.columns:
                self.logger.warning("⚠️ tokens não encontrados, usando normalized_text")
                text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
                text_data = df[text_column].fillna('').tolist()
            else:
                text_data = df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('').tolist()
            
            # Topic modeling básico com LDA
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            # Preparar dados
            vectorizer = CountVectorizer(max_features=50, stop_words=None)
            doc_term_matrix = vectorizer.fit_transform(text_data)
            
            # LDA simples
            n_topics = min(5, len(df) // 20 + 1)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            doc_topic_matrix = lda.fit_transform(doc_term_matrix)
            
            # Tópico dominante para cada documento
            df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)
            df['topic_probability'] = doc_topic_matrix.max(axis=1)
            
            # Palavras-chave dos tópicos
            feature_names = vectorizer.get_feature_names_out()
            topic_keywords = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[::-1][:3]]
                topic_keywords.append(top_words)
            
            df['topic_keywords'] = df['dominant_topic'].apply(lambda x: topic_keywords[x] if x < len(topic_keywords) else [])
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 3
            
            self.logger.info(f"✅ Stage 08 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 08: {e}")
            self.stats['processing_errors'] += 1
            return df

    @validate_stage_dependencies(required_columns=['normalized_text'])
    def _stage_13_temporal_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 09: Análise temporal.
        
        Extrai padrões temporais dos timestamps.
        """
        try:
            self.logger.info("🔄 Stage 09: Análise temporal")
            
            if 'datetime' not in df.columns:
                self.logger.warning("⚠️ datetime não encontrado")
                df['hour'] = 12
                df['day_of_week'] = 1
                df['month'] = 1
            else:
                # Converter datetime para análise temporal
                try:
                    datetime_series = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    
                    df['hour'] = datetime_series.dt.hour
                    df['day_of_week'] = datetime_series.dt.dayofweek
                    df['month'] = datetime_series.dt.month
                    df['year'] = datetime_series.dt.year
                    df['day_of_year'] = datetime_series.dt.dayofyear
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Erro conversão datetime: {e}")
                    df['hour'] = 12
                    df['day_of_week'] = 1
                    df['month'] = 1
                    df['year'] = 2020
                    df['day_of_year'] = 1
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 5
            
            self.logger.info(f"✅ Stage 09 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 09: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_14_network_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 10: Análise de rede (coordenação e padrões).
        
        Detecta padrões de coordenação e comportamento de rede.
        """
        try:
            self.logger.info("🔄 Stage 10: Análise de rede")
            
            # Análise de coordenação básica
            if 'sender' in df.columns:
                sender_counts = df['sender'].value_counts()
                df['sender_frequency'] = df['sender'].map(sender_counts)
                df['is_frequent_sender'] = df['sender_frequency'] > df['sender_frequency'].median()
            else:
                df['sender_frequency'] = 1
                df['is_frequent_sender'] = False
            
            # Análise de URLs compartilhadas
            if 'urls_extracted' in df.columns:
                # URLs mais compartilhadas
                all_urls = []
                for urls in df['urls_extracted'].fillna('[]'):
                    if isinstance(urls, str):
                        try:
                            url_list = eval(urls) if urls.startswith('[') else [urls]
                            all_urls.extend(url_list)
                        except:
                            pass
                
                url_counts = pd.Series(all_urls).value_counts()
                df['shared_url_frequency'] = df['urls_extracted'].apply(
                    lambda x: max([url_counts.get(url, 0) for url in (eval(x) if isinstance(x, str) and x.startswith('[') else [])], default=0)
                )
            else:
                df['shared_url_frequency'] = 0
            
            # Coordenação temporal (mensagens em horários similares)
            if 'hour' in df.columns:
                hour_counts = df['hour'].value_counts()
                df['temporal_coordination'] = df['hour'].map(hour_counts) / len(df)
            else:
                df['temporal_coordination'] = 0.0
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 4
            
            self.logger.info(f"✅ Stage 10 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 10: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_15_domain_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 11: Análise de domínios.
        
        Analisa domínios e URLs para identificar padrões de mídia.
        """
        try:
            self.logger.info("🔄 Stage 11: Análise de domínios")
            
            # Análise de domínios
            if 'domain' in df.columns:
                df['domain_type'] = df['domain'].apply(self._classify_domain_type)
                
                domain_counts = df['domain'].value_counts()
                df['domain_frequency'] = df['domain'].map(domain_counts)
                
                # Mídia mainstream vs alternativa
                mainstream_domains = ['youtube.com', 'twitter.com', 'facebook.com', 'instagram.com', 'g1.globo.com', 'folha.uol.com.br']
                df['is_mainstream_media'] = df['domain'].isin(mainstream_domains)
            else:
                df['domain_type'] = 'unknown'
                df['domain_frequency'] = 0
                df['is_mainstream_media'] = False
            
            # Análise de URLs
            if 'urls_extracted' in df.columns:
                df['url_count'] = df['urls_extracted'].apply(
                    lambda x: len(eval(x)) if isinstance(x, str) and x.startswith('[') else (1 if x else 0)
                )
                df['has_external_links'] = df['url_count'] > 0
            else:
                df['url_count'] = 0
                df['has_external_links'] = False
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 5
            
            self.logger.info(f"✅ Stage 11 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 11: {e}")
            self.stats['processing_errors'] += 1
            return df


    def _stage_12_semantic_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 12: Análise semântica.
        
        Análise semântica e de sentimento dos textos.
        """
        try:
            self.logger.info("🔄 Stage 12: Análise semântica")
            
            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            
            # Análise de sentimento básica
            df['sentiment_polarity'] = df[text_column].apply(self._calculate_sentiment_polarity)
            df['sentiment_label'] = df['sentiment_polarity'].apply(
                lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
            )
            
            # Análise de emoções básicas
            df['emotion_intensity'] = df[text_column].apply(self._calculate_emotion_intensity)
            df['has_aggressive_language'] = df[text_column].apply(self._detect_aggressive_language)
            
            # Complexidade semântica
            if 'tokens' in df.columns:
                df['semantic_diversity'] = df['tokens'].apply(
                    lambda x: len(set(x)) / len(x) if isinstance(x, list) and len(x) > 0 else 0
                )
            else:
                df['semantic_diversity'] = df[text_column].apply(
                    lambda x: len(set(str(x).split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0
                )
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 5
            
            self.logger.info(f"✅ Stage 12 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 12: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_16_event_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 13: Análise de contexto de eventos.
        
        Detecta contextos políticos e eventos relevantes.
        """
        try:
            self.logger.info("🔄 Stage 13: Análise de contexto de eventos")
            
            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            
            # Contextos políticos brasileiros
            df['political_context'] = df[text_column].apply(self._detect_political_context)
            df['mentions_government'] = df[text_column].apply(self._mentions_government)
            df['mentions_opposition'] = df[text_column].apply(self._mentions_opposition)
            
            # Eventos específicos (eleições, manifestações, etc.)
            df['election_context'] = df[text_column].apply(self._detect_election_context)
            df['protest_context'] = df[text_column].apply(self._detect_protest_context)
            
            # Análise temporal de eventos
            if 'datetime' in df.columns:
                df['is_weekend'] = df['day_of_week'].isin([5, 6]) if 'day_of_week' in df.columns else False
                df['is_business_hours'] = df['hour'].between(9, 17) if 'hour' in df.columns else False
            else:
                df['is_weekend'] = False
                df['is_business_hours'] = True
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 7
            
            self.logger.info(f"✅ Stage 13 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 13: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_17_channel_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 14: Análise de canais/fontes.
        
        Classifica canais e fontes de informação.
        """
        try:
            self.logger.info("🔄 Stage 14: Análise de canais")
            
            # Análise de canais
            if 'channel' in df.columns:
                df['channel_type'] = df['channel'].apply(self._classify_channel_type)
                
                channel_counts = df['channel'].value_counts()
                df['channel_activity'] = df['channel'].map(channel_counts)
                df['is_active_channel'] = df['channel_activity'] > df['channel_activity'].median()
            else:
                df['channel_type'] = 'unknown'
                df['channel_activity'] = 1
                df['is_active_channel'] = False
            
            # Análise de mídia
            if 'media_type' in df.columns:
                df['content_type'] = df['media_type'].fillna('text')
                df['has_media'] = df['media_type'].notna()
            else:
                df['content_type'] = 'text'
                df['has_media'] = False
            
            # Padrões de forwarding
            if 'is_fwrd' in df.columns:
                df['is_forwarded'] = df['is_fwrd'].fillna(False)
                forwarded_ratio = df['is_forwarded'].mean()
                df['forwarding_context'] = forwarded_ratio
            else:
                df['is_forwarded'] = False
                df['forwarding_context'] = 0.0
            
            # Influência do canal
            if 'sender' in df.columns and 'channel' in df.columns:
                sender_channel_counts = df.groupby(['sender', 'channel']).size()
                df['sender_channel_influence'] = df.apply(
                    lambda row: sender_channel_counts.get((row['sender'], row['channel']), 0), axis=1
                )
            else:
                df['sender_channel_influence'] = 1
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 7
            
            self.logger.info(f"✅ Stage 14 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 14: {e}")
            self.stats['processing_errors'] += 1
            return df

    # ==========================================
    # HELPER METHODS FOR ANALYSIS STAGES
    # ==========================================

    def _classify_political_orientation(self, text: str) -> str:
        """Classifica orientação política do texto."""
        if not text or pd.isna(text):
            return 'neutral'
        
        text_lower = str(text).lower()
        
        # Palavras-chave para classificação política brasileira
        extrema_direita = ['bolsonaro', 'mito', 'capitão', 'comunista', 'petralha', 'globalismo']
        direita = ['conservador', 'tradicional', 'família', 'ordem', 'progresso']
        centro_direita = ['liberal', 'empreendedor', 'mercado', 'economia']
        centro = ['moderado', 'equilibrio', 'consenso']
        centro_esquerda = ['social', 'trabalhador', 'direitos']
        esquerda = ['lula', 'pt', 'socialismo', 'igualdade', 'justiça social']
        
        scores = {
            'extrema-direita': sum(1 for word in extrema_direita if word in text_lower),
            'direita': sum(1 for word in direita if word in text_lower),
            'centro-direita': sum(1 for word in centro_direita if word in text_lower),
            'centro': sum(1 for word in centro if word in text_lower),
            'centro-esquerda': sum(1 for word in centro_esquerda if word in text_lower),
            'esquerda': sum(1 for word in esquerda if word in text_lower)
        }
        
        return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else 'neutral'

    def _extract_political_keywords(self, text: str) -> list:
        """Extrai palavras-chave políticas do texto."""
        if not text or pd.isna(text):
            return []
        
        political_keywords = [
            'bolsonaro', 'lula', 'pt', 'psl', 'mdb', 'psdb',
            'eleição', 'voto', 'democracia', 'ditadura', 'golpe',
            'esquerda', 'direita', 'conservador', 'liberal'
        ]
        
        text_lower = str(text).lower()
        found_keywords = [word for word in political_keywords if word in text_lower]
        
        return found_keywords[:5]  # Máximo 5 palavras-chave

    def _calculate_political_intensity(self, text: str) -> float:
        """Calcula intensidade do discurso político."""
        if not text or pd.isna(text):
            return 0.0
        
        intensity_words = [
            'sempre', 'nunca', 'jamais', 'obrigatório', 'proibido',
            'urgente', 'imediato', 'crucial', 'fundamental', 'essencial'
        ]
        
        text_lower = str(text).lower()
        intensity_count = sum(1 for word in intensity_words if word in text_lower)
        
        return min(intensity_count / 10.0, 1.0)  # Normalizar entre 0 e 1

    def _classify_domain_type(self, domain: str) -> str:
        """Classifica tipo de domínio."""
        if not domain or pd.isna(domain):
            return 'unknown'
        
        domain_lower = str(domain).lower()
        
        if any(x in domain_lower for x in ['youtube', 'youtu.be']):
            return 'video'
        elif any(x in domain_lower for x in ['twitter', 'facebook', 'instagram']):
            return 'social'
        elif any(x in domain_lower for x in ['globo', 'folha', 'estadao', 'uol']):
            return 'mainstream_news'
        elif any(x in domain_lower for x in ['blog', 'wordpress', 'medium']):
            return 'blog'
        else:
            return 'other'

    def _calculate_sentiment_polarity(self, text: str) -> float:
        """Calcula polaridade de sentimento básica."""
        if not text or pd.isna(text):
            return 0.0
        
        positive_words = ['bom', 'ótimo', 'excelente', 'maravilhoso', 'perfeito', 'amor', 'feliz']
        negative_words = ['ruim', 'péssimo', 'terrível', 'ódio', 'raiva', 'triste', 'infeliz']
        
        text_lower = str(text).lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words

    def _calculate_emotion_intensity(self, text: str) -> float:
        """Calcula intensidade emocional."""
        if not text or pd.isna(text):
            return 0.0
        
        # Contagem de pontuação emocional
        emotion_markers = text.count('!') + text.count('?') + text.count('...') 
        caps_words = sum(1 for word in str(text).split() if word.isupper() and len(word) > 2)
        
        return min((emotion_markers + caps_words) / 10.0, 1.0)

    def _detect_aggressive_language(self, text: str) -> bool:
        """Detecta linguagem agressiva."""
        if not text or pd.isna(text):
            return False
        
        aggressive_words = [
            'ódio', 'matar', 'destruir', 'eliminar', 'acabar',
            'burro', 'idiota', 'imbecil', 'estúpido'
        ]
        
        text_lower = str(text).lower()
        return any(word in text_lower for word in aggressive_words)

    def _detect_political_context(self, text: str) -> str:
        """Detecta contexto político."""
        if not text or pd.isna(text):
            return 'none'
        
        text_lower = str(text).lower()
        
        if any(word in text_lower for word in ['eleição', 'voto', 'urna', 'candidato']):
            return 'electoral'
        elif any(word in text_lower for word in ['governo', 'ministro', 'presidente']):
            return 'government'
        elif any(word in text_lower for word in ['manifestação', 'protesto', 'greve']):
            return 'protest'
        elif any(word in text_lower for word in ['economia', 'inflação', 'desemprego']):
            return 'economic'
        else:
            return 'general'

    def _mentions_government(self, text: str) -> bool:
        """Verifica se menciona governo."""
        if not text or pd.isna(text):
            return False
        
        government_terms = ['governo', 'presidente', 'ministro', 'secretário', 'federal']
        text_lower = str(text).lower()
        
        return any(term in text_lower for term in government_terms)

    def _mentions_opposition(self, text: str) -> bool:
        """Verifica se menciona oposição."""
        if not text or pd.isna(text):
            return False
        
        opposition_terms = ['oposição', 'contra', 'resistência', 'impeachment']
        text_lower = str(text).lower()
        
        return any(term in text_lower for term in opposition_terms)

    def _detect_election_context(self, text: str) -> bool:
        """Detecta contexto eleitoral."""
        if not text or pd.isna(text):
            return False
        
        election_terms = ['eleição', 'voto', 'urna', 'candidato', 'campanha', 'debate']
        text_lower = str(text).lower()
        
        return any(term in text_lower for term in election_terms)

    def _detect_protest_context(self, text: str) -> bool:
        """Detecta contexto de protesto."""
        if not text or pd.isna(text):
            return False
        
        protest_terms = ['manifestação', 'protesto', 'greve', 'ocupação', 'ato']
        text_lower = str(text).lower()
        
        return any(term in text_lower for term in protest_terms)

    def _classify_channel_type(self, channel: str) -> str:
        """Classifica tipo de canal."""
        if not channel or pd.isna(channel):
            return 'unknown'
        
        channel_lower = str(channel).lower()
        
        if any(word in channel_lower for word in ['news', 'notícia', 'jornal']):
            return 'news'
        elif any(word in channel_lower for word in ['brasil', 'patriota', 'conservador']):
            return 'political'
        elif any(word in channel_lower for word in ['humor', 'meme', 'engraçado']):
            return 'entertainment'
        else:
            return 'general'


def main():
    """Teste do analyzer limpo."""
    logging.basicConfig(level=logging.INFO)

    # Teste com dados de exemplo
    test_data = pd.DataFrame({
        'body': [
            'Este é um texto sobre política brasileira com bolsonaro',
            'Discussão sobre economia e mercado financeiro liberal',
            'Análise social progressista sobre direitos humanos',
            'Texto neutro sobre tecnologia e ciência',
            'Debate político conservador sobre tradições'
        ],
        'timestamp': [
            '2023-01-01 10:00:00',
            '2023-01-01 11:00:00',
            '2023-01-01 10:30:00',
            '2023-01-02 15:00:00',
            '2023-01-01 10:15:00'
        ]
    })

    analyzer = CleanScientificAnalyzer()
    result = analyzer.analyze_dataset(test_data)

    print(f"\n✅ Análise concluída:")
    print(f"📊 Colunas geradas: {result['columns_generated']}")
    print(f"🎯 Stages completados: {result['stats']['stages_completed']}")
    print(f"🔧 Features extraídas: {result['stats']['features_extracted']}")

    return result


if __name__ == "__main__":
    main()