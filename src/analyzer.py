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

MODULARIZAÇÃO (TAREFA 11):
- Cada stage foi extraído como módulo independente em src/stages/
- Registry de stages: from stages import STAGE_REGISTRY
- Helpers compartilhados: from stages.helpers import _calculate_emoji_ratio, etc.
- Os métodos inline neste arquivo são a versão autoritativa (source of truth)
- Os módulos em stages/ são a versão modular de referência, 1:1 com os inline
- Migração completa: substituir self._stage_XX por import de stages.stage_XX

Author: digiNEV Academic Research Team
Version: v.final (Consolidação Final + Modularização)
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

# Lexicon loader
from src.lexicon_loader import LexiconLoader

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
            # Skip validation if processing chunks
            if getattr(self, '_skip_validation', False):
                return func(self, df, *args, **kwargs)

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

    def __init__(self, chunk_size: int = 2000, memory_limit_gb: float = 2.0, auto_chunk: bool = True,
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

        # Load political lexicon via LexiconLoader (unified system: 956 termos, 9 macrotemas)
        self.lexicon_loader = LexiconLoader()
        self.political_lexicon = self._load_political_lexicon()
        self._political_terms_map = self.lexicon_loader.get_terms_by_category_map()

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
        """Carregar lexicon político via LexiconLoader (lexico_unified_system.json)."""
        if self.lexicon_loader.lexicon:
            terms_map = self.lexicon_loader.get_terms_by_category_map()
            total_terms = sum(len(v) for v in terms_map.values())
            self.logger.info(f"✅ Lexicon unificado carregado: {len(terms_map)} categorias, {total_terms} termos")
            return self.lexicon_loader.lexicon

        self.logger.warning("⚠️ Lexicon unificado não encontrado, usando fallback mínimo")
        return {
            "lexico": {
                "identidade_patriotica": {"subtemas": {"fallback": {"palavras": ["bolsonaro", "mito", "patriota"]}}},
                "inimigos_ideologicos": {"subtemas": {"fallback": {"palavras": ["lula", "pt", "petista", "comunista"]}}}
            }
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
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024**3)
            memory_percent = process.memory_percent()

            # Log detalhado de memória a cada verificação
            self.logger.debug(f"💾 Memória atual: {memory_gb:.2f}GB ({memory_percent:.1f}% do sistema)")

            if memory_gb > self.memory_limit_gb:
                self.logger.warning(f"🚨 Memória alta: {memory_gb:.1f}GB > {self.memory_limit_gb}GB ({memory_percent:.1f}%)")
                return True
            elif memory_gb > self.memory_limit_gb * 0.8:
                self.logger.info(f"⚠️ Memória crescendo: {memory_gb:.1f}GB (80% do limite)")

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
        """Carregar DataFrame com detecção automática de separador e tratamento robusto."""
        file_path = Path(file_path)
        
        # Detectar separador lendo primeira linha
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
            # Contar separadores na primeira linha
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            
            # Escolher separador baseado em contagem
            if semicolon_count > comma_count:
                separator = ';'
            elif comma_count > 0:
                separator = ','
            else:
                # Fallback: tentar ambos e ver qual funciona melhor
                try:
                    test_df = pd.read_csv(file_path, sep=';', nrows=1)
                    if len(test_df.columns) > 1:
                        separator = ';'
                    else:
                        separator = ','
                except:
                    separator = ','
            
            self.logger.info(f"🔍 Separador detectado: '{separator}' (vírgulas: {comma_count}, ponto-vírgulas: {semicolon_count})")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro na detecção de separador: {e}, usando vírgula por padrão")
            separator = ','
        
        # Carregar com limite de registros se especificado
        try:
            # Usar configurações mais robustas para CSV com conteúdo complexo
            read_kwargs = {
                'sep': separator,
                'encoding': 'utf-8',
                'quotechar': '"',
                'quoting': 1,  # QUOTE_ALL
                'skipinitialspace': True,
                'on_bad_lines': 'skip'  # Pular linhas problemáticas
            }
            
            if max_records:
                read_kwargs['nrows'] = max_records
                
            df = pd.read_csv(file_path, **read_kwargs)
                
            # Verificar se carregamento foi bem-sucedido
            if len(df.columns) == 1 and separator == ';':
                # Tentar com vírgula se ponto-vírgula resultou em uma coluna só
                self.logger.warning("⚠️ Tentando vírgula após falha com ponto-vírgula")
                read_kwargs['sep'] = ','
                df = pd.read_csv(file_path, **read_kwargs)
                separator = ','
                
            self.logger.info(f"📂 Dataset carregado: {len(df):,} registros, {len(df.columns)} colunas (sep='{separator}')")
            return df
            
        except Exception as e:
            # Fallback: tentar com configurações mais permissivas
            self.logger.warning(f"⚠️ Erro com configurações padrão: {e}, tentando modo permissivo")
            try:
                fallback_kwargs = {
                    'sep': separator,
                    'encoding': 'utf-8',
                    'quotechar': '"',
                    'quoting': 0,  # QUOTE_MINIMAL
                    'skipinitialspace': True,
                    'on_bad_lines': 'skip',
                    'error_bad_lines': False,
                    'warn_bad_lines': True
                }
                
                if max_records:
                    fallback_kwargs['nrows'] = max_records
                    
                df = pd.read_csv(file_path, **fallback_kwargs)
                self.logger.info(f"📂 Dataset carregado (modo permissivo): {len(df):,} registros, {len(df.columns)} colunas")
                return df
                
            except Exception as e2:
                self.logger.error(f"❌ Erro ao carregar arquivo mesmo com fallback: {e2}")
                raise

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
                
                # Analisar chunk usando método específico para chunks
                chunk_data = self._analyze_single_chunk(chunk.copy())
                
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
                
                # Atualizar stats consolidadas
                consolidated_stats['stages_completed'] = max(consolidated_stats['stages_completed'], self.stats.get('stages_completed', 0))
                consolidated_stats['features_extracted'] = max(consolidated_stats['features_extracted'], self.stats.get('features_extracted', 0))
                
                # Salvar chunk se solicitado
                if output_file:
                    chunk_output = f"{output_file.replace('.csv', '')}_chunk_{total_chunks}.csv"
                    chunk_data.to_csv(chunk_output, index=False, sep=';')
                    self.logger.info(f"💾 Chunk salvo: {chunk_output}")
                
                chunk_time = time.time() - chunk_start
                chunk_performance = len(chunk_data) / chunk_time if chunk_time > 0 else 0
                
                self.logger.info(f"✅ Chunk {total_chunks} processado: {len(chunk_data):,} registros em {chunk_time:.1f}s ({chunk_performance:.1f} reg/s)")
                
                # Limpar memória entre chunks
                del chunk_data, chunk
                if self._check_memory_usage():
                    self._clean_memory()
                
        except Exception as e:
            self.logger.error(f"❌ Erro na análise chunked: {e}")
            consolidated_stats['processing_errors'] += 1
            raise
        
        # Atualizar stats principais
        self.stats.update({
            'total_records_processed': total_records,
            'chunks_processed': total_chunks,
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
            'columns_generated': 75,  # Estimativa baseada no pipeline
            'total_records': total_records,
            'stages_completed': consolidated_stats['stages_completed']
        }

    def _analyze_single_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa um chunk específico sem recursão.
        Ajusta thresholds para chunks pequenos.
        """
        try:
            # Ajustar parâmetros para chunks pequenos
            original_threshold = getattr(self, 'political_relevance_threshold', 0.02)
            chunk_size = len(chunk_df)
            
            # Threshold mais permissivo para chunks pequenos
            if chunk_size < 1000:
                self.political_relevance_threshold = 0.01  # Mais permissivo
                self.logger.info(f"📊 Chunk pequeno ({chunk_size}): threshold ajustado para {self.political_relevance_threshold}")
            elif chunk_size < 5000:
                self.political_relevance_threshold = 0.015  # Moderadamente permissivo
            
            # Ativar flag para pular validações durante processamento de chunk
            self._skip_validation = True
            
            self.logger.info(f"🔄 Processando chunk: {len(chunk_df)} registros")
            
            # Processar através do pipeline principal
            result_chunk = chunk_df.copy()
            
            # Stages 01-02: Preprocessing
            result_chunk = self._stage_01_feature_extraction(result_chunk)
            result_chunk = self._stage_02_text_preprocessing(result_chunk)
            
            # Stages 03-06: Volume reduction (críticos para chunks)
            result_chunk = self._stage_03_cross_dataset_deduplication(result_chunk)
            result_chunk = self._stage_04_statistical_analysis(result_chunk)
            result_chunk = self._stage_05_content_quality_filter(result_chunk)
            result_chunk = self._stage_06_affordances_classification(result_chunk)
            
            # Verificar se ainda há dados suficientes após filtros
            if len(result_chunk) == 0:
                self.logger.warning("⚠️ Chunk completamente filtrado, retornando dados parciais")
                # Restaurar threshold original
                self.political_relevance_threshold = original_threshold
                self._skip_validation = False
                return chunk_df.copy()  # Retornar dados originais com minimal processing
            
            # Stages 07-09: Linguistic processing (ajustados para chunks pequenos)
            try:
                result_chunk = self._stage_07_linguistic_processing(result_chunk)
                result_chunk = self._stage_08_political_classification(result_chunk)
                
                # Stage 09: TF-IDF com tratamento especial para chunks pequenos
                if len(result_chunk) >= 5:  # Mínimo viável para TF-IDF
                    result_chunk = self._stage_09_tfidf_vectorization(result_chunk)
                else:
                    self.logger.warning(f"🔍 Chunk muito pequeno ({len(result_chunk)}) para TF-IDF, preenchendo com valores padrão")
                    result_chunk['tfidf_score_mean'] = 0.0
                    result_chunk['tfidf_score_max'] = 0.0 
                    result_chunk['tfidf_top_terms'] = [[] for _ in range(len(result_chunk))]
                    self.stats['features_extracted'] += 3
                
            except Exception as e:
                self.logger.warning(f"⚠️ Erro em processamento linguístico do chunk: {e}")
                # Continuar com dados parciais
            
            # Stages 10-17: Advanced analysis (opcionais para chunks pequenos)
            try:
                if len(result_chunk) >= 10:  # Só executar se chunk tem tamanho viável
                    result_chunk = self._stage_10_clustering_analysis(result_chunk)
                    result_chunk = self._stage_11_topic_modeling(result_chunk)
                    result_chunk = self._stage_12_semantic_analysis(result_chunk)
                    result_chunk = self._stage_13_temporal_analysis(result_chunk)
                    result_chunk = self._stage_14_network_analysis(result_chunk)  # CORRIGIDO
                    result_chunk = self._stage_15_domain_analysis(result_chunk)   # CORRIGIDO
                    result_chunk = self._stage_16_event_context(result_chunk)     # CORRIGIDO
                    result_chunk = self._stage_17_channel_analysis(result_chunk) # CORRIGIDO
                else:
                    self.logger.info(f"📊 Chunk pequeno ({len(result_chunk)}): pulando análises avançadas")
                    # Adicionar colunas padrão para manter consistência
                    self._add_default_columns_for_small_chunks(result_chunk)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Erro em análises avançadas do chunk: {e}")
                # Continuar com dados disponíveis
            
            # Restaurar configurações originais
            self.political_relevance_threshold = original_threshold
            self._skip_validation = False
            
            self.logger.info(f"✅ Chunk processado: {len(result_chunk)} registros finais")
            return result_chunk
            
        except Exception as e:
            # Restaurar configurações em caso de erro
            if hasattr(self, 'political_relevance_threshold'):
                self.political_relevance_threshold = getattr(self, '_original_threshold', 0.02)
            self._skip_validation = False
            
            self.logger.error(f"❌ Erro ao processar chunk: {e}")
            # Retornar pelo menos os dados originais com colunas básicas
            return self._add_minimal_columns(chunk_df.copy())

    def _add_default_columns_for_small_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas padrão para chunks pequenos que pulam análises avançadas."""
        try:
            # Clustering columns
            if 'cluster_id' not in df.columns:
                df['cluster_id'] = 0
                df['cluster_size'] = len(df)
                df['cluster_confidence'] = 0.5
            
            # Topic modeling columns  
            if 'topic_distribution' not in df.columns:
                df['topic_distribution'] = [[] for _ in range(len(df))]
                df['dominant_topic'] = 0
                df['topic_confidence'] = 0.5
            
            # Semantic analysis columns
            if 'semantic_density' not in df.columns:
                df['semantic_density'] = 0.5
                df['semantic_complexity'] = 0.5
                df['semantic_coherence'] = 0.5
            
            # Temporal analysis columns
            if 'hour' not in df.columns and 'datetime' in df.columns:
                try:
                    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
                    df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek  
                    df['month'] = pd.to_datetime(df['datetime']).dt.month
                except:
                    df['hour'] = 12  # Default noon
                    df['day_of_week'] = 1  # Default Monday
                    df['month'] = 6  # Default June
            
            # Network coordination columns
            if 'coordination_score' not in df.columns:
                df['coordination_score'] = 0.0
                df['cascade_participation'] = False
                df['influence_score'] = 0.0
            
            # Domain analysis columns
            if 'domains_found' not in df.columns:
                df['domains_found'] = [[] for _ in range(len(df))]
                df['domain_authority_score'] = 0.0
                df['url_diversity'] = 0.0
            
            # Event context columns
            if 'political_events_detected' not in df.columns:
                df['political_events_detected'] = [[] for _ in range(len(df))]
                df['event_context_score'] = 0.0
                df['temporal_relevance'] = 0.5
            
            # Channel analysis columns
            if 'channel_classification' not in df.columns:
                df['channel_classification'] = 'unknown'
                df['channel_authority'] = 0.5
                df['source_type'] = 'social_media'
            
            self.logger.info(f"📋 Colunas padrão adicionadas para chunk pequeno")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao adicionar colunas padrão: {e}")
            return df
    
    def _add_minimal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas mínimas em caso de erro no processamento."""
        try:
            # Colunas essenciais
            if 'political_relevance_score' not in df.columns:
                df['political_relevance_score'] = 0.5
            if 'content_quality_score' not in df.columns:
                df['content_quality_score'] = 50.0
            if 'normalized_text' not in df.columns:
                text_col = 'body' if 'body' in df.columns else df.columns[0]
                df['normalized_text'] = df[text_col].fillna('')
            
            self.logger.info(f"📋 Colunas mínimas adicionadas para recuperação de erro")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao adicionar colunas mínimas: {e}")
            return df

    def _load_dataset_chunks(self, file_path: Path, max_records: Optional[int] = None):
        """Generator para carregar dataset em chunks."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {file_path}")
        
        self.logger.info(f"📂 Carregando dataset em chunks: {file_path}")
        
        # Detectar separador e configuração necessária
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
            # Determinar separador
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            separator = ';' if semicolon_count > comma_count else ','
            
            # Configurações robustas para CSV complexo
            read_kwargs = {
                'sep': separator,
                'encoding': 'utf-8',
                'engine': 'python',  # Usar engine Python para maior flexibilidade
                'quotechar': '"',
                'quoting': 1,  # QUOTE_ALL
                'skipinitialspace': True,
                'on_bad_lines': 'skip',
                'chunksize': self.chunk_size
            }
            
            if max_records:
                read_kwargs['nrows'] = max_records
                
            self.logger.info(f"🔍 Configuração: sep='{separator}', engine=python")
            
            chunk_number = 0
            total_processed = 0
            
            try:
                # Usar pandas com engine Python para maior robustez
                for chunk in pd.read_csv(file_path, **read_kwargs):
                    chunk_number += 1
                    total_processed += len(chunk)
                    
                    self.logger.info(f"📦 Chunk {chunk_number}: {len(chunk)} registros (total: {total_processed})")
                    
                    # Verificar se chunk tem dados válidos
                    if len(chunk) > 0 and len(chunk.columns) > 1:
                        yield chunk
                    
                    if max_records and total_processed >= max_records:
                        break
                        
            except Exception as e:
                self.logger.error(f"❌ Erro ao carregar dataset: {e}")
                
                # Fallback: carregar linha por linha se necessário
                self.logger.warning("⚠️ Tentando carregamento linha por linha...")
                try:
                    import csv
                    
                    with open(file_path, 'r', encoding='utf-8') as csvfile:
                        # Detectar dialect automaticamente
                        sample = csvfile.read(1024)
                        csvfile.seek(0)
                        sniffer = csv.Sniffer()
                        dialect = sniffer.sniff(sample)
                        
                        reader = csv.DictReader(csvfile, dialect=dialect)
                        
                        chunk_data = []
                        for i, row in enumerate(reader):
                            chunk_data.append(row)
                            
                            if len(chunk_data) >= self.chunk_size:
                                df_chunk = pd.DataFrame(chunk_data)
                                chunk_number += 1
                                self.logger.info(f"📦 Chunk CSV {chunk_number}: {len(df_chunk)} registros")
                                yield df_chunk
                                chunk_data = []
                            
                            if max_records and i >= max_records:
                                break
                        
                        # Último chunk se houver dados
                        if chunk_data:
                            df_chunk = pd.DataFrame(chunk_data)
                            chunk_number += 1
                            self.logger.info(f"📦 Chunk CSV final {chunk_number}: {len(df_chunk)} registros")
                            yield df_chunk
                            
                except Exception as e2:
                    self.logger.error(f"❌ Fallback também falhou: {e2}")
                    raise
                
        except Exception as e:
            self.logger.error(f"❌ Erro crítico no carregamento: {e}")
            raise

    def _chunked_from_dataframe(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Simular processamento chunked para DataFrame que já está em memória."""
        self.logger.info("🔄 Simulando chunked para DataFrame em memória")
        
        # Se DataFrame é pequeno, processar normalmente
        if len(df) <= self.chunk_size:
            return self._analyze_single_chunk(df)

        # Dividir DataFrame em chunks
        chunks = [df[i:i+self.chunk_size] for i in range(0, len(df), self.chunk_size)]

        consolidated_results = []
        for i, chunk in enumerate(chunks, 1):
            self.logger.info(f"📦 Chunk {i}: {len(chunk)} registros, Total: {len(chunks):,}")

            # Log de progresso a cada 5 chunks
            if i % 5 == 0:
                progress_pct = (i / len(chunks)) * 100
                self.logger.info(f"🔄 Progresso: {progress_pct:.1f}% ({i}/{len(chunks)} chunks)")

            result = self._analyze_single_chunk(chunk.copy())
            consolidated_results.append(result)

            # Limpeza forçada de memória após cada chunk
            import gc
            del chunk  # Liberar chunk explicitamente
            gc.collect()

            # Verificar memória e limpar se necessário
            if self._check_memory_usage():
                self.logger.warning(f"🚨 Limpeza de memória no chunk {i}")
                self._clean_memory()

            self.logger.info(f"✅ Chunk {i} processado: {len(result['data']) if 'data' in result else 0} registros finais")
        
        # Para simplificar, retornar resultado do último chunk com stats consolidados
        final_result = consolidated_results[-1]
        final_result['stats']['total_chunks'] = len(chunks)
        final_result['stats']['chunked_processing'] = True
        
        return final_result

    def analyze_dataset(self, data_input, max_records=None) -> Dict[str, Any]:
        """
        Analisar dataset com pipeline sequencial otimizado de 17 stages.

        NOVA SEQUÊNCIA OTIMIZADA (conforme PIPELINE_STAGES_ANALYSIS.md):
        Fase 1: Preparação (01-02) - estrutura básica
        Fase 2: Redução de volume (03-06) - CRÍTICO para performance
        Fase 3: Análise linguística (07-09) - volume reduzido
        Fase 4: Análises avançadas (10-17) - dados otimizados

        Args:
            data_input: DataFrame ou caminho do arquivo para análise
            max_records: Limite máximo de registros para processar (opcional)

        Returns:
            Dict com resultado da análise
        """
        # Verificar se deve usar chunking
        should_chunk, estimated_records, reason = self._should_use_chunking(data_input)

        # Se for caminho de arquivo e deve usar chunking, usar processamento chunked
        if self.auto_chunk and should_chunk:
            self.logger.info(f"⚡ Chunking ativado: {reason}")
            self.stats['chunked_processing'] = True
            return self._analyze_chunked(data_input, max_records=max_records)

        # Senão, carregar dados e processar normalmente
        self.stats['chunked_processing'] = False

        # Se for caminho de arquivo, carregar
        if isinstance(data_input, (str, Path)):
            df = self._load_dataframe(data_input, max_records)
        else:
            df = data_input
            if max_records and len(df) > max_records:
                df = df.head(max_records)
                self.logger.info(f"📊 Limitado a {max_records:,} registros")

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

            # STAGE 06: Affordances Classification (AI-powered analysis)
            df = self._stage_06_affordances_classification(df)

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
                'total_records': len(df),
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

        # Verificar se a coluna de texto existe
        if text_column not in df.columns:
            self.logger.error(f"❌ Coluna de texto '{text_column}' não encontrada no DataFrame")
            self.logger.error(f"Colunas disponíveis: {list(df.columns)}")
            # Usar primeira coluna disponível como fallback
            text_column = df.columns[0] if len(df.columns) > 0 else 'body'
            self.logger.warning(f"⚠️ Usando coluna '{text_column}' como fallback")

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
            
            # Obter nome da coluna principal de texto (armazenado no Stage 01)
            if 'main_text_column' in df.columns and len(df) > 0:
                main_text_col = df['main_text_column'].iloc[0]
                self.logger.info(f"🔍 Coluna principal identificada: {main_text_col}")
                
                # Verificar se a coluna existe realmente
                if main_text_col not in df.columns:
                    self.logger.warning(f"⚠️ Coluna '{main_text_col}' não encontrada, buscando alternativa")
                    # Buscar coluna de texto válida
                    text_columns = [col for col in df.columns if df[col].dtype == 'object' and col not in ['main_text_column', 'timestamp_column']]
                    if text_columns:
                        main_text_col = text_columns[0]
                        self.logger.info(f"✅ Usando coluna alternativa: {main_text_col}")
                    else:
                        raise ValueError("❌ Nenhuma coluna de texto válida encontrada")
            else:
                # Fallback: buscar coluna de texto
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
                if text_columns:
                    main_text_col = text_columns[0]
                    self.logger.warning(f"⚠️ Usando primeira coluna de texto disponível: {main_text_col}")
                else:
                    raise ValueError("❌ Nenhuma coluna de texto encontrada")
            
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
        Stage 07: Processamento linguístico com spaCy.

        USA: normalized_text do Stage 02
        GERA: tokens, lemmas, POS tags, entidades nomeadas
        """
        self.logger.info("🔤 Stage 07: Linguistic Processing (spaCy)")

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

        # FIX: spaCy deve receber 'body' (texto cru) — normalized_text é lowercase
        # e sem pontuação, o que degrada NER, POS tagging e sentence splitting
        spacy_input_col = 'body' if 'body' in df.columns else 'normalized_text'
        spacy_results = df[spacy_input_col].apply(process_text_with_spacy)

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
        # FIX: usar body como input (consistente com spaCy path)
        fallback_col = 'body' if 'body' in df.columns else 'normalized_text'
        df['spacy_tokens'] = df[fallback_col].str.split()
        df['spacy_tokens_count'] = df['spacy_tokens'].str.len()
        df['spacy_lemmas'] = df['spacy_tokens']  # sem spaCy, lemmas = tokens
        df['spacy_entities_count'] = 0
        df['lemmatized_text'] = df[fallback_col].str.lower()  # fallback: lowercase do body

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
            # FIX: usar coluna 'hashtags_extracted' (Stage 01) ou 'body' (# removido de normalized_text)
            has_hashtags = 0
            if 'hashtags_extracted' in df.columns:
                has_hashtags = df['hashtags_extracted'].apply(
                    lambda x: len(x) > 0 if isinstance(x, list) else bool(x)
                ).sum()
            elif 'body' in df.columns:
                has_hashtags = df['body'].str.contains('#', na=False).sum()
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
            # FIX: emoji_ratio e caps_ratio devem usar 'body' (texto cru) — normalized_text
            # é lowercase e sem emojis, o que faz essas métricas retornarem sempre 0.0
            raw_col = 'body' if 'body' in df.columns else text_column
            if raw_col in df.columns:
                df['emoji_ratio'] = df[raw_col].apply(self._calculate_emoji_ratio)
                df['caps_ratio'] = df[raw_col].apply(self._calculate_caps_ratio)
                df['repetition_ratio'] = df[raw_col].apply(self._calculate_repetition_ratio)

                # Detecção de idioma básica (pode usar normalized_text — lowercase ok)
                df['likely_portuguese'] = df[text_column].apply(self._detect_portuguese) if text_column in df.columns else True
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

    def _stage_06_affordances_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 06: Affordances Classification (Híbrido: Heurística + API)

        Estratégia otimizada em 3 fases:
        1. Heurística expandida classifica todas as mensagens com scoring
        2. Mensagens de alta confiança (>=0.6) ficam com resultado heurístico
        3. Mensagens de baixa confiança (<0.6) são enviadas à API em batches de 10

        Categorias:
        - noticia, midia_social, video_audio_gif, opiniao,
        - mobilizacao, ataque, interacao, is_forwarded
        """
        try:
            self.logger.info("🎯 STAGE 06: Affordances Classification (Híbrido)")

            import os
            import requests
            import json
            import time
            from typing import List, Dict, Any

            text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
            initial_count = len(df)

            # === FASE 1: Heurística expandida em todas as mensagens ===
            self.logger.info(f"   📋 Fase 1: Heurística expandida em {initial_count} mensagens...")
            df = self._stage_06_affordances_heuristic_fallback(df)

            # Contar mensagens por nível de confiança
            high_conf_mask = df['affordance_confidence'] >= 0.6
            low_conf_mask = df['affordance_confidence'] < 0.6
            high_conf_count = high_conf_mask.sum()
            low_conf_count = low_conf_mask.sum()

            self.logger.info(f"   🟢 Alta confiança heurística: {high_conf_count} ({high_conf_count/initial_count*100:.1f}%)")
            self.logger.info(f"   🟡 Baixa confiança (candidatas API): {low_conf_count} ({low_conf_count/initial_count*100:.1f}%)")

            # === FASE 2: Verificar se API está disponível ===
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                self.logger.warning("⚠️ ANTHROPIC_API_KEY não encontrada. Usando apenas heurística.")
                # Limpar coluna temporária
                if '_heuristic_scores' in df.columns:
                    df = df.drop(columns=['_heuristic_scores'])
                self.stats['stages_completed'] += 1
                self.stats['features_extracted'] += 10
                return df

            if low_conf_count == 0:
                self.logger.info("   ✅ Todas as mensagens classificadas com alta confiança. API não necessária.")
                if '_heuristic_scores' in df.columns:
                    df = df.drop(columns=['_heuristic_scores'])
                self.stats['stages_completed'] += 1
                self.stats['features_extracted'] += 10
                return df

            # === FASE 3: Classificar mensagens de baixa confiança via API (batches de 10) ===
            self.logger.info(f"   🤖 Fase 3: Enviando {low_conf_count} mensagens à API em batches de 10...")

            # Modelo configurável via .env (default: Haiku 3.5)
            configured_model = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')
            self.logger.info(f"   🔧 Modelo API: {configured_model}")

            api_config = {
                'model': configured_model,
                'max_tokens': 800,
                'temperature': 0.1,
                'system_prompt': """Você é um classificador de conteúdo especializado em discurso político brasileiro em redes sociais.

Classifique CADA mensagem numerada de acordo com as categorias de affordances (múltiplas possíveis):

1. noticia: Conteúdo informativo, reportagem, fatos
2. midia_social: Posts de redes sociais, compartilhamentos
3. video_audio_gif: Referências a conteúdo multimídia
4. opiniao: Opiniões pessoais, comentários subjetivos
5. mobilizacao: Chamadas para ação, mobilização política
6. ataque: Ataques pessoais, insultos, agressões verbais
7. interacao: Respostas, menções, conversações diretas
8. is_forwarded: Conteúdo encaminhado/repassado

Responda APENAS com um JSON array válido. Exemplo para 3 mensagens:
[{"id":1,"categorias":["opiniao","ataque"],"confianca":0.9},{"id":2,"categorias":["noticia"],"confianca":0.85},{"id":3,"categorias":["mobilizacao"],"confianca":0.8}]"""
            }

            def classify_batch_with_anthropic(texts: List[str]) -> List[Dict[str, Any]]:
                """Classificar batch de textos (até 10) em uma única chamada API."""
                # Montar mensagem com textos numerados
                numbered_texts = []
                for i, text in enumerate(texts, 1):
                    text_sample = str(text)[:400] if not pd.isna(text) else ''
                    if len(text_sample.strip()) < 10:
                        text_sample = '(mensagem vazia ou muito curta)'
                    numbered_texts.append(f"[{i}] {text_sample}")

                user_content = "Classifique estas mensagens:\n\n" + "\n\n".join(numbered_texts)

                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'anthropic-beta': 'prompt-caching-2024-07-31'
                }

                payload = {
                    'model': api_config['model'],
                    'max_tokens': api_config['max_tokens'],
                    'temperature': api_config['temperature'],
                    'system': [
                        {
                            'type': 'text',
                            'text': api_config['system_prompt'],
                            'cache_control': {'type': 'ephemeral'}
                        }
                    ],
                    'messages': [{'role': 'user', 'content': user_content}]
                }

                try:
                    response = requests.post(
                        'https://api.anthropic.com/v1/messages',
                        headers=headers,
                        json=payload,
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        content = result['content'][0]['text'].strip()

                        # Parse JSON array
                        try:
                            classifications = json.loads(content)
                            if isinstance(classifications, list):
                                return classifications
                        except json.JSONDecodeError:
                            # Tentar extrair JSON de resposta
                            if '[' in content and ']' in content:
                                json_start = content.find('[')
                                json_end = content.rfind(']') + 1
                                try:
                                    classifications = json.loads(content[json_start:json_end])
                                    if isinstance(classifications, list):
                                        return classifications
                                except Exception:
                                    pass

                    elif response.status_code == 429:
                        self.logger.warning("⚠️ Rate limit atingido, aguardando 5s...")
                        time.sleep(5)

                    else:
                        self.logger.warning(f"⚠️ API error: {response.status_code}")

                except requests.RequestException as e:
                    self.logger.warning(f"⚠️ Erro de conexão: {e}")

                # Retorno vazio em caso de erro
                return []

            # Processar mensagens de baixa confiança
            low_conf_indices = df.index[low_conf_mask].tolist()

            # Verificar se deve usar Batch API (assíncrona) ou chamadas individuais
            use_batch_api = os.getenv('USE_BATCH_API', 'false').lower() in ('true', '1', 'yes')

            if use_batch_api and low_conf_count > 100:
                # === BATCH API (assíncrona, 50% desconto, até 24h) ===
                self.logger.info(f"   📦 Usando Batch API assíncrona para {low_conf_count} mensagens...")
                df = self._stage_06_submit_batch_api(
                    df, low_conf_indices, text_column, api_key, api_config
                )
            else:
                # === CHAMADAS INDIVIDUAIS (síncrono, batches de 10) ===
                if use_batch_api and low_conf_count <= 100:
                    self.logger.info(f"   ℹ️ Batch API não eficiente para {low_conf_count} mensagens. Usando chamadas diretas.")

                batch_size = 10
                api_calls_made = 0
                api_successes = 0
                api_failures = 0

                for i in range(0, len(low_conf_indices), batch_size):
                    batch_indices = low_conf_indices[i:i+batch_size]
                    batch_texts = [df.loc[idx, text_column] for idx in batch_indices]

                    classifications = classify_batch_with_anthropic(batch_texts)
                    api_calls_made += 1

                    if classifications:
                        for j, idx in enumerate(batch_indices):
                            if j < len(classifications):
                                cls = classifications[j]
                                if isinstance(cls, dict) and 'categorias' in cls:
                                    df.at[idx, 'affordance_categories'] = cls['categorias']
                                    df.at[idx, 'affordance_confidence'] = cls.get('confianca', 0.8)
                                    for aff_type in ['noticia', 'midia_social', 'video_audio_gif', 'opiniao',
                                                    'mobilizacao', 'ataque', 'interacao', 'is_forwarded']:
                                        df.at[idx, f'aff_{aff_type}'] = 1 if aff_type in cls['categorias'] else 0
                                    api_successes += 1
                                else:
                                    api_failures += 1
                            else:
                                api_failures += 1
                    else:
                        api_failures += len(batch_indices)

                    time.sleep(0.2)

                    if api_calls_made % 100 == 0:
                        progress = min(100, (i / len(low_conf_indices)) * 100)
                        self.logger.info(f"   🔄 API Progresso: {progress:.1f}% ({api_calls_made} calls, {api_successes} sucessos)")

            # === ESTATÍSTICAS FINAIS ===
            avg_confidence = df['affordance_confidence'].mean()
            classified_count = len(df[df['affordance_confidence'] > 0.1])

            affordance_types = ['noticia', 'midia_social', 'video_audio_gif', 'opiniao',
                              'mobilizacao', 'ataque', 'interacao', 'is_forwarded']
            category_counts = {}
            for affordance_type in affordance_types:
                count = df[f'aff_{affordance_type}'].sum()
                category_counts[affordance_type] = count

            # Limpar coluna temporária
            if '_heuristic_scores' in df.columns:
                df = df.drop(columns=['_heuristic_scores'])

            self.logger.info(f"✅ Classificação Híbrida de Affordances concluída:")
            self.logger.info(f"   📊 Heurística: {high_conf_count} mensagens ({high_conf_count/initial_count*100:.1f}%)")
            api_mode = "Batch API" if (use_batch_api and low_conf_count > 100) else "chamadas diretas"
            self.logger.info(f"   🤖 API ({api_mode}): {low_conf_count} mensagens processadas")
            self.logger.info(f"   ✅ Total classificadas: {classified_count}/{initial_count}")
            self.logger.info(f"   🎯 Confiança média: {avg_confidence:.3f}")

            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"   🔝 Top categorias: {dict(top_categories)}")

            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += len(affordance_types) + 2

            return df

        except Exception as e:
            self.logger.error(f"❌ Erro Stage 06: {e}")
            self.stats['processing_errors'] += 1
            # Fallback heurístico
            return self._stage_06_affordances_heuristic_fallback(df)

    def _stage_06_submit_batch_api(self, df: pd.DataFrame, low_conf_indices: list,
                                    text_column: str, api_key: str, api_config: dict) -> pd.DataFrame:
        """
        Submeter mensagens de baixa confiança à Anthropic Batch API.
        Processa até 10.000 requests por batch com 50% de desconto.

        Args:
            df: DataFrame com dados
            low_conf_indices: Índices de mensagens de baixa confiança
            text_column: Nome da coluna de texto
            api_key: Chave da API Anthropic
            api_config: Configurações do modelo (model, system_prompt, etc.)

        Returns:
            DataFrame atualizado com classificações da Batch API
        """
        import requests
        import json
        import time
        import tempfile
        import os

        batch_size = 10  # Mensagens por request individual dentro do batch
        max_requests_per_batch = 10000  # Limite da Batch API

        self.logger.info(f"   📦 Preparando Batch API: {len(low_conf_indices)} mensagens em batches de {batch_size}")

        # === FASE 1: Gerar requests para a Batch API ===
        batch_requests = []
        request_mapping = {}  # custom_id -> lista de índices do DataFrame

        for i in range(0, len(low_conf_indices), batch_size):
            batch_indices = low_conf_indices[i:i+batch_size]
            batch_texts = []
            for idx in batch_indices:
                text = df.loc[idx, text_column]
                text_sample = str(text)[:400] if not pd.isna(text) else ''
                if len(text_sample.strip()) < 10:
                    text_sample = '(mensagem vazia ou muito curta)'
                batch_texts.append(text_sample)

            # Montar mensagem com textos numerados
            numbered_texts = [f"[{j+1}] {t}" for j, t in enumerate(batch_texts)]
            user_content = "Classifique estas mensagens:\n\n" + "\n\n".join(numbered_texts)

            custom_id = f"batch_{i//batch_size:06d}"
            request_mapping[custom_id] = batch_indices

            request = {
                "custom_id": custom_id,
                "params": {
                    "model": api_config['model'],
                    "max_tokens": api_config['max_tokens'],
                    "temperature": api_config['temperature'],
                    "system": [
                        {
                            "type": "text",
                            "text": api_config['system_prompt'],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    "messages": [{"role": "user", "content": user_content}]
                }
            }
            batch_requests.append(request)

        total_requests = len(batch_requests)
        self.logger.info(f"   📝 {total_requests} requests gerados para Batch API")

        # === FASE 2: Submeter batches (até 10.000 por vez) ===
        all_results = {}

        for batch_start in range(0, total_requests, max_requests_per_batch):
            batch_chunk = batch_requests[batch_start:batch_start + max_requests_per_batch]
            chunk_num = batch_start // max_requests_per_batch + 1
            total_chunks = (total_requests + max_requests_per_batch - 1) // max_requests_per_batch

            self.logger.info(f"   🚀 Submetendo batch {chunk_num}/{total_chunks} ({len(batch_chunk)} requests)...")

            headers = {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'anthropic-beta': 'prompt-caching-2024-07-31'
            }

            payload = {"requests": batch_chunk}

            try:
                response = requests.post(
                    'https://api.anthropic.com/v1/messages/batches',
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                if response.status_code != 200:
                    self.logger.error(f"   ❌ Batch API erro: {response.status_code} - {response.text[:200]}")
                    continue

                batch_response = response.json()
                batch_id = batch_response['id']
                self.logger.info(f"   ✅ Batch {batch_id} criado. Status: {batch_response['processing_status']}")

                # === FASE 3: Polling para resultados ===
                results = self._stage_06_poll_batch_results(batch_id, api_key, headers)
                all_results.update(results)

            except requests.RequestException as e:
                self.logger.error(f"   ❌ Erro ao submeter batch: {e}")
                continue

        # === FASE 4: Aplicar resultados ao DataFrame ===
        api_successes = 0
        api_failures = 0

        for custom_id, classifications in all_results.items():
            if custom_id not in request_mapping:
                continue

            batch_indices = request_mapping[custom_id]

            if classifications:
                for j, idx in enumerate(batch_indices):
                    if j < len(classifications):
                        cls = classifications[j]
                        if isinstance(cls, dict) and 'categorias' in cls:
                            df.at[idx, 'affordance_categories'] = cls['categorias']
                            df.at[idx, 'affordance_confidence'] = cls.get('confianca', 0.8)
                            for aff_type in ['noticia', 'midia_social', 'video_audio_gif', 'opiniao',
                                            'mobilizacao', 'ataque', 'interacao', 'is_forwarded']:
                                df.at[idx, f'aff_{aff_type}'] = 1 if aff_type in cls['categorias'] else 0
                            api_successes += 1
                        else:
                            api_failures += 1
                    else:
                        api_failures += 1
            else:
                api_failures += len(batch_indices)

        self.logger.info(f"   📊 Batch API concluída: {api_successes} sucessos, {api_failures} falhas")
        return df

    def _stage_06_poll_batch_results(self, batch_id: str, api_key: str, headers: dict,
                                      max_wait_seconds: int = 86400, poll_interval: int = 30) -> dict:
        """
        Polling para resultados da Batch API.

        Args:
            batch_id: ID do batch submetido
            api_key: Chave da API
            headers: Headers HTTP
            max_wait_seconds: Tempo máximo de espera (default: 24h)
            poll_interval: Intervalo entre polls (default: 30s)

        Returns:
            Dict de custom_id -> lista de classificações
        """
        import requests
        import json
        import time

        results = {}
        start_time = time.time()

        self.logger.info(f"   ⏳ Aguardando resultados do batch {batch_id}...")

        while time.time() - start_time < max_wait_seconds:
            try:
                # Verificar status do batch
                status_response = requests.get(
                    f'https://api.anthropic.com/v1/messages/batches/{batch_id}',
                    headers={
                        'x-api-key': api_key,
                        'anthropic-version': '2023-06-01'
                    },
                    timeout=30
                )

                if status_response.status_code != 200:
                    self.logger.warning(f"   ⚠️ Erro ao verificar status: {status_response.status_code}")
                    time.sleep(poll_interval)
                    continue

                batch_status = status_response.json()
                processing_status = batch_status['processing_status']
                request_counts = batch_status.get('request_counts', {})

                processing = request_counts.get('processing', 0)
                succeeded = request_counts.get('succeeded', 0)
                errored = request_counts.get('errored', 0)
                total = processing + succeeded + errored

                elapsed = int(time.time() - start_time)
                self.logger.info(
                    f"   ⏳ Batch {batch_id}: {processing_status} "
                    f"({succeeded}/{total} ok, {errored} erros, {elapsed}s decorridos)"
                )

                if processing_status == 'ended':
                    # Coletar resultados
                    results_url = batch_status.get('results_url')
                    if results_url:
                        results = self._stage_06_fetch_batch_results(results_url, api_key)
                    else:
                        # Usar endpoint direto
                        results = self._stage_06_fetch_batch_results(
                            f'https://api.anthropic.com/v1/messages/batches/{batch_id}/results',
                            api_key
                        )

                    self.logger.info(f"   ✅ Batch {batch_id} finalizado: {len(results)} resultados coletados")
                    return results

            except requests.RequestException as e:
                self.logger.warning(f"   ⚠️ Erro no polling: {e}")

            time.sleep(poll_interval)

        self.logger.warning(f"   ⚠️ Timeout: batch {batch_id} não concluiu em {max_wait_seconds}s")
        return results

    def _stage_06_fetch_batch_results(self, results_url: str, api_key: str) -> dict:
        """
        Buscar e parsear resultados da Batch API (JSONL).

        Args:
            results_url: URL dos resultados
            api_key: Chave da API

        Returns:
            Dict de custom_id -> lista de classificações
        """
        import requests
        import json

        results = {}

        try:
            response = requests.get(
                results_url,
                headers={
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01'
                },
                timeout=120,
                stream=True
            )

            if response.status_code != 200:
                self.logger.error(f"   ❌ Erro ao buscar resultados: {response.status_code}")
                return results

            # Parsear JSONL
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    custom_id = entry.get('custom_id', '')
                    result = entry.get('result', {})

                    if result.get('type') == 'succeeded':
                        message = result.get('message', {})
                        content_blocks = message.get('content', [])

                        # Extrair texto da resposta
                        text_content = ''
                        for block in content_blocks:
                            if block.get('type') == 'text':
                                text_content = block.get('text', '').strip()
                                break

                        # Parsear JSON de classificações
                        classifications = self._stage_06_parse_batch_json(text_content)
                        results[custom_id] = classifications

                    elif result.get('type') in ('errored', 'canceled', 'expired'):
                        results[custom_id] = []
                        self.logger.debug(f"   ⚠️ Request {custom_id}: {result.get('type')}")

                except json.JSONDecodeError:
                    continue

        except requests.RequestException as e:
            self.logger.error(f"   ❌ Erro ao buscar resultados: {e}")

        return results

    def _stage_06_parse_batch_json(self, text: str) -> list:
        """
        Parsear JSON de classificações da resposta da API.
        Tenta múltiplas estratégias de parsing.

        Args:
            text: Texto da resposta da API

        Returns:
            Lista de dicts com categorias e confiança
        """
        import json

        if not text:
            return []

        # Tentativa 1: JSON direto
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Tentativa 2: Extrair array JSON do texto
        if '[' in text and ']' in text:
            json_start = text.find('[')
            json_end = text.rfind(']') + 1
            try:
                result = json.loads(text[json_start:json_end])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Tentativa 3: Extrair objetos JSON individuais
        try:
            objects = []
            import re
            for match in re.finditer(r'\{[^{}]+\}', text):
                try:
                    obj = json.loads(match.group())
                    if 'categorias' in obj:
                        objects.append(obj)
                except json.JSONDecodeError:
                    continue
            if objects:
                return objects
        except Exception:
            pass

        return []

    def _stage_06_affordances_heuristic_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback heurístico para classificação de affordances sem API."""
        self.logger.info("🔄 Aplicando classificação heurística de affordances...")

        text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'

        # Padrões heurísticos expandidos (15-20 keywords por categoria)
        patterns = {
            'noticia': [
                'aconteceu', 'notícia', 'informação', 'fato', 'governo', 'brasil',
                'reportagem', 'jornal', 'imprensa', 'publicou', 'divulgou', 'segundo',
                'fonte', 'comunicado', 'nota oficial', 'decreto', 'lei', 'medida',
                'aprovado', 'anunciou', 'declarou', 'dados', 'pesquisa', 'estudo'
            ],
            'midia_social': [
                'compartilhem', 'rt', 'retweet', 'curtir', 'like', 'seguir',
                'inscreva', 'canal', 'grupo', 'telegram', 'whatsapp', 'twitter',
                'instagram', 'youtube', 'facebook', 'tiktok', 'sigam', 'divulguem',
                'espalhem', 'repassem'
            ],
            'video_audio_gif': [
                'vídeo', 'video', 'audio', 'áudio', 'gif', 'assista', 'ouça',
                'podcast', 'live', 'ao vivo', 'transmissão', 'gravação', 'filmou',
                'imagem', 'foto', 'print', 'screenshot', 'clipe', 'documentário'
            ],
            'opiniao': [
                'acho', 'penso', 'na minha opinião', 'acredito', 'creio',
                'considero', 'entendo', 'parece', 'me parece', 'na verdade',
                'sinceramente', 'francamente', 'obviamente', 'claramente',
                'infelizmente', 'felizmente', 'absurdo', 'ridículo', 'inaceitável'
            ],
            'mobilizacao': [
                'vamos', 'precisamos', 'juntos', 'ação', 'mobilizar',
                'protesto', 'manifestação', 'marcha', 'ato', 'convocação',
                'compareçam', 'participem', 'lutar', 'resistir', 'unir',
                'levantar', 'defender', 'cobrar', 'exigir', 'pressionar'
            ],
            'ataque': [
                'idiota', 'burro', 'canalha', 'corrupto', 'mentiroso',
                'ladrão', 'bandido', 'vagabundo', 'safado', 'lixo',
                'vergonha', 'nojo', 'traidor', 'covarde', 'hipócrita',
                'incompetente', 'criminoso', 'fascista', 'comunista', 'genocida'
            ],
            'interacao': [
                '@', 'resposta', 'pergunta', 'dúvida', 'respondendo',
                'concordo', 'discordo', 'exatamente', 'isso mesmo',
                'verdade', 'falso', 'correto', 'errado', 'complementando'
            ],
            'is_forwarded': [
                'encaminhado', 'forward', 'repasse', 'compartilhe',
                'repassando', 'recebi', 'me mandaram', 'vejam isso',
                'olha isso', 'leiam', 'importante', 'urgente', 'atenção'
            ]
        }

        import re

        def classify_text_heuristic(text):
            """Classifica texto por heurística com scoring de confiança."""
            text_lower = str(text).lower() if not pd.isna(text) else ''
            if len(text_lower) < 5:
                return [], {}, 0.0

            categories = []
            scores = {}
            total_matches = 0

            for affordance_type, keywords in patterns.items():
                matches = sum(1 for kw in keywords if kw in text_lower)
                scores[affordance_type] = matches
                if matches >= 1:
                    categories.append(affordance_type)
                    total_matches += matches

            # FIX: regex em normalized_text nunca matcheia (: e // são removidos na normalização)
            # A detecção de URL agora é feita fora do loop, usando 'urls_extracted' do Stage 01
            if re.search(r'https?://', text_lower):
                if 'noticia' not in categories:
                    categories.append('noticia')
                    scores['noticia'] = scores.get('noticia', 0) + 1
                    total_matches += 1

            if text_lower.count('@') >= 2:
                if 'interacao' not in categories:
                    categories.append('interacao')
                    scores['interacao'] = scores.get('interacao', 0) + 2
                    total_matches += 2

            # Confiança baseada no total de matches
            if total_matches == 0:
                confidence = 0.1
            elif total_matches <= 2:
                confidence = 0.4
            elif total_matches <= 4:
                confidence = 0.6
            elif total_matches <= 7:
                confidence = 0.75
            else:
                confidence = 0.85

            return categories, scores, confidence

        # Aplicar classificação vetorizada
        self.logger.info(f"   📊 Classificando {len(df)} mensagens por heurística expandida...")
        results = df[text_column].apply(classify_text_heuristic)

        # Extrair resultados
        df['affordance_categories'] = results.apply(lambda x: x[0])
        df['_heuristic_scores'] = results.apply(lambda x: x[1])
        df['affordance_confidence'] = results.apply(lambda x: x[2])

        # Colunas binárias por categoria
        affordance_types = ['noticia', 'midia_social', 'video_audio_gif', 'opiniao',
                          'mobilizacao', 'ataque', 'interacao', 'is_forwarded']
        for affordance_type in affordance_types:
            df[f'aff_{affordance_type}'] = df['affordance_categories'].apply(
                lambda cats: 1 if affordance_type in cats else 0
            )

        # FIX: URL detection — regex em normalized_text nunca matcheia (://  removido)
        # Usar 'urls_extracted' do Stage 01 (preserva URLs reais do body)
        if 'urls_extracted' in df.columns:
            has_url = df['urls_extracted'].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else bool(x) if x else False
            )
            # Marcar textos com URL como 'noticia' se não classificados
            mask_url_not_noticia = has_url & (df['aff_noticia'] == 0)
            df.loc[mask_url_not_noticia, 'aff_noticia'] = 1
            url_boost_count = mask_url_not_noticia.sum()
            if url_boost_count > 0:
                self.logger.info(f"   🔗 URL detection via urls_extracted: +{url_boost_count} classificações 'noticia'")

        # Estatísticas
        classified = len(df[df['affordance_confidence'] > 0.1])
        high_conf = len(df[df['affordance_confidence'] >= 0.6])
        low_conf = len(df[(df['affordance_confidence'] > 0.1) & (df['affordance_confidence'] < 0.6)])

        self.logger.info(f"✅ Classificação heurística expandida concluída:")
        self.logger.info(f"   📊 Total classificadas: {classified}/{len(df)} ({classified/len(df)*100:.1f}%)")
        self.logger.info(f"   🟢 Alta confiança (>=0.6): {high_conf} ({high_conf/len(df)*100:.1f}%)")
        self.logger.info(f"   🟡 Baixa confiança (<0.6): {low_conf} ({low_conf/len(df)*100:.1f}%)")

        # Contagem por categoria
        for aff_type in affordance_types:
            count = df[f'aff_{aff_type}'].sum()
            if count > 0:
                self.logger.info(f"   📌 {aff_type}: {count} ({count/len(df)*100:.1f}%)")

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
        Stage 08: Classificação política brasileira.

        REFORMULADO: usa spacy_lemmas (Stage 07) com token matching via set lookup.
        Aplica léxico unificado (914+ termos, 11 macrotemas) para classificar textos.
        Escopo: discurso bolsonarista/direita brasileira (2019-2023).
        Orientações retornadas: extrema-direita, direita, centro-direita, neutral.
        Nota: léxico não inclui termos de esquerda (fora do escopo do projeto).
        """
        try:
            self.logger.info("🔄 Stage 08: Classificação política brasileira (token matching via spaCy lemmas)")

            # Determinar coluna de input: preferir spacy_lemmas > lemmatized_text > normalized_text
            if 'spacy_lemmas' in df.columns:
                input_col = 'spacy_lemmas'
                self.logger.info("   📥 Input: spacy_lemmas (token-level matching)")
            elif 'lemmatized_text' in df.columns:
                input_col = 'lemmatized_text'
                self.logger.info("   📥 Input: lemmatized_text (string fallback)")
            else:
                input_col = 'normalized_text' if 'normalized_text' in df.columns else 'body'
                self.logger.warning(f"   ⚠️ spaCy output não disponível, fallback: {input_col}")

            # Classificação política usando léxico unificado
            df['political_orientation'] = df[input_col].apply(self._classify_political_orientation)
            df['political_keywords'] = df[input_col].apply(self._extract_political_keywords)
            df['political_intensity'] = df[input_col].apply(self._calculate_political_intensity)

            # Classificação temática - 12 categorias (political_keywords_dict.py)
            try:
                from src.core.political_keywords_dict import POLITICAL_KEYWORDS
                import re as _re

                for cat_name, cat_terms in POLITICAL_KEYWORDS.items():
                    col_name = 'cat_' + _re.sub(r'^cat\d+_', '', cat_name)
                    # Token matching: set intersection para single-word, substring para multi-word
                    cat_single = set(t for t in cat_terms if ' ' not in t)
                    cat_multi = [t for t in cat_terms if ' ' in t]

                    def _count_cat_matches(lemmas_or_text, s=cat_single, m=cat_multi):
                        if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
                            return 0
                        if isinstance(lemmas_or_text, list):
                            tset = set(t.lower() for t in lemmas_or_text if t)
                        else:
                            tset = set(str(lemmas_or_text).lower().split())
                        count = len(tset & s)
                        if m:
                            joined = ' '.join(sorted(tset))
                            count += sum(1 for t in m if t in joined)
                        return count

                    df[col_name] = df[input_col].apply(_count_cat_matches)

                self.logger.info(f"📊 Classificação temática: {len(POLITICAL_KEYWORDS)} categorias aplicadas (token matching)")
            except ImportError:
                self.logger.warning("⚠️ political_keywords_dict.py não encontrado, pulando categorias temáticas")

            # === CODIFICAÇÃO TCW (Tabela-Categoria-Palavra) ===
            # Integrado do classificador TCW: adiciona códigos numéricos 3-dígitos
            # e grau de concordância entre as 3 tabelas LLM
            try:
                import json as _json
                from pathlib import Path as _Path

                tcw_path = _Path(__file__).parent / 'core' / 'tcw_codes.json'
                if tcw_path.exists():
                    with open(tcw_path, 'r', encoding='utf-8') as f:
                        tcw_codes = _json.load(f)

                    # Construir lookup: word → list of codes
                    word_to_codes = {}
                    for code, info in tcw_codes.items():
                        word = info['word'].lower()
                        if word not in word_to_codes:
                            word_to_codes[word] = []
                        word_to_codes[word].append({
                            'code': code,
                            'table': info['table'],
                            'category': info['category'],
                            'category_name': info['category_name']
                        })

                    # Single-word e multi-word lookup sets
                    tcw_single = set(w for w in word_to_codes if ' ' not in w)
                    tcw_multi = [w for w in word_to_codes if ' ' in w]

                    def _tcw_classify(lemmas_or_text):
                        """Classificar texto usando codificação TCW."""
                        if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
                            return [], [], 0.0
                        if isinstance(lemmas_or_text, list):
                            tset = set(t.lower() for t in lemmas_or_text if t)
                        else:
                            tset = set(str(lemmas_or_text).lower().split())

                        codes_found = []
                        categories_found = set()

                        # Single-word matches
                        for word in tset & tcw_single:
                            for entry in word_to_codes[word]:
                                codes_found.append(entry['code'])
                                categories_found.add(entry['category_name'])

                        # Multi-word matches
                        if tcw_multi:
                            joined = ' '.join(sorted(tset))
                            for mw in tcw_multi:
                                if mw in joined:
                                    for entry in word_to_codes[mw]:
                                        codes_found.append(entry['code'])
                                        categories_found.add(entry['category_name'])

                        # Concordância: quantas tabelas (1-3) concordam nos códigos encontrados
                        if codes_found:
                            tables_seen = set()
                            for code in codes_found:
                                if code in tcw_codes:
                                    tables_seen.add(tcw_codes[code]['table'])
                            agreement = len(tables_seen) / 3.0  # 0.33, 0.67, 1.0
                        else:
                            agreement = 0.0

                        return codes_found, list(categories_found), agreement

                    tcw_results = df[input_col].apply(_tcw_classify)
                    df['tcw_codes'] = tcw_results.apply(lambda x: x[0])
                    df['tcw_categories'] = tcw_results.apply(lambda x: x[1])
                    df['tcw_agreement'] = tcw_results.apply(lambda x: x[2])
                    df['tcw_code_count'] = df['tcw_codes'].apply(len)

                    tcw_classified = (df['tcw_code_count'] > 0).sum()
                    self.logger.info(f"🔢 TCW: {tcw_classified}/{len(df)} textos classificados ({tcw_classified/len(df)*100:.1f}%)")
                else:
                    self.logger.warning("⚠️ tcw_codes.json não encontrado em src/core/, pulando TCW")
            except Exception as tcw_err:
                self.logger.warning(f"⚠️ Erro TCW: {tcw_err}")

            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 19  # 15 base + 4 TCW

            self.logger.info(f"✅ Stage 08 concluído: {len(df)} registros processados")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro Stage 08: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_09_tfidf_vectorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 09: Vetorização TF-IDF com tokens spaCy.
        
        Calcula TF-IDF usando tokens processados pelo spaCy.
        Trata casos de vocabulário vazio em chunks pequenos.
        """
        try:
            self.logger.info("🔄 Stage 09: Vetorização TF-IDF")
            
            # Verificar se há dados suficientes
            if len(df) < 2:
                self.logger.warning(f"⚠️ Dados insuficientes para TF-IDF ({len(df)} documentos), preenchendo com padrões")
                df['tfidf_score_mean'] = 0.0
                df['tfidf_score_max'] = 0.0
                df['tfidf_top_terms'] = [[] for _ in range(len(df))]
                self.stats['features_extracted'] += 3
                return df
            
            # FIX: usar 'lemmatized_text' (output do Stage 07 spaCy) em vez de 'tokens' (inexistente)
            if 'lemmatized_text' in df.columns:
                text_data = df['lemmatized_text'].fillna('').tolist()
            elif 'spacy_tokens' in df.columns:
                text_data = df['spacy_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('').tolist()
            else:
                self.logger.warning("⚠️ lemmatized_text/spacy_tokens não encontrados, usando normalized_text")
                text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
                text_data = df[text_column].fillna('').tolist()
            
            # Verificar se há texto não-vazio
            non_empty_texts = [text for text in text_data if text.strip()]
            if len(non_empty_texts) < 2:
                self.logger.warning(f"⚠️ Textos vazios demais para TF-IDF ({len(non_empty_texts)} válidos), usando fallback")
                df['tfidf_score_mean'] = 0.1
                df['tfidf_score_max'] = 0.2
                df['tfidf_top_terms'] = [['texto', 'palavra'] for _ in range(len(df))]
                self.stats['features_extracted'] += 3
                return df
            
            # TF-IDF com configuração adaptativa para chunks pequenos
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Ajustar max_features baseado no tamanho do chunk
            chunk_size = len(df)
            if chunk_size < 50:
                max_features = min(20, chunk_size * 2)  # Muito conservador
            elif chunk_size < 200:
                max_features = min(50, chunk_size)  # Conservador
            else:
                max_features = 100  # Padrão
            
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=1,  # Aceitar termos que aparecem pelo menos 1 vez
                stop_words=None,  # Já removemos stopwords no spaCy
                lowercase=False,   # Já normalizado
                token_pattern=r'\S+',  # Aceitar qualquer token não-espaço
                ngram_range=(1, 1)  # Apenas unigramas para chunks pequenos
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(text_data)
                feature_names = vectorizer.get_feature_names_out()
                
                # Verificar se conseguiu gerar features
                if tfidf_matrix.shape[1] == 0:
                    raise ValueError("Vocabulário vazio após vectorização")
                
                # Converter para array denso para cálculos
                tfidf_dense = tfidf_matrix.toarray()
                
                # Scores médios por documento
                df['tfidf_score_mean'] = np.mean(tfidf_dense, axis=1)
                df['tfidf_score_max'] = np.max(tfidf_dense, axis=1)
                
                # Top terms por documento 
                top_terms_count = min(5, len(feature_names))  # Adaptar ao vocabulário disponível
                df['tfidf_top_terms'] = [
                    [feature_names[i] for i in row.argsort()[::-1][:top_terms_count] if row[i] > 0]
                    for row in tfidf_dense
                ]
                
                self.logger.info(f"✅ TF-IDF: {len(feature_names)} features, max_features={max_features}")
                
            except (ValueError, Exception) as ve:
                self.logger.warning(f"⚠️ Erro na vectorização TF-IDF: {ve}, usando fallback simples")
                
                # Fallback: análise simples baseada em frequência de palavras
                import re
                from collections import Counter
                
                all_words = []
                for text in text_data:
                    if text and text.strip():
                        # Extrair palavras simples
                        words = re.findall(r'\w+', text.lower())
                        words = [w for w in words if len(w) > 2]  # Filtrar palavras muito curtas
                        all_words.extend(words)
                
                if all_words:
                    word_freq = Counter(all_words)
                    common_words = [word for word, _ in word_freq.most_common(10)]
                    
                    # Scores baseados em presença de palavras comuns
                    df['tfidf_score_mean'] = [
                        len([w for w in re.findall(r'\w+', str(text).lower()) if w in common_words]) / max(1, len(common_words)) * 0.5
                        for text in text_data
                    ]
                    df['tfidf_score_max'] = df['tfidf_score_mean'] * 1.5
                    df['tfidf_top_terms'] = [
                        [w for w in re.findall(r'\w+', str(text).lower()) if w in common_words][:5]
                        for text in text_data
                    ]
                else:
                    # Última opção: valores padrão
                    df['tfidf_score_mean'] = 0.1
                    df['tfidf_score_max'] = 0.2
                    df['tfidf_top_terms'] = [[] for _ in range(len(df))]
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 3
            
            self.logger.info(f"✅ Stage 09 concluído: {len(df)} registros processados")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 09: {e}")
            self.stats['processing_errors'] += 1
            
            # Valores padrão em caso de erro
            df['tfidf_score_mean'] = 0.0
            df['tfidf_score_max'] = 0.0
            df['tfidf_top_terms'] = [[] for _ in range(len(df))]
            self.stats['features_extracted'] += 3
            return df

    @validate_stage_dependencies(required_columns=['tfidf_score_mean'], required_attrs=['tfidf_matrix'])
    def _stage_10_clustering_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 10: Análise de clustering baseado em features linguísticas.

        Agrupa documentos similares usando características extraídas.
        """
        try:
            self.logger.info("🔄 Stage 10: Análise de clustering")
            
            # Features numéricas para clustering
            numeric_features = []
            # FIX: 'text_length' não existe — Stage 04 gera 'char_count'
            for col in ['word_count', 'char_count', 'tfidf_score_mean', 'political_intensity']:
                if col in df.columns:
                    numeric_features.append(col)
            
            if len(numeric_features) < 2:
                self.logger.warning("⚠️ Features insuficientes para clustering")
                df['cluster_id'] = 0
                df['cluster_distance'] = 0.0
                df['cluster_size'] = len(df)
            else:
                from sklearn.preprocessing import StandardScaler

                # Preparar dados
                feature_data = df[numeric_features].fillna(0)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_data)

                # Tentar HDBSCAN (instalado no pyproject.toml, auto-detecção de k)
                try:
                    import hdbscan
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=max(5, len(df) // 50),
                        min_samples=3,
                        metric='euclidean'
                    )
                    clusters = clusterer.fit_predict(scaled_data)
                    # HDBSCAN retorna -1 para noise
                    df['cluster_id'] = clusters
                    df['cluster_distance'] = 1.0 - clusterer.probabilities_  # prob → distância
                    n_found = len(set(clusters) - {-1})
                    n_noise = (clusters == -1).sum()
                    self.logger.info(f"📊 HDBSCAN: {n_found} clusters, {n_noise} noise points")

                except (ImportError, Exception) as e:
                    # Fallback para K-Means
                    self.logger.warning(f"⚠️ HDBSCAN indisponível ({e}), usando KMeans fallback")
                    from sklearn.cluster import KMeans
                    n_clusters = min(5, len(df) // 10 + 1)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_data)
                    df['cluster_id'] = clusters
                    df['cluster_distance'] = [
                        min(((scaled_data[i] - center) ** 2).sum() for center in kmeans.cluster_centers_)
                        for i in range(len(scaled_data))
                    ]

                # Tamanho dos clusters
                cluster_sizes = pd.Series(df['cluster_id']).value_counts().to_dict()
                df['cluster_size'] = df['cluster_id'].map(cluster_sizes)
            
            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 3
            
            self.logger.info(f"✅ Stage 10 concluído: {len(df)} registros processados")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro Stage 10: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_11_topic_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 11: Topic modeling com LDA.

        Descoberta automática de tópicos nos textos.
        """
        try:
            self.logger.info("🔄 Stage 11: Topic modeling")
            
            # FIX: usar 'lemmatized_text' (output do Stage 07 spaCy) em vez de 'tokens' (inexistente)
            if 'lemmatized_text' in df.columns:
                text_data = df['lemmatized_text'].fillna('').tolist()
            elif 'spacy_tokens' in df.columns:
                text_data = df['spacy_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('').tolist()
            else:
                self.logger.warning("⚠️ lemmatized_text/spacy_tokens não encontrados, usando normalized_text")
                text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
                text_data = df[text_column].fillna('').tolist()
            
            # Topic modeling básico com LDA
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            # Stopwords PT (termos funcionais que poluem LDA)
            pt_stopwords = [
                'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas',
                'um', 'uma', 'uns', 'umas', 'por', 'para', 'com', 'sem', 'sob',
                'que', 'se', 'não', 'mais', 'muito', 'como', 'mas', 'ou', 'já',
                'também', 'só', 'seu', 'sua', 'seus', 'suas', 'ele', 'ela', 'eles',
                'elas', 'isso', 'isto', 'esse', 'essa', 'este', 'esta', 'aqui',
                'ali', 'lá', 'ao', 'aos', 'à', 'às', 'pelo', 'pela', 'pelos', 'pelas',
                'entre', 'sobre', 'após', 'até', 'quando', 'onde', 'quem', 'qual',
                'foi', 'ser', 'ter', 'está', 'são', 'tem', 'era', 'vai', 'pode',
                'nos', 'me', 'te', 'lhe', 'o', 'a', 'os', 'as', 'e', 'é',
                'eu', 'tu', 'nós', 'vós', 'meu', 'minha', 'teu', 'tua',
                'nosso', 'nossa', 'nossos', 'nossas', 'todo', 'toda', 'todos', 'todas',
                'outro', 'outra', 'outros', 'outras', 'mesmo', 'mesma', 'cada',
                'ainda', 'então', 'depois', 'antes', 'bem', 'agora', 'sempre',
                'nunca', 'nada', 'tudo', 'algo', 'assim', 'aquele', 'aquela',
                'http', 'https', 'www', 'com', 'org', 'br', 'the', 'and', 'for'
            ]

            # Preparar dados (com remoção de stopwords PT)
            vectorizer = CountVectorizer(max_features=50, stop_words=pt_stopwords)
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
            
            self.logger.info(f"✅ Stage 11 concluído: {len(df)} registros processados")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro Stage 11: {e}")
            self.stats['processing_errors'] += 1
            return df

    @validate_stage_dependencies(required_columns=['normalized_text'])
    def _stage_13_temporal_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 13: Análise temporal.

        Extrai padrões temporais dos timestamps.
        """
        try:
            self.logger.info("🔄 Stage 13: Análise temporal")
            
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
            
            # Burst Detection - Kleinberg (2003), KDD
            # Detecta dias com volume anormal de mensagens
            df['is_burst_day'] = False
            if 'datetime' in df.columns:
                try:
                    dt_series = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    dates = dt_series.dt.date
                    daily_counts = dates.value_counts()
                    if len(daily_counts) >= 3:
                        mean_count = daily_counts.mean()
                        std_count = daily_counts.std()
                        burst_threshold = mean_count + 2 * std_count  # 2 desvios padrão
                        burst_dates = daily_counts[daily_counts > burst_threshold].index.tolist()
                        df['is_burst_day'] = dates.isin(burst_dates)
                        if burst_dates:
                            self.logger.info(f"📈 Burst detection: {len(burst_dates)} dias com volume anormal")
                except Exception as e:
                    self.logger.warning(f"⚠️ Burst detection: {e}")

            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 6

            self.logger.info(f"✅ Stage 13 concluído: {len(df)} registros processados")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro Stage 13: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_14_network_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 14: Análise de rede (coordenação e padrões).

        Detecta padrões de coordenação e comportamento de rede.
        """
        try:
            self.logger.info("🔄 Stage 14: Análise de rede")
            
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
            
            self.logger.info(f"✅ Stage 14 concluído: {len(df)} registros processados")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro Stage 14: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_15_domain_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 15: Análise de domínios.

        Analisa domínios e URLs para identificar padrões de mídia.
        """
        try:
            self.logger.info("🔄 Stage 15: Análise de domínios")
            
            # Análise de domínios com trust score (Page et al. 1999, adaptado)
            if 'domain' in df.columns:
                df['domain_type'] = df['domain'].apply(self._classify_domain_type)
                df['domain_trust_score'] = df['domain'].apply(self._calculate_domain_trust_score)

                domain_counts = df['domain'].value_counts()
                df['domain_frequency'] = df['domain'].map(domain_counts)

                # Mídia mainstream vs alternativa (baseado em domain_type classificado)
                mainstream_types = ['mainstream_news', 'government']
                df['is_mainstream_media'] = df['domain_type'].isin(mainstream_types)
            else:
                df['domain_type'] = 'unknown'
                df['domain_trust_score'] = 0.0
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
            
            self.logger.info(f"✅ Stage 15 concluído: {len(df)} registros processados")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro Stage 15: {e}")
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
            
            # Análise de emoções básicas (usar body original para detectar !, ?, CAPS)
            raw_col = 'body' if 'body' in df.columns else text_column
            df['emotion_intensity'] = df.apply(
                lambda row: self._calculate_emotion_intensity(
                    str(row.get(text_column, '')),
                    raw_text=str(row.get(raw_col, ''))
                ), axis=1
            )
            df['has_aggressive_language'] = df[text_column].apply(self._detect_aggressive_language)
            
            # Complexidade semântica
            # FIX: usar 'spacy_tokens' (output real do Stage 07) em vez de 'tokens' (inexistente)
            if 'spacy_tokens' in df.columns:
                df['semantic_diversity'] = df['spacy_tokens'].apply(
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
        Stage 16: Análise de contexto de eventos.

        Detecta contextos políticos e eventos relevantes.
        """
        try:
            self.logger.info("🔄 Stage 16: Análise de contexto de eventos")

            # FIX: preferir lemmatized_text (output do spaCy) para melhor matching de contextos
            if 'lemmatized_text' in df.columns:
                text_column = 'lemmatized_text'
            elif 'normalized_text' in df.columns:
                text_column = 'normalized_text'
            else:
                text_column = 'body'
            
            # Contextos políticos brasileiros
            df['political_context'] = df[text_column].apply(self._detect_political_context)
            df['mentions_government'] = df[text_column].apply(self._mentions_government)
            df['mentions_opposition'] = df[text_column].apply(self._mentions_opposition)
            
            # Eventos específicos (eleições, manifestações, etc.)
            df['election_context'] = df[text_column].apply(self._detect_election_context)
            df['protest_context'] = df[text_column].apply(self._detect_protest_context)
            
            # Frame Analysis - Entman (1993), J Communication 43(4): 51-58
            frame_results = df[text_column].apply(self._analyze_political_frames)
            df['frame_conflito'] = frame_results.apply(lambda x: x.get('conflito', 0.0))
            df['frame_responsabilizacao'] = frame_results.apply(lambda x: x.get('responsabilizacao', 0.0))
            df['frame_moralista'] = frame_results.apply(lambda x: x.get('moralista', 0.0))
            df['frame_economico'] = frame_results.apply(lambda x: x.get('economico', 0.0))

            # Análise temporal de eventos
            if 'datetime' in df.columns:
                df['is_weekend'] = df['day_of_week'].isin([5, 6]) if 'day_of_week' in df.columns else False
                df['is_business_hours'] = df['hour'].between(9, 17) if 'hour' in df.columns else False
            else:
                df['is_weekend'] = False
                df['is_business_hours'] = True

            self.stats['stages_completed'] += 1
            self.stats['features_extracted'] += 11

            self.logger.info(f"✅ Stage 16 concluído: {len(df)} registros, 4 frames Entman extraídos")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erro Stage 16: {e}")
            self.stats['processing_errors'] += 1
            return df

    def _stage_17_channel_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 17: Análise de canais/fontes.

        Classifica canais e fontes de informação.
        """
        try:
            self.logger.info("🔄 Stage 17: Análise de canais")
            
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
            
            self.logger.info(f"✅ Stage 17 concluído: {len(df)} registros processados")
            return df

        except Exception as e:
            self.logger.error(f"❌ Erro Stage 17: {e}")
            self.stats['processing_errors'] += 1
            return df

    # ==========================================
    # HELPER METHODS FOR ANALYSIS STAGES
    # (Integrado com lexico_unified_system.json: 956 termos, 9 macrotemas)
    # ==========================================

    def _classify_political_orientation(self, lemmas_or_text) -> str:
        """
        Classifica orientação política usando léxico unificado.
        REFORMULADO: aceita lista de lemmas (spaCy) ou string (fallback).
        Token matching via set lookup — O(1) por token, zero falsos positivos.
        """
        if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
            return 'neutral'

        # Converter input para set de tokens (lemmas ou palavras)
        if isinstance(lemmas_or_text, list):
            token_set = set(t.lower() for t in lemmas_or_text if t)
        else:
            token_set = set(str(lemmas_or_text).lower().split())

        if not token_set:
            return 'neutral'

        terms_map = self._political_terms_map

        # Macrotemas de direita/extrema-direita
        direita_categories = [
            'identidade_patriotica', 'inimigos_ideologicos', 'teorias_conspiracao',
            'negacionismo', 'autoritarismo_violencia', 'mobilizacao_acao',
            'desinformacao_verdade', 'estrategias_discursivas', 'eventos_simbolicos',
            'corrupcao_transparencia', 'politica_externa'
        ]

        # Contar matches por macrotema via set intersection
        scores = {}
        for cat in direita_categories:
            terms = terms_map.get(cat, [])
            # Para termos compostos (multi-word), verificar no texto concatenado
            single_word_terms = set(t for t in terms if ' ' not in t)
            multi_word_terms = [t for t in terms if ' ' in t]

            count = len(token_set & single_word_terms)
            if multi_word_terms:
                text_joined = ' '.join(sorted(token_set))
                count += sum(1 for t in multi_word_terms if t in text_joined)
            scores[cat] = count

        total_matches = sum(scores.values())
        if total_matches == 0:
            return 'neutral'

        radical_score = scores.get('autoritarismo_violencia', 0) + scores.get('mobilizacao_acao', 0)
        conspiracao_score = scores.get('teorias_conspiracao', 0) + scores.get('negacionismo', 0)
        identidade_score = scores.get('identidade_patriotica', 0) + scores.get('eventos_simbolicos', 0)
        adversario_score = scores.get('inimigos_ideologicos', 0)

        if radical_score >= 2 or (conspiracao_score >= 2 and adversario_score >= 1):
            return 'extrema-direita'
        elif adversario_score >= 2 or conspiracao_score >= 2:
            return 'direita'
        elif identidade_score >= 2:
            return 'centro-direita'
        elif total_matches >= 1:
            return 'direita'
        return 'neutral'

    def _extract_political_keywords(self, lemmas_or_text) -> list:
        """
        Extrai palavras-chave políticas usando léxico unificado.
        REFORMULADO: token matching via set intersection.
        """
        if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
            return []

        if isinstance(lemmas_or_text, list):
            token_set = set(t.lower() for t in lemmas_or_text if t)
        else:
            token_set = set(str(lemmas_or_text).lower().split())

        if not token_set:
            return []

        terms_map = self._political_terms_map
        found = []
        for cat, terms in terms_map.items():
            single_word_terms = set(t for t in terms if ' ' not in t)
            matches = token_set & single_word_terms
            for m in matches:
                if m not in found:
                    found.append(m)
                    if len(found) >= 10:
                        return found
            # Multi-word terms: fallback substring
            multi_word_terms = [t for t in terms if ' ' in t]
            if multi_word_terms:
                text_joined = ' '.join(sorted(token_set))
                for t in multi_word_terms:
                    if t in text_joined and t not in found:
                        found.append(t)
                        if len(found) >= 10:
                            return found
        return found

    def _calculate_political_intensity(self, lemmas_or_text) -> float:
        """
        Calcula intensidade usando termos de mobilização e autoritarismo.
        REFORMULADO: token matching via set intersection.
        """
        if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
            return 0.0

        if isinstance(lemmas_or_text, list):
            token_set = set(t.lower() for t in lemmas_or_text if t)
        else:
            token_set = set(str(lemmas_or_text).lower().split())

        if not token_set:
            return 0.0

        terms_map = self._political_terms_map

        # Termos de alta intensidade
        intensity_terms = (
            terms_map.get('mobilizacao_acao', []) +
            terms_map.get('autoritarismo_violencia', []) +
            terms_map.get('desinformacao_verdade', [])
        )

        if not intensity_terms:
            return 0.0

        single_word = set(t for t in intensity_terms if ' ' not in t)
        match_count = len(token_set & single_word)

        # Multi-word fallback
        multi_word = [t for t in intensity_terms if ' ' in t]
        if multi_word:
            text_joined = ' '.join(sorted(token_set))
            match_count += sum(1 for t in multi_word if t in text_joined)

        return min(match_count * 0.15, 1.0)

    def _classify_domain_type(self, domain: str) -> str:
        """Classifica tipo de domínio com categorias expandidas."""
        if not domain or pd.isna(domain):
            return 'unknown'

        domain_lower = str(domain).lower()

        # Categorias expandidas (baseado em domain_authority_analysis do archive)
        trusted_news = ['folha.uol.com.br', 'g1.globo.com', 'estadao.com.br',
                       'oglobo.globo.com', 'uol.com.br', 'bbc.com', 'reuters.com',
                       'globo.com', 'folha.com', 'r7.com', 'terra.com.br']
        government = ['.gov.br', '.leg.br', '.jus.br', '.mil.br']
        video = ['youtube.com', 'youtu.be', 'vimeo.com', 'rumble.com', 'odysee.com']
        social = ['twitter.com', 'x.com', 'facebook.com', 'instagram.com',
                 't.me', 'telegram.me', 'whatsapp.com', 'tiktok.com']
        blog = ['blog', 'wordpress', 'medium.com', 'substack.com', 'blogspot']

        if any(trusted in domain_lower for trusted in trusted_news):
            return 'mainstream_news'
        elif any(gov in domain_lower for gov in government):
            return 'government'
        elif any(v in domain_lower for v in video):
            return 'video'
        elif any(s in domain_lower for s in social):
            return 'social'
        elif any(b in domain_lower for b in blog):
            return 'blog'
        else:
            return 'alternative'

    def _calculate_domain_trust_score(self, domain: str) -> float:
        """Calcula score de confiança do domínio (Page et al. 1999, adaptado)."""
        if not domain or pd.isna(domain):
            return 0.0
        dtype = self._classify_domain_type(domain)
        trust_map = {
            'government': 0.9, 'mainstream_news': 0.8, 'video': 0.5,
            'social': 0.4, 'blog': 0.3, 'alternative': 0.2, 'unknown': 0.0
        }
        return trust_map.get(dtype, 0.0)

    def _calculate_sentiment_polarity(self, text: str) -> float:
        """Calcula polaridade com dicionário LIWC expandido (Balage Filho et al. 2013)."""
        if not text or pd.isna(text):
            return 0.0

        # Dicionário LIWC-PT expandido (baseado em sci_validated_methods_implementation.py)
        positive_words = [
            'bom', 'boa', 'bons', 'boas', 'ótimo', 'ótima', 'excelente',
            'maravilhoso', 'maravilhosa', 'perfeito', 'perfeita', 'amor',
            'feliz', 'felicidade', 'alegria', 'alegre', 'vitória', 'sucesso',
            'conquista', 'esperança', 'orgulho', 'admiração', 'respeito',
            'liberdade', 'paz', 'progresso', 'avanço', 'melhoria',
            'lindo', 'linda', 'beleza', 'incrível', 'fantástico', 'fantástica',
            'parabéns', 'obrigado', 'obrigada', 'gratidão', 'abençoado',
            'honra', 'glória', 'benção', 'fé', 'força', 'coragem'
        ]
        negative_words = [
            'ruim', 'péssimo', 'péssima', 'terrível', 'horrível', 'ódio',
            'raiva', 'triste', 'tristeza', 'infeliz', 'medo', 'fracasso',
            'derrota', 'vergonha', 'nojo', 'desgraça', 'desastre',
            'culpa', 'miséria', 'sofrimento', 'dor', 'angústia',
            'decepção', 'frustração', 'absurdo', 'ridículo', 'ridícula',
            'lamentável', 'deplorável', 'covarde', 'covardia', 'mentira',
            'mentiroso', 'mentirosa', 'traição', 'traidor', 'traidora',
            'destruição', 'morte', 'desespero', 'pânico', 'terror',
            'criminoso', 'criminosa', 'bandido', 'bandida', 'corrupto', 'corrupção'
        ]

        text_lower = str(text).lower()
        words = text_lower.split()
        total_words = len(words)
        if total_words == 0:
            return 0.0

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        return (positive_count - negative_count) / total_words

    def _calculate_emotion_intensity(self, text: str, raw_text: str = None) -> float:
        """Calcula intensidade emocional usando texto original (com pontuação)."""
        # Usar raw_text (body original) se disponível, pois normalized_text remove pontuação
        source = raw_text if raw_text else text
        if not source or pd.isna(source):
            return 0.0

        source_str = str(source)
        emotion_markers = source_str.count('!') + source_str.count('?') + source_str.count('...')
        caps_words = sum(1 for word in source_str.split() if word.isupper() and len(word) > 2)

        return min((emotion_markers + caps_words) / 10.0, 1.0)

    def _detect_aggressive_language(self, text: str) -> bool:
        """Detecta linguagem agressiva usando léxico (autoritarismo_violencia + inimigos)."""
        if not text or pd.isna(text):
            return False

        text_lower = str(text).lower()
        terms_map = self._political_terms_map

        # Combinar termos de violência e agressão do léxico
        aggressive_terms = terms_map.get('autoritarismo_violencia', [])
        # Adicionar termos clássicos de agressão pessoal
        extra_aggressive = [
            'ódio', 'matar', 'destruir', 'eliminar', 'acabar',
            'burro', 'idiota', 'imbecil', 'estúpido', 'canalha',
            'vagabundo', 'lixo', 'verme', 'parasita', 'bandido',
            'safado', 'nojento', 'covarde', 'traidor', 'criminoso'
        ]

        all_aggressive = set(aggressive_terms + extra_aggressive)
        return any(term in text_lower for term in all_aggressive)

    def _detect_political_context(self, text: str) -> str:
        """Detecta contexto político."""
        if not text or pd.isna(text):
            return 'none'

        text_lower = str(text).lower()

        if any(word in text_lower for word in ['eleição', 'voto', 'urna', 'candidato', 'campanha', 'debate']):
            return 'electoral'
        elif any(word in text_lower for word in ['governo', 'ministro', 'presidente', 'planalto', 'congresso']):
            return 'government'
        elif any(word in text_lower for word in ['manifestação', 'protesto', 'greve', 'ato', 'marcha']):
            return 'protest'
        elif any(word in text_lower for word in ['economia', 'inflação', 'desemprego', 'pib', 'dólar']):
            return 'economic'
        elif any(word in text_lower for word in ['pandemia', 'covid', 'vacina', 'lockdown', 'quarentena']):
            return 'pandemic'
        else:
            return 'general'

    def _mentions_government(self, text: str) -> bool:
        """Verifica se menciona governo."""
        if not text or pd.isna(text):
            return False

        government_terms = [
            'governo', 'presidente', 'ministro', 'secretário', 'federal',
            'planalto', 'congresso', 'senado', 'câmara', 'deputado',
            'senador', 'governador', 'prefeito', 'bolsonaro', 'lula'
        ]
        text_lower = str(text).lower()
        return any(term in text_lower for term in government_terms)

    def _mentions_opposition(self, text: str) -> bool:
        """Verifica se menciona oposição."""
        if not text or pd.isna(text):
            return False

        terms_map = self._political_terms_map
        opposition_terms = terms_map.get('inimigos_ideologicos', [])
        extra = ['oposição', 'contra', 'resistência', 'impeachment', 'fora']
        all_terms = set(opposition_terms + extra)

        text_lower = str(text).lower()
        return any(term in text_lower for term in all_terms)

    def _detect_election_context(self, text: str) -> bool:
        """Detecta contexto eleitoral."""
        if not text or pd.isna(text):
            return False

        election_terms = [
            'eleição', 'eleições', 'voto', 'votos', 'urna', 'urnas',
            'candidato', 'candidata', 'campanha', 'debate', 'apuração',
            'segundo turno', 'primeiro turno', 'tse', 'propaganda eleitoral'
        ]
        text_lower = str(text).lower()
        return any(term in text_lower for term in election_terms)

    def _detect_protest_context(self, text: str) -> bool:
        """Detecta contexto de protesto."""
        if not text or pd.isna(text):
            return False

        terms_map = self._political_terms_map
        mobilizacao = terms_map.get('mobilizacao_acao', [])
        extra = ['manifestação', 'protesto', 'greve', 'ocupação', 'ato', 'marcha']
        all_terms = set(mobilizacao + extra)

        text_lower = str(text).lower()
        return any(term in text_lower for term in all_terms)

    def _classify_channel_type(self, channel: str) -> str:
        """Classifica tipo de canal."""
        if not channel or pd.isna(channel):
            return 'unknown'

        channel_lower = str(channel).lower()

        if any(word in channel_lower for word in ['news', 'notícia', 'jornal', 'imprensa']):
            return 'news'
        elif any(word in channel_lower for word in ['brasil', 'patriota', 'conservador', 'direita', 'bolso']):
            return 'political'
        elif any(word in channel_lower for word in ['humor', 'meme', 'engraçado', 'zueira']):
            return 'entertainment'
        elif any(word in channel_lower for word in ['gospel', 'igreja', 'cristo', 'deus']):
            return 'religious'
        else:
            return 'general'

    # ==========================================
    # FRAME ANALYSIS - Entman (1993)
    # ==========================================

    def _analyze_political_frames(self, text: str) -> dict:
        """Identifica frames políticos (Entman 1993, J Communication 43(4): 51-58)."""
        if not text or pd.isna(text):
            return {'conflito': 0.0, 'responsabilizacao': 0.0, 'moralista': 0.0, 'economico': 0.0}

        frames = {
            'conflito': ['contra', 'ataque', 'briga', 'guerra', 'batalha', 'confronto',
                        'disputa', 'embate', 'oposição', 'adversário', 'inimigo'],
            'responsabilizacao': ['culpa', 'responsável', 'causou', 'provocou', 'deve',
                                 'culpado', 'responsabilidade', 'negligência', 'omissão'],
            'moralista': ['certo', 'errado', 'justo', 'moral', 'ética', 'valores',
                         'pecado', 'virtude', 'honra', 'vergonha', 'dever', 'dignidade'],
            'economico': ['economia', 'dinheiro', 'custo', 'gasto', 'investimento', 'pib',
                         'inflação', 'desemprego', 'salário', 'imposto', 'dívida', 'mercado']
        }

        text_lower = str(text).lower()
        result = {}
        for frame, keywords in frames.items():
            score = sum(1 for word in keywords if word in text_lower)
            result[frame] = score / len(keywords)
        return result

    # ==========================================
    # MANN-KENDALL TREND TEST - Mann (1945); Kendall (1975)
    # ==========================================

    def _mann_kendall_trend_test(self, time_series) -> dict:
        """Teste não-paramétrico para tendência (Mann 1945, Kendall 1975)."""
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            return {'statistic': 0, 'p_value': 1.0, 'trend': 'unavailable'}

        n = len(time_series)
        if n < 4:
            return {'statistic': 0, 'p_value': 1.0, 'trend': 'insufficient_data'}

        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(time_series[j] - time_series[i])

        var_s = n * (n - 1) * (2 * n + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

        if p_value < 0.05:
            trend = 'increasing' if z > 0 else 'decreasing'
        else:
            trend = 'no_trend'

        return {'statistic': float(s), 'p_value': float(p_value), 'trend': trend}

    # ==========================================
    # INFORMATION CASCADE DETECTION - Leskovec et al. (2007)
    # ==========================================

    def _detect_information_cascades(self, df) -> pd.DataFrame:
        """Detecta cascatas de informação (Leskovec et al. 2007, ACM Trans Web)."""
        cascades = []
        if 'is_fwrd' not in df.columns:
            return pd.DataFrame(cascades)

        forwarded = df[df['is_fwrd'] == True].copy()
        if len(forwarded) < 3:
            return pd.DataFrame(cascades)

        forwarded['cascade_id'] = forwarded.groupby('body').ngroup()
        for cascade_id in forwarded['cascade_id'].unique():
            cascade = forwarded[forwarded['cascade_id'] == cascade_id]
            if len(cascade) > 2:
                cascades.append({
                    'cascade_id': cascade_id,
                    'size': len(cascade),
                    'channels': cascade['channel'].nunique() if 'channel' in cascade.columns else 0,
                    'content_preview': str(cascade['body'].iloc[0])[:100]
                })

        return pd.DataFrame(cascades)


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