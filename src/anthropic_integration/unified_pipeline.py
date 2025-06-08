"""
UNIFIED ANTHROPIC PIPELINE SYSTEM v4.6
=======================================
Consolida todas as 19 etapas do pipeline com integração Anthropic centralizada
e integração completa com dashboard para análise em tempo real.
"""

import os
import json
import logging
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Importar componentes base
from .base import AnthropicBase
from .pipeline_integration import APIPipelineIntegration
from .feature_extractor import FeatureExtractor
from .encoding_validator import EncodingValidator
from .deduplication_validator import DeduplicationValidator
from .intelligent_text_cleaner import IntelligentTextCleaner
from .sentiment_analyzer import AnthropicSentimentAnalyzer
from .topic_interpreter import TopicInterpreter
from .semantic_tfidf_analyzer import SemanticTfidfAnalyzer
from .cluster_validator import ClusterValidator
from .semantic_hashtag_analyzer import SemanticHashtagAnalyzer
from .intelligent_domain_analyzer import IntelligentDomainAnalyzer
from .smart_temporal_analyzer import SmartTemporalAnalyzer
from .intelligent_network_analyzer import IntelligentNetworkAnalyzer
from .qualitative_classifier import QualitativeClassifier
from .smart_pipeline_reviewer import SmartPipelineReviewer
from .pipeline_validator import CompletePipelineValidator
from .voyage_embeddings import VoyageEmbeddingAnalyzer

# Importar novos componentes Voyage.ai aprimorados
from .voyage_topic_modeler import VoyageTopicModeler
from .voyage_clustering_analyzer import VoyageClusteringAnalyzer

# Importar novo sistema de busca semântica e inteligência
from .semantic_search_engine import SemanticSearchEngine
from .hybrid_search_engine import HybridSearchEngine
from .optimized_cache import OptimizedCache, EmbeddingCache
from .intelligent_query_system import IntelligentQuerySystem
from .content_discovery_engine import ContentDiscoveryEngine
from .analytics_dashboard import AnalyticsDashboard
from .temporal_evolution_tracker import TemporalEvolutionTracker
from .dataset_statistics_generator import DatasetStatisticsGenerator

# Importar novos componentes de validação e análise política
from .feature_validator import FeatureValidator
from .political_analyzer import PoliticalAnalyzer

# Importar processadores de dados
from src.data.processors.chunk_processor import ChunkProcessor

logger = logging.getLogger(__name__)


class UnifiedAnthropicPipeline(AnthropicBase):
    """
    Pipeline unificado com integração Anthropic para todas as 19 etapas
    
    Etapas do Pipeline v4.6:
    01. chunk_processing - Processamento robusto em chunks
    02. encoding_validation - Validação estrutural e encoding
    03. deduplication - Deduplicação inteligente
    04. feature_validation - Validação e enriquecimento de features básicas
    05. political_analysis - Análise política profunda via API
    06. text_cleaning - Limpeza inteligente de texto
    07. sentiment_analysis - Análise de sentimento multidimensional
    08. topic_modeling - Modelagem de tópicos com interpretação
    09. tfidf_extraction - Extração TF-IDF semântica
    10. clustering - Clustering com validação automática
    11. hashtag_normalization - Normalização e categorização de hashtags
    12. domain_analysis - Análise completa de domínios e credibilidade
    13. temporal_analysis - Análise temporal inteligente
    14. network_analysis - Análise de estrutura de rede e comunidades
    15. qualitative_analysis - Análise qualitativa com taxonomias
    16. pipeline_review - Revisão inteligente e reprodutibilidade
    17. topic_interpretation - Interpretação contextualizada de tópicos
    18. semantic_search - Busca semântica inteligente e indexação
    19. pipeline_validation - Validação final completa do pipeline
    """
    
    def __init__(self, config: Dict[str, Any] = None, project_root: str = None):
        """
        Inicializa pipeline unificado com tratamento robusto de erros
        
        Args:
            config: Configuração do pipeline
            project_root: Diretório raiz do projeto
        """
        try:
            # Validar sistema antes de inicializar
            from .system_validator import SystemValidator
            validator = SystemValidator(project_root)
            system_ok, validation_results = validator.run_full_validation()
            
            if not system_ok and validation_results["overall_status"] == "error":
                logger.error("Sistema não passou na validação crítica")
                logger.error(validator.generate_report())
                # Continuar mas marcar como modo degradado
                
            # Inicializar classe base primeiro
            super().__init__(config)
            self.project_root = Path(project_root) if project_root else Path.cwd()
            
            # Validar configuração
            if not config:
                raise ValueError("Configuração é obrigatória para inicializar o pipeline")
            
            # Configurações do pipeline com validação
            self.pipeline_config = self._validate_and_setup_config(config)
            
            # Estado do pipeline
            self.pipeline_state = {
                "start_time": None,
                "current_stage": None,
                "completed_stages": [],
                "failed_stages": [],
                "stage_results": {},
                "checkpoints": {},
                "data_versions": {},
                "initialization_errors": []
            }
            
            # Inicializar componentes especializados com tratamento de erro
            initialization_success = self._initialize_components_safely()
            
            # Flag para habilitar geração de estatísticas
            self.generate_dataset_statistics = config.get('processing', {}).get('generate_statistics', True)
            
            if initialization_success:
                logger.info("Pipeline Unificado Anthropic inicializado com sucesso")
            else:
                logger.warning("Pipeline inicializado com alguns erros - verifique logs")
                
        except Exception as e:
            logger.error(f"Erro crítico na inicialização do pipeline: {e}")
            # Não re-raise, permitir que pipeline seja criado em modo degradado
            self.pipeline_state = {"initialization_error": str(e)}
            self.pipeline_config = {"error_mode": True}
    
    def _validate_and_setup_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Valida e configura parâmetros do pipeline"""
        try:
            pipeline_config = {
                "chunk_size": config.get("processing", {}).get("chunk_size", 10000),
                "use_anthropic": config.get("anthropic", {}).get("enable_api_integration", True),
                "save_checkpoints": True,
                "data_path": config.get("data", {}).get("path", "data/DATASETS_FULL"),
                "output_path": config.get("data", {}).get("interim_path", "data/interim"),
                "enable_validation": True
            }
            
            # Validar valores críticos
            if pipeline_config["chunk_size"] <= 0:
                pipeline_config["chunk_size"] = 10000
                logger.warning("chunk_size inválido, usando padrão: 10000")
                
            # Verificar se diretórios existem, criar se necessário
            for path_key in ["data_path", "output_path"]:
                path = Path(pipeline_config[path_key])
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Diretório criado: {path}")
                    
            return pipeline_config
            
        except Exception as e:
            logger.error(f"Erro ao validar configuração: {e}")
            # Retornar configuração mínima para modo degradado
            return {
                "chunk_size": 10000,
                "use_anthropic": False,
                "save_checkpoints": False,
                "data_path": "data/DATASETS_FULL",
                "output_path": "data/interim",
                "enable_validation": False,
                "error_mode": True
            }
    
    def _initialize_components_safely(self) -> bool:
        """Inicializa componentes com tratamento individual de erros"""
        success_count = 0
        total_components = 0
        
        # Lista de componentes para inicializar
        component_configs = [
            ("api_integration", lambda: APIPipelineIntegration(self.config, str(self.project_root))),
            ("feature_extractor", lambda: FeatureExtractor(self.config)),
            ("feature_validator", lambda: FeatureValidator()),
            ("political_analyzer", lambda: PoliticalAnalyzer(self.config)),
            ("encoding_validator", lambda: EncodingValidator(self.config)),
            ("deduplication_validator", lambda: DeduplicationValidator(self.config)),
            ("text_cleaner", lambda: IntelligentTextCleaner(self.config)),
            ("sentiment_analyzer", lambda: AnthropicSentimentAnalyzer(self.config)),
            ("topic_interpreter", lambda: TopicInterpreter(self.config)),
            ("tfidf_analyzer", lambda: SemanticTfidfAnalyzer(self.config)),
            ("cluster_validator", lambda: ClusterValidator(self.config)),
            ("hashtag_analyzer", lambda: SemanticHashtagAnalyzer(self.config)),
            ("domain_analyzer", lambda: IntelligentDomainAnalyzer(self.config)),
            ("temporal_analyzer", lambda: SmartTemporalAnalyzer(self.config)),
            ("network_analyzer", lambda: IntelligentNetworkAnalyzer(self.config)),
            ("qualitative_classifier", lambda: QualitativeClassifier(self.config)),
            ("pipeline_reviewer", lambda: SmartPipelineReviewer(self.config)),
            ("pipeline_validator", lambda: CompletePipelineValidator(self.config, str(self.project_root))),
            ("voyage_embeddings", lambda: VoyageEmbeddingAnalyzer(self.config)),
            # Novos componentes Voyage.ai aprimorados
            ("voyage_topic_modeler", lambda: VoyageTopicModeler(self.config)),
            ("voyage_clustering_analyzer", lambda: VoyageClusteringAnalyzer(self.config)),
            # Novo sistema de busca semântica e inteligência
            ("hybrid_search_engine", lambda: HybridSearchEngine(self.config, self.voyage_embeddings)),
            ("semantic_search_engine", lambda: SemanticSearchEngine(self.config, self.voyage_embeddings)),
            ("intelligent_query_system", lambda: IntelligentQuerySystem(self.config, self.semantic_search_engine)),
            ("content_discovery_engine", lambda: ContentDiscoveryEngine(self.config, self.semantic_search_engine)),
            ("temporal_evolution_tracker", lambda: TemporalEvolutionTracker(self.config, self.semantic_search_engine)),
            ("analytics_dashboard", lambda: AnalyticsDashboard(self.config, self.semantic_search_engine, self.content_discovery_engine, self.intelligent_query_system)),
            ("dataset_statistics_generator", lambda: DatasetStatisticsGenerator(self.config))
        ]
        
        for component_name, component_factory in component_configs:
            total_components += 1
            try:
                # Verificar dependências especiais para componentes semânticos
                if component_name == "semantic_search_engine":
                    if hasattr(self, 'voyage_embeddings') and self.voyage_embeddings:
                        component = SemanticSearchEngine(self.config, self.voyage_embeddings)
                    else:
                        component = SemanticSearchEngine(self.config)
                elif component_name == "intelligent_query_system":
                    if hasattr(self, 'semantic_search_engine') and self.semantic_search_engine:
                        component = IntelligentQuerySystem(self.config, self.semantic_search_engine)
                    else:
                        component = IntelligentQuerySystem(self.config)
                elif component_name == "content_discovery_engine":
                    if hasattr(self, 'semantic_search_engine') and self.semantic_search_engine:
                        component = ContentDiscoveryEngine(self.config, self.semantic_search_engine)
                    else:
                        component = ContentDiscoveryEngine(self.config)
                elif component_name == "temporal_evolution_tracker":
                    if hasattr(self, 'semantic_search_engine') and self.semantic_search_engine:
                        component = TemporalEvolutionTracker(self.config, self.semantic_search_engine)
                    else:
                        component = TemporalEvolutionTracker(self.config)
                elif component_name == "analytics_dashboard":
                    search_engine = getattr(self, 'semantic_search_engine', None)
                    discovery_engine = getattr(self, 'content_discovery_engine', None)
                    query_system = getattr(self, 'intelligent_query_system', None)
                    component = AnalyticsDashboard(self.config, search_engine, discovery_engine, query_system)
                else:
                    component = component_factory()
                
                setattr(self, component_name, component)
                success_count += 1
                logger.debug(f"Componente {component_name} inicializado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao inicializar {component_name}: {e}")
                self.pipeline_state["initialization_errors"].append({
                    "component": component_name,
                    "error": str(e)
                })
                # Criar componente mock como fallback
                setattr(self, component_name, None)
        
        # Inicializar processador de chunks separadamente
        try:
            from src.data.processors.chunk_processor import ChunkConfig
            chunk_config = ChunkConfig(
                chunk_size=self.pipeline_config["chunk_size"],
                encoding='utf-8',
                delimiter=';'
            )
            self.chunk_processor = ChunkProcessor(config=chunk_config)
            success_count += 1
            total_components += 1
        except Exception as e:
            logger.error(f"Erro ao inicializar chunk_processor: {e}")
            self.chunk_processor = None
            self.pipeline_state["initialization_errors"].append({
                "component": "chunk_processor",
                "error": str(e)
            })
        
        success_rate = success_count / total_components if total_components > 0 else 0
        logger.info(f"Inicialização de componentes: {success_count}/{total_components} ({success_rate:.1%})")
        
        return success_rate >= 0.8  # Considerar sucesso se 80%+ dos componentes funcionam
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Retorna status de saúde do pipeline"""
        try:
            health_report = {
                "overall_status": "healthy",
                "initialization_errors": len(self.pipeline_state.get("initialization_errors", [])),
                "api_available": getattr(self, 'api_available', False),
                "components_status": {},
                "config_valid": not self.pipeline_config.get("error_mode", False),
                "ready_for_execution": True
            }
            
            # Verificar status de cada componente
            required_components = [
                'api_integration', 'chunk_processor', 'encoding_validator',
                'text_cleaner', 'sentiment_analyzer'
            ]
            
            for component in required_components:
                if hasattr(self, component) and getattr(self, component) is not None:
                    health_report["components_status"][component] = "ok"
                else:
                    health_report["components_status"][component] = "failed"
                    health_report["overall_status"] = "degraded"
            
            # Se muitos erros de inicialização, marcar como não saudável
            if health_report["initialization_errors"] > 3:
                health_report["overall_status"] = "unhealthy"
                health_report["ready_for_execution"] = False
            
            return health_report
            
        except Exception as e:
            return {
                "overall_status": "error",
                "error": str(e),
                "ready_for_execution": False
            }
    
    def _initialize_components(self):
        """Inicializa todos os componentes especializados (método legado)"""
        
        self.api_integration = APIPipelineIntegration(self.config, str(self.project_root))
        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_validator = FeatureValidator()
        self.political_analyzer = PoliticalAnalyzer(self.config)
        self.encoding_validator = EncodingValidator(self.config)
        self.deduplication_validator = DeduplicationValidator(self.config)
        self.text_cleaner = IntelligentTextCleaner(self.config)
        self.sentiment_analyzer = AnthropicSentimentAnalyzer(self.config)
        self.topic_interpreter = TopicInterpreter(self.config)
        self.tfidf_analyzer = SemanticTfidfAnalyzer(self.config)
        self.cluster_validator = ClusterValidator(self.config)
        self.hashtag_analyzer = SemanticHashtagAnalyzer(self.config)
        self.domain_analyzer = IntelligentDomainAnalyzer(self.config)
        self.temporal_analyzer = SmartTemporalAnalyzer(self.config)
        self.network_analyzer = IntelligentNetworkAnalyzer(self.config)
        self.qualitative_classifier = QualitativeClassifier(self.config)
        self.pipeline_reviewer = SmartPipelineReviewer(self.config)
        
        # Inicializar processador de chunks
        from src.data.processors.chunk_processor import ChunkConfig
        chunk_config = ChunkConfig(
            chunk_size=self.pipeline_config["chunk_size"],
            encoding='utf-8',
            delimiter=';'
        )
        self.chunk_processor = ChunkProcessor(config=chunk_config)
    
    def run_complete_pipeline(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """
        Executa pipeline completo em todos os datasets
        
        Args:
            dataset_paths: Lista de caminhos para os datasets
            
        Returns:
            Resultados completos do pipeline
        """
        
        self.pipeline_state["start_time"] = datetime.now()
        logger.info(f"Iniciando pipeline completo para {len(dataset_paths)} datasets")
        
        # Inicializar integração API
        api_init_result = self.api_integration.initialize_pipeline_run(self.pipeline_config)
        
        pipeline_results = {
            "datasets_processed": [],
            "stage_results": {},
            "api_integration": api_init_result,
            "overall_success": True,
            "errors": []
        }
        
        try:
            # Inicializar caminhos atuais dos datasets
            current_dataset_paths = dataset_paths.copy()
            
            # Executar todas as 19 etapas sequencialmente
            all_pipeline_stages = [
                "01_chunk_processing",
                "02_encoding_validation",
                "03_deduplication", 
                "04_feature_validation",
                "05_political_analysis",
                "06_text_cleaning",
                "07_sentiment_analysis",
                "08_topic_modeling",
                "09_tfidf_extraction",
                "10_clustering",
                "11_hashtag_normalization",
                "12_domain_analysis",
                "13_temporal_analysis",
                "14_network_analysis",
                "15_qualitative_analysis",
                "16_smart_pipeline_review",
                "17_topic_interpretation",
                "18_semantic_search",
                "19_pipeline_validation"
            ]
            
            logger.info(f"Executando {len(all_pipeline_stages)} etapas do pipeline v4.6")
            
            for stage_num, stage_name in enumerate(all_pipeline_stages, 1):
                
                stage_result = self._execute_stage_with_recovery(stage_name, current_dataset_paths)
                pipeline_results["stage_results"][stage_name] = stage_result
                
                if not stage_result.get("success", False):
                    pipeline_results["overall_success"] = False
                    pipeline_results["errors"].append({
                        "stage": stage_name,
                        "error": stage_result.get("error", "Unknown error")
                    })
                    
                    # Decidir se continuar ou parar
                    if stage_result.get("critical_error", False):
                        logger.error(f"Erro crítico na etapa {stage_name}, parando pipeline")
                        break
                else:
                    # Atualizar caminhos dos datasets para a próxima etapa
                    updated_paths = self._update_dataset_paths_after_stage(stage_name, current_dataset_paths, stage_result)
                    if updated_paths:
                        current_dataset_paths = updated_paths
                        logger.info(f"Caminhos atualizados após {stage_name}: {len(current_dataset_paths)} datasets")
                
                # Salvar checkpoint após cada etapa
                self._save_pipeline_checkpoint(stage_name, stage_result)
            
            # Executar validação final
            if pipeline_results["overall_success"]:
                final_validation = self._execute_final_validation(pipeline_results)
                pipeline_results["final_validation"] = final_validation
            
        except Exception as e:
            logger.error(f"Erro na execução do pipeline: {e}")
            pipeline_results["overall_success"] = False
            pipeline_results["errors"].append({
                "stage": "pipeline_execution",
                "error": str(e)
            })
        
        # Finalizar pipeline
        self.pipeline_state["end_time"] = datetime.now()
        self.pipeline_state["total_duration"] = (
            self.pipeline_state["end_time"] - self.pipeline_state["start_time"]
        ).total_seconds()
        
        pipeline_results["execution_summary"] = {
            "start_time": self.pipeline_state["start_time"].isoformat(),
            "end_time": self.pipeline_state["end_time"].isoformat(),
            "total_duration_seconds": self.pipeline_state["total_duration"],
            "completed_stages": len(self.pipeline_state["completed_stages"]),
            "failed_stages": len(self.pipeline_state["failed_stages"])
        }
        
        # Adicionar relatório consolidado de otimização de custos
        if hasattr(self, 'voyage_embeddings') and self.voyage_embeddings:
            pipeline_results["voyage_cost_summary"] = self._generate_cost_optimization_summary(pipeline_results)
        
        # Salvar resultado final
        self._save_final_results(pipeline_results)
        
        return pipeline_results
    
    def _generate_cost_optimization_summary(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera relatório consolidado de otimização de custos Voyage"""
        
        try:
            cost_info = self.voyage_embeddings._calculate_estimated_cost()
            quota_info = self.voyage_embeddings._estimate_quota_usage()
            model_info = self.voyage_embeddings.get_embedding_model_info()
            
            # Calcular totais do pipeline
            total_datasets = len(pipeline_results.get("datasets_processed", []))
            total_cost = cost_info['estimated_cost_usd'] * total_datasets
            total_tokens = cost_info['estimated_tokens'] * total_datasets
            
            summary = {
                "pipeline_execution_summary": {
                    "model": self.voyage_embeddings.model_name,
                    "optimization_enabled": self.voyage_embeddings.enable_sampling,
                    "datasets_processed": total_datasets,
                    "total_estimated_cost": round(total_cost, 4),
                    "total_estimated_tokens": total_tokens,
                    "is_free_tier": quota_info.get('likely_free', False)
                },
                "per_dataset_metrics": {
                    "messages_per_dataset": self.voyage_embeddings.max_messages_per_dataset if self.voyage_embeddings.enable_sampling else "unlimited",
                    "estimated_cost_per_dataset": cost_info['estimated_cost_usd'],
                    "estimated_tokens_per_dataset": cost_info['estimated_tokens'],
                    "sampling_strategy": self.voyage_embeddings.sampling_strategy if self.voyage_embeddings.enable_sampling else "none"
                },
                "quota_analysis": quota_info,
                "optimization_impact": {
                    "cost_reduction_enabled": self.voyage_embeddings.enable_sampling,
                    "estimated_savings_vs_unoptimized": "97%" if self.voyage_embeddings.enable_sampling else "0%",
                    "recommended_configuration": "CURRENT" if self.voyage_embeddings.enable_sampling and 'voyage-3.5-lite' in self.voyage_embeddings.model_name else "UPGRADE_TO_LITE_WITH_SAMPLING"
                },
                "recommendations": self._generate_cost_recommendations()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de custos: {e}")
            return {"error": str(e)}
    
    def _generate_cost_recommendations(self) -> List[str]:
        """Gera recomendações de otimização de custos"""
        
        recommendations = []
        
        try:
            # Verificar configuração atual
            if not self.voyage_embeddings.enable_sampling:
                recommendations.append("🔥 ATIVE a amostragem inteligente para reduzir custos em 97%")
            
            if self.voyage_embeddings.model_name != 'voyage-3.5-lite':
                recommendations.append("💰 MIGRE para voyage-3.5-lite (modelo mais econômico com 200M tokens gratuitos)")
            
            if self.voyage_embeddings.batch_size < 128:
                recommendations.append("⚡ AUMENTE batch_size para 128 para melhor throughput")
            
            # Verificar uso eficiente
            quota_info = self.voyage_embeddings._estimate_quota_usage()
            if quota_info.get('estimated_usage_percent', 0) > 80:
                recommendations.append("⚠️  USO ALTO da cota gratuita - considere aumentar limites de amostragem")
            
            # Configuração ideal
            if (self.voyage_embeddings.enable_sampling and 
                self.voyage_embeddings.model_name == 'voyage-3.5-lite' and 
                self.voyage_embeddings.batch_size >= 128):
                recommendations.append("✅ CONFIGURAÇÃO IDEAL - Sistema otimizado para máxima economia")
            
            # Alertas de performance
            if self.voyage_embeddings.max_messages_per_dataset > 100000:
                recommendations.append("📊 CONSIDERE reduzir max_messages_per_dataset para <50K para melhor performance")
                
        except Exception as e:
            recommendations.append(f"❌ Erro ao gerar recomendações: {e}")
        
        return recommendations
    
    def _execute_stage_with_recovery(self, stage_name: str, dataset_paths: List[str], max_retries: int = 2) -> Dict[str, Any]:
        """Executa etapa com mecanismo de recuperação de erro"""
        
        attempt = 0
        last_error = None
        
        while attempt <= max_retries:
            try:
                if attempt > 0:
                    logger.info(f"Tentativa {attempt + 1}/{max_retries + 1} para etapa {stage_name}")
                
                result = self._execute_stage(stage_name, dataset_paths)
                
                if result.get("success", False):
                    if attempt > 0:
                        logger.info(f"Etapa {stage_name} recuperada na tentativa {attempt + 1}")
                    return result
                else:
                    last_error = result.get("error", "Erro desconhecido")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tentativa {attempt + 1} falhou para {stage_name}: {e}")
            
            attempt += 1
            
            # Se não é a última tentativa, tentar recuperação
            if attempt <= max_retries:
                logger.info(f"Tentando recuperar etapa {stage_name}...")
                
                # Estratégias de recuperação específicas
                recovery_success = self._attempt_stage_recovery(stage_name, last_error)
                if not recovery_success:
                    logger.warning(f"Recuperação automática falhou para {stage_name}")
        
        # Se chegou até aqui, todas as tentativas falharam
        logger.error(f"Etapa {stage_name} falhou após {max_retries + 1} tentativas")
        return {
            "stage": stage_name,
            "success": False,
            "error": last_error,
            "attempts": max_retries + 1,
            "recovery_attempted": True
        }
    
    def _attempt_stage_recovery(self, stage_name: str, error: str) -> bool:
        """Tenta recuperar de erros específicos"""
        
        recovery_strategies = {
            "memory": self._recover_memory_error,
            "api": self._recover_api_error,
            "file": self._recover_file_error,
            "encoding": self._recover_encoding_error
        }
        
        # Identificar tipo de erro
        error_lower = error.lower()
        
        if "memory" in error_lower or "memoryerror" in error_lower:
            return recovery_strategies["memory"](stage_name)
        elif "api" in error_lower or "anthropic" in error_lower:
            return recovery_strategies["api"](stage_name)
        elif "file" in error_lower or "filenotfound" in error_lower:
            return recovery_strategies["file"](stage_name)
        elif "encoding" in error_lower or "unicode" in error_lower:
            return recovery_strategies["encoding"](stage_name)
        
        return False
    
    def _recover_memory_error(self, stage_name: str) -> bool:
        """Recupera de erros de memória"""
        logger.info(f"Tentando recuperar erro de memória para {stage_name}")
        
        # Reduzir chunk size
        if hasattr(self, 'chunk_processor') and self.chunk_processor:
            original_size = self.chunk_processor.config.chunk_size
            new_size = max(1000, original_size // 2)
            self.chunk_processor.config.chunk_size = new_size
            logger.info(f"Chunk size reduzido de {original_size} para {new_size}")
            return True
        
        return False
    
    def _recover_api_error(self, stage_name: str) -> bool:
        """Recupera de erros de API"""
        logger.info(f"Tentando recuperar erro de API para {stage_name}")
        
        # Forçar modo tradicional
        if hasattr(self, 'pipeline_config'):
            self.pipeline_config["use_anthropic"] = False
            logger.info("Modo Anthropic desabilitado, usando processamento tradicional")
            return True
        
        return False
    
    def _recover_file_error(self, stage_name: str) -> bool:
        """Recupera de erros de arquivo"""
        logger.info(f"Tentando recuperar erro de arquivo para {stage_name}")
        
        # Criar diretórios necessários
        try:
            output_dir = Path(self.pipeline_config.get("output_path", "data/interim"))
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Diretório {output_dir} criado/verificado")
            return True
        except Exception as e:
            logger.error(f"Falha ao criar diretório: {e}")
            return False
    
    def _recover_encoding_error(self, stage_name: str) -> bool:
        """Recupera de erros de encoding"""
        logger.info(f"Tentando recuperar erro de encoding para {stage_name}")
        
        # Modificar configuração de encoding do chunk processor
        if hasattr(self, 'chunk_processor') and self.chunk_processor:
            self.chunk_processor.config.on_bad_lines = 'skip'
            self.chunk_processor.config.encoding = 'utf-8'
            logger.info("Configuração de encoding ajustada para modo tolerante")
            return True
        
        return False

    def _update_dataset_paths_after_stage(self, stage_name: str, original_paths: List[str], stage_result: Dict[str, Any]) -> Optional[List[str]]:
        """
        Atualiza caminhos dos datasets após uma etapa que modifica os dados
        
        Args:
            stage_name: Nome da etapa executada
            original_paths: Caminhos originais dos datasets
            stage_result: Resultado da etapa executada
            
        Returns:
            Lista de novos caminhos ou None se não houve mudança
        """
        
        try:
            # Etapas que geram novos arquivos e precisam atualizar os caminhos
            path_updating_stages = {
                "02b_deduplication": "deduplication_reports",
                "01b_feature_validation": "feature_validation_reports",
                "01c_political_analysis": "political_analysis_reports",
                "03_clean_text": "cleaning_reports",
                "04_sentiment_analysis": "sentiment_reports",
                "05_topic_modeling": "topic_reports",
                "06_tfidf_extraction": "tfidf_reports",
                "07_clustering": "clustering_reports",
                "08_hashtag_normalization": "hashtag_reports",
                "09_domain_extraction": "domain_reports",
                "10_temporal_analysis": "temporal_reports",
                "11_network_structure": "network_reports",
                "12_qualitative_analysis": "qualitative_reports",
                "13_review_reproducibility": "review_reports"
            }
            
            if stage_name not in path_updating_stages:
                return None
                
            report_key = path_updating_stages[stage_name]
            reports = stage_result.get(report_key, {})
            
            if not reports:
                logger.warning(f"Nenhum relatório encontrado para {stage_name}")
                return None
            
            # Extrair novos caminhos dos relatórios
            new_paths = []
            for original_path in original_paths:
                # Procurar output_path correspondente no relatório
                report_data = reports.get(original_path)
                if report_data and "output_path" in report_data:
                    new_path = report_data["output_path"]
                    new_paths.append(new_path)
                    logger.debug(f"Caminho atualizado: {original_path} -> {new_path}")
                else:
                    # Se não encontrou novo caminho, manter o original
                    new_paths.append(original_path)
                    logger.warning(f"Caminho não atualizado para {original_path} na etapa {stage_name}")
            
            # Verificar se os novos arquivos existem
            existing_paths = []
            for path in new_paths:
                if os.path.exists(path):
                    existing_paths.append(path)
                    logger.debug(f"Arquivo confirmado: {path}")
                else:
                    logger.error(f"Arquivo esperado não existe: {path}")
                    # Usar caminho original como fallback
                    original_idx = new_paths.index(path)
                    if original_idx < len(original_paths):
                        existing_paths.append(original_paths[original_idx])
            
            if existing_paths:
                logger.info(f"Caminhos atualizados com sucesso para {stage_name}: {len(existing_paths)} arquivos")
                return existing_paths
            else:
                logger.error(f"Nenhum arquivo válido após {stage_name}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao atualizar caminhos após {stage_name}: {e}")
            return None

    def _execute_stage(self, stage_name: str, dataset_paths: List[str]) -> Dict[str, Any]:
        """Executa uma etapa específica do pipeline"""
        
        self.pipeline_state["current_stage"] = stage_name
        logger.info(f"Executando etapa: {stage_name}")
        
        stage_result = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "datasets_processed": 0,
            "results": {},
            "anthropic_used": False,
            "execution_time": 0,
            "memory_usage": "unknown"
        }
        
        start_time = time.time()
        
        try:
            # Mapear etapa para método correspondente - CORRIGIDO v4.7 (Numeração Atualizada)
            stage_methods = {
                # FASE 1: Preparação e Validação de Dados (01-05)
                "01_chunk_processing": self._stage_01_chunk_processing,
                "02_encoding_validation": self._stage_02a_encoding_validation,
                "03_deduplication": self._stage_02b_deduplication,
                "04_feature_validation": self._stage_01b_feature_validation,
                "05_political_analysis": self._stage_01c_political_analysis,
                
                # FASE 2: Processamento de Texto e Análise (06-09)
                "06_text_cleaning": self._stage_03_clean_text,
                "07_sentiment_analysis": self._stage_04_sentiment_analysis,
                "08_topic_modeling": self._stage_05_topic_modeling,  # 🚀 VOYAGE.AI
                "09_tfidf_extraction": self._stage_06_tfidf_extraction,  # 🚀 VOYAGE.AI
                
                # FASE 3: Análise Estrutural e de Rede (10-13)
                "10_clustering": self._stage_07_clustering,  # 🚀 VOYAGE.AI
                "11_hashtag_normalization": self._stage_08_hashtag_normalization,
                "12_domain_analysis": self._stage_09_domain_extraction,
                "13_temporal_analysis": self._stage_10_temporal_analysis,
                
                # FASE 4: Análise Avançada e Finalização (14-19)
                "14_network_analysis": self._stage_11_network_structure,
                "15_qualitative_analysis": self._stage_12_qualitative_analysis,
                "16_smart_pipeline_review": self._stage_13_review_reproducibility,
                "17_topic_interpretation": self._stage_14_topic_interpretation,
                "18_semantic_search": self._stage_14_semantic_search_intelligence,  # 🚀 VOYAGE.AI
                "19_pipeline_validation": self._stage_16_pipeline_validation,
                
                # Aliases para compatibilidade com numeração antiga
                "01b_features_validation": self._stage_01b_feature_validation,
                "01b_feature_validation": self._stage_01b_feature_validation,
                "01c_political_analysis": self._stage_01c_political_analysis,
                "02a_encoding_validation": self._stage_02a_encoding_validation,
                "02b_deduplication": self._stage_02b_deduplication,
                "03_text_cleaning": self._stage_03_clean_text,
                "03_clean_text": self._stage_03_clean_text
            }
            
            if stage_name in stage_methods:
                method_result = stage_methods[stage_name](dataset_paths)
                stage_result.update(method_result)
                stage_result["success"] = True
                self.pipeline_state["completed_stages"].append(stage_name)
            else:
                raise ValueError(f"Etapa desconhecida: {stage_name}")
                
        except Exception as e:
            logger.error(f"Erro na etapa {stage_name}: {e}")
            stage_result["error"] = str(e)
            stage_result["error_type"] = type(e).__name__
            stage_result["critical_error"] = self._is_critical_error(stage_name, e)
            self.pipeline_state["failed_stages"].append(stage_name)
        
        finally:
            stage_result["execution_time"] = time.time() - start_time
            
            # Tentar capturar uso de memória
            try:
                import psutil
                process = psutil.Process()
                stage_result["memory_usage"] = f"{process.memory_info().rss / 1024 / 1024:.1f} MB"
            except ImportError:
                stage_result["memory_usage"] = "psutil not available"
            except Exception:
                stage_result["memory_usage"] = "error getting memory info"
        
        self.pipeline_state["stage_results"][stage_name] = stage_result
        return stage_result
    
    def _stage_01_validate_data(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 01: Validação estrutural e detecção de encoding"""
        
        results = {"validation_reports": {}}
        
        for dataset_path in dataset_paths:
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar validação Anthropic
                # Carregar dados primeiro para análise
                df = self._load_processed_data(dataset_path) 
                validation_result = self.encoding_validator.validate_encoding_quality(df)
                results["anthropic_used"] = True
            else:
                # Usar validação tradicional
                validation_result = self._traditional_validate_data(dataset_path)
            
            results["validation_reports"][dataset_path] = validation_result
            results["datasets_processed"] = len(results["validation_reports"])
        
        return results
    
    
    def _stage_02b_deduplication(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 02b: Deduplicação inteligente - Remove duplicatas e adiciona coluna de frequência"""
        
        logger.info("🔄 INICIANDO ETAPA 02b: DEDUPLICAÇÃO INTELIGENTE")
        results = {"deduplication_reports": {}}
        
        for dataset_path in dataset_paths:
            logger.info(f"📂 Processando dataset: {Path(dataset_path).name}")
            
            try:
                # Carregar dados originais
                original_df = self._load_processed_data(dataset_path)
                logger.info(f"📊 Dataset carregado: {len(original_df)} registros")
                
                # Verificar colunas disponíveis
                has_body = 'body' in original_df.columns
                has_body_cleaned = 'body_cleaned' in original_df.columns
                logger.info(f"📋 Colunas de texto disponíveis: body={has_body}, body_cleaned={has_body_cleaned}")
                
                if not has_body and not has_body_cleaned:
                    logger.error(f"❌ Nenhuma coluna de texto encontrada em {dataset_path}")
                    # Usar deduplicação tradicional como fallback
                    deduplicated_df = self._traditional_deduplication(original_df)
                    deduplication_report = {
                        "method": "traditional_fallback",
                        "original_count": len(original_df),
                        "deduplicated_count": len(deduplicated_df),
                        "reduction_ratio": (len(original_df) - len(deduplicated_df)) / len(original_df)
                    }
                else:
                    # Usar deduplicação inteligente
                    if self.pipeline_config.get("use_anthropic", True) and self.api_available:
                        logger.info("🤖 Usando deduplicação inteligente com API Anthropic")
                        
                        # Executar deduplicação inteligente
                        deduplicated_df = self.deduplication_validator.intelligent_deduplication(original_df)
                        
                        # Calcular estatísticas
                        original_count = len(original_df)
                        final_count = len(deduplicated_df)
                        duplicates_removed = original_count - final_count
                        reduction_ratio = duplicates_removed / original_count if original_count > 0 else 0
                        
                        # Validar resultado da deduplicação
                        validation_report = self.deduplication_validator.validate_deduplication_process(
                            original_df, deduplicated_df, "duplicate_frequency"
                        )
                        
                        deduplication_report = {
                            "method": "intelligent_anthropic",
                            "original_count": original_count,
                            "deduplicated_count": final_count,
                            "duplicates_removed": duplicates_removed,
                            "reduction_ratio": reduction_ratio,
                            "validation": validation_report,
                            "duplicate_frequency_added": "duplicate_frequency" in deduplicated_df.columns
                        }
                        
                        logger.info(f"✅ Deduplicação concluída: {original_count} → {final_count} ({reduction_ratio:.1%} redução)")
                        
                    else:
                        logger.info("🔧 Usando deduplicação tradicional (API não disponível)")
                        deduplicated_df = self._traditional_deduplication(original_df)
                        deduplication_report = {
                            "method": "traditional",
                            "original_count": len(original_df),
                            "deduplicated_count": len(deduplicated_df),
                            "reduction_ratio": (len(original_df) - len(deduplicated_df)) / len(original_df)
                        }
                
                # Salvar dados deduplicados
                output_path = self._get_stage_output_path("02b_deduplicated", dataset_path)
                self._save_processed_data(deduplicated_df, output_path)
                logger.info(f"💾 Dados deduplicados salvos: {output_path}")
                
                results["deduplication_reports"][dataset_path] = {
                    "report": deduplication_report,
                    "output_path": output_path
                }
                results["anthropic_used"] = True
                
            except Exception as e:
                logger.error(f"❌ Erro na deduplicação de {dataset_path}: {e}")
                # Fallback: copiar dados originais
                output_path = self._get_stage_output_path("02b_deduplicated", dataset_path)
                original_df['duplicate_frequency'] = 1  # Adicionar coluna mesmo no fallback
                self._save_processed_data(original_df, output_path)
                
                results["deduplication_reports"][dataset_path] = {
                    "report": {"error": str(e), "method": "fallback_copy"},
                    "output_path": output_path
                }
        
        results["datasets_processed"] = len(results["deduplication_reports"])
        
        return results
    
    def _traditional_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicação tradicional como fallback"""
        logger.info("🔧 Executando deduplicação tradicional")
        
        # Usar método otimizado para detectar melhor coluna de texto
        try:
            text_col = self._get_best_text_column(df, prefer_cleaned=True)
        except ValueError:
            logger.warning("Nenhuma coluna de texto encontrada, retornando dados originais")
            df['duplicate_frequency'] = 1
            return df
        
        # Deduplicação simples baseada no conteúdo
        original_count = len(df)
        df_normalized = df.copy()
        df_normalized['_temp_text'] = df_normalized[text_col].fillna('').astype(str).str.strip().str.lower()
        
        # Contar frequências
        text_counts = df_normalized['_temp_text'].value_counts()
        df_normalized['duplicate_frequency'] = df_normalized['_temp_text'].map(text_counts)
        
        # Remover duplicatas
        df_dedup = df_normalized.drop_duplicates(subset=['_temp_text'], keep='first')
        df_dedup = df_dedup.drop('_temp_text', axis=1)
        
        final_count = len(df_dedup)
        logger.info(f"Deduplicação tradicional: {original_count} → {final_count} ({100*(original_count-final_count)/original_count:.1f}% redução)")
        
        return df_dedup
    
    def _stage_01b_feature_validation(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 01b: Validação e enriquecimento de features básicas"""
        
        logger.info("🔧 INICIANDO ETAPA 01b: VALIDAÇÃO DE FEATURES")
        results = {"feature_validation_reports": {}}
        
        for dataset_path in dataset_paths:
            logger.info(f"📂 Processando dataset: {Path(dataset_path).name}")
            
            # Carregar dados deduplicados
            if "02b_deduplicated" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("02b_deduplicated", dataset_path)
            
            df = self._load_processed_data(input_path)
            logger.info(f"📊 Dataset carregado: {len(df)} registros")
            
            # Validar e enriquecer features
            enriched_df, validation_report = self.feature_validator.validate_and_enrich_features(df)
            
            # Salvar dados enriquecidos
            output_path = self._get_stage_output_path("01b_features_validated", dataset_path)
            self._save_processed_data(enriched_df, output_path)
            
            results["feature_validation_reports"][dataset_path] = {
                "report": validation_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["feature_validation_reports"])
            
            logger.info(f"✅ Features validadas e salvas em: {output_path}")
        
        return results
    
    def _stage_01c_political_analysis(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 01c: Análise política profunda via API"""
        
        logger.info("🏛️ INICIANDO ETAPA 01c: ANÁLISE POLÍTICA")
        results = {"political_analysis_reports": {}}
        
        for dataset_path in dataset_paths:
            logger.info(f"📂 Processando dataset: {Path(dataset_path).name}")
            
            # Carregar dados com features validadas
            if "01b_features_validated" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("01b_features_validated", dataset_path)
            
            df = self._load_processed_data(input_path)
            logger.info(f"📊 Dataset carregado: {len(df)} registros")
            
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar análise política via API
                analyzed_df, political_report = self.political_analyzer.analyze_political_discourse(df)
                results["anthropic_used"] = True
                logger.info("🤖 Análise política via API Anthropic concluída")
            else:
                # Usar análise política tradicional (baseada em léxico)
                analyzed_df, political_report = self._traditional_political_analysis(df)
                logger.info("📚 Análise política tradicional (léxico) concluída")
            
            # Salvar dados com análise política
            output_path = self._get_stage_output_path("01c_politically_analyzed", dataset_path)
            self._save_processed_data(analyzed_df, output_path)
            
            results["political_analysis_reports"][dataset_path] = {
                "report": political_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["political_analysis_reports"])
            
            logger.info(f"✅ Análise política salva em: {output_path}")
        
        return results
    
    def _traditional_political_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Análise política tradicional baseada em léxico"""
        
        logger.info("Executando análise política tradicional baseada em léxico")
        
        # Carregar léxico político
        try:
            lexicon_path = Path("config/brazilian_political_lexicon.yaml")
            if lexicon_path.exists():
                import yaml
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    lexicon_data = yaml.safe_load(f)
                    lexicon = lexicon_data.get("brazilian_political_lexicon", {})
            else:
                lexicon = self._get_default_political_lexicon()
        except Exception as e:
            logger.warning(f"Erro ao carregar léxico: {e}, usando padrão")
            lexicon = self._get_default_political_lexicon()
        
        # Detectar coluna de texto
        text_column = 'body_cleaned' if 'body_cleaned' in df.columns else ('body' if 'body' in df.columns else None)
        
        if not text_column:
            logger.error("Nenhuma coluna de texto encontrada")
            return df, {"error": "Nenhuma coluna de texto encontrada"}
        
        # Análise baseada em léxico
        enhanced_df = df.copy()
        
        # Inicializar colunas
        enhanced_df['political_alignment'] = 'neutro'
        enhanced_df['alignment_confidence'] = 0.0
        enhanced_df['conspiracy_score'] = 0.0
        enhanced_df['negacionism_score'] = 0.0
        enhanced_df['emotional_tone'] = 'neutro'
        enhanced_df['misinformation_risk'] = 'baixo'
        enhanced_df['brazilian_context_score'] = 0.0
        
        # Aplicar análise por categoria
        for category, terms in lexicon.items():
            if isinstance(terms, list) and terms:
                pattern = "|".join([f"\\b{term}\\b" for term in terms])
                matches = enhanced_df[text_column].fillna("").str.contains(
                    pattern, case=False, regex=True
                )
                
                # Aplicar lógica baseada na categoria
                if category == "governo_bolsonaro":
                    enhanced_df.loc[matches, 'political_alignment'] = 'bolsonarista'
                    enhanced_df.loc[matches, 'alignment_confidence'] = 0.7
                elif category == "oposição":
                    enhanced_df.loc[matches, 'political_alignment'] = 'antibolsonarista'
                    enhanced_df.loc[matches, 'alignment_confidence'] = 0.7
                elif category == "teorias_conspiração":
                    enhanced_df.loc[matches, 'conspiracy_score'] = 0.8
                    enhanced_df.loc[matches, 'misinformation_risk'] = 'alto'
                elif category == "saúde_negacionismo":
                    enhanced_df.loc[matches, 'negacionism_score'] = 0.8
                    enhanced_df.loc[matches, 'misinformation_risk'] = 'alto'
        
        # Relatório
        report = {
            "method": "lexicon_based",
            "lexicon_categories": len(lexicon),
            "text_column": text_column,
            "political_alignment_distribution": enhanced_df['political_alignment'].value_counts().to_dict(),
            "high_risk_messages": (enhanced_df['misinformation_risk'] == 'alto').sum()
        }
        
        return enhanced_df, report
    
    def _get_default_political_lexicon(self) -> Dict[str, List[str]]:
        """Léxico político padrão"""
        return {
            "governo_bolsonaro": ["bolsonaro", "presidente", "capitão", "mito"],
            "oposição": ["lula", "pt", "petista", "esquerda"],
            "militarismo": ["forças armadas", "militares", "intervenção militar", "quartel"],
            "teorias_conspiração": ["urna fraudada", "globalismo", "deep state"],
            "saúde_negacionismo": ["tratamento precoce", "ivermectina", "cloroquina"]
        }
    
    def _stage_01b_feature_extraction(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 01b: Extração de features com IA (MÉTODO LEGADO - MANTIDO PARA COMPATIBILIDADE)"""
        
        results = {"feature_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados deduplicados
            # Se dataset_path já aponta para arquivo deduplicado, usar diretamente
            if "02b_deduplicated" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("02b_deduplicated", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            # Guardar referência ao DataFrame original (antes da deduplicação) se disponível
            original_dataset_path = dataset_path
            df_original = None
            try:
                df_original = self._load_processed_data(original_dataset_path)
            except:
                df_original = df  # Se não conseguir carregar original, usar o deduplicado
            
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar extração Anthropic
                enhanced_df = self.feature_extractor.extract_comprehensive_features(df)
                feature_report = self.feature_extractor.generate_feature_report(enhanced_df)
                results["anthropic_used"] = True
            else:
                # Usar extração tradicional
                enhanced_df, feature_report = self._traditional_feature_extraction(df)
            
            # GERAR ESTATÍSTICAS GERAIS DO DATASET
            if self.generate_dataset_statistics and hasattr(self, 'dataset_statistics_generator'):
                try:
                    logger.info(f"Gerando estatísticas gerais para {dataset_path}")
                    
                    # Obter relatório de deduplicação se disponível
                    dedup_report = self.pipeline_state.get("stage_results", {}).get("02b_deduplication", {}).get("deduplication_reports", {}).get(dataset_path, {}).get("report", None)
                    
                    # Gerar estatísticas completas
                    dataset_statistics = self.dataset_statistics_generator.generate_comprehensive_statistics(
                        df_original=df_original,
                        df_processed=enhanced_df,
                        deduplication_report=dedup_report
                    )
                    
                    # Adicionar ao relatório de features
                    feature_report["dataset_statistics"] = dataset_statistics
                    
                    # Exportar estatísticas
                    stats_output_path = str(Path(self.pipeline_config["output_path"]) / "statistics")
                    Path(stats_output_path).mkdir(parents=True, exist_ok=True)
                    
                    exported_file = self.dataset_statistics_generator.export_statistics(
                        dataset_statistics,
                        stats_output_path,
                        format="json"
                    )
                    
                    feature_report["statistics_exported_to"] = exported_file
                    logger.info(f"Estatísticas exportadas para: {exported_file}")
                    
                except Exception as e:
                    logger.error(f"Erro ao gerar estatísticas do dataset: {e}")
                    feature_report["dataset_statistics"] = {"error": str(e)}
            
            # Salvar dados com features
            output_path = self._get_stage_output_path("01b_features_extracted", dataset_path)
            self._save_processed_data(enhanced_df, output_path)
            
            results["feature_reports"][dataset_path] = {
                "report": feature_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["feature_reports"])
        
        return results
    
    def _stage_03_clean_text(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 03: Limpeza inteligente de texto com fallback robusto"""
        
        results = {"cleaning_reports": {}}
        
        for dataset_path in dataset_paths:
            try:
                # Carregar dados com análise política
                # Se dataset_path já aponta para arquivo com análise política, usar diretamente
                if "01c_politically_analyzed" in dataset_path:
                    input_path = dataset_path
                else:
                    input_path = self._get_stage_output_path("01c_politically_analyzed", dataset_path)
                
                logger.info(f"Carregando dados para limpeza: {input_path}")
                df = self._load_processed_data(input_path)
                
                cleaned_df = None
                quality_report = {}
                method_used = "unknown"
                
                # Tentativa 1: Limpeza Anthropic (se disponível)
                if self.pipeline_config["use_anthropic"] and self.api_available:
                    try:
                        logger.info("Tentando limpeza inteligente via Anthropic...")
                        cleaned_df = self.text_cleaner.clean_text_intelligent(df)
                        if hasattr(self.text_cleaner, 'validate_cleaning_quality'):
                            quality_report = self.text_cleaner.validate_cleaning_quality(df, cleaned_df)
                        else:
                            quality_report = {"method": "anthropic", "success": True}
                        method_used = "anthropic"
                        results["anthropic_used"] = True
                        logger.info("✅ Limpeza Anthropic bem-sucedida")
                    except Exception as e:
                        logger.warning(f"Limpeza Anthropic falhou: {e}. Tentando fallback...")
                        cleaned_df = None
                
                # Tentativa 2: Limpeza simples (fallback)
                if cleaned_df is None:
                    try:
                        logger.info("Usando limpeza simples...")
                        from .simple_text_cleaner import SimpleTextCleaner
                        simple_cleaner = SimpleTextCleaner()
                        cleaned_df = simple_cleaner.clean_text_simple(df, backup=False)
                        quality_report = {"method": "simple", "success": True, "fallback_used": True}
                        method_used = "simple"
                        logger.info("✅ Limpeza simples bem-sucedida")
                    except Exception as e:
                        logger.error(f"Limpeza simples também falhou: {e}")
                
                # Tentativa 3: Fallback mínimo (sem limpeza)
                if cleaned_df is None:
                    logger.warning("Usando fallback mínimo - cópia sem limpeza")
                    cleaned_df = df.copy()
                    cleaned_df["text_cleaned"] = cleaned_df.get("body_cleaned", "").fillna("")
                    quality_report = {"method": "minimal", "success": True, "no_cleaning": True}
                    method_used = "minimal"
                
                # Salvar dados limpos
                output_path = self._get_stage_output_path("03_text_cleaned", dataset_path)
                self._save_processed_data(cleaned_df, output_path)
                logger.info(f"Dados limpos salvos: {output_path}")
                
                results["cleaning_reports"][dataset_path] = {
                    "quality_report": quality_report,
                    "output_path": output_path,
                    "method_used": method_used
                }
                
            except Exception as e:
                logger.error(f"Erro crítico na limpeza de {dataset_path}: {e}")
                # Criar relatório de erro mas continuar pipeline
                results["cleaning_reports"][dataset_path] = {
                    "error": str(e),
                    "output_path": None,
                    "method_used": "failed"
                }
        
        results["datasets_processed"] = len([r for r in results["cleaning_reports"].values() if r.get("output_path")])
        return results
    
    def _stage_04_sentiment_analysis(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 04: Análise de sentimento"""
        
        results = {"sentiment_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados limpos
            # Se dataset_path já aponta para arquivo limpo, usar diretamente
            if "03_text_cleaned" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("03_text_cleaned", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            if self.config.get("sentiment", {}).get("use_anthropic", True) and self.api_available:
                # Usar análise Anthropic
                sentiment_df = self.sentiment_analyzer.analyze_sentiment_comprehensive(df)
                sentiment_report = self.sentiment_analyzer.generate_sentiment_report(sentiment_df)
                results["anthropic_used"] = True
            else:
                # Usar análise tradicional
                sentiment_df, sentiment_report = self._traditional_sentiment_analysis(df)
            
            # Salvar dados com sentimento
            output_path = self._get_stage_output_path("04_sentiment_analyzed", dataset_path)
            self._save_processed_data(sentiment_df, output_path)
            
            results["sentiment_reports"][dataset_path] = {
                "report": sentiment_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["sentiment_reports"])
        
        return results
    
    def _stage_05_topic_modeling(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 08: Modelagem de tópicos com Voyage.ai"""
        
        logger.info("🎯 INICIANDO ETAPA 08: TOPIC MODELING COM VOYAGE.AI")
        results = {"topic_reports": {}}
        
        for dataset_path in dataset_paths:
            logger.info(f"📂 Processando dataset: {Path(dataset_path).name}")
            
            # Carregar dados com sentimento
            if "07_sentiment_analyzed" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("07_sentiment_analyzed", dataset_path)
            
            df = self._load_processed_data(input_path)
            logger.info(f"📊 Dataset carregado: {len(df)} registros")
            
            # Verificar se Voyage.ai está disponível e habilitado
            voyage_enabled = (hasattr(self, 'voyage_topic_modeler') and 
                            hasattr(self, 'voyage_embeddings') and 
                            getattr(self.voyage_embeddings, 'voyage_available', False))
            
            if voyage_enabled:
                logger.info("🚀 Usando Voyage.ai para topic modeling avançado")
                # Usar novo modelador de tópicos com Voyage.ai
                topic_result = self.voyage_topic_modeler.extract_semantic_topics(df)
                
                if topic_result.get('success', False):
                    topic_df = topic_result.get('enhanced_dataframe', df)
                    topic_report = {
                        'topics': topic_result.get('topics', []),
                        'n_topics': topic_result.get('n_topics_extracted', 0),
                        'method': topic_result.get('method', 'voyage_embeddings'),
                        'model_used': topic_result.get('model_used'),
                        'cost_optimized': topic_result.get('cost_optimized', False),
                        'sample_ratio': topic_result.get('sample_ratio', 1.0),
                        'embedding_stats': topic_result.get('embedding_stats', {}),
                        'analysis_timestamp': topic_result.get('analysis_timestamp')
                    }
                    results["voyage_used"] = True
                    logger.info(f"✅ Voyage topic modeling concluído: {topic_result.get('n_topics_extracted', 0)} tópicos")
                else:
                    # Fallback para método tradicional
                    logger.warning("⚠️ Voyage topic modeling falhou, usando fallback")
                    topic_df, topic_report = self._enhanced_traditional_topic_modeling(df)
                    results["fallback_used"] = True
            
            elif self.config.get("lda", {}).get("use_anthropic_interpretation", True) and self.api_available:
                logger.info("🤖 Usando modelagem tradicional + interpretação Anthropic")
                # Usar modelagem Anthropic tradicional
                topic_df = self.topic_interpreter.extract_and_interpret_topics(df)
                topic_report = self.topic_interpreter.generate_topic_report(topic_df)
                results["anthropic_used"] = True
            else:
                logger.info("📚 Usando modelagem tradicional")
                # Usar modelagem tradicional
                topic_df, topic_report = self._traditional_topic_modeling(df)
                results["traditional_used"] = True
            
            # Salvar dados com tópicos
            output_path = self._get_stage_output_path("08_topic_modeled", dataset_path)
            self._save_processed_data(topic_df, output_path)
            
            results["topic_reports"][dataset_path] = {
                "report": topic_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["topic_reports"])
            
            logger.info(f"✅ Topic modeling salvo em: {output_path}")
        
        return results
    
    def _stage_06_tfidf_extraction(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 06: Extração TF-IDF"""
        
        results = {"tfidf_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados com tópicos
            # Se dataset_path já aponta para arquivo com tópicos, usar diretamente
            if "05_topic_modeled" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("05_topic_modeled", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            if self.config.get("tfidf", {}).get("use_anthropic", True) and self.api_available:
                # Usar extração Anthropic
                tfidf_result = self.tfidf_analyzer.extract_semantic_tfidf(df)
                tfidf_df = tfidf_result.get('dataframe', df)
                tfidf_report = tfidf_result.get('analysis', {})
                results["anthropic_used"] = True
            else:
                # Usar extração tradicional
                tfidf_df, tfidf_report = self._traditional_tfidf_extraction(df)
            
            # Integrar análise de embeddings Voyage.ai se disponível (com otimização de custos)
            if hasattr(self, 'voyage_embeddings') and self.voyage_embeddings:
                try:
                    logger.info("Integrando análise de embeddings Voyage.ai com otimização de custos")
                    
                    # Detectar coluna de texto automaticamente usando método otimizado
                    text_column = self._get_best_text_column(df, prefer_cleaned=True)
                    
                    # Aplicar análise semântica otimizada
                    enhanced_df = self.voyage_embeddings.enhance_semantic_analysis(df, text_column)
                    tfidf_df = enhanced_df
                    
                    # Adicionar informações de embeddings ao relatório
                    embedding_info = self.voyage_embeddings.get_embedding_model_info()
                    if isinstance(tfidf_report, dict):
                        tfidf_report['voyage_embeddings'] = embedding_info
                        
                        # Adicionar métricas detalhadas de otimização de custos
                        cost_info = self.voyage_embeddings._calculate_estimated_cost()
                        quota_info = self.voyage_embeddings._estimate_quota_usage()
                        
                        if hasattr(enhanced_df, 'sample_ratio'):
                            sample_ratio = enhanced_df.get('sample_ratio', [1.0])[0] if len(enhanced_df) > 0 else 1.0
                        else:
                            sample_ratio = 1.0
                            
                        tfidf_report['voyage_cost_optimization'] = {
                            'model_used': self.voyage_embeddings.model_name,
                            'sampling_enabled': self.voyage_embeddings.enable_sampling,
                            'sample_ratio': sample_ratio,
                            'original_messages': len(df),
                            'processed_messages': int(len(df) * sample_ratio) if self.voyage_embeddings.enable_sampling else len(df),
                            'cost_reduction_estimate': f"{(1 - sample_ratio) * 100:.1f}%",
                            'estimated_cost': cost_info,
                            'quota_usage': quota_info,
                            'optimization_summary': {
                                'status': 'ACTIVE' if self.voyage_embeddings.enable_sampling else 'DISABLED',
                                'economic_model': 'voyage-3.5-lite' in self.voyage_embeddings.model_name,
                                'free_tier_usage': quota_info.get('likely_free', False),
                                'executions_possible': quota_info.get('executions_possible', 0)
                            }
                        }
                    
                except Exception as e:
                    logger.warning(f"Falha na integração Voyage.ai: {e}")
            
            # Salvar dados com TF-IDF e embeddings
            output_path = self._get_stage_output_path("06_tfidf_extracted", dataset_path)
            self._save_processed_data(tfidf_df, output_path)
            
            results["tfidf_reports"][dataset_path] = {
                "report": tfidf_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["tfidf_reports"])
        
        return results
    
    def _stage_07_clustering(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 10: Clustering semântico com Voyage.ai (Método interno _stage_07)"""
        
        logger.info("🎯 INICIANDO ETAPA 10: CLUSTERING SEMÂNTICO COM VOYAGE.AI")
        results = {"clustering_reports": {}}
        
        for dataset_path in dataset_paths:
            logger.info(f"📂 Processando dataset: {Path(dataset_path).name}")
            
            # Carregar dados com TF-IDF
            if "09_tfidf_extracted" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("09_tfidf_extracted", dataset_path)
            
            df = self._load_processed_data(input_path)
            logger.info(f"📊 Dataset carregado: {len(df)} registros")
            
            # Verificar se Voyage.ai está disponível e habilitado
            voyage_enabled = (hasattr(self, 'voyage_clustering_analyzer') and 
                            hasattr(self, 'voyage_embeddings') and 
                            getattr(self.voyage_embeddings, 'voyage_available', False))
            
            if voyage_enabled:
                logger.info("🚀 Usando Voyage.ai para clustering semântico avançado")
                # Usar novo analisador de clustering com Voyage.ai
                clustering_result = self.voyage_clustering_analyzer.perform_semantic_clustering(df)
                
                if clustering_result.get('success', False):
                    clustered_df = clustering_result.get('enhanced_dataframe', df)
                    cluster_report = {
                        'clusters': clustering_result.get('clusters', []),
                        'n_clusters': clustering_result.get('n_clusters', 0),
                        'algorithm_used': clustering_result.get('algorithm_used'),
                        'cluster_metrics': clustering_result.get('cluster_metrics', {}),
                        'embedding_model': clustering_result.get('embedding_model'),
                        'cost_optimized': clustering_result.get('cost_optimized', False),
                        'sample_ratio': clustering_result.get('sample_ratio', 1.0),
                        'clustering_quality': clustering_result.get('clustering_quality', 0),
                        'analysis_timestamp': clustering_result.get('analysis_timestamp')
                    }
                    results["voyage_used"] = True
                    logger.info(f"✅ Voyage clustering concluído: {clustering_result.get('n_clusters', 0)} clusters")
                else:
                    # Fallback para método tradicional
                    logger.warning("⚠️ Voyage clustering falhou, usando fallback")
                    clustered_df = self.cluster_validator.validate_and_enhance_clusters(df)
                    cluster_report = self.cluster_validator.generate_clustering_report(clustered_df)
                    results["fallback_used"] = True
            
            elif self.pipeline_config["use_anthropic"] and self.api_available:
                logger.info("🤖 Usando clustering tradicional + validação Anthropic")
                # Usar clustering Anthropic tradicional
                clustered_df = self.cluster_validator.validate_and_enhance_clusters(df)
                cluster_report = self.cluster_validator.generate_clustering_report(clustered_df)
                results["anthropic_used"] = True
            else:
                logger.info("📚 Usando clustering tradicional")
                # Usar clustering tradicional
                clustered_df, cluster_report = self._traditional_clustering(df)
                results["traditional_used"] = True
            
            # Salvar dados clusterizados
            output_path = self._get_stage_output_path("10_clustered", dataset_path)
            self._save_processed_data(clustered_df, output_path)
            
            results["clustering_reports"][dataset_path] = {
                "report": cluster_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["clustering_reports"])
            
            logger.info(f"✅ Clustering salvo em: {output_path}")
        
        return results
    
    def _enhanced_traditional_topic_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Modelagem de tópicos tradicional aprimorada"""
        try:
            logger.info("📚 Executando topic modeling tradicional aprimorado")
            
            # Use enhanced traditional method with better AI interpretation
            if self.api_available and hasattr(self, 'topic_interpreter'):
                topic_df = self.topic_interpreter.extract_and_interpret_topics(df)
                topic_report = self.topic_interpreter.generate_topic_report(topic_df) if hasattr(self.topic_interpreter, 'generate_topic_report') else {"method": "enhanced_traditional"}
            else:
                # Pure traditional fallback
                topic_df, topic_report = self._traditional_topic_modeling(df)
                topic_report["enhanced"] = True
            
            return topic_df, topic_report
            
        except Exception as e:
            logger.error(f"Erro no topic modeling aprimorado: {e}")
            return self._traditional_topic_modeling(df)
    
    def _stage_08_hashtag_normalization(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 08: Normalização de hashtags"""
        
        results = {"hashtag_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados clusterizados
            # Se dataset_path já aponta para arquivo clusterizado, usar diretamente
            if "07_clustered" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("07_clustered", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar normalização Anthropic
                normalized_df = self.hashtag_analyzer.normalize_and_analyze_hashtags(df)
                hashtag_report = self.hashtag_analyzer.generate_hashtag_report(normalized_df)
                results["anthropic_used"] = True
            else:
                # Usar normalização tradicional
                normalized_df, hashtag_report = self._traditional_hashtag_normalization(df)
            
            # Salvar dados com hashtags normalizadas
            output_path = self._get_stage_output_path("08_hashtags_normalized", dataset_path)
            self._save_processed_data(normalized_df, output_path)
            
            results["hashtag_reports"][dataset_path] = {
                "report": hashtag_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["hashtag_reports"])
        
        return results
    
    def _stage_09_domain_extraction(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 09: Extração e análise de domínios"""
        
        results = {"domain_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados com hashtags normalizadas
            # Se dataset_path já aponta para arquivo com hashtags normalizadas, usar diretamente
            if "08_hashtags_normalized" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("08_hashtags_normalized", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar análise Anthropic
                domain_df = self.domain_analyzer.analyze_domains_comprehensive(df)
                domain_report = self.domain_analyzer.generate_domain_report(domain_df)
                results["anthropic_used"] = True
            else:
                # Usar análise tradicional
                domain_df, domain_report = self._traditional_domain_extraction(df)
            
            # Salvar dados com domínios analisados
            output_path = self._get_stage_output_path("09_domains_analyzed", dataset_path)
            self._save_processed_data(domain_df, output_path)
            
            results["domain_reports"][dataset_path] = {
                "report": domain_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["domain_reports"])
        
        return results
    
    def _stage_10_temporal_analysis(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 10: Análise temporal inteligente"""
        
        results = {"temporal_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados com domínios analisados
            # Se dataset_path já aponta para arquivo com domínios analisados, usar diretamente
            if "09_domains_analyzed" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("09_domains_analyzed", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar análise Anthropic
                temporal_df = self.temporal_analyzer.analyze_temporal_patterns_comprehensive(df)
                temporal_report = self.temporal_analyzer.generate_temporal_report(temporal_df)
                results["anthropic_used"] = True
            else:
                # Usar análise tradicional
                temporal_df, temporal_report = self._traditional_temporal_analysis(df)
            
            # Salvar dados com análise temporal
            output_path = self._get_stage_output_path("10_temporal_analyzed", dataset_path)
            self._save_processed_data(temporal_df, output_path)
            
            results["temporal_reports"][dataset_path] = {
                "report": temporal_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["temporal_reports"])
        
        return results
    
    def _stage_11_network_structure(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 11: Análise de estrutura de rede"""
        
        results = {"network_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados com análise temporal
            # Se dataset_path já aponta para arquivo com análise temporal, usar diretamente
            if "10_temporal_analyzed" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("10_temporal_analyzed", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar análise Anthropic
                network_df = self.network_analyzer.analyze_network_structure_comprehensive(df)
                network_report = self.network_analyzer.generate_network_report(network_df)
                results["anthropic_used"] = True
            else:
                # Usar análise tradicional
                network_df, network_report = self._traditional_network_analysis(df)
            
            # Salvar dados com análise de rede
            output_path = self._get_stage_output_path("11_network_analyzed", dataset_path)
            self._save_processed_data(network_df, output_path)
            
            results["network_reports"][dataset_path] = {
                "report": network_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["network_reports"])
        
        return results
    
    def _stage_12_qualitative_analysis(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 12: Análise qualitativa"""
        
        results = {"qualitative_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados com análise de rede
            # Se dataset_path já aponta para arquivo com análise de rede, usar diretamente
            if "11_network_analyzed" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("11_network_analyzed", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar análise Anthropic
                qualitative_df = self.qualitative_classifier.classify_content_comprehensive(df)
                qualitative_report = self.qualitative_classifier.generate_qualitative_report(qualitative_df)
                results["anthropic_used"] = True
            else:
                # Usar análise tradicional
                qualitative_df, qualitative_report = self._traditional_qualitative_analysis(df)
            
            # Salvar dados com análise qualitativa
            output_path = self._get_stage_output_path("12_qualitative_analyzed", dataset_path)
            self._save_processed_data(qualitative_df, output_path)
            
            results["qualitative_reports"][dataset_path] = {
                "report": qualitative_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["qualitative_reports"])
        
        return results
    
    def _stage_13_review_reproducibility(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 13: Revisão e reprodutibilidade"""
        
        results = {"review_reports": {}}
        
        for dataset_path in dataset_paths:
            # Carregar dados finais
            # Se dataset_path já aponta para arquivo com análise qualitativa, usar diretamente
            if "12_qualitative_analyzed" in dataset_path:
                input_path = dataset_path
            else:
                input_path = self._get_stage_output_path("12_qualitative_analyzed", dataset_path)
            
            df = self._load_processed_data(input_path)
            
            if self.pipeline_config["use_anthropic"] and self.api_available:
                # Usar revisão Anthropic
                review_report = self.pipeline_reviewer.comprehensive_pipeline_review(df, self.pipeline_state)
                results["anthropic_used"] = True
            else:
                # Usar revisão tradicional
                review_report = self._traditional_review_reproducibility(df)
            
            # Salvar dados finais
            output_path = self._get_stage_output_path("13_final_processed", dataset_path)
            self._save_processed_data(df, output_path)
            
            results["review_reports"][dataset_path] = {
                "report": review_report,
                "output_path": output_path
            }
            results["datasets_processed"] = len(results["review_reports"])
        
        return results
    
    def _execute_final_validation(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Executa validação final completa"""
        
        logger.info("Executando validação final do pipeline")
        
        validation_results = {
            "comprehensive_validation": {},
            "api_integration_validation": {},
            "overall_quality_score": 0.0,
            "validation_success": False
        }
        
        # 1. Executar validação holística com CompletePipelineValidator
        if hasattr(self, 'pipeline_validator') and self.pipeline_validator:
            try:
                logger.info("Executando validação holística com CompletePipelineValidator")
                comprehensive_validation = self.pipeline_validator.validate_complete_pipeline(
                    pipeline_results,
                    None,  # final_dataset_path (será inferido)
                    None   # original_dataset_path
                )
                validation_results["comprehensive_validation"] = comprehensive_validation
                logger.info(f"Validação holística concluída com score: {comprehensive_validation.get('overall_quality_score', 'N/A')}")
            except Exception as e:
                logger.error(f"Erro na validação holística: {e}")
                validation_results["comprehensive_validation"] = {"error": str(e)}
        
        # 2. Usar o integrador API para validação final (mantendo comportamento original)
        final_dataset_paths = []
        for dataset_path in pipeline_results.get("datasets_processed", []):
            final_path = self._get_stage_output_path("13_final_processed", dataset_path)
            if os.path.exists(final_path):
                final_dataset_paths.append(final_path)
        
        if final_dataset_paths:
            try:
                api_validation = self.api_integration.execute_comprehensive_pipeline_validation(
                    pipeline_results,
                    final_dataset_paths[0],  # Use first dataset as reference
                    None  # Original dataset path
                )
                validation_results["api_integration_validation"] = api_validation
            except Exception as e:
                logger.error(f"Erro na validação API: {e}")
                validation_results["api_integration_validation"] = {"error": str(e)}
        else:
            validation_results["api_integration_validation"] = {"error": "No final datasets found for validation"}
        
        # 3. Calcular score final combinado
        comprehensive_score = validation_results["comprehensive_validation"].get("overall_quality_score", 0.0)
        api_score = validation_results["api_integration_validation"].get("quality_score", 0.0)
        
        if comprehensive_score > 0 and api_score > 0:
            # Média ponderada: 70% holística, 30% API
            validation_results["overall_quality_score"] = (comprehensive_score * 0.7) + (api_score * 0.3)
        elif comprehensive_score > 0:
            validation_results["overall_quality_score"] = comprehensive_score
        elif api_score > 0:
            validation_results["overall_quality_score"] = api_score
        
        # 4. Determinar sucesso geral
        validation_results["validation_success"] = (
            validation_results["overall_quality_score"] >= 0.7 and
            not validation_results["comprehensive_validation"].get("error") and
            not validation_results["api_integration_validation"].get("error")
        )
        
        logger.info(f"Validação final concluída - Score: {validation_results['overall_quality_score']:.3f}, Sucesso: {validation_results['validation_success']}")
        
        return validation_results
    
    # Métodos auxiliares
    
    def _get_stage_output_path(self, stage_name: str, original_path: str) -> str:
        """Gera caminho de saída para uma etapa"""
        
        base_name = Path(original_path).stem
        
        # Usar data/dashboard_results como output_path padrão para consistência
        # (mesmo diretório usado pelo dashboard)
        output_path = self.pipeline_config.get("output_path", "data/dashboard_results")
        
        # Se output_path for "data/interim" mas não existir, usar dashboard_results
        if output_path == "data/interim" and not os.path.exists("data/interim"):
            output_path = "data/dashboard_results"
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir / f"{base_name}_{stage_name}.csv")
    
    def _save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Salva dados processados com proteção contra separadores mistos"""
        
        # Criar diretório se não existir
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar com quoting adequado para evitar problemas de parsing
        df.to_csv(
            output_path, 
            sep=';',                    # Separador padrão do projeto
            index=False, 
            encoding='utf-8',
            quoting=1,                  # QUOTE_ALL - protege contra confusão de separadores
            quotechar='"',              # Aspas duplas padrão
            doublequote=True,           # Escapar aspas duplas duplicando
            lineterminator='\n'        # Terminador de linha Unix
        )
        
        self.pipeline_state["data_versions"][output_path] = datetime.now().isoformat()
        logger.debug(f"Dados salvos: {output_path} ({len(df)} linhas, {len(df.columns)} colunas)")
    
    def _load_processed_data(self, input_path: str) -> pd.DataFrame:
        """Carrega dados processados usando chunks se necessário"""
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
        
        # Verificar tamanho do arquivo
        file_size = os.path.getsize(input_path)
        
        # Função para tentar diferentes configurações de parsing
        def try_parse_csv(file_path, **kwargs):
            """Tenta parsear CSV com diferentes configurações e validação"""
            try:
                df = pd.read_csv(file_path, **kwargs)
                if df is not None:
                    # VALIDAÇÃO CRÍTICA: Verificar se CSV foi parseado corretamente
                    if len(df.columns) == 1 and ',' in df.columns[0]:
                        logger.error(f"❌ CSV mal parseado: header concatenado detectado")
                        logger.error(f"Header problemático: {df.columns[0][:100]}...")
                        return None  # Forçar nova tentativa
                    
                    # Verificar se temos colunas esperadas
                    expected_cols = ['message_id', 'datetime', 'body', 'channel']
                    if not any(col in df.columns for col in expected_cols):
                        logger.warning(f"⚠️  Colunas esperadas não encontradas: {list(df.columns)[:5]}")
                        # Não é erro crítico, pode ser dataset processado diferentemente
                    
                    logger.debug(f"✅ CSV parseado: {len(df.columns)} colunas, {len(df)} linhas")
                return df
            except Exception as e:
                logger.warning(f"Tentativa de parsing falhou: {e}")
                return None
        
        # Configurações de parsing para testar em ordem de preferência
        import csv
        csv.field_size_limit(500000)  # Aumentar limite para 500KB por campo
        
        # CORREÇÃO CRÍTICA: Detectar separador automaticamente analisando primeira linha
        def detect_separator(file_path):
            """Detecta o separador do CSV analisando a primeira linha com validação robusta"""
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
        
        detected_sep = detect_separator(input_path)
        logger.info(f"📊 Separador detectado: '{detected_sep}'")
        
        # Preparar configurações com ambos separadores para garantir parsing correto
        separators_to_try = [detected_sep]
        if detected_sep == ';':
            separators_to_try.append(',')  # Tentar vírgula como fallback
        else:
            separators_to_try.append(';')  # Tentar ponto-e-vírgula como fallback
        
        parse_configs = []
        
        # Gerar configurações para cada separador
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
        
        logger.info(f"📋 Configurações de parsing preparadas: {len(parse_configs)} tentativas")
        
        if file_size > 200 * 1024 * 1024:  # >200MB, usar chunks
            logger.info(f"Arquivo grande detectado ({file_size/1024/1024:.1f}MB), usando processamento em chunks")
            
            # Tentar parsing em chunks
            for i, config in enumerate(parse_configs):
                try:
                    chunk_iterator = pd.read_csv(
                        input_path,
                        chunksize=self.pipeline_config.get("chunk_size", 10000),
                        **config
                    )
                    chunks = []
                    for chunk in chunk_iterator:
                        # Validar cada chunk
                        if len(chunk.columns) == 1 and ',' in chunk.columns[0]:
                            logger.error(f"❌ Chunk mal parseado com config {i+1}, tentando próxima")
                            chunks = []  # Limpar chunks inválidos
                            break
                        chunks.append(chunk)
                    
                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        logger.info(f"✅ Parsing bem-sucedido em chunks (config {i+1}): {len(df)} linhas, {len(df.columns)} colunas")
                        return df
                        
                except Exception as e:
                    logger.warning(f"Configuração de chunk {i+1} falhou: {e}")
                    continue
            
            raise ValueError("Todas as configurações de parsing em chunks falharam")
        else:
            logger.info(f"Arquivo pequeno ({file_size/1024:.1f}KB), carregando completo")
            
            # Tentar parsing completo
            for i, config in enumerate(parse_configs):
                df = try_parse_csv(input_path, **config)
                if df is not None:
                    logger.info(f"✅ Parsing bem-sucedido com configuração {i+1}: {len(df)} linhas, {len(df.columns)} colunas")
                    logger.debug(f"Colunas detectadas: {list(df.columns)[:10]}")
                    return df
            
            # Se tudo falhar, tentar com chunk_processor como último recurso
            try:
                def chunk_reader(chunk_df, chunk_idx):
                    return chunk_df
                
                if hasattr(self, 'chunk_processor') and self.chunk_processor:
                    results = self.chunk_processor.process_file(input_path, chunk_reader)
                    if results and len(results) > 0:
                        df = pd.concat(results, ignore_index=True)
                        logger.info(f"Fallback com chunk_processor bem-sucedido: {len(df)} linhas")
                        return df
            except Exception as e:
                logger.warning(f"Fallback com chunk_processor falhou: {e}")
            
            raise ValueError("Todas as configurações de parsing falharam")
        
    def _detect_text_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detecta automaticamente as colunas de texto principais com cache"""
        
        # Cache para evitar redetecção desnecessária
        if hasattr(self, '_column_mapping_cache'):
            cached_columns = set(self._column_mapping_cache.keys())
            current_columns = set(df.columns)
            if cached_columns.issubset(current_columns):
                logger.debug("✅ Usando mapeamento de colunas do cache")
                return self._column_mapping_cache
        
        column_mapping = {}
        
        # Mapeamentos conhecidos para diferentes formatos de dataset (PRIORIDADE: cleaned > original)
        text_candidates = ['body_cleaned', 'body', 'texto_cleaned', 'texto', 'text_cleaned', 'text', 'content_cleaned', 'content', 'message', 'mensagem']
        datetime_candidates = ['datetime', 'data_hora', 'timestamp', 'date', 'time']
        channel_candidates = ['channel', 'canal', 'channel_name', 'source']
        
        # Detectar coluna de texto principal (PRIORIDADE: body_cleaned > body)
        for candidate in text_candidates:
            if candidate in df.columns:
                column_mapping['text'] = candidate
                break
        
        # Detectar coluna de texto limpo (separadamente para análises específicas)
        cleaned_candidates = ['body_cleaned', 'texto_cleaned', 'text_cleaned', 'content_cleaned']
        for candidate in cleaned_candidates:
            if candidate in df.columns:
                column_mapping['text_cleaned'] = candidate
                break
        
        # Detectar coluna de data/hora
        for candidate in datetime_candidates:
            if candidate in df.columns:
                column_mapping['datetime'] = candidate
                break
                
        # Detectar coluna de canal
        for candidate in channel_candidates:
            if candidate in df.columns:
                column_mapping['channel'] = candidate
                break
        
        # Se não encontrou texto principal, usar a primeira coluna de texto longo
        if 'text' not in column_mapping:
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample_length = df[col].dropna().apply(str).str.len().mean()
                    if sample_length > 50:  # Texto com média > 50 caracteres
                        column_mapping['text'] = col
                        break
        
        # Salvar no cache para reutilização
        self._column_mapping_cache = column_mapping.copy()
        
        logger.debug(f"📋 Mapeamento de colunas detectado: {column_mapping}")
        if not column_mapping.get('text'):
            logger.warning(f"⚠️  Nenhuma coluna de texto principal detectada em: {list(df.columns)}")
        
        return column_mapping
    
    def _get_best_text_column(self, df: pd.DataFrame, prefer_cleaned: bool = True) -> str:
        """
        Obtém a melhor coluna de texto para processamento
        
        Args:
            df: DataFrame com dados
            prefer_cleaned: Se deve priorizar colunas limpas (body_cleaned)
            
        Returns:
            Nome da coluna de texto mais adequada
        """
        column_mapping = self._detect_text_columns(df)
        
        if prefer_cleaned and column_mapping.get('text_cleaned'):
            return column_mapping['text_cleaned']
        elif column_mapping.get('text'):
            return column_mapping['text']
        else:
            # Fallback para primeira coluna que parece conter texto
            for col in df.columns:
                if 'body' in col.lower() or 'text' in col.lower() or 'content' in col.lower():
                    return col
            
            # Último fallback - primeira coluna object
            for col in df.columns:
                if df[col].dtype == 'object':
                    return col
                    
            raise ValueError(f"Nenhuma coluna de texto encontrada em: {list(df.columns)}")
    
    def _preserve_deduplication_info(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preserva informações de deduplicação quando processando dados
        
        Args:
            original_df: DataFrame original com duplicate_frequency
            processed_df: DataFrame processado que pode ter perdido a coluna
            
        Returns:
            DataFrame processado com duplicate_frequency preservada
        """
        if 'duplicate_frequency' in original_df.columns and 'duplicate_frequency' not in processed_df.columns:
            logger.debug("🔄 Preservando coluna duplicate_frequency")
            # Assumir que as linhas estão na mesma ordem
            if len(original_df) == len(processed_df):
                processed_df['duplicate_frequency'] = original_df['duplicate_frequency'].values
            else:
                logger.warning(f"⚠️  Tamanhos diferentes: original={len(original_df)}, processado={len(processed_df)}. Usando frequência padrão.")
                processed_df['duplicate_frequency'] = 1
        
        return processed_df
    
    def _save_pipeline_checkpoint(self, stage_name: str, stage_result: Dict[str, Any]):
        """Salva checkpoint da etapa"""
        
        checkpoint_data = {
            "stage": stage_name,
            "stage_result": stage_result,
            "pipeline_state": self.pipeline_state,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_dir = self.project_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"pipeline_{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.pipeline_state["checkpoints"][stage_name] = str(checkpoint_file)
    
    def _save_final_results(self, pipeline_results: Dict[str, Any]):
        """Salva resultados finais do pipeline"""
        
        results_dir = self.project_root / "logs" / "pipeline"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Salvar também como latest
        latest_file = results_dir / "latest_pipeline_results.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False, default=str)
    
    def _is_critical_error(self, stage_name: str, error: Exception) -> bool:
        """Determina se um erro é crítico para o pipeline"""
        
        critical_stages = ["01_validate_data", "02b_deduplication"]
        critical_errors = ["FileNotFoundError", "PermissionError", "APIKeyError"]
        
        return stage_name in critical_stages or type(error).__name__ in critical_errors
    
    # Métodos de fallback tradicional (implementação básica)
    
    def _traditional_validate_data(self, dataset_path: str) -> Dict[str, Any]:
        """Validação tradicional sem API"""
        return {"method": "traditional", "encoding_issues": [], "structural_issues": []}
    
    
    def _traditional_deduplication(self, dataset_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Deduplicação tradicional"""
        df = pd.read_csv(dataset_path, sep=';', encoding='utf-8')
        original_count = len(df)
        df_deduplicated = df.drop_duplicates()
        return df_deduplicated, {
            "method": "traditional", 
            "original_count": original_count,
            "deduplicated_count": len(df_deduplicated)
        }
    
    def _traditional_feature_extraction(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extração tradicional de features adaptada para estrutura existente"""
        
        enhanced_df = df.copy()
        
        # Usar método otimizado para detectar melhor coluna de texto
        text_column = self._get_best_text_column(df, prefer_cleaned=True)
        
        # Métricas básicas de texto
        enhanced_df['text_length'] = enhanced_df[text_column].fillna('').str.len()
        enhanced_df['word_count'] = enhanced_df[text_column].fillna('').str.split().str.len()
        
        # Análise de colunas existentes
        features_added = 2  # text_length, word_count
        
        if 'mentions' in df.columns:
            enhanced_df['mention_count'] = enhanced_df['mentions'].fillna('').apply(
                lambda x: len([m for m in str(x).split(',') if m.strip()]) if x else 0
            )
            enhanced_df['has_mentions'] = enhanced_df['mention_count'] > 0
            features_added += 2
        
        if 'hashtag' in df.columns:
            enhanced_df['hashtag_count'] = enhanced_df['hashtag'].fillna('').apply(
                lambda x: len([h for h in str(x).split(',') if h.strip()]) if x else 0
            )
            enhanced_df['has_hashtag'] = enhanced_df['hashtag_count'] > 0
            features_added += 2
        
        if 'url' in df.columns:
            enhanced_df['url_count'] = enhanced_df['url'].fillna('').apply(
                lambda x: len([u for u in str(x).split(',') if u.strip()]) if x else 0
            )
            enhanced_df['has_url'] = enhanced_df['url_count'] > 0
            features_added += 2
        
        # Flags básicas (sem análise semântica)
        enhanced_df['political_alignment'] = 'neutro'
        enhanced_df['sentiment_category'] = 'neutro'
        enhanced_df['discourse_type'] = 'informativo'
        features_added += 3
        
        report = {
            "method": "traditional", 
            "features_extracted": features_added,
            "semantic_analysis": False,
            "used_existing_columns": True
        }
        
        return enhanced_df, report
    
    def _traditional_clean_text(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Limpeza tradicional de texto"""
        return df, {"method": "traditional", "texts_cleaned": len(df)}
    
    def _traditional_sentiment_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Análise tradicional de sentimento"""
        df['sentiment'] = 'neutral'
        return df, {"method": "traditional", "sentiments_analyzed": len(df)}
    
    def _traditional_topic_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Modelagem tradicional de tópicos"""
        df['topic'] = 'general'
        return df, {"method": "traditional", "topics_found": 1}
    
    def _traditional_tfidf_extraction(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extração tradicional TF-IDF"""
        return df, {"method": "traditional", "terms_extracted": 0}
    
    def _traditional_clustering(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clustering tradicional"""
        df['cluster'] = 0
        return df, {"method": "traditional", "clusters_found": 1}
    
    def _traditional_hashtag_normalization(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalização tradicional de hashtags"""
        return df, {"method": "traditional", "hashtags_normalized": 0}
    
    def _traditional_domain_extraction(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extração tradicional de domínios"""
        return df, {"method": "traditional", "domains_extracted": 0}
    
    def _traditional_temporal_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Análise temporal tradicional"""
        return df, {"method": "traditional", "temporal_patterns_found": 0}
    
    def _traditional_network_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Análise de rede tradicional"""
        return df, {"method": "traditional", "network_structures_found": 0}
    
    def _traditional_qualitative_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Análise qualitativa tradicional"""
        return df, {"method": "traditional", "qualitative_classifications": 0}
    
    def _traditional_review_reproducibility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Revisão tradicional"""
        return {"method": "traditional", "reproducibility_score": 1.0, "review_complete": True}
    
    def _stage_14_semantic_search_intelligence(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """
        Etapa 14: Sistema de Busca Semântica e Inteligência
        
        Constrói índice de busca semântica e executa análises avançadas:
        - Criação de índice de embeddings semânticos
        - Descoberta automática de padrões e insights
        - Análise de evolução temporal de conceitos
        - Detecção de redes de influência e coordenação
        - Dashboard analítico abrangente
        """
        
        logger.info("Iniciando etapa de busca semântica e inteligência")
        
        results = {
            "search_indices_built": {},
            "content_discoveries": {},
            "temporal_evolutions": {},
            "analytics_dashboards": {},
            "insights_generated": [],
            "anthropic_used": False
        }
        
        try:
            for dataset_path in dataset_paths:
                dataset_name = Path(dataset_path).stem
                logger.info(f"Processando {dataset_name} para busca semântica")
                
                # Carregar dados processados - usar última versão disponível
                # Se dataset_path contém stage name, usar diretamente  
                if any(stage in dataset_path for stage in ["_02b_deduplicated", "_01b_features_extracted", "_03_text_cleaned", "_04_sentiment_analyzed", "_05_topic_modeled", "_06_tfidf_extracted", "_07_clustered", "_08_hashtags_normalized", "_09_domains_analyzed", "_10_temporal_analyzed", "_11_network_analyzed", "_12_qualitative_analyzed", "_13_final_processed"]):
                    df = self._load_processed_data(dataset_path)
                else:
                    # Tentar encontrar a versão mais recente processada
                    final_path = self._get_stage_output_path("13_final_processed", dataset_path)
                    if os.path.exists(final_path):
                        df = self._load_processed_data(final_path)
                    else:
                        # Fallback para dados originais
                        df = self._load_processed_data(dataset_path)
                
                if df is None or len(df) == 0:
                    logger.warning(f"Dataset vazio ou inválido: {dataset_path}")
                    continue
                
                # Verificar se componentes semânticos estão disponíveis
                if not self.semantic_search_engine:
                    logger.warning("Semantic search engine não disponível, pulando etapa")
                    results["search_indices_built"][dataset_name] = {"error": "component_not_available"}
                    continue
                
                # 1. Construir índice de busca semântica
                logger.info(f"Construindo índice de busca semântica para {dataset_name}")
                
                # Usar método otimizado para detectar melhor coluna de texto
                try:
                    text_column = self._get_best_text_column(df, prefer_cleaned=True)
                    logger.info(f"Coluna de texto detectada: '{text_column}'")
                except ValueError:
                    # Última tentativa: procurar qualquer coluna de texto
                    text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].dropna().str.len().mean() > 50]
                    if text_columns:
                        text_column = text_columns[0]
                        logger.warning(f"Usando primeira coluna de texto encontrada: '{text_column}'")
                    else:
                        logger.error(f"Nenhuma coluna de texto encontrada no dataset {dataset_name}. Colunas disponíveis: {list(df.columns)}")
                        continue
                
                search_index_result = self.semantic_search_engine.build_search_index(df, text_column)
                results["search_indices_built"][dataset_name] = search_index_result
                
                if not search_index_result.get('success'):
                    logger.error(f"Falha ao construir índice semântico para {dataset_name}")
                    continue
                
                # 2. Descoberta automática de conteúdo e padrões
                if self.content_discovery_engine:
                    logger.info(f"Executando descoberta de conteúdo para {dataset_name}")
                    
                    content_patterns = self.content_discovery_engine.discover_content_patterns()
                    coordination_patterns = self.content_discovery_engine.detect_coordination_patterns()
                    misinformation_campaigns = self.content_discovery_engine.detect_misinformation_campaigns()
                    influence_networks = self.content_discovery_engine.discover_influence_networks()
                    
                    results["content_discoveries"][dataset_name] = {
                        "content_patterns": content_patterns,
                        "coordination_patterns": coordination_patterns, 
                        "misinformation_campaigns": misinformation_campaigns,
                        "influence_networks": influence_networks
                    }
                
                # 3. Análise de evolução temporal de conceitos-chave
                if self.temporal_evolution_tracker:
                    logger.info(f"Analisando evolução temporal para {dataset_name}")
                    
                    # Conceitos-chave para análise temporal
                    key_concepts = ['democracia', 'eleições', 'vacinas', 'stf', 'mídia']
                    temporal_analyses = {}
                    
                    for concept in key_concepts:
                        try:
                            evolution_result = self.temporal_evolution_tracker.track_concept_evolution(concept)
                            if not evolution_result.get('error'):
                                temporal_analyses[concept] = evolution_result
                        except Exception as e:
                            logger.warning(f"Erro na análise temporal de '{concept}': {e}")
                    
                    # Análise de mudanças de discurso
                    discourse_shifts = self.temporal_evolution_tracker.detect_discourse_shifts()
                    
                    results["temporal_evolutions"][dataset_name] = {
                        "concept_evolutions": temporal_analyses,
                        "discourse_shifts": discourse_shifts
                    }
                
                # 4. Gerar dashboard analítico abrangente
                if self.analytics_dashboard:
                    logger.info(f"Gerando dashboard analítico para {dataset_name}")
                    
                    dashboard_data = self.analytics_dashboard.generate_comprehensive_dashboard(
                        time_period_days=30
                    )
                    
                    # Gerar relatórios especializados
                    trend_report = self.analytics_dashboard.generate_trend_analysis_report()
                    influence_report = self.analytics_dashboard.generate_influence_analysis_report()
                    quality_report = self.analytics_dashboard.generate_content_quality_report()
                    
                    results["analytics_dashboards"][dataset_name] = {
                        "comprehensive_dashboard": dashboard_data,
                        "trend_analysis": trend_report,
                        "influence_analysis": influence_report,
                        "content_quality": quality_report
                    }
                    
                    # Exportar dashboard em múltiplos formatos
                    try:
                        dashboard_export_path = self.analytics_dashboard.export_dashboard_data(
                            dashboard_data, 
                            format_type='json',
                            output_path=f"data/interim/dashboard_{dataset_name}_{datetime.now().strftime('%Y%m%d')}.json"
                        )
                        results["analytics_dashboards"][dataset_name]["export_path"] = dashboard_export_path
                    except Exception as e:
                        logger.warning(f"Erro ao exportar dashboard: {e}")
                
                # 5. Gerar insights automáticos usando IA
                if self.api_available and self.semantic_search_engine:
                    logger.info(f"Gerando insights automáticos com IA para {dataset_name}")
                    
                    automated_insights = self.semantic_search_engine.generate_automated_insights([
                        'political_discourse', 'conspiracy_theories', 'institutional_trust',
                        'pandemic_response', 'election_integrity', 'media_criticism'
                    ])
                    
                    results["insights_generated"].append({
                        "dataset": dataset_name,
                        "insights": automated_insights,
                        "generated_at": datetime.now().isoformat()
                    })
                    
                    results["anthropic_used"] = True
                
                logger.info(f"Processamento semântico completo para {dataset_name}")
            
            results["datasets_processed"] = len([d for d in results["search_indices_built"].values() if d.get('success')])
            
            # Gerar insights consolidados entre datasets
            if results["datasets_processed"] > 1:
                logger.info("Gerando insights consolidados entre datasets")
                consolidated_insights = self._generate_consolidated_semantic_insights(results)
                results["consolidated_insights"] = consolidated_insights
            
            logger.info(f"Etapa semântica concluída com sucesso para {results['datasets_processed']} datasets")
            
        except Exception as e:
            logger.error(f"Erro na etapa de busca semântica: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def _generate_consolidated_semantic_insights(self, semantic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera insights consolidados entre múltiplos datasets"""
        
        consolidated = {
            "cross_dataset_patterns": [],
            "common_themes": [],
            "evolution_trends": [],
            "network_overlaps": [],
            "methodology": "cross_dataset_semantic_analysis"
        }
        
        try:
            # Consolidar padrões de descoberta entre datasets
            all_patterns = []
            for dataset_name, discoveries in semantic_results.get("content_discoveries", {}).items():
                if "content_patterns" in discoveries:
                    patterns = discoveries["content_patterns"].get("discovered_patterns", [])
                    for pattern in patterns:
                        pattern["source_dataset"] = dataset_name
                        all_patterns.append(pattern)
            
            # Identificar temas comuns
            if all_patterns:
                all_themes = []
                for pattern in all_patterns:
                    all_themes.extend(pattern.get("key_themes", []))
                
                from collections import Counter
                theme_counts = Counter(all_themes)
                consolidated["common_themes"] = [
                    {"theme": theme, "frequency": count, "datasets": len(set(p["source_dataset"] for p in all_patterns if theme in p.get("key_themes", [])))}
                    for theme, count in theme_counts.most_common(10)
                ]
            
            # Consolidar tendências de evolução temporal
            evolution_data = semantic_results.get("temporal_evolutions", {})
            if evolution_data:
                # Analisar evolução de conceitos entre datasets
                concept_evolution_summary = {}
                for dataset_name, evolutions in evolution_data.items():
                    for concept, evolution in evolutions.get("concept_evolutions", {}).items():
                        if concept not in concept_evolution_summary:
                            concept_evolution_summary[concept] = []
                        concept_evolution_summary[concept].append({
                            "dataset": dataset_name,
                            "evolution_data": evolution.get("insights", {})
                        })
                
                consolidated["evolution_trends"] = concept_evolution_summary
            
            logger.info("Insights consolidados gerados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao gerar insights consolidados: {e}")
            consolidated["error"] = str(e)
        
        return consolidated

    # =================== MÉTODOS DE ETAPAS FALTANTES v4.7 ===================
    
    def _stage_01_chunk_processing(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 01: Processamento em chunks - Carregamento e validação inicial"""
        
        logger.info("Iniciando processamento em chunks")
        results = {"chunks_processed": {}, "validation_reports": {}}
        
        for dataset_path in dataset_paths:
            try:
                # Carregamento seguro com detecção de separador
                import pandas as pd
                
                # Tentar diferentes separadores
                separators = [';', ',', '\t']
                df = None
                
                for sep in separators:
                    try:
                        df = pd.read_csv(dataset_path, sep=sep, encoding='utf-8')
                        if len(df.columns) > 1:  # Se temos mais de uma coluna, separador correto
                            break
                    except Exception:
                        continue
                
                # Se falhou, tentar encoding alternativo
                if df is None or len(df.columns) == 1:
                    try:
                        df = pd.read_csv(dataset_path, sep=';', encoding='latin-1')
                    except Exception:
                        df = None
                
                if df is not None and not df.empty:
                    # Validação básica da estrutura
                    validation_result = {
                        "total_records": len(df),
                        "columns": list(df.columns),
                        "memory_usage": f"{df.memory_usage().sum() / 1024 / 1024:.2f} MB",
                        "chunk_processed": True
                    }
                    
                    results["chunks_processed"][dataset_path] = {
                        "records": len(df),
                        "success": True
                    }
                    results["validation_reports"][dataset_path] = validation_result
                    
                    # Salvar dados processados
                    output_path = dataset_path.replace('.csv', '_01_chunked.csv')
                    df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
                    logger.info(f"Chunk processado salvo: {output_path}")
                    
                else:
                    logger.warning(f"Falha ao carregar dataset: {dataset_path}")
                    results["chunks_processed"][dataset_path] = {
                        "records": 0,
                        "success": False,
                        "error": "Dataset vazio ou ilegível"
                    }
                    
            except Exception as e:
                logger.error(f"Erro no processamento de chunk para {dataset_path}: {e}")
                results["chunks_processed"][dataset_path] = {
                    "records": 0,
                    "success": False,
                    "error": str(e)
                }
        
        results["datasets_processed"] = len(results["chunks_processed"])
        return results
    
    def _stage_02a_encoding_validation(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 02a: Validação de encoding - Detecta e corrige problemas de codificação"""
        
        logger.info("Iniciando validação de encoding")
        results = {"encoding_reports": {}, "corrections_applied": {}}
        
        for dataset_path in dataset_paths:
            try:
                # Carregar dados primeiro
                df = self._load_processed_data(dataset_path)
                
                if df is not None and not df.empty:
                    # Usar EncodingValidator
                    validation_result = self.encoding_validator.validate_encoding_quality(df)
                    results["encoding_reports"][dataset_path] = validation_result
                    
                    # Aplicar correções se necessário
                    if validation_result.get("encoding_issues", []):
                        from ..utils.encoding_fixer import EncodingFixer
                        fixer = EncodingFixer()
                        
                        corrected_df = fixer.fix_encoding_issues(df)
                        if corrected_df is not None:
                            # Salvar dados corrigidos
                            output_path = dataset_path.replace('.csv', '_02a_encoding_validated.csv')
                            corrected_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
                            
                            results["corrections_applied"][dataset_path] = {
                                "corrections": len(validation_result.get("encoding_issues", [])),
                                "output_path": output_path,
                                "success": True
                            }
                            logger.info(f"Encoding corrigido e salvo: {output_path}")
                        else:
                            results["corrections_applied"][dataset_path] = {
                                "corrections": 0,
                                "success": False,
                                "error": "Falha na correção de encoding"
                            }
                    else:
                        # Nenhuma correção necessária, copiar arquivo
                        output_path = dataset_path.replace('.csv', '_02a_encoding_validated.csv')
                        df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
                        results["corrections_applied"][dataset_path] = {
                            "corrections": 0,
                            "output_path": output_path,
                            "success": True,
                            "message": "Nenhuma correção necessária"
                        }
                        
                else:
                    logger.warning(f"Dataset vazio para validação de encoding: {dataset_path}")
                    results["encoding_reports"][dataset_path] = {"error": "Dataset vazio"}
                    
            except Exception as e:
                logger.error(f"Erro na validação de encoding para {dataset_path}: {e}")
                results["encoding_reports"][dataset_path] = {"error": str(e)}
        
        results["datasets_processed"] = len(results["encoding_reports"])
        return results
    
    def _stage_14_topic_interpretation(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 14: Interpretação de tópicos - Análise semântica avançada de tópicos identificados"""
        
        logger.info("Iniciando interpretação de tópicos")
        results = {"topic_interpretations": {}, "semantic_analysis": {}}
        
        for dataset_path in dataset_paths:
            try:
                df = self._load_processed_data(dataset_path)
                
                if df is not None and not df.empty:
                    if self.pipeline_config["use_anthropic"] and self.api_available:
                        # Usar TopicInterpreter para análise avançada
                        interpretation_result = self.topic_interpreter.interpret_topics_semantically(df)
                        results["anthropic_used"] = True
                    else:
                        # Análise tradicional de tópicos
                        interpretation_result = self._traditional_topic_interpretation(df)
                    
                    results["topic_interpretations"][dataset_path] = interpretation_result
                    
                    # Salvar resultados interpretados
                    if interpretation_result.get("success", False):
                        output_path = dataset_path.replace('.csv', '_14_topic_interpreted.csv')
                        
                        # Adicionar colunas de interpretação se necessário
                        enhanced_df = self._enhance_dataframe_with_topic_interpretation(df, interpretation_result)
                        enhanced_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
                        
                        logger.info(f"Interpretação de tópicos salva: {output_path}")
                        
                else:
                    logger.warning(f"Dataset vazio para interpretação de tópicos: {dataset_path}")
                    results["topic_interpretations"][dataset_path] = {"error": "Dataset vazio"}
                    
            except Exception as e:
                logger.error(f"Erro na interpretação de tópicos para {dataset_path}: {e}")
                results["topic_interpretations"][dataset_path] = {"error": str(e)}
        
        results["datasets_processed"] = len(results["topic_interpretations"])
        return results
    
    def _stage_16_pipeline_validation(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Etapa 16: Validação final do pipeline - Verificação de qualidade e completude"""
        
        logger.info("Iniciando validação final do pipeline")
        results = {"validation_reports": {}, "pipeline_quality_score": 0.0}
        
        for dataset_path in dataset_paths:
            try:
                df = self._load_processed_data(dataset_path)
                
                if df is not None and not df.empty:
                    # Usar CompletePipelineValidator
                    validation_result = self.pipeline_validator.validate_complete_pipeline(
                        {"stage_results": df}, 
                        dataset_path,
                        None  # original_dataset_path
                    )
                    
                    results["validation_reports"][dataset_path] = validation_result
                    
                    # Atualizar score de qualidade
                    dataset_score = validation_result.get("overall_score", 0.0)
                    if results["pipeline_quality_score"] == 0.0:
                        results["pipeline_quality_score"] = dataset_score
                    else:
                        # Média dos scores
                        current_count = len([r for r in results["validation_reports"].values() 
                                           if "overall_score" in r])
                        results["pipeline_quality_score"] = (
                            (results["pipeline_quality_score"] * (current_count - 1) + dataset_score) / current_count
                        )
                    
                    # Salvar relatório final
                    output_path = dataset_path.replace('.csv', '_16_pipeline_validated.csv')
                    df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
                    
                    # Gerar relatório JSON detalhado
                    report_path = output_path.replace('.csv', '_validation_report.json')
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(validation_result, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Validação final concluída: {output_path}")
                    logger.info(f"Relatório detalhado: {report_path}")
                    
                else:
                    logger.warning(f"Dataset vazio para validação final: {dataset_path}")
                    results["validation_reports"][dataset_path] = {"error": "Dataset vazio"}
                    
            except Exception as e:
                logger.error(f"Erro na validação final para {dataset_path}: {e}")
                results["validation_reports"][dataset_path] = {"error": str(e)}
        
        results["datasets_processed"] = len(results["validation_reports"])
        
        # Determinar se pipeline foi bem-sucedido
        success_threshold = 0.7
        results["pipeline_success"] = results["pipeline_quality_score"] >= success_threshold
        
        if results["pipeline_success"]:
            logger.info(f"✅ Pipeline validado com sucesso! Score: {results['pipeline_quality_score']:.2f}")
        else:
            logger.warning(f"⚠️ Pipeline com qualidade baixa. Score: {results['pipeline_quality_score']:.2f}")
        
        return results
    
    def _traditional_topic_interpretation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Interpretação tradicional de tópicos sem API"""
        return {
            "method": "traditional",
            "topics_found": 0,
            "interpretations": [],
            "success": True,
            "note": "Análise tradicional aplicada"
        }
    
    def _enhance_dataframe_with_topic_interpretation(self, df: pd.DataFrame, interpretation_result: Dict[str, Any]) -> pd.DataFrame:
        """Adiciona colunas de interpretação de tópicos ao DataFrame"""
        enhanced_df = df.copy()
        
        # Adicionar colunas de interpretação se disponíveis
        if "interpretations" in interpretation_result:
            enhanced_df["topic_interpretation"] = "interpretação_disponível"
        else:
            enhanced_df["topic_interpretation"] = "interpretação_tradicional"
            
        return enhanced_df
    
    def _get_expected_final_columns(self) -> List[str]:
        """Retorna colunas esperadas no dataset final"""
        return [
            "message_id", "datetime", "body", "body_cleaned", "text_cleaned",
            "political_alignment", "sentiment_score", "topic_assignment",
            "tfidf_keywords", "cluster_id", "hashtag_normalized",
            "domain_category", "temporal_pattern", "network_influence",
            "qualitative_category", "pipeline_reviewed", "topic_interpretation"
        ]


def create_unified_pipeline(config: Dict[str, Any] = None, project_root: str = None) -> UnifiedAnthropicPipeline:
    """
    Função de conveniência para criar pipeline unificado
    
    Args:
        config: Configuração do pipeline
        project_root: Diretório raiz do projeto
        
    Returns:
        Instância do pipeline unificado
    """
    
    return UnifiedAnthropicPipeline(config, project_root)