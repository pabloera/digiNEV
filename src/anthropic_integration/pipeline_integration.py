"""
Integração Completa das Validações API no Sistema de Pipeline
Mantém checkpoints, logs e integra todos os componentes API desenvolvidos.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Importar todos os componentes desenvolvidos
from .api_error_handler import APIErrorHandler, APIQualityChecker
from .deduplication_validator import DeduplicationValidator
from .encoding_validator import EncodingValidator
from .feature_extractor import FeatureExtractor
from .intelligent_text_cleaner import IntelligentTextCleaner
from .pipeline_validator import CompletePipelineValidator

logger = logging.getLogger(__name__)


class APIPipelineIntegration:
    """
    Classe principal de integração das validações API no pipeline

    Responsabilidades:
    - Orquestrar todas as validações API
    - Manter checkpoints e estado do pipeline
    - Integrar com o sistema de logs existente
    - Coordenar retry automático e escalação
    - Fornecer interface unificada para o pipeline
    """

    def __init__(self, config: Dict[str, Any] = None, project_root: str = None):
        self.config = config or {}
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Inicializar componentes
        self.error_handler = APIErrorHandler(str(self.project_root))
        self.quality_checker = APIQualityChecker(config)

        # Inicializar validadores específicos
        self.feature_extractor = FeatureExtractor(config)
        self.encoding_validator = EncodingValidator(config)
        self.deduplication_validator = DeduplicationValidator(config)
        self.text_cleaner = IntelligentTextCleaner(config)
        self.pipeline_validator = CompletePipelineValidator(config, str(self.project_root))

        # Estado do pipeline
        self.pipeline_state = {
            "start_time": None,
            "current_stage": None,
            "completed_stages": [],
            "failed_stages": [],
            "api_validations": {},
            "checkpoints": {}
        }

        # Configurações de integração
        self.integration_config = {
            "enable_api_validations": self.config.get("anthropic", {}).get("enable_api_integration", True),
            "auto_validation": True,
            "save_checkpoints": True,
            "escalate_failures": True,
            "comprehensive_final_validation": True
        }

        logger.info("API Pipeline Integration inicializada")

    def initialize_pipeline_run(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Inicializa execução do pipeline com validações API"""

        self.pipeline_state = {
            "start_time": datetime.now(),
            "current_stage": None,
            "completed_stages": [],
            "failed_stages": [],
            "api_validations": {},
            "checkpoints": {},
            "config": pipeline_config
        }

        logger.info("Execução do pipeline inicializada com validações API")

        # Salvar checkpoint inicial
        if self.integration_config["save_checkpoints"]:
            self._save_checkpoint("pipeline_start", self.pipeline_state)

        return {
            "success": True,
            "pipeline_id": self.pipeline_state["start_time"].strftime("%Y%m%d_%H%M%S"),
            "api_integration_enabled": self.integration_config["enable_api_validations"]
        }

    def validate_stage_01b_features(self, df, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Valida extração de features com API (Stage 01b)"""

        stage_name = "01b_feature_extraction"
        self.pipeline_state["current_stage"] = stage_name

        logger.info(f"Iniciando validação API para {stage_name}")

        validation_result = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "api_validation": {},
            "issues_detected": [],
            "corrections_applied": [],
            "recommendations": []
        }

        try:
            if self.integration_config["enable_api_validations"]:
                # Executar extração de features via API
                enhanced_df = self.feature_extractor.extract_comprehensive_features(df)

                # Gerar relatório de features
                feature_report = self.feature_extractor.generate_feature_report(enhanced_df)

                validation_result["api_validation"] = feature_report
                validation_result["success"] = True
                validation_result["enhanced_data"] = enhanced_df

                # Verificar se correções são necessárias
                if feature_report.get("quality_metrics", {}).get("avg_coverage", 0) < 80:
                    corrections = self.feature_extractor.correct_extraction_errors(enhanced_df)
                    validation_result["corrections_applied"] = corrections

            else:
                validation_result["success"] = True
                validation_result["api_validation"] = {"note": "API validation disabled"}

            self.pipeline_state["completed_stages"].append(stage_name)

        except Exception as e:
            logger.error(f"Erro na validação do {stage_name}: {e}")
            validation_result["error"] = str(e)
            self.pipeline_state["failed_stages"].append(stage_name)

            # Escalar para usuário se configurado
            if self.integration_config["escalate_failures"]:
                self._escalate_stage_failure(stage_name, str(e))

        # Salvar resultados
        self.pipeline_state["api_validations"][stage_name] = validation_result
        self._save_checkpoint(f"{stage_name}_validation", validation_result)

        return validation_result

    def validate_stage_02_encoding(self, df, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Valida correção de encoding com API (Stage 02)"""

        stage_name = "02_encoding_validation"
        self.pipeline_state["current_stage"] = stage_name

        logger.info(f"Iniciando validação API para {stage_name}")

        validation_result = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "api_validation": {},
            "issues_detected": [],
            "corrections_applied": []
        }

        try:
            if self.integration_config["enable_api_validations"]:
                # Validar qualidade do encoding
                encoding_report = self.encoding_validator.validate_encoding_quality(df)

                validation_result["api_validation"] = encoding_report
                validation_result["success"] = True

                # Aplicar correções se necessário
                if encoding_report.get("overall_quality_score", 1.0) < 0.8:
                    corrected_df, correction_report = self.encoding_validator.detect_and_fix_encoding_issues(df)
                    validation_result["corrected_data"] = corrected_df
                    validation_result["corrections_applied"] = correction_report

            else:
                validation_result["success"] = True
                validation_result["api_validation"] = {"note": "API validation disabled"}

            self.pipeline_state["completed_stages"].append(stage_name)

        except Exception as e:
            logger.error(f"Erro na validação do {stage_name}: {e}")
            validation_result["error"] = str(e)
            self.pipeline_state["failed_stages"].append(stage_name)

            if self.integration_config["escalate_failures"]:
                self._escalate_stage_failure(stage_name, str(e))

        self.pipeline_state["api_validations"][stage_name] = validation_result
        self._save_checkpoint(f"{stage_name}_validation", validation_result)

        return validation_result

    def validate_stage_02b_deduplication(self, original_df, deduplicated_df, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Valida deduplicação com API (Stage 02b)"""

        stage_name = "02b_deduplication_validation"
        self.pipeline_state["current_stage"] = stage_name

        logger.info(f"Iniciando validação API para {stage_name}")

        validation_result = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "api_validation": {},
            "issues_detected": []
        }

        try:
            if self.integration_config["enable_api_validations"]:
                # Validar processo de deduplicação
                dedup_report = self.deduplication_validator.validate_deduplication_process(
                    original_df, deduplicated_df
                )

                validation_result["api_validation"] = dedup_report
                validation_result["success"] = True

                # Verificar se há problemas de qualidade
                quality_score = dedup_report.get("quality_assessment", {}).get("overall_score", 1.0)
                if quality_score < 0.7:
                    validation_result["issues_detected"].append(f"Qualidade da deduplicação baixa: {quality_score}")

            else:
                validation_result["success"] = True
                validation_result["api_validation"] = {"note": "API validation disabled"}

            self.pipeline_state["completed_stages"].append(stage_name)

        except Exception as e:
            logger.error(f"Erro na validação do {stage_name}: {e}")
            validation_result["error"] = str(e)
            self.pipeline_state["failed_stages"].append(stage_name)

            if self.integration_config["escalate_failures"]:
                self._escalate_stage_failure(stage_name, str(e))

        self.pipeline_state["api_validations"][stage_name] = validation_result
        self._save_checkpoint(f"{stage_name}_validation", validation_result)

        return validation_result

    def execute_stage_03_text_cleaning(self, df, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Executa limpeza de texto inteligente via API (Stage 03 - SUBSTITUTO)"""

        stage_name = "03_intelligent_text_cleaning"
        self.pipeline_state["current_stage"] = stage_name

        logger.info(f"Executando limpeza inteligente via API para {stage_name}")

        execution_result = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "api_execution": {},
            "data_processed": False
        }

        try:
            if self.integration_config["enable_api_validations"]:
                # Executar limpeza inteligente via API (SUBSTITUI processamento Python)
                cleaned_df = self.text_cleaner.clean_text_intelligent(df)

                # Validar qualidade da limpeza
                quality_report = self.text_cleaner.validate_cleaning_quality(df, cleaned_df)

                execution_result["api_execution"] = {
                    "cleaning_completed": True,
                    "quality_report": quality_report,
                    "records_processed": len(cleaned_df)
                }
                execution_result["success"] = True
                execution_result["data_processed"] = True
                execution_result["cleaned_data"] = cleaned_df

            else:
                execution_result["success"] = True
                execution_result["api_execution"] = {"note": "API execution disabled - using traditional method"}

            self.pipeline_state["completed_stages"].append(stage_name)

        except Exception as e:
            logger.error(f"Erro na execução do {stage_name}: {e}")
            execution_result["error"] = str(e)
            self.pipeline_state["failed_stages"].append(stage_name)

            if self.integration_config["escalate_failures"]:
                self._escalate_stage_failure(stage_name, str(e))

        self.pipeline_state["api_validations"][stage_name] = execution_result
        self._save_checkpoint(f"{stage_name}_execution", execution_result)

        return execution_result

    def execute_comprehensive_pipeline_validation(
        self,
        pipeline_results: Dict[str, Any],
        final_dataset_path: str,
        original_dataset_path: str = None
    ) -> Dict[str, Any]:
        """Executa validação completa pós-pipeline"""

        stage_name = "comprehensive_pipeline_validation"
        self.pipeline_state["current_stage"] = stage_name

        logger.info("Iniciando validação completa pós-pipeline")

        validation_result = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "comprehensive_report": {},
            "escalation_needed": False
        }

        try:
            if self.integration_config["comprehensive_final_validation"]:
                # Executar validação completa
                comprehensive_report = self.pipeline_validator.validate_complete_pipeline(
                    pipeline_results,
                    final_dataset_path,
                    original_dataset_path
                )

                validation_result["comprehensive_report"] = comprehensive_report
                validation_result["success"] = True

                # Verificar se escalação é necessária
                overall_assessment = comprehensive_report.get("overall_assessment", {})
                if not overall_assessment.get("ready_for_analysis", True):
                    validation_result["escalation_needed"] = True

                    # Escalar problemas críticos
                    escalation_info = self.pipeline_validator.escalate_critical_issues(comprehensive_report)
                    validation_result["escalation_info"] = escalation_info

            else:
                validation_result["success"] = True
                validation_result["comprehensive_report"] = {"note": "Comprehensive validation disabled"}

            self.pipeline_state["completed_stages"].append(stage_name)

        except Exception as e:
            logger.error(f"Erro na validação completa: {e}")
            validation_result["error"] = str(e)
            self.pipeline_state["failed_stages"].append(stage_name)

        # Finalizar pipeline
        self.pipeline_state["end_time"] = datetime.now()
        self.pipeline_state["total_duration"] = (
            self.pipeline_state["end_time"] - self.pipeline_state["start_time"]
        ).total_seconds()

        self.pipeline_state["api_validations"][stage_name] = validation_result
        self._save_final_checkpoint(validation_result)

        return validation_result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Retorna status atual do pipeline"""

        return {
            "current_stage": self.pipeline_state.get("current_stage"),
            "completed_stages": self.pipeline_state.get("completed_stages", []),
            "failed_stages": self.pipeline_state.get("failed_stages", []),
            "start_time": self.pipeline_state.get("start_time"),
            "duration": (
                (datetime.now() - self.pipeline_state["start_time"]).total_seconds()
                if self.pipeline_state.get("start_time") else 0
            ),
            "api_integration_enabled": self.integration_config["enable_api_validations"],
            "total_api_validations": len(self.pipeline_state.get("api_validations", {}))
        }

    def get_api_validation_summary(self) -> Dict[str, Any]:
        """Retorna resumo das validações API executadas"""

        summary = {
            "total_validations": len(self.pipeline_state.get("api_validations", {})),
            "successful_validations": 0,
            "failed_validations": 0,
            "stages_with_issues": [],
            "stages_with_corrections": [],
            "overall_api_success_rate": 0.0
        }

        api_validations = self.pipeline_state.get("api_validations", {})

        for stage, validation in api_validations.items():
            if validation.get("success", False):
                summary["successful_validations"] += 1
            else:
                summary["failed_validations"] += 1

            if validation.get("issues_detected"):
                summary["stages_with_issues"].append(stage)

            if validation.get("corrections_applied"):
                summary["stages_with_corrections"].append(stage)

        if summary["total_validations"] > 0:
            summary["overall_api_success_rate"] = summary["successful_validations"] / summary["total_validations"]

        return summary

    def _save_checkpoint(self, checkpoint_name: str, data: Dict[str, Any]):
        """Salva checkpoint do pipeline"""

        if not self.integration_config["save_checkpoints"]:
            return

        try:
            checkpoint_dir = self.project_root / "logs" / "pipeline" / "api_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_file = checkpoint_dir / f"{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            checkpoint_data = {
                "checkpoint_name": checkpoint_name,
                "timestamp": datetime.now().isoformat(),
                "pipeline_state": self.pipeline_state,
                "data": data
            }

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)

            self.pipeline_state["checkpoints"][checkpoint_name] = str(checkpoint_file)

        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint {checkpoint_name}: {e}")

    def _save_final_checkpoint(self, final_validation: Dict[str, Any]):
        """Salva checkpoint final com resumo completo"""

        final_checkpoint = {
            "pipeline_summary": self.pipeline_state,
            "api_validation_summary": self.get_api_validation_summary(),
            "final_validation": final_validation,
            "integration_config": self.integration_config
        }

        self._save_checkpoint("pipeline_final", final_checkpoint)

        # Salvar também como latest
        try:
            latest_file = self.project_root / "logs" / "pipeline" / "latest_api_integration_result.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(final_checkpoint, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Erro ao salvar resultado latest: {e}")

    def _escalate_stage_failure(self, stage_name: str, error_message: str):
        """Escala falha de etapa para usuário"""

        logger.warning(f"Escalando falha da etapa {stage_name} para usuário")

        try:
            # Criar erro artificial para escalação
            error = type('StageFailure', (), {
                'stage': stage_name,
                'operation': 'api_validation',
                'error_type': 'StageExecutionError',
                'error_message': error_message,
                'timestamp': datetime.now(),
                'retry_count': 0
            })()

            context = {
                "stage": stage_name,
                "pipeline_state": self.pipeline_state,
                "integration_config": self.integration_config
            }

            escalation_info = self.error_handler.escalate_to_user(error, context)

            # Salvar escalação no estado do pipeline
            if "escalations" not in self.pipeline_state:
                self.pipeline_state["escalations"] = []

            self.pipeline_state["escalations"].append({
                "stage": stage_name,
                "escalation_info": escalation_info
            })

        except Exception as e:
            logger.error(f"Erro ao escalar falha da etapa {stage_name}: {e}")

    def load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Carrega checkpoint específico"""

        try:
            checkpoint_file = self.pipeline_state.get("checkpoints", {}).get(checkpoint_name)

            if checkpoint_file and os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

            return {}

        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint {checkpoint_name}: {e}")
            return {}

    def resume_from_checkpoint(self, checkpoint_name: str) -> bool:
        """Resume pipeline a partir de checkpoint"""

        try:
            checkpoint_data = self.load_checkpoint(checkpoint_name)

            if checkpoint_data:
                self.pipeline_state = checkpoint_data.get("pipeline_state", self.pipeline_state)
                logger.info(f"Pipeline resumido a partir do checkpoint {checkpoint_name}")
                return True

            return False

        except Exception as e:
            logger.error(f"Erro ao resumir pipeline do checkpoint {checkpoint_name}: {e}")
            return False

    def cleanup_old_checkpoints(self, days_to_keep: int = 7):
        """Remove checkpoints antigos"""

        try:
            checkpoint_dir = self.project_root / "logs" / "pipeline" / "api_checkpoints"

            if checkpoint_dir.exists():
                cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)

                for checkpoint_file in checkpoint_dir.glob("*.json"):
                    if checkpoint_file.stat().st_mtime < cutoff_time:
                        checkpoint_file.unlink()
                        logger.info(f"Checkpoint antigo removido: {checkpoint_file}")

        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints antigos: {e}")


# Função de conveniência para integração fácil
def initialize_api_pipeline_integration(config: Dict[str, Any] = None, project_root: str = None) -> APIPipelineIntegration:
    """
    Função de conveniência para inicializar integração API

    Args:
        config: Configuração do projeto
        project_root: Diretório raiz do projeto

    Returns:
        Instância configurada de APIPipelineIntegration
    """

    integration = APIPipelineIntegration(config, project_root)

    logger.info("API Pipeline Integration inicializada via função de conveniência")

    return integration
