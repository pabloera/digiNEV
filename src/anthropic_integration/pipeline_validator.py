"""
Validador Completo de Pipeline via API Anthropic
Implementa validação pós-execução com verificação de qualidade, sugestões de reprocessamento e tratamento de erros.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .api_error_handler import APIErrorHandler, APIQualityChecker
from .base import AnthropicBase
from .deduplication_validator import DeduplicationValidator
from .encoding_validator import EncodingValidator
from .feature_extractor import FeatureExtractor
from .intelligent_text_cleaner import IntelligentTextCleaner

logger = logging.getLogger(__name__)

class CompletePipelineValidator(AnthropicBase):
    """
    Validador completo de pipeline usando API Anthropic

    Capacidades:
    - Validação holística de todo o pipeline
    - Detecção de inconsistências entre etapas
    - Sugestões inteligentes de reprocessamento
    - Análise de qualidade geral dos dados
    - Identificação de problemas não detectados por validadores individuais
    - Recomendações estratégicas para melhorias
    """

    def __init__(self, config: Dict[str, Any] = None, project_root: str = None):
        # 🔧 UPGRADE: Usar enhanced model configuration para validation
        super().__init__(config, stage_operation="validation")
        self.error_handler = APIErrorHandler(project_root)
        self.quality_checker = APIQualityChecker(config)
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Inicializar validadores específicos
        self.feature_extractor = FeatureExtractor(config)
        self.encoding_validator = EncodingValidator(config)
        self.deduplication_validator = DeduplicationValidator(config)
        self.text_cleaner = IntelligentTextCleaner(config)

        # Configurações de validação
        self.validation_config = {
            "comprehensive_analysis": True,
            "cross_stage_validation": True,
            "quality_thresholds": {
                "overall_quality_min": 0.7,
                "stage_quality_min": 0.6,
                "data_integrity_min": 0.8
            },
            "reprocessing_suggestions": True,
            "auto_issue_detection": True
        }

        # Métricas esperadas para cada etapa
        self.expected_metrics = {
            "01_validate_data": {
                "completion_rate": 1.0,
                "error_rate_max": 0.05
            },
            "02_fix_encoding": {
                "encoding_quality_min": 0.9,
                "correction_rate_expected": 0.1
            },
            "02b_deduplication": {
                "reduction_rate_min": 0.05,
                "reduction_rate_max": 0.6,
                "media_exclusion_rate_min": 0.9
            },
            "01b_feature_extraction": {
                "feature_coverage_min": 0.8,
                "extraction_success_min": 0.9
            },
            "03_clean_text": {
                "cleaning_success_min": 0.9,
                "preservation_rate_min": 0.8
            }
        }

    def validate_complete_pipeline(
        self,
        pipeline_results: Dict[str, Any],
        final_dataset_path: str,
        original_dataset_path: str = None
    ) -> Dict[str, Any]:
        """
        Validação completa e abrangente do pipeline

        Args:
            pipeline_results: Resultados de todas as etapas do pipeline
            final_dataset_path: Caminho para dataset final
            original_dataset_path: Caminho para dataset original (opcional)

        Returns:
            Relatório completo de validação com recomendações
        """
        logger.info("Iniciando validação completa do pipeline")

        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_overview": {},
            "stage_validations": {},
            "cross_stage_analysis": {},
            "data_quality_assessment": {},
            "issue_analysis": {},
            "reprocessing_recommendations": [],
            "overall_assessment": {}
        }

        try:
            # 1. Análise geral do pipeline
            validation_report["pipeline_overview"] = self._analyze_pipeline_overview(pipeline_results)

            # 2. Validação de cada etapa
            validation_report["stage_validations"] = self._validate_individual_stages(pipeline_results)

            # 3. Análise cruzada entre etapas
            validation_report["cross_stage_analysis"] = self._analyze_cross_stage_consistency(pipeline_results)

            # 4. Análise da qualidade dos dados finais
            if os.path.exists(final_dataset_path):
                final_df = pd.read_csv(final_dataset_path, sep=';', encoding='utf-8', nrows=10000)  # Amostra
                validation_report["data_quality_assessment"] = self._assess_final_data_quality(final_df)

            # 5. Análise comparativa com dataset original
            if original_dataset_path and os.path.exists(original_dataset_path):
                original_df = pd.read_csv(original_dataset_path, sep=';', encoding='utf-8', nrows=10000)
                validation_report["data_transformation_analysis"] = self._analyze_data_transformation(
                    original_df, final_df if 'final_df' in locals() else None
                )

            # 6. Análise inteligente via API
            api_analysis = self._comprehensive_analysis_via_api(validation_report)
            validation_report["api_intelligence"] = api_analysis

            # 7. Detecção automática de problemas
            validation_report["issue_analysis"] = self._detect_pipeline_issues(validation_report)

            # 8. Geração de recomendações
            validation_report["reprocessing_recommendations"] = self._generate_reprocessing_recommendations(validation_report)

            # 9. Avaliação geral
            validation_report["overall_assessment"] = self._calculate_overall_assessment(validation_report)

            # 10. Salvar relatório
            self._save_validation_report(validation_report)

        except Exception as e:
            logger.error(f"Erro na validação do pipeline: {e}")
            validation_report["critical_error"] = str(e)

        logger.info("Validação completa do pipeline concluída")
        return validation_report

    def _analyze_pipeline_overview(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa visão geral do pipeline"""

        overview = {
            "stages_executed": [],
            "total_execution_time": 0,
            "stages_with_errors": [],
            "overall_success_rate": 0,
            "data_flow_integrity": True
        }

        total_stages = 0
        successful_stages = 0

        for stage_name, stage_result in pipeline_results.items():
            total_stages += 1
            overview["stages_executed"].append(stage_name)

            if isinstance(stage_result, dict):
                # Analisar resultado da etapa
                if stage_result.get("success", True):
                    successful_stages += 1
                else:
                    overview["stages_with_errors"].append(stage_name)

                # Somar tempo de execução
                execution_time = stage_result.get("execution_time", 0)
                if isinstance(execution_time, (int, float)):
                    overview["total_execution_time"] += execution_time

        overview["overall_success_rate"] = successful_stages / total_stages if total_stages > 0 else 0

        return overview

    def _validate_individual_stages(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Valida cada etapa individualmente"""

        stage_validations = {}

        for stage_name, stage_result in pipeline_results.items():
            validation = {
                "stage_success": False,
                "meets_expectations": False,
                "quality_metrics": {},
                "issues_detected": [],
                "specific_validation": {}
            }

            if isinstance(stage_result, dict):
                validation["stage_success"] = stage_result.get("success", True)

                # Validação específica por tipo de etapa
                if "encoding" in stage_name.lower():
                    validation["specific_validation"] = self._validate_encoding_stage(stage_result)
                elif "deduplication" in stage_name.lower() or "dedup" in stage_name.lower():
                    validation["specific_validation"] = self._validate_deduplication_stage(stage_result)
                elif "feature" in stage_name.lower():
                    validation["specific_validation"] = self._validate_feature_extraction_stage(stage_result)
                elif "clean" in stage_name.lower():
                    validation["specific_validation"] = self._validate_text_cleaning_stage(stage_result)

                # Verificar se atende expectativas
                expected_metrics = self.expected_metrics.get(stage_name, {})
                validation["meets_expectations"] = self._check_stage_expectations(stage_result, expected_metrics)

            stage_validations[stage_name] = validation

        return stage_validations

    def _validate_encoding_stage(self, stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validação específica para etapa de encoding"""

        validation = {
            "encoding_quality_score": stage_result.get("encoding_quality_score", 0),
            "corrections_applied": stage_result.get("corrections_applied", 0),
            "issues_remaining": stage_result.get("issues_remaining", 0),
            "validation_passed": False
        }

        # Critérios de validação
        if validation["encoding_quality_score"] >= 0.9:
            validation["validation_passed"] = True

        return validation

    def _validate_deduplication_stage(self, stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validação específica para etapa de deduplicação"""

        validation = {
            "reduction_ratio": stage_result.get("reduction_ratio", 0),
            "media_exclusion_rate": stage_result.get("media_exclusion_rate", 0),
            "duplicate_count_accuracy": stage_result.get("duplicate_count_accuracy", 0),
            "validation_passed": False
        }

        # Critérios de validação
        if (0.05 <= validation["reduction_ratio"] <= 0.6 and
            validation["media_exclusion_rate"] >= 0.9):
            validation["validation_passed"] = True

        return validation

    def _validate_feature_extraction_stage(self, stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validação específica para extração de features"""

        validation = {
            "features_extracted": stage_result.get("features_extracted", 0),
            "extraction_success_rate": stage_result.get("extraction_success_rate", 0),
            "feature_coverage": stage_result.get("feature_coverage", 0),
            "validation_passed": False
        }

        # Critérios de validação
        if (validation["extraction_success_rate"] >= 0.9 and
            validation["feature_coverage"] >= 0.8):
            validation["validation_passed"] = True

        return validation

    def _validate_text_cleaning_stage(self, stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validação específica para limpeza de texto"""

        validation = {
            "cleaning_success_rate": stage_result.get("cleaning_success_rate", 0),
            "preservation_rate": stage_result.get("preservation_rate", 0),
            "quality_improvement": stage_result.get("quality_improvement", 0),
            "validation_passed": False
        }

        # Critérios de validação
        if (validation["cleaning_success_rate"] >= 0.9 and
            validation["preservation_rate"] >= 0.8):
            validation["validation_passed"] = True

        return validation

    def _check_stage_expectations(self, stage_result: Dict[str, Any], expected_metrics: Dict[str, Any]) -> bool:
        """Verifica se etapa atende expectativas"""

        for metric_name, expected_value in expected_metrics.items():
            actual_value = stage_result.get(metric_name)

            if actual_value is None:
                continue

            # Verificar diferentes tipos de critérios
            if metric_name.endswith("_min") and actual_value < expected_value:
                return False
            elif metric_name.endswith("_max") and actual_value > expected_value:
                return False
            elif metric_name.endswith("_expected") and abs(actual_value - expected_value) > 0.1:
                return False

        return True

    def _analyze_cross_stage_consistency(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa consistência entre etapas"""

        analysis = {
            "data_flow_consistency": True,
            "metric_coherence": True,
            "stage_dependencies_met": True,
            "inconsistencies_detected": [],
            "data_size_progression": {}
        }

        # Analisar progressão do tamanho dos dados
        sizes = {}
        for stage_name, stage_result in pipeline_results.items():
            if isinstance(stage_result, dict):
                size = stage_result.get("output_records", stage_result.get("dataset_size"))
                if size:
                    sizes[stage_name] = size

        analysis["data_size_progression"] = sizes

        # Verificar consistência lógica
        if "02b_deduplication" in sizes and "01_validate_data" in sizes:
            reduction = (sizes["01_validate_data"] - sizes["02b_deduplication"]) / sizes["01_validate_data"]
            if reduction < 0.01 or reduction > 0.8:
                analysis["inconsistencies_detected"].append(
                    f"Redução suspeita na deduplicação: {reduction:.1%}"
                )
                analysis["data_flow_consistency"] = False

        return analysis

    def _assess_final_data_quality(self, final_df: pd.DataFrame) -> Dict[str, Any]:
        """Avalia qualidade dos dados finais"""

        assessment = {
            "total_records": len(final_df),
            "column_completeness": {},
            "data_types_correct": True,
            "outliers_detected": [],
            "quality_score": 0.0,
            "critical_issues": []
        }

        # Análise de completude das colunas
        for col in final_df.columns:
            non_null_count = final_df[col].notna().sum()
            completeness = non_null_count / len(final_df)
            assessment["column_completeness"][col] = round(completeness, 3)

            # Identificar colunas críticas com baixa completude
            if col in ["body", "body_cleaned", "channel", "datetime"] and completeness < 0.95:
                assessment["critical_issues"].append(f"Coluna crítica {col} com baixa completude: {completeness:.1%}")

        # Verificar tipos de dados essenciais
        expected_columns = ["body", "body_cleaned", "channel"]
        missing_columns = [col for col in expected_columns if col not in final_df.columns]
        if missing_columns:
            assessment["critical_issues"].append(f"Colunas essenciais ausentes: {missing_columns}")
            assessment["data_types_correct"] = False

        # Calcular score de qualidade
        avg_completeness = sum(assessment["column_completeness"].values()) / len(assessment["column_completeness"])
        assessment["quality_score"] = avg_completeness * (0.8 if assessment["data_types_correct"] else 0.5)

        return assessment

    def _analyze_data_transformation(self, original_df: pd.DataFrame, final_df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa transformação dos dados do original ao final"""

        if final_df is None:
            return {"error": "Dataset final não disponível"}

        analysis = {
            "size_change": {
                "original": len(original_df),
                "final": len(final_df),
                "reduction_ratio": (len(original_df) - len(final_df)) / len(original_df)
            },
            "column_changes": {
                "original_columns": len(original_df.columns),
                "final_columns": len(final_df.columns),
                "new_columns": [col for col in final_df.columns if col not in original_df.columns],
                "removed_columns": [col for col in original_df.columns if col not in final_df.columns]
            },
            "data_integrity": True,
            "transformation_quality": 0.0
        }

        # Verificar integridade dos dados
        if analysis["size_change"]["reduction_ratio"] > 0.8:
            analysis["data_integrity"] = False

        # Calcular qualidade da transformação
        new_features_count = len(analysis["column_changes"]["new_columns"])
        size_preservation = 1 - min(analysis["size_change"]["reduction_ratio"], 0.8)

        analysis["transformation_quality"] = (size_preservation + (new_features_count / 20)) / 2

        return analysis

    def _comprehensive_analysis_via_api(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Análise abrangente via API"""

        # Preparar resumo para API
        summary = {
            "pipeline_overview": validation_report.get("pipeline_overview", {}),
            "stage_success_rates": {},
            "critical_issues": [],
            "data_quality_metrics": validation_report.get("data_quality_assessment", {})
        }

        # Extrair taxas de sucesso das etapas
        stage_validations = validation_report.get("stage_validations", {})
        for stage, validation in stage_validations.items():
            summary["stage_success_rates"][stage] = validation.get("stage_success", False)

        # Usar error handler para análise com retry
        result = self.error_handler.execute_with_retry(
            self._analyze_pipeline_intelligence_api,
            stage="pipeline_validation",
            operation="comprehensive_analysis",
            summary=summary
        )

        if result.success:
            return result.data
        else:
            logger.warning(f"Falha na análise via API: {result.error.error_message}")
            return {"error": result.error.error_message}

    def _analyze_pipeline_intelligence_api(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Usa API para análise inteligente do pipeline"""

        prompt = f"""
Analise este relatório de validação de pipeline de processamento de dados do Telegram brasileiro:

RESUMO DO PIPELINE:
{json.dumps(summary, indent=2, ensure_ascii=False)}

Como especialista em pipeline de dados, avalie:

1. QUALIDADE GERAL: O pipeline foi executado adequadamente?
2. PROBLEMAS CRÍTICOS: Há problemas que requerem reprocessamento?
3. ETAPAS PROBLEMÁTICAS: Quais etapas precisam de atenção?
4. INTEGRIDADE DOS DADOS: Os dados finais são confiáveis?
5. RECOMENDAÇÕES: Que melhorias devem ser implementadas?

CONTEXTO:
- Pipeline para análise de discurso político brasileiro
- Dataset do Telegram (2019-2023)
- Etapas: validação, encoding, deduplicação, extração de features, limpeza

Responda em formato JSON:
{{
  "overall_assessment": {{
    "pipeline_quality": "excelente|bom|satisfatorio|ruim|critico",
    "data_reliability": "alta|media|baixa|critica",
    "ready_for_analysis": true/false,
    "confidence_score": 0.0-1.0
  }},
  "critical_issues_identified": [
    "problema1", "problema2"
  ],
  "stages_requiring_attention": [
    {{
      "stage": "nome_stage",
      "issue": "descrição do problema",
      "severity": "alto|medio|baixo",
      "action_required": "reprocessar|ajustar|monitorar"
    }}
  ],
  "data_quality_concerns": [
    "preocupacao1", "preocupacao2"
  ],
  "strategic_recommendations": [
    {{
      "recommendation": "descrição da recomendação",
      "priority": "alta|media|baixa",
      "impact": "alto|medio|baixo",
      "effort": "alto|medio|baixo"
    }}
  ],
  "next_steps": [
    "acao1", "acao2"
  ]
}}
"""

        try:
            response = self.create_message(
                prompt,
                stage="pipeline_validation",
                operation="intelligence_analysis",
                temperature=0.2
            )

            return self.parse_json_response(response)

        except Exception as e:
            logger.error(f"Erro na análise inteligente via API: {e}")
            return {}

    def _detect_pipeline_issues(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta problemas automaticamente no pipeline"""

        issues = {
            "critical_issues": [],
            "warning_issues": [],
            "quality_degradation": [],
            "data_integrity_problems": [],
            "performance_issues": []
        }

        # Analisar problemas críticos
        overall_success_rate = validation_report.get("pipeline_overview", {}).get("overall_success_rate", 1.0)
        if overall_success_rate < 0.8:
            issues["critical_issues"].append(f"Taxa de sucesso baixa do pipeline: {overall_success_rate:.1%}")

        # Analisar qualidade dos dados
        data_quality = validation_report.get("data_quality_assessment", {})
        quality_score = data_quality.get("quality_score", 1.0)
        if quality_score < 0.7:
            issues["quality_degradation"].append(f"Score de qualidade baixo: {quality_score:.2f}")

        # Analisar problemas de integridade
        critical_issues = data_quality.get("critical_issues", [])
        if critical_issues:
            issues["data_integrity_problems"].extend(critical_issues)

        # Analisar inconsistências entre etapas
        cross_stage = validation_report.get("cross_stage_analysis", {})
        if not cross_stage.get("data_flow_consistency", True):
            issues["data_integrity_problems"].append("Inconsistência no fluxo de dados entre etapas")

        inconsistencies = cross_stage.get("inconsistencies_detected", [])
        if inconsistencies:
            issues["warning_issues"].extend(inconsistencies)

        return issues

    def _generate_reprocessing_recommendations(self, validation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera recomendações de reprocessamento"""

        recommendations = []

        # Analisar resultados da API
        api_analysis = validation_report.get("api_intelligence", {})
        if "stages_requiring_attention" in api_analysis:
            for stage_issue in api_analysis["stages_requiring_attention"]:
                if stage_issue.get("action_required") == "reprocessar":
                    recommendations.append({
                        "type": "reprocess_stage",
                        "stage": stage_issue.get("stage"),
                        "reason": stage_issue.get("issue"),
                        "priority": stage_issue.get("severity", "medio"),
                        "estimated_effort": "medio"
                    })

        # Analisar problemas detectados
        issues = validation_report.get("issue_analysis", {})
        critical_issues = issues.get("critical_issues", [])

        if critical_issues:
            recommendations.append({
                "type": "full_reprocess",
                "reason": "Problemas críticos detectados",
                "priority": "alta",
                "estimated_effort": "alto",
                "issues": critical_issues
            })

        # Analisar qualidade dos dados
        data_quality = validation_report.get("data_quality_assessment", {})
        if data_quality.get("quality_score", 1.0) < 0.5:
            recommendations.append({
                "type": "quality_improvement",
                "reason": "Qualidade dos dados abaixo do aceitável",
                "priority": "alta",
                "estimated_effort": "medio",
                "suggested_actions": [
                    "Revisar parâmetros de limpeza",
                    "Melhorar detecção de encoding",
                    "Ajustar critérios de deduplicação"
                ]
            })

        return recommendations

    def _calculate_overall_assessment(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula avaliação geral do pipeline"""

        assessment = {
            "overall_score": 0.0,
            "quality_level": "ruim",
            "ready_for_analysis": False,
            "confidence_level": "baixo",
            "summary": "",
            "key_metrics": {}
        }

        # Calcular score geral
        pipeline_success = validation_report.get("pipeline_overview", {}).get("overall_success_rate", 0)
        data_quality = validation_report.get("data_quality_assessment", {}).get("quality_score", 0)

        # Score ponderado
        assessment["overall_score"] = (pipeline_success * 0.4 + data_quality * 0.6)

        # Determinar nível de qualidade
        if assessment["overall_score"] >= 0.9:
            assessment["quality_level"] = "excelente"
            assessment["ready_for_analysis"] = True
            assessment["confidence_level"] = "alto"
        elif assessment["overall_score"] >= 0.7:
            assessment["quality_level"] = "bom"
            assessment["ready_for_analysis"] = True
            assessment["confidence_level"] = "medio"
        elif assessment["overall_score"] >= 0.5:
            assessment["quality_level"] = "satisfatorio"
            assessment["ready_for_analysis"] = True
            assessment["confidence_level"] = "medio"
        else:
            assessment["quality_level"] = "ruim"
            assessment["ready_for_analysis"] = False
            assessment["confidence_level"] = "baixo"

        # Usar resultado da API se disponível
        api_assessment = validation_report.get("api_intelligence", {}).get("overall_assessment", {})
        if api_assessment:
            assessment["ready_for_analysis"] = api_assessment.get("ready_for_analysis", assessment["ready_for_analysis"])
            api_confidence = api_assessment.get("confidence_score", 0)
            if api_confidence > 0:
                assessment["overall_score"] = (assessment["overall_score"] + api_confidence) / 2

        # Métricas chave
        assessment["key_metrics"] = {
            "pipeline_success_rate": pipeline_success,
            "data_quality_score": data_quality,
            "stages_completed": len(validation_report.get("stage_validations", {})),
            "critical_issues_count": len(validation_report.get("issue_analysis", {}).get("critical_issues", [])),
            "reprocessing_recommendations": len(validation_report.get("reprocessing_recommendations", []))
        }

        # Resumo
        if assessment["ready_for_analysis"]:
            assessment["summary"] = f"Pipeline executado com qualidade {assessment['quality_level']}. Dados prontos para análise."
        else:
            assessment["summary"] = f"Pipeline com problemas de qualidade. Reprocessamento recomendado."

        return assessment

    def _save_validation_report(self, validation_report: Dict[str, Any]):
        """Salva relatório de validação"""

        try:
            # Salvar relatório completo
            report_file = self.project_root / "logs" / "pipeline" / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, ensure_ascii=False)

            logger.info(f"Relatório de validação salvo: {report_file}")

            # Salvar resumo executivo
            summary_file = self.project_root / "logs" / "pipeline" / "latest_validation_summary.json"
            summary = {
                "timestamp": validation_report["timestamp"],
                "overall_assessment": validation_report.get("overall_assessment", {}),
                "critical_issues": validation_report.get("issue_analysis", {}).get("critical_issues", []),
                "reprocessing_needed": len(validation_report.get("reprocessing_recommendations", [])) > 0
            }

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")

    def escalate_critical_issues(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Escala problemas críticos para o usuário"""

        critical_issues = validation_report.get("issue_analysis", {}).get("critical_issues", [])

        if critical_issues:
            escalation_info = self.error_handler.escalate_to_user(
                error=type('CriticalPipelineIssue', (), {
                    'stage': 'pipeline_validation',
                    'operation': 'complete_validation',
                    'error_type': 'CriticalQualityIssue',
                    'error_message': f"Problemas críticos detectados: {'; '.join(critical_issues)}",
                    'timestamp': datetime.now(),
                    'retry_count': 0
                })(),
                context={
                    "validation_report": validation_report,
                    "reprocessing_recommendations": validation_report.get("reprocessing_recommendations", [])
                }
            )

            return escalation_info

        return {}
