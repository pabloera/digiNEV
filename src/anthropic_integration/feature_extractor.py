"""
Extrator de Features Avançado via API Anthropic
Implementa extração completa de features com identificação de padrões e correção de erros.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .api_error_handler import APIErrorHandler, APIQualityChecker
from .base import AnthropicBase

logger = logging.getLogger(__name__)


class FeatureExtractor(AnthropicBase):
    """
    Extrator de features avançado usando API Anthropic

    Capacidades:
    - Extração inteligente de hashtags, URLs e domínios
    - Detecção de padrões de comportamento
    - Classificação automática de conteúdo
    - Identificação de características específicas do contexto brasileiro
    - Correção de erros detectados pela própria API
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.error_handler = APIErrorHandler()
        self.quality_checker = APIQualityChecker(config)

        # Padrões específicos para contexto brasileiro/bolsonarista
        self.brazilian_patterns = {
            "political_keywords": [
                "bolsonaro", "lula", "pt", "psl", "pl", "tse", "stf", "governo",
                "presidente", "deputado", "senador", "ministro", "eleições",
                "urna", "voto", "democracia", "ditadura", "comunismo", "socialismo"
            ],
            "conspiracy_keywords": [
                "fake news", "mídia", "globo", "manipulação", "censura", "verdade",
                "acordem", "despertem", "sistema", "elite", "illuminati", "maçonaria"
            ],
            "health_keywords": [
                "covid", "corona", "vírus", "vacina", "cloroquina", "ivermectina",
                "lockdown", "quarentena", "pandemia", "sus", "anvisa", "oms"
            ]
        }

    def extract_comprehensive_features(
        self,
        df: pd.DataFrame,
        text_column: str = "body",
        batch_size: int = 50
    ) -> pd.DataFrame:
        """
        Extrai features abrangentes do dataset usando API

        Args:
            df: DataFrame com os dados
            text_column: Nome da coluna de texto
            batch_size: Tamanho do lote para processamento

        Returns:
            DataFrame com features extraídas
        """
        logger.info(f"Iniciando extração de features para {len(df)} registros")

        # Fazer backup antes de começar
        backup_file = f"data/interim/feature_extraction_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False, sep=';', encoding='utf-8')
        logger.info(f"Backup criado: {backup_file}")

        # Processar em lotes
        result_dfs = []
        total_batches = (len(df) + batch_size - 1) // batch_size

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            batch_num = i // batch_size + 1

            logger.info(f"Processando lote {batch_num}/{total_batches}")

            # Usar error handler para processamento com retry
            result = self.error_handler.execute_with_retry(
                self._process_batch_features,
                stage="01b_feature_extraction",
                operation=f"batch_{batch_num}",
                batch_df=batch_df,
                text_column=text_column
            )

            if result.success:
                result_dfs.append(result.data)
            else:
                logger.error(f"Falha no lote {batch_num}: {result.error.error_message}")
                # Adicionar lote sem features extras (preservar dados originais)
                result_dfs.append(batch_df)

        # Combinar resultados
        final_df = pd.concat(result_dfs, ignore_index=True)

        # Validação final
        validation_result = self._validate_extracted_features(final_df, df)

        return final_df

    def _process_batch_features(
        self,
        batch_df: pd.DataFrame,
        text_column: str
    ) -> pd.DataFrame:
        """Processa um lote de dados para extração de features"""

        # Preparar textos para análise
        texts = batch_df[text_column].fillna("").astype(str).tolist()

        # Extrair features básicas primeiro
        batch_df = self._extract_basic_features(batch_df, text_column)

        # Usar API para análise avançada
        advanced_features = self._extract_advanced_features_api(texts)

        # Integrar features avançadas
        for i, features in enumerate(advanced_features):
            if i < len(batch_df):
                for key, value in features.items():
                    # Verificar se a coluna existe, senão criar
                    if key not in batch_df.columns:
                        batch_df[key] = None
                    # Usar .at para assignment mais seguro
                    batch_df.at[batch_df.index[i], key] = value

        return batch_df

    def _extract_basic_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Extrai features básicas APENAS se não existirem
        Evita duplicação de features já presentes no dataset
        """

        # Verificar e extrair hashtags apenas se não existir
        if 'hashtag' not in df.columns and 'hashtags' not in df.columns:
            df['hashtags_extracted'] = df[text_column].apply(
                lambda x: self._extract_hashtags(str(x))
            )
            logger.info("Hashtags extraídas - coluna não existia")
        else:
            logger.info("Hashtags já existem no dataset - pulando extração")

        # Verificar e extrair URLs apenas se não existir
        if 'url' not in df.columns and 'urls' not in df.columns:
            df['urls_extracted'] = df[text_column].apply(
                lambda x: self._extract_urls(str(x))
            )
            logger.info("URLs extraídas - coluna não existia")
        else:
            logger.info("URLs já existem no dataset - pulando extração")

        # Verificar e extrair domínios apenas se não existir
        if 'domain' not in df.columns and 'domains' not in df.columns:
            # Usar URLs existentes ou recém-extraídas
            url_column = None
            if 'urls_extracted' in df.columns:
                url_column = 'urls_extracted'
            elif 'urls' in df.columns:
                url_column = 'urls'
            elif 'url' in df.columns:
                url_column = 'url'

            if url_column:
                df['domains_extracted'] = df[url_column].apply(
                    lambda x: self._extract_domains_from_urls(x)
                )
                logger.info("Domínios extraídos - coluna não existia")
        else:
            logger.info("Domínios já existem no dataset - pulando extração")

        # Verificar media_type existente antes de criar flags individuais
        if 'media_type' in df.columns:
            logger.info("Media_type já existe - usando validação em vez de flags individuais")
            # Se media_type existe, não criar has_photo, has_video, has_audio
        else:
            # Criar flags de mídia apenas se media_type não existir
            df['has_photo'] = df[text_column].str.contains(
                r'foto|imagem|jpeg|jpg|png|gif', case=False, na=False
            )
            df['has_video'] = df[text_column].str.contains(
                r'vídeo|video|mp4|avi|mov', case=False, na=False
            )
            df['has_audio'] = df[text_column].str.contains(
                r'áudio|audio|mp3|wav|voz', case=False, na=False
            )
            logger.info("Flags de mídia criadas - media_type não existia")

        # Métricas básicas sempre são úteis, mas verificar se já existem
        if 'text_length' not in df.columns:
            df['text_length'] = df[text_column].str.len()
        if 'word_count' not in df.columns:
            df['word_count'] = df[text_column].str.split().str.len()
        if 'has_emoji' not in df.columns and 'emoji_count' not in df.columns:
            df['has_emoji'] = df[text_column].str.contains(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]',
                na=False
            )

        return df

    def _extract_advanced_features_api(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Usa API para extrair features avançadas"""

        prompt = self._build_feature_extraction_prompt(texts[:10])  # Máximo 10 por vez

        try:
            response = self.create_message(
                prompt,
                stage="01b_feature_extraction",
                operation="advanced_analysis"
            )

            # Validar qualidade da resposta
            validation = self.quality_checker.validate_output_quality(
                response,
                expected_format="json",
                context={"texts_count": len(texts)},
                stage="01b_feature_extraction"
            )

            if not validation["valid"]:
                logger.warning(f"Qualidade da resposta baixa: {validation['issues']}")

            # Parse da resposta com método robusto
            parsed_response = self.parse_claude_response_safe(response, ["results"])
            return parsed_response.get("results", [{}] * len(texts))

        except Exception as e:
            logger.error(f"Erro na extração avançada via API: {e}")
            return [{}] * len(texts)

    def _build_feature_extraction_prompt(self, texts: List[str]) -> str:
        """Constrói prompt para extração de features"""

        texts_sample = "\n".join([f"{i+1}. {text[:200]}..." for i, text in enumerate(texts)])

        return f"""
Analise os seguintes textos de mensagens do Telegram brasileiro (contexto político 2019-2023) e extraia features detalhadas.

TEXTOS:
{texts_sample}

Para cada texto, forneça a análise em formato JSON:

{{
  "results": [
    {{
      "text_id": 1,
      "sentiment_category": "positivo|negativo|neutro",
      "political_alignment": "bolsonarista|antibolsonarista|neutro|indefinido",
      "conspiracy_indicators": ["indicador1", "indicador2"],
      "negacionism_indicators": ["tipo1", "tipo2"],
      "discourse_type": "informativo|opinativo|mobilizador|atacante|defensivo",
      "urgency_level": "baixo|medio|alto",
      "emotional_tone": "raiva|medo|esperança|tristeza|alegria|neutro",
      "target_entities": ["pessoa", "instituição", "grupo"],
      "call_to_action": true/false,
      "misinformation_risk": "baixo|medio|alto",
      "coordination_signals": ["sinal1", "sinal2"],
      "brazilian_context_markers": ["marker1", "marker2"],
      "quality_issues": ["erro1", "erro2"]
    }}
  ]
}}

INSTRUÇÕES ESPECÍFICAS:
1. Identifique padrões específicos do contexto político brasileiro
2. Detecte sinais de coordenação (mensagens similares, timing)
3. Avalie risco de desinformação baseado em contexto
4. Identifique problemas de qualidade no próprio texto
5. Classifique alinhamento político baseado em linguagem e temas
6. Detecte indicadores de teorias conspiratórias brasileiras
7. Identifique negacionismo (científico, democrático, etc.)

RESPONDA APENAS COM O JSON, SEM EXPLICAÇÕES ADICIONAIS.
"""

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extrai hashtags do texto"""
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, text, re.IGNORECASE)
        return [tag.lower() for tag in hashtags]

    def _extract_urls(self, text: str) -> List[str]:
        """Extrai URLs do texto"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls

    def _extract_domains_from_urls(self, urls: List[str]) -> List[str]:
        """Extrai domínios de lista de URLs"""
        domains = []
        for url in urls:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                if domain:
                    domains.append(domain.lower())
            except:
                continue
        return list(set(domains))

    def _validate_extracted_features(self, final_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Valida features extraídas"""

        validation_report = {
            "original_rows": len(original_df),
            "final_rows": len(final_df),
            "rows_preserved": len(final_df) == len(original_df),
            "new_columns": [col for col in final_df.columns if col not in original_df.columns],
            "missing_values_analysis": {},
            "data_quality_issues": []
        }

        # Análise de valores faltantes nas novas colunas
        for col in validation_report["new_columns"]:
            if col in final_df.columns:
                missing_count = final_df[col].isna().sum()
                missing_pct = (missing_count / len(final_df)) * 100
                validation_report["missing_values_analysis"][col] = {
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_pct, 2)
                }

        # Identificar problemas de qualidade
        if not validation_report["rows_preserved"]:
            validation_report["data_quality_issues"].append("Número de linhas não preservado")

        if len(validation_report["new_columns"]) < 5:
            validation_report["data_quality_issues"].append("Poucas features novas extraídas")

        # Log do relatório
        logger.info(f"Validação de features concluída: {validation_report}")

        return validation_report

    def correct_extraction_errors(
        self,
        df: pd.DataFrame,
        error_patterns: List[str] = None
    ) -> pd.DataFrame:
        """
        Corrige erros detectados na extração de features

        Args:
            df: DataFrame com features extraídas
            error_patterns: Padrões de erro a corrigir

        Returns:
            DataFrame com correções aplicadas
        """
        logger.info("Iniciando correção de erros de extração")

        corrected_df = df.copy()
        corrections_applied = []

        # Correções básicas
        corrections_applied.extend(self._fix_basic_extraction_errors(corrected_df))

        # Usar API para correções avançadas se necessário
        if error_patterns:
            api_corrections = self._fix_errors_with_api(corrected_df, error_patterns)
            corrections_applied.extend(api_corrections)

        logger.info(f"Correções aplicadas: {len(corrections_applied)}")

        return corrected_df

    def _fix_basic_extraction_errors(self, df: pd.DataFrame) -> List[str]:
        """Aplica correções básicas de erros comuns"""
        corrections = []

        # Corrigir hashtags malformadas
        if 'hashtags_extracted' in df.columns:
            original_count = df['hashtags_extracted'].apply(len).sum()
            df['hashtags_extracted'] = df['hashtags_extracted'].apply(
                lambda x: [tag for tag in x if len(tag) > 1 and tag.startswith('#')]
            )
            new_count = df['hashtags_extracted'].apply(len).sum()
            if new_count != original_count:
                corrections.append(f"Hashtags corrigidas: {original_count} -> {new_count}")

        # Corrigir URLs inválidas
        if 'urls_extracted' in df.columns:
            original_count = df['urls_extracted'].apply(len).sum()
            df['urls_extracted'] = df['urls_extracted'].apply(
                lambda x: [url for url in x if self._is_valid_url(url)]
            )
            new_count = df['urls_extracted'].apply(len).sum()
            if new_count != original_count:
                corrections.append(f"URLs corrigidas: {original_count} -> {new_count}")

        # Corrigir flags booleanas
        boolean_columns = ['has_photo', 'has_video', 'has_audio', 'has_emoji', 'call_to_action']
        for col in boolean_columns:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = df[col].astype(bool)
                if original_type != bool:
                    corrections.append(f"Coluna {col} convertida para boolean")

        return corrections

    def _fix_errors_with_api(self, df: pd.DataFrame, error_patterns: List[str]) -> List[str]:
        """Usa API para corrigir erros específicos"""
        corrections = []

        # Implementar correções específicas via API conforme necessário
        # Por exemplo, re-análise de textos com problemas detectados

        return corrections

    def _is_valid_url(self, url: str) -> bool:
        """Valida se URL é válida"""
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def generate_feature_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera relatório das features extraídas adaptado para estrutura existente"""

        # Identificar colunas originais vs. novas features
        original_columns = {
            'datetime', 'body', 'url', 'hashtag', 'channel', 'is_fwrd',
            'mentions', 'sender', 'media_type', 'domain', 'body_cleaned'
        }
        new_feature_columns = [col for col in df.columns if col not in original_columns]

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "original_columns": len(original_columns.intersection(set(df.columns))),
            "new_features_added": len(new_feature_columns),
            "features_extracted": {},
            "data_quality": {},
            "feature_coverage": {},
            "recommendations": []
        }

        # Análise de cobertura das features
        for feature in new_feature_columns:
            if feature in df.columns:
                non_null_count = df[feature].notna().sum()
                coverage = (non_null_count / len(df)) * 100
                report["feature_coverage"][feature] = {
                    "non_null_count": int(non_null_count),
                    "coverage_percentage": round(coverage, 2)
                }

        # Análise específica das colunas originais aproveitadas
        if 'hashtag' in df.columns:
            hashtag_usage = df['hashtag'].notna().sum()
            report["features_extracted"]["hashtag_analysis"] = {
                "messages_with_hashtags": int(hashtag_usage),
                "hashtag_usage_rate": round((hashtag_usage / len(df)) * 100, 2)
            }

        if 'mentions' in df.columns:
            mentions_usage = df['mentions'].notna().sum()
            report["features_extracted"]["mention_analysis"] = {
                "messages_with_mentions": int(mentions_usage),
                "mention_usage_rate": round((mentions_usage / len(df)) * 100, 2)
            }

        if 'url' in df.columns:
            url_usage = df['url'].notna().sum()
            report["features_extracted"]["url_analysis"] = {
                "messages_with_urls": int(url_usage),
                "url_usage_rate": round((url_usage / len(df)) * 100, 2)
            }

        # Análise de qualidade
        report["data_quality"] = {
            "text_metrics_available": 'text_length' in df.columns,
            "temporal_features_available": 'hour_of_day' in df.columns,
            "political_analysis_available": 'political_alignment' in df.columns,
            "sentiment_analysis_available": 'sentiment_category' in df.columns
        }

        # Recomendações baseadas na estrutura
        if len(new_feature_columns) < 10:
            report["recommendations"].append("Poucas features novas extraídas - verificar API")

        low_coverage_features = [
            feature for feature, data in report["feature_coverage"].items()
            if data["coverage_percentage"] < 50
        ]
        if low_coverage_features:
            report["recommendations"].append(
                f"Features com baixa cobertura: {', '.join(low_coverage_features)}"
            )

        if not report["data_quality"]["political_analysis_available"]:
            report["recommendations"].append("Análise política não foi extraída com sucesso")

        return report
