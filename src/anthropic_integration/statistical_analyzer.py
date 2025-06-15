"""
Statistical Analyzer para Análise Dual (Antes/Depois da Limpeza)
Gera estatísticas detalhadas de hashtags, canais, URLs e padrões de encaminhamento.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .api_error_handler import APIErrorHandler, APIQualityChecker
from .base import AnthropicBase

logger = logging.getLogger(__name__)

class StatisticalAnalyzer(AnthropicBase):
    """
    Analisador estatístico para comparação antes/depois da limpeza

    Capacidades:
    - Análise de hashtags (antes/depois da limpeza)
    - Análise de canais e distribuição
    - Análise de URLs e domínios
    - Análise de padrões de encaminhamento
    - Comparação de impacto da limpeza
    - Relatórios detalhados para dashboard
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.error_handler = APIErrorHandler()
        self.quality_checker = APIQualityChecker(config)

        # Padrões para extração de elementos
        self.patterns = {
            "hashtags": r'#[\w\u00C0-\u00FF]+',
            "mentions": r'@[\w\u00C0-\u00FF]+',
            "urls": r'https?://[^\s<>"{}|\\^`\[\]]+',
            "domains": r'https?://(?:www\.)?([^/\s<>"{}|\\^`\[\]]+)',
            "forwards": r'(encaminhada|forwarded|reencaminhada|compartilhada)',
            "emojis": r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]',
            "phone_numbers": r'\+?55\s*\(?[1-9]{2}\)?\s*[0-9]{4,5}-?[0-9]{4}',
            "timestamps": r'\d{1,2}:\d{2}(?::\d{2})?'
        }

        # Categorias para análise de conteúdo
        self.content_categories = {
            "political": ["bolsonaro", "lula", "eleição", "voto", "política", "presidente", "brasil"],
            "media": ["foto", "vídeo", "áudio", "imagem", "link", "arquivo"],
            "social": ["família", "amigos", "grupo", "pessoal", "privado"],
            "news": ["notícia", "jornal", "reportagem", "mídia", "imprensa"],
            "conspiracy": ["fake", "mentira", "conspiração", "teoria", "verdade"],
            "religious": ["deus", "igreja", "oração", "fé", "religião", "jesus"]
        }

    def analyze_pre_cleaning_statistics(
        self,
        df: pd.DataFrame,
        text_column: str = "body",
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Análise estatística antes da limpeza

        Args:
            df: DataFrame para analisar
            text_column: Coluna de texto principal
            output_file: Arquivo para salvar relatório (opcional)

        Returns:
            Dicionário com estatísticas pré-limpeza
        """
        logger.info(f"Iniciando análise estatística PRÉ-limpeza para {len(df)} registros")

        analysis = {
            "analysis_type": "pre_cleaning",
            "timestamp": datetime.now().isoformat(),
            "dataset_info": self._get_dataset_info(df),
            "text_statistics": self._analyze_text_statistics(df, text_column),
            "hashtag_analysis": self._analyze_hashtags(df, text_column),
            "channel_analysis": self._analyze_channels(df),
            "url_analysis": self._analyze_urls(df, text_column),
            "forward_analysis": self._analyze_forwards(df, text_column),
            "temporal_analysis": self._analyze_temporal_patterns(df),
            "content_categorization": self._categorize_content(df, text_column),
            "quality_metrics": self._calculate_quality_metrics(df, text_column)
        }

        # Salvar relatório se especificado
        if output_file:
            self._save_analysis_report(analysis, output_file)

        logger.info("Análise pré-limpeza concluída")
        return analysis

    def analyze_post_cleaning_statistics(
        self,
        df: pd.DataFrame,
        text_column: str = "body_cleaned",
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Análise estatística após a limpeza

        Args:
            df: DataFrame para analisar
            text_column: Coluna de texto limpo
            output_file: Arquivo para salvar relatório (opcional)

        Returns:
            Dicionário com estatísticas pós-limpeza
        """
        logger.info(f"Iniciando análise estatística PÓS-limpeza para {len(df)} registros")

        analysis = {
            "analysis_type": "post_cleaning",
            "timestamp": datetime.now().isoformat(),
            "dataset_info": self._get_dataset_info(df),
            "text_statistics": self._analyze_text_statistics(df, text_column),
            "hashtag_analysis": self._analyze_hashtags(df, text_column),
            "channel_analysis": self._analyze_channels(df),
            "url_analysis": self._analyze_urls(df, text_column),
            "forward_analysis": self._analyze_forwards(df, text_column),
            "temporal_analysis": self._analyze_temporal_patterns(df),
            "content_categorization": self._categorize_content(df, text_column),
            "quality_metrics": self._calculate_quality_metrics(df, text_column),
            "cleaning_validation": self._validate_cleaning_effectiveness(df, text_column)
        }

        # Salvar relatório se especificado
        if output_file:
            self._save_analysis_report(analysis, output_file)

        logger.info("Análise pós-limpeza concluída")
        return analysis

    def compare_before_after_cleaning(
        self,
        pre_analysis: Dict[str, Any],
        post_analysis: Dict[str, Any],
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Compara estatísticas antes e depois da limpeza

        Args:
            pre_analysis: Análise pré-limpeza
            post_analysis: Análise pós-limpeza
            output_file: Arquivo para salvar comparação

        Returns:
            Relatório de comparação detalhado
        """
        logger.info("Gerando comparação antes/depois da limpeza")

        comparison = {
            "comparison_type": "before_after_cleaning",
            "timestamp": datetime.now().isoformat(),
            "dataset_changes": self._compare_dataset_info(pre_analysis, post_analysis),
            "text_changes": self._compare_text_statistics(pre_analysis, post_analysis),
            "hashtag_changes": self._compare_hashtags(pre_analysis, post_analysis),
            "url_changes": self._compare_urls(pre_analysis, post_analysis),
            "quality_improvements": self._compare_quality_metrics(pre_analysis, post_analysis),
            "cleaning_effectiveness": self._assess_cleaning_effectiveness(pre_analysis, post_analysis),
            "recommendations": self._generate_cleaning_recommendations(pre_analysis, post_analysis)
        }

        # Salvar comparação se especificado
        if output_file:
            self._save_analysis_report(comparison, output_file)

        logger.info("Comparação concluída")
        return comparison

    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Obtém informações básicas do dataset"""

        info = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "null_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        return info

    def _analyze_text_statistics(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Analisa estatísticas de texto"""

        if text_column not in df.columns:
            return {"error": f"Coluna '{text_column}' não encontrada"}

        text_series = df[text_column].fillna("").astype(str)

        # Estatísticas básicas
        lengths = text_series.str.len()
        word_counts = text_series.str.split().str.len().fillna(0)

        stats = {
            "total_messages": len(text_series),
            "non_empty_messages": (lengths > 0).sum(),
            "empty_messages": (lengths == 0).sum(),
            "total_characters": lengths.sum(),
            "total_words": word_counts.sum(),
            "length_statistics": {
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "mean_length": round(lengths.mean(), 2),
                "median_length": round(lengths.median(), 2),
                "std_length": round(lengths.std(), 2)
            },
            "word_statistics": {
                "min_words": int(word_counts.min()),
                "max_words": int(word_counts.max()),
                "mean_words": round(word_counts.mean(), 2),
                "median_words": round(word_counts.median(), 2)
            },
            "content_distribution": {
                "very_short": int((lengths <= 10).sum()),
                "short": int(((lengths > 10) & (lengths <= 50)).sum()),
                "medium": int(((lengths > 50) & (lengths <= 200)).sum()),
                "long": int(((lengths > 200) & (lengths <= 500)).sum()),
                "very_long": int((lengths > 500).sum())
            }
        }

        return stats

    def _analyze_hashtags(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Analisa hashtags no texto"""

        if text_column not in df.columns:
            return {"error": f"Coluna '{text_column}' não encontrada"}

        text_series = df[text_column].fillna("").astype(str)

        # Extrair hashtags
        all_hashtags = []
        for text in text_series:
            hashtags = re.findall(self.patterns["hashtags"], text, re.IGNORECASE)
            all_hashtags.extend([tag.lower() for tag in hashtags])

        hashtag_counts = Counter(all_hashtags)

        analysis = {
            "total_hashtags": len(all_hashtags),
            "unique_hashtags": len(hashtag_counts),
            "messages_with_hashtags": sum(1 for text in text_series if re.search(self.patterns["hashtags"], text)),
            "hashtag_density": round(len(all_hashtags) / len(text_series), 4),
            "top_hashtags": dict(hashtag_counts.most_common(20)),
            "hashtag_length_stats": {
                "min_length": min(len(tag) for tag in all_hashtags) if all_hashtags else 0,
                "max_length": max(len(tag) for tag in all_hashtags) if all_hashtags else 0,
                "mean_length": round(sum(len(tag) for tag in all_hashtags) / len(all_hashtags), 2) if all_hashtags else 0
            }
        }

        return analysis

    def _analyze_channels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa distribuição por canais"""

        # Detectar coluna de canal
        channel_candidates = ['channel', 'canal', 'chat', 'group', 'grupo']
        channel_column = None

        for candidate in channel_candidates:
            if candidate in df.columns:
                channel_column = candidate
                break

        if not channel_column:
            return {"error": "Coluna de canal não encontrada"}

        channel_series = df[channel_column].fillna("unknown")
        channel_counts = channel_series.value_counts()

        analysis = {
            "total_channels": len(channel_counts),
            "channel_distribution": channel_counts.to_dict(),
            "top_channels": dict(channel_counts.head(10)),
            "channel_statistics": {
                "most_active_channel": channel_counts.index[0] if len(channel_counts) > 0 else None,
                "least_active_channel": channel_counts.index[-1] if len(channel_counts) > 0 else None,
                "mean_messages_per_channel": round(channel_counts.mean(), 2),
                "median_messages_per_channel": round(channel_counts.median(), 2)
            }
        }

        return analysis

    def _analyze_urls(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Analisa URLs e domínios"""

        if text_column not in df.columns:
            return {"error": f"Coluna '{text_column}' não encontrada"}

        text_series = df[text_column].fillna("").astype(str)

        # Extrair URLs e domínios
        all_urls = []
        all_domains = []

        for text in text_series:
            urls = re.findall(self.patterns["urls"], text)
            all_urls.extend(urls)

            domains = re.findall(self.patterns["domains"], text)
            all_domains.extend([domain.lower() for domain in domains])

        url_counts = Counter(all_urls)
        domain_counts = Counter(all_domains)

        analysis = {
            "total_urls": len(all_urls),
            "unique_urls": len(url_counts),
            "unique_domains": len(domain_counts),
            "messages_with_urls": sum(1 for text in text_series if re.search(self.patterns["urls"], text)),
            "url_density": round(len(all_urls) / len(text_series), 4),
            "top_domains": dict(domain_counts.most_common(10)),
            "top_urls": dict(list(url_counts.most_common(5))),  # Limitar URLs por privacidade
            "domain_categories": self._categorize_domains(domain_counts)
        }

        return analysis

    def _analyze_forwards(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Analisa padrões de encaminhamento"""

        if text_column not in df.columns:
            return {"error": f"Coluna '{text_column}' não encontrada"}

        text_series = df[text_column].fillna("").astype(str)

        # Detectar mensagens encaminhadas
        forward_pattern = self.patterns["forwards"]
        forwarded_messages = text_series.str.contains(forward_pattern, case=False, regex=True)

        analysis = {
            "total_forwarded": int(forwarded_messages.sum()),
            "forward_percentage": round((forwarded_messages.sum() / len(text_series)) * 100, 2),
            "non_forwarded": int((~forwarded_messages).sum()),
            "forward_indicators": {
                "encaminhada": int(text_series.str.contains("encaminhada", case=False).sum()),
                "forwarded": int(text_series.str.contains("forwarded", case=False).sum()),
                "compartilhada": int(text_series.str.contains("compartilhada", case=False).sum())
            }
        }

        return analysis

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões temporais"""

        # Detectar coluna de datetime
        datetime_candidates = ['datetime', 'timestamp', 'date', 'created_at']
        datetime_column = None

        for candidate in datetime_candidates:
            if candidate in df.columns:
                datetime_column = candidate
                break

        if not datetime_column:
            return {"error": "Coluna de datetime não encontrada"}

        try:
            datetime_series = pd.to_datetime(df[datetime_column], errors='coerce')

            # Análise por período
            hourly_counts = datetime_series.dt.hour.value_counts().sort_index()
            daily_counts = datetime_series.dt.dayofweek.value_counts().sort_index()
            monthly_counts = datetime_series.dt.month.value_counts().sort_index()

            analysis = {
                "date_range": {
                    "start_date": str(datetime_series.min()),
                    "end_date": str(datetime_series.max()),
                    "total_days": (datetime_series.max() - datetime_series.min()).days
                },
                "hourly_distribution": hourly_counts.to_dict(),
                "daily_distribution": daily_counts.to_dict(),
                "monthly_distribution": monthly_counts.to_dict(),
                "peak_activity": {
                    "peak_hour": int(hourly_counts.idxmax()),
                    "peak_day": int(daily_counts.idxmax()),
                    "peak_month": int(monthly_counts.idxmax())
                }
            }

        except Exception as e:
            analysis = {"error": f"Erro na análise temporal: {str(e)}"}

        return analysis

    def _categorize_content(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Categoriza conteúdo por temas"""

        if text_column not in df.columns:
            return {"error": f"Coluna '{text_column}' não encontrada"}

        text_series = df[text_column].fillna("").astype(str).str.lower()

        categorization = {}

        for category, keywords in self.content_categories.items():
            # Contar mensagens que contêm palavras-chave da categoria
            category_pattern = '|'.join(keywords)
            matches = text_series.str.contains(category_pattern, regex=True, na=False)

            categorization[category] = {
                "message_count": int(matches.sum()),
                "percentage": round((matches.sum() / len(text_series)) * 100, 2),
                "keywords_found": [kw for kw in keywords if text_series.str.contains(kw, na=False).any()]
            }

        return categorization

    def _calculate_quality_metrics(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Calcula métricas de qualidade do texto"""

        if text_column not in df.columns:
            return {"error": f"Coluna '{text_column}' não encontrada"}

        text_series = df[text_column].fillna("").astype(str)

        # Métricas de qualidade
        metrics = {
            "completeness": {
                "non_empty_ratio": round((text_series.str.len() > 0).mean(), 4),
                "meaningful_content_ratio": round((text_series.str.len() > 10).mean(), 4)
            },
            "encoding_quality": {
                "suspicious_chars": int(text_series.str.contains('[��]', regex=True).sum()),
                "control_chars": int(text_series.str.contains(r'[\x00-\x1f]', regex=True).sum())
            },
            "content_diversity": {
                "unique_messages": text_series.nunique(),
                "duplicate_ratio": round(1 - (text_series.nunique() / len(text_series)), 4)
            },
            "linguistic_quality": {
                "avg_sentence_length": round(text_series.str.split().str.len().mean(), 2),
                "punctuation_usage": round(text_series.str.contains(r'[.!?]').mean(), 4)
            }
        }

        return metrics

    def _validate_cleaning_effectiveness(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Valida efetividade da limpeza de texto"""

        if text_column not in df.columns:
            return {"error": f"Coluna '{text_column}' não encontrada"}

        text_series = df[text_column].fillna("").astype(str)

        validation = {
            "cleaning_artifacts": {
                "double_spaces": int(text_series.str.contains(r'  +').sum()),
                "leading_trailing_spaces": int((text_series != text_series.str.strip()).sum()),
                "empty_after_cleaning": int((text_series.str.strip() == "").sum())
            },
            "preserved_elements": {
                "hashtags_preserved": int(text_series.str.contains(self.patterns["hashtags"]).sum()),
                "mentions_preserved": int(text_series.str.contains(self.patterns["mentions"]).sum()),
                "urls_preserved": int(text_series.str.contains(self.patterns["urls"]).sum())
            },
            "unwanted_elements": {
                "html_tags": int(text_series.str.contains(r'<[^>]+>').sum()),
                "special_chars": int(text_series.str.contains(r'[^\w\s#@.,:;!?-]').sum())
            }
        }

        return validation

    def _categorize_domains(self, domain_counts: Counter) -> Dict[str, List[str]]:
        """Categoriza domínios por tipo"""

        categories = {
            "social_media": ["facebook.com", "instagram.com", "twitter.com", "tiktok.com", "youtube.com"],
            "news": ["globo.com", "uol.com.br", "folha.uol.com.br", "g1.globo.com", "estadao.com.br"],
            "messaging": ["whatsapp.com", "telegram.org", "t.me"],
            "government": [".gov.br", "planalto.gov.br", "tse.jus.br"],
            "unknown": []
        }

        categorized = {cat: [] for cat in categories.keys()}

        for domain in domain_counts.keys():
            categorized_flag = False
            for category, known_domains in categories.items():
                if category != "unknown" and any(known in domain for known in known_domains):
                    categorized[category].append(domain)
                    categorized_flag = True
                    break

            if not categorized_flag:
                categorized["unknown"].append(domain)

        return categorized

    def _compare_dataset_info(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        """Compara informações básicas do dataset"""

        pre_info = pre.get("dataset_info", {})
        post_info = post.get("dataset_info", {})

        return {
            "record_count_change": post_info.get("total_records", 0) - pre_info.get("total_records", 0),
            "memory_usage_change_mb": post_info.get("memory_usage_mb", 0) - pre_info.get("memory_usage_mb", 0),
            "columns_added": list(set(post_info.get("column_names", [])) - set(pre_info.get("column_names", []))),
            "columns_removed": list(set(pre_info.get("column_names", [])) - set(post_info.get("column_names", [])))
        }

    def _compare_text_statistics(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        """Compara estatísticas de texto"""

        pre_stats = pre.get("text_statistics", {})
        post_stats = post.get("text_statistics", {})

        return {
            "character_reduction": pre_stats.get("total_characters", 0) - post_stats.get("total_characters", 0),
            "word_count_change": post_stats.get("total_words", 0) - pre_stats.get("total_words", 0),
            "average_length_change": post_stats.get("length_statistics", {}).get("mean_length", 0) - pre_stats.get("length_statistics", {}).get("mean_length", 0),
            "empty_messages_change": post_stats.get("empty_messages", 0) - pre_stats.get("empty_messages", 0)
        }

    def _compare_hashtags(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        """Compara análise de hashtags"""

        pre_hashtags = pre.get("hashtag_analysis", {})
        post_hashtags = post.get("hashtag_analysis", {})

        return {
            "total_hashtags_change": post_hashtags.get("total_hashtags", 0) - pre_hashtags.get("total_hashtags", 0),
            "unique_hashtags_change": post_hashtags.get("unique_hashtags", 0) - pre_hashtags.get("unique_hashtags", 0),
            "hashtag_density_change": post_hashtags.get("hashtag_density", 0) - pre_hashtags.get("hashtag_density", 0),
            "hashtags_preserved": len(set(post_hashtags.get("top_hashtags", {}).keys()) & set(pre_hashtags.get("top_hashtags", {}).keys())),
            "hashtags_lost": len(set(pre_hashtags.get("top_hashtags", {}).keys()) - set(post_hashtags.get("top_hashtags", {}).keys()))
        }

    def _compare_urls(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        """Compara análise de URLs"""

        pre_urls = pre.get("url_analysis", {})
        post_urls = post.get("url_analysis", {})

        return {
            "total_urls_change": post_urls.get("total_urls", 0) - pre_urls.get("total_urls", 0),
            "unique_urls_change": post_urls.get("unique_urls", 0) - pre_urls.get("unique_urls", 0),
            "unique_domains_change": post_urls.get("unique_domains", 0) - pre_urls.get("unique_domains", 0),
            "url_density_change": post_urls.get("url_density", 0) - pre_urls.get("url_density", 0)
        }

    def _compare_quality_metrics(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        """Compara métricas de qualidade"""

        pre_quality = pre.get("quality_metrics", {})
        post_quality = post.get("quality_metrics", {})

        return {
            "completeness_improvement": {
                "non_empty_ratio_change": post_quality.get("completeness", {}).get("non_empty_ratio", 0) - pre_quality.get("completeness", {}).get("non_empty_ratio", 0),
                "meaningful_content_change": post_quality.get("completeness", {}).get("meaningful_content_ratio", 0) - pre_quality.get("completeness", {}).get("meaningful_content_ratio", 0)
            },
            "encoding_improvements": {
                "suspicious_chars_removed": pre_quality.get("encoding_quality", {}).get("suspicious_chars", 0) - post_quality.get("encoding_quality", {}).get("suspicious_chars", 0),
                "control_chars_removed": pre_quality.get("encoding_quality", {}).get("control_chars", 0) - post_quality.get("encoding_quality", {}).get("control_chars", 0)
            },
            "duplicate_reduction": post_quality.get("content_diversity", {}).get("duplicate_ratio", 0) - pre_quality.get("content_diversity", {}).get("duplicate_ratio", 0)
        }

    def _assess_cleaning_effectiveness(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        """Avalia efetividade geral da limpeza"""

        effectiveness_score = 0.0
        factors = []

        # Fator 1: Redução de caracteres suspeitos
        pre_suspicious = pre.get("quality_metrics", {}).get("encoding_quality", {}).get("suspicious_chars", 0)
        post_suspicious = post.get("quality_metrics", {}).get("encoding_quality", {}).get("suspicious_chars", 0)

        if pre_suspicious > 0:
            char_reduction = (pre_suspicious - post_suspicious) / pre_suspicious
            effectiveness_score += char_reduction * 0.3
            factors.append(f"Redução de caracteres suspeitos: {char_reduction:.2%}")

        # Fator 2: Preservação de conteúdo importante
        hashtag_preservation = self._compare_hashtags(pre, post).get("hashtags_preserved", 0)
        if hashtag_preservation > 0:
            effectiveness_score += 0.2
            factors.append(f"Hashtags preservadas: {hashtag_preservation}")

        # Fator 3: Melhoria na completude
        completeness_change = self._compare_quality_metrics(pre, post).get("completeness_improvement", {}).get("meaningful_content_change", 0)
        if completeness_change >= 0:
            effectiveness_score += 0.2
            factors.append("Completude mantida ou melhorada")

        # Fator 4: Redução adequada de tamanho
        char_reduction = self._compare_text_statistics(pre, post).get("character_reduction", 0)
        total_chars = pre.get("text_statistics", {}).get("total_characters", 1)
        reduction_ratio = char_reduction / total_chars if total_chars > 0 else 0

        if 0.05 <= reduction_ratio <= 0.3:  # 5-30% de redução é considerado adequado
            effectiveness_score += 0.3
            factors.append(f"Redução adequada de caracteres: {reduction_ratio:.2%}")

        return {
            "effectiveness_score": min(1.0, effectiveness_score),
            "effectiveness_level": ("excelente" if effectiveness_score >= 0.8 else
                                    "bom" if effectiveness_score >= 0.6 else
                                    "satisfatorio" if effectiveness_score >= 0.4 else "insatisfatorio"),
            "contributing_factors": factors
        }

    def _generate_cleaning_recommendations(self, pre: Dict[str, Any], post: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na comparação"""

        recommendations = []

        # Análise de perdas
        hashtag_loss = self._compare_hashtags(pre, post).get("hashtags_lost", 0)
        if hashtag_loss > 5:
            recommendations.append(f"Revisar processo de limpeza - {hashtag_loss} hashtags foram perdidas")

        url_loss = self._compare_urls(pre, post).get("total_urls_change", 0)
        if url_loss < -10:  # Perda significativa de URLs
            recommendations.append("Verificar se URLs importantes não foram removidas inadequadamente")

        # Análise de qualidade
        effectiveness = self._assess_cleaning_effectiveness(pre, post)
        if effectiveness.get("effectiveness_score", 0) < 0.6:
            recommendations.append("Efetividade da limpeza abaixo do esperado - revisar parâmetros")

        # Análise de conteúdo
        char_reduction = self._compare_text_statistics(pre, post).get("character_reduction", 0)
        total_chars = pre.get("text_statistics", {}).get("total_characters", 1)
        reduction_ratio = char_reduction / total_chars if total_chars > 0 else 0

        if reduction_ratio > 0.4:
            recommendations.append("Redução excessiva de caracteres - verificar se conteúdo importante foi preservado")
        elif reduction_ratio < 0.05:
            recommendations.append("Redução mínima de caracteres - considerar limpeza mais agressiva")

        if not recommendations:
            recommendations.append("Processo de limpeza executado com sucesso - nenhuma ação adicional necessária")

        return recommendations

    def _save_analysis_report(self, analysis: Dict[str, Any], output_file: str):
        """Salva relatório de análise em arquivo"""

        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Relatório salvo: {output_file}")

        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")

    # TDD Phase 3 Methods - Standard statistical analysis interface
    def generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        TDD interface: Generate comprehensive statistics from DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict with comprehensive statistics
        """
        try:
            logger.info(f"📊 TDD statistical analysis started for {len(df)} records")
            
            statistics = {
                'timestamp': datetime.now().isoformat(),
                'total_messages': len(df),
                'unique_channels': self._calculate_unique_channels(df),
                'date_range': self._calculate_date_range(df),
                'avg_message_length': self._calculate_avg_message_length(df),
                'temporal_patterns': self._generate_temporal_patterns(df),
                'channel_stats': self._generate_channel_stats(df),
                'content_stats': self._generate_content_stats(df),
                'data_quality': self._assess_data_quality(df),
                'distribution_metrics': self._calculate_distribution_metrics(df)
            }
            
            logger.info(f"✅ TDD statistical analysis completed: {len(statistics)} metric categories generated")
            
            return statistics
            
        except Exception as e:
            logger.error(f"TDD statistical analysis error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'total_messages': len(df) if df is not None else 0
            }
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """TDD interface alias for generate_statistics."""
        return self.generate_statistics(df)
    
    def _calculate_unique_channels(self, df: pd.DataFrame) -> int:
        """Calculate number of unique channels."""
        channel_candidates = ['channel', 'canal', 'source', 'from']
        
        for candidate in channel_candidates:
            if candidate in df.columns:
                return int(df[candidate].nunique())
        
        return 1  # Default if no channel column found
    
    def _calculate_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate date range from DataFrame."""
        date_candidates = ['date', 'datetime', 'timestamp', 'created_at', 'sent_at']
        
        for candidate in date_candidates:
            if candidate in df.columns:
                try:
                    date_series = pd.to_datetime(df[candidate], errors='coerce').dropna()
                    if len(date_series) > 0:
                        return {
                            'start_date': date_series.min().isoformat(),
                            'end_date': date_series.max().isoformat(),
                            'days_span': (date_series.max() - date_series.min()).days,
                            'date_column_used': candidate
                        }
                except:
                    continue
        
        return {
            'start_date': None,
            'end_date': None,
            'days_span': 0,
            'date_column_used': None
        }
    
    def _calculate_avg_message_length(self, df: pd.DataFrame) -> float:
        """Calculate average message length."""
        text_candidates = ['body', 'text', 'content', 'message', 'mensagem']
        
        for candidate in text_candidates:
            if candidate in df.columns:
                lengths = df[candidate].fillna('').astype(str).str.len()
                return float(lengths.mean())
        
        return 0.0
    
    def _generate_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate temporal pattern analysis."""
        date_candidates = ['date', 'datetime', 'timestamp', 'created_at', 'sent_at']
        
        for candidate in date_candidates:
            if candidate in df.columns:
                try:
                    date_series = pd.to_datetime(df[candidate], errors='coerce').dropna()
                    if len(date_series) > 0:
                        # Messages by hour
                        hourly_counts = date_series.dt.hour.value_counts().sort_index()
                        
                        # Messages by day of week
                        daily_counts = date_series.dt.day_name().value_counts()
                        
                        # Peak activity
                        peak_hour = hourly_counts.idxmax() if not hourly_counts.empty else None
                        peak_day = daily_counts.idxmax() if not daily_counts.empty else None
                        
                        return {
                            'messages_by_hour': hourly_counts.to_dict(),
                            'messages_by_day': daily_counts.to_dict(),
                            'peak_activity': {
                                'peak_hour': int(peak_hour) if peak_hour is not None else None,
                                'peak_day': peak_day,
                                'peak_hour_count': int(hourly_counts.max()) if not hourly_counts.empty else 0
                            },
                            'temporal_distribution': 'available'
                        }
                except:
                    continue
        
        return {
            'messages_by_hour': {},
            'messages_by_day': {},
            'peak_activity': None,
            'temporal_distribution': 'unavailable'
        }
    
    def _generate_channel_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate channel-specific statistics."""
        channel_candidates = ['channel', 'canal', 'source', 'from']
        
        for candidate in channel_candidates:
            if candidate in df.columns:
                channel_counts = df[candidate].value_counts()
                
                stats = {}
                for channel, count in channel_counts.items():
                    stats[str(channel)] = {
                        'message_count': int(count),
                        'percentage': round((count / len(df)) * 100, 2)
                    }
                
                return stats
        
        # Default single channel if no channel column
        return {
            'unknown_channel': {
                'message_count': len(df),
                'percentage': 100.0
            }
        }
    
    def _generate_content_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate content-specific statistics."""
        text_candidates = ['body', 'text', 'content', 'message', 'mensagem']
        
        content_stats = {
            'avg_length': 0.0,
            'url_frequency': 0.0,
            'hashtag_frequency': 0.0,
            'mention_frequency': 0.0,
            'total_words': 0,
            'unique_words': 0,
            'content_diversity': 0.0
        }
        
        for candidate in text_candidates:
            if candidate in df.columns:
                text_series = df[candidate].fillna('').astype(str)
                
                # Average length
                content_stats['avg_length'] = float(text_series.str.len().mean())
                
                # URL frequency
                url_pattern = r'https?://[^\s]+'
                urls_per_message = text_series.str.count(url_pattern)
                content_stats['url_frequency'] = float(urls_per_message.mean())
                
                # Hashtag frequency
                hashtag_pattern = r'#\w+'
                hashtags_per_message = text_series.str.count(hashtag_pattern)
                content_stats['hashtag_frequency'] = float(hashtags_per_message.mean())
                
                # Mention frequency
                mention_pattern = r'@\w+'
                mentions_per_message = text_series.str.count(mention_pattern)
                content_stats['mention_frequency'] = float(mentions_per_message.mean())
                
                # Word statistics
                all_words = []
                for text in text_series:
                    words = str(text).lower().split()
                    all_words.extend(words)
                
                content_stats['total_words'] = len(all_words)
                content_stats['unique_words'] = len(set(all_words))
                content_stats['content_diversity'] = len(set(all_words)) / len(all_words) if all_words else 0.0
                
                break
        
        return content_stats
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_metrics = {
            'completeness': 0.0,
            'consistency': 0.0,
            'validity': 0.0,
            'missing_data_percentage': 0.0,
            'duplicate_percentage': 0.0
        }
        
        if len(df) > 0:
            # Completeness (non-null values)
            non_null_counts = df.count()
            total_cells = len(df) * len(df.columns)
            non_null_cells = non_null_counts.sum()
            quality_metrics['completeness'] = float(non_null_cells / total_cells) if total_cells > 0 else 0.0
            
            # Missing data percentage
            missing_cells = total_cells - non_null_cells
            quality_metrics['missing_data_percentage'] = float(missing_cells / total_cells * 100) if total_cells > 0 else 0.0
            
            # Duplicate percentage
            unique_rows = df.drop_duplicates()
            duplicate_count = len(df) - len(unique_rows)
            quality_metrics['duplicate_percentage'] = float(duplicate_count / len(df) * 100)
            
            # Basic validity check (assuming text columns should have reasonable length)
            text_candidates = ['body', 'text', 'content', 'message', 'mensagem']
            valid_text_ratio = 1.0
            
            for candidate in text_candidates:
                if candidate in df.columns:
                    text_lengths = df[candidate].fillna('').astype(str).str.len()
                    valid_texts = (text_lengths > 0) & (text_lengths < 10000)  # Reasonable text length
                    valid_text_ratio = float(valid_texts.mean())
                    break
            
            quality_metrics['validity'] = valid_text_ratio
            quality_metrics['consistency'] = 0.8  # Placeholder - would need domain-specific logic
        
        return quality_metrics
    
    def _calculate_distribution_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate distribution metrics for numerical columns."""
        distribution_metrics = {}
        
        # Find numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numerical_columns:
            if df[col].notna().sum() > 0:  # Only analyze columns with data
                series = df[col].dropna()
                distribution_metrics[col] = {
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'count': int(len(series))
                }
        
        # Calculate text length distribution if text column exists
        text_candidates = ['body', 'text', 'content', 'message', 'mensagem']
        for candidate in text_candidates:
            if candidate in df.columns:
                text_lengths = df[candidate].fillna('').astype(str).str.len()
                distribution_metrics['text_length_distribution'] = {
                    'mean': float(text_lengths.mean()),
                    'median': float(text_lengths.median()),
                    'std': float(text_lengths.std()),
                    'min': int(text_lengths.min()),
                    'max': int(text_lengths.max()),
                    'count': int(len(text_lengths))
                }
                break
        
        return distribution_metrics
