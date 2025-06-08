"""
Dataset Statistics Generator - Análise Estatística Completa
Gera estatísticas gerais do dataset após feature extraction
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter
import json

from .base import AnthropicBase

logger = logging.getLogger(__name__)


class DatasetStatisticsGenerator(AnthropicBase):
    """
    Gerador de Estatísticas Completas do Dataset
    
    Executado após feature extraction para fornecer visão geral completa:
    - Estatísticas de deduplicação
    - Frequências de features
    - Distribuições temporais
    - Análise de canais
    - Métricas de qualidade
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.statistics_config = {
            "top_n_items": 20,  # Top N para rankings
            "temporal_granularity": "daily",  # daily, weekly, monthly
            "generate_plots": False,  # Desabilitado por padrão
            "export_format": "json",  # json, csv, html
            "calculate_advanced_metrics": True
        }
    
    def generate_comprehensive_statistics(
        self,
        df_original: pd.DataFrame,
        df_processed: pd.DataFrame,
        deduplication_report: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Gera estatísticas completas do dataset
        
        Args:
            df_original: DataFrame original (antes da deduplicação)
            df_processed: DataFrame processado (após deduplicação e features)
            deduplication_report: Relatório de deduplicação
            
        Returns:
            Dicionário com estatísticas completas
        """
        logger.info("Gerando estatísticas completas do dataset")
        
        statistics = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "original_size": len(df_original),
                "processed_size": len(df_processed),
                "reduction_rate": round((1 - len(df_processed) / len(df_original)) * 100, 2) if len(df_original) > 0 else 0
            },
            "basic_statistics": self._calculate_basic_statistics(df_processed),
            "feature_frequencies": self._calculate_feature_frequencies(df_processed),
            "temporal_distribution": self._calculate_temporal_distribution(df_processed),
            "channel_analysis": self._analyze_channels(df_processed),
            "text_statistics": self._calculate_text_statistics(df_processed),
            "deduplication_summary": self._summarize_deduplication(df_original, df_processed, deduplication_report),
            "quality_metrics": self._calculate_quality_metrics(df_processed),
            "content_patterns": self._analyze_content_patterns(df_processed)
        }
        
        # Adicionar análise avançada se habilitada
        if self.statistics_config["calculate_advanced_metrics"]:
            statistics["advanced_metrics"] = self._calculate_advanced_metrics(df_processed)
        
        # Gerar recomendações baseadas nas estatísticas
        statistics["insights_and_recommendations"] = self._generate_insights(statistics)
        
        logger.info(f"Estatísticas geradas: {len(df_processed)} mensagens analisadas")
        
        return statistics
    
    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas básicas do dataset"""
        
        stats = {
            "total_messages": len(df),
            "unique_channels": df['channel'].nunique() if 'channel' in df.columns else 0,
            "date_range": {
                "start": str(df['datetime'].min()) if 'datetime' in df.columns else None,
                "end": str(df['datetime'].max()) if 'datetime' in df.columns else None,
                "days_covered": (df['datetime'].max() - df['datetime'].min()).days if 'datetime' in df.columns else 0
            },
            "message_types": {
                "forwarded": int(df['is_fwrd'].sum()) if 'is_fwrd' in df.columns else 0,
                "with_media": int((df['media_type'] != 'none').sum()) if 'media_type' in df.columns else 0,
                "with_urls": int((df['url'].notna()).sum()) if 'url' in df.columns else 0,
                "with_hashtags": int((df['hashtag'].notna()).sum()) if 'hashtag' in df.columns else 0,
                "with_mentions": int((df['mentions'].notna()).sum()) if 'mentions' in df.columns else 0
            }
        }
        
        # Calcular percentuais
        total = stats["total_messages"]
        if total > 0:
            stats["message_types_percentage"] = {
                k: round(v / total * 100, 2) 
                for k, v in stats["message_types"].items()
            }
        
        return stats
    
    def _calculate_feature_frequencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula frequências das principais features"""
        
        frequencies = {}
        
        # Top hashtags
        if 'hashtag' in df.columns:
            hashtags = []
            for tags in df['hashtag'].dropna():
                if isinstance(tags, str):
                    hashtags.extend([tag.strip() for tag in tags.split(',') if tag.strip()])
            
            hashtag_counter = Counter(hashtags)
            frequencies["top_hashtags"] = [
                {"hashtag": tag, "count": count, "percentage": round(count / len(df) * 100, 2)}
                for tag, count in hashtag_counter.most_common(self.statistics_config["top_n_items"])
            ]
        
        # Top mentions
        if 'mentions' in df.columns:
            mentions = []
            for mention_list in df['mentions'].dropna():
                if isinstance(mention_list, str):
                    mentions.extend([m.strip() for m in mention_list.split(',') if m.strip()])
            
            mention_counter = Counter(mentions)
            frequencies["top_mentions"] = [
                {"mention": mention, "count": count, "percentage": round(count / len(df) * 100, 2)}
                for mention, count in mention_counter.most_common(self.statistics_config["top_n_items"])
            ]
        
        # Top domains
        if 'domain' in df.columns:
            domain_counter = Counter(df['domain'].dropna())
            frequencies["top_domains"] = [
                {"domain": domain, "count": count, "percentage": round(count / len(df) * 100, 2)}
                for domain, count in domain_counter.most_common(self.statistics_config["top_n_items"])
            ]
        
        # Top channels
        if 'channel' in df.columns:
            channel_counter = Counter(df['channel'])
            frequencies["top_channels"] = [
                {"channel": channel, "count": count, "percentage": round(count / len(df) * 100, 2)}
                for channel, count in channel_counter.most_common(self.statistics_config["top_n_items"])
            ]
        
        return frequencies
    
    def _calculate_temporal_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula distribuição temporal das mensagens"""
        
        if 'datetime' not in df.columns:
            return {"error": "No datetime column found"}
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        distribution = {
            "by_year": df.groupby(df['datetime'].dt.year).size().to_dict(),
            "by_month": df.groupby(df['datetime'].dt.to_period('M').astype(str)).size().to_dict(),
            "by_day_of_week": df.groupby(df['datetime'].dt.day_name()).size().to_dict(),
            "by_hour": df.groupby(df['datetime'].dt.hour).size().to_dict(),
            "peak_activity": {
                "busiest_day": df.groupby(df['datetime'].dt.date).size().idxmax().strftime('%Y-%m-%d'),
                "busiest_hour": int(df.groupby(df['datetime'].dt.hour).size().idxmax()),
                "busiest_day_of_week": df.groupby(df['datetime'].dt.day_name()).size().idxmax()
            }
        }
        
        # Adicionar tendências
        daily_counts = df.groupby(df['datetime'].dt.date).size()
        distribution["trends"] = {
            "average_messages_per_day": round(daily_counts.mean(), 2),
            "std_messages_per_day": round(daily_counts.std(), 2),
            "max_messages_in_day": int(daily_counts.max()),
            "min_messages_in_day": int(daily_counts.min())
        }
        
        return distribution
    
    def _analyze_channels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa distribuição e características dos canais"""
        
        if 'channel' not in df.columns:
            return {"error": "No channel column found"}
        
        channel_stats = df.groupby('channel').agg({
            'body': 'count',
            'is_fwrd': 'sum' if 'is_fwrd' in df.columns else lambda x: 0,
            'hashtag': lambda x: x.notna().sum() if 'hashtag' in df.columns else 0,
            'url': lambda x: x.notna().sum() if 'url' in df.columns else 0
        }).rename(columns={'body': 'message_count'})
        
        channel_analysis = {
            "total_channels": len(channel_stats),
            "channel_distribution": {
                "top_10_channels_message_share": round(
                    channel_stats.nlargest(10, 'message_count')['message_count'].sum() / len(df) * 100, 2
                ),
                "channels_with_10+_messages": len(channel_stats[channel_stats['message_count'] >= 10]),
                "channels_with_100+_messages": len(channel_stats[channel_stats['message_count'] >= 100]),
                "channels_with_1000+_messages": len(channel_stats[channel_stats['message_count'] >= 1000])
            },
            "channel_characteristics": []
        }
        
        # Características dos top canais
        for channel, row in channel_stats.nlargest(10, 'message_count').iterrows():
            channel_analysis["channel_characteristics"].append({
                "channel": channel,
                "messages": int(row['message_count']),
                "forwarded_ratio": round(row.get('is_fwrd', 0) / row['message_count'] * 100, 2),
                "hashtag_usage": round(row.get('hashtag', 0) / row['message_count'] * 100, 2),
                "url_sharing": round(row.get('url', 0) / row['message_count'] * 100, 2)
            })
        
        return channel_analysis
    
    def _calculate_text_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas relacionadas ao texto"""
        
        text_col = 'body_cleaned' if 'body_cleaned' in df.columns else 'body'
        
        if text_col not in df.columns:
            return {"error": "No text column found"}
        
        # Calcular métricas de texto
        df['text_length'] = df[text_col].fillna('').str.len()
        df['word_count'] = df[text_col].fillna('').str.split().str.len()
        
        text_stats = {
            "length_statistics": {
                "mean": round(df['text_length'].mean(), 2),
                "median": round(df['text_length'].median(), 2),
                "std": round(df['text_length'].std(), 2),
                "min": int(df['text_length'].min()),
                "max": int(df['text_length'].max()),
                "percentiles": {
                    "25%": int(df['text_length'].quantile(0.25)),
                    "50%": int(df['text_length'].quantile(0.50)),
                    "75%": int(df['text_length'].quantile(0.75)),
                    "95%": int(df['text_length'].quantile(0.95))
                }
            },
            "word_count_statistics": {
                "mean": round(df['word_count'].mean(), 2),
                "median": round(df['word_count'].median(), 2),
                "total_words": int(df['word_count'].sum()),
                "empty_messages": int((df['text_length'] == 0).sum()),
                "short_messages_(<50_chars)": int((df['text_length'] < 50).sum()),
                "long_messages_(>500_chars)": int((df['text_length'] > 500).sum())
            }
        }
        
        # Limpar colunas temporárias
        df.drop(['text_length', 'word_count'], axis=1, inplace=True, errors='ignore')
        
        return text_stats
    
    def _summarize_deduplication(
        self, 
        df_original: pd.DataFrame, 
        df_processed: pd.DataFrame,
        dedup_report: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Resumo do processo de deduplicação"""
        
        total_removed = len(df_original) - len(df_processed)
        
        summary = {
            "original_messages": len(df_original),
            "unique_messages": len(df_processed),
            "duplicates_removed": total_removed,
            "deduplication_rate": round(total_removed / len(df_original) * 100, 2) if len(df_original) > 0 else 0,
            "deduplication_method": "intelligent" if dedup_report else "traditional"
        }
        
        # Adicionar detalhes do relatório se disponível
        if dedup_report:
            summary["advanced_deduplication"] = {
                "exact_duplicates": dedup_report.get("exact_duplicates", 0),
                "fuzzy_duplicates": dedup_report.get("fuzzy_duplicates", 0),
                "media_preserved": dedup_report.get("media_preserved", 0),
                "quality_score": dedup_report.get("quality_score", 0)
            }
        
        return summary
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula métricas de qualidade do dataset"""
        
        quality = {
            "completeness": {},
            "data_quality_flags": {},
            "content_richness": {}
        }
        
        # Completeness
        total = len(df)
        for col in df.columns:
            non_null = df[col].notna().sum()
            quality["completeness"][col] = round(non_null / total * 100, 2)
        
        # Quality flags
        text_col = 'body_cleaned' if 'body_cleaned' in df.columns else 'body'
        if text_col in df.columns:
            quality["data_quality_flags"] = {
                "empty_messages": int((df[text_col].fillna('').str.strip() == '').sum()),
                "very_short_messages_(<10_chars)": int((df[text_col].fillna('').str.len() < 10).sum()),
                "potential_spam_(repeated_chars)": int(
                    df[text_col].fillna('').str.contains(r'(.)\1{9,}', regex=True).sum()
                ),
                "messages_with_only_urls": int(
                    (df[text_col].fillna('').str.strip().str.match(r'^https?://\S+$')).sum()
                )
            }
        
        # Content richness
        quality["content_richness"] = {
            "messages_with_multiple_features": int(
                ((df['hashtag'].notna() if 'hashtag' in df.columns else False) & 
                 (df['url'].notna() if 'url' in df.columns else False) & 
                 (df['mentions'].notna() if 'mentions' in df.columns else False)).sum()
            ),
            "average_features_per_message": round(
                sum([
                    df['hashtag'].notna().sum() if 'hashtag' in df.columns else 0,
                    df['url'].notna().sum() if 'url' in df.columns else 0,
                    df['mentions'].notna().sum() if 'mentions' in df.columns else 0,
                    df['media_type'].notna().sum() if 'media_type' in df.columns else 0
                ]) / (total * 4) * 100, 2
            ) if total > 0 else 0
        }
        
        return quality
    
    def _analyze_content_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões de conteúdo"""
        
        patterns = {
            "media_distribution": {},
            "forwarding_patterns": {},
            "engagement_indicators": {}
        }
        
        # Distribuição de mídia
        if 'media_type' in df.columns:
            media_counts = df['media_type'].value_counts()
            patterns["media_distribution"] = {
                str(media): int(count) for media, count in media_counts.items()
            }
        
        # Padrões de encaminhamento
        if 'is_fwrd' in df.columns:
            fwd_messages = df[df['is_fwrd'] == True]
            patterns["forwarding_patterns"] = {
                "total_forwarded": len(fwd_messages),
                "forwarding_rate": round(len(fwd_messages) / len(df) * 100, 2) if len(df) > 0 else 0,
                "channels_that_forward_most": []
            }
            
            if 'channel' in df.columns and len(fwd_messages) > 0:
                fwd_by_channel = fwd_messages.groupby('channel').size().nlargest(10)
                patterns["forwarding_patterns"]["channels_that_forward_most"] = [
                    {"channel": channel, "forwarded_count": int(count)}
                    for channel, count in fwd_by_channel.items()
                ]
        
        # Indicadores de engajamento
        if all(col in df.columns for col in ['hashtag', 'mentions', 'url']):
            patterns["engagement_indicators"] = {
                "avg_hashtags_per_message": round(
                    df['hashtag'].fillna('').apply(
                        lambda x: len([h for h in str(x).split(',') if h.strip()]) if x else 0
                    ).mean(), 2
                ),
                "avg_mentions_per_message": round(
                    df['mentions'].fillna('').apply(
                        lambda x: len([m for m in str(x).split(',') if m.strip()]) if x else 0
                    ).mean(), 2
                ),
                "avg_urls_per_message": round(
                    df['url'].fillna('').apply(
                        lambda x: len([u for u in str(x).split(',') if u.strip()]) if x else 0
                    ).mean(), 2
                )
            }
        
        return patterns
    
    def _calculate_advanced_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula métricas avançadas usando análise de conteúdo"""
        
        advanced = {
            "political_content_indicators": {},
            "coordination_patterns": {},
            "temporal_anomalies": {}
        }
        
        # Indicadores de conteúdo político (se features foram extraídas)
        if 'political_alignment' in df.columns:
            alignment_counts = df['political_alignment'].value_counts()
            advanced["political_content_indicators"]["alignment_distribution"] = {
                str(align): int(count) for align, count in alignment_counts.items()
            }
        
        # Padrões de coordenação
        if 'datetime' in df.columns and 'channel' in df.columns:
            # Detectar bursts de atividade
            df['datetime'] = pd.to_datetime(df['datetime'])
            hourly_counts = df.groupby([
                df['datetime'].dt.date,
                df['datetime'].dt.hour,
                'channel'
            ]).size()
            
            # Encontrar picos anormais (>3 desvios padrão)
            mean_activity = hourly_counts.mean()
            std_activity = hourly_counts.std()
            anomalies = hourly_counts[hourly_counts > mean_activity + 3 * std_activity]
            
            advanced["temporal_anomalies"] = {
                "detected_bursts": len(anomalies),
                "average_hourly_activity": round(mean_activity, 2),
                "peak_burst_size": int(anomalies.max()) if len(anomalies) > 0 else 0
            }
        
        return advanced
    
    def _generate_insights(self, statistics: Dict[str, Any]) -> List[str]:
        """Gera insights e recomendações baseados nas estatísticas"""
        
        insights = []
        
        # Insights sobre deduplicação
        dedup_rate = statistics["metadata"].get("reduction_rate", 0)
        if dedup_rate > 20:
            insights.append(f"Alta taxa de duplicação detectada ({dedup_rate}%) - considere investigar fontes de spam")
        
        # Insights sobre canais
        if "channel_analysis" in statistics:
            top_share = statistics["channel_analysis"]["channel_distribution"].get("top_10_channels_message_share", 0)
            if top_share > 80:
                insights.append(f"Alta concentração em poucos canais ({top_share}% em top 10) - possível câmara de eco")
        
        # Insights sobre conteúdo
        if "basic_statistics" in statistics:
            fwd_rate = statistics["basic_statistics"]["message_types_percentage"].get("forwarded", 0)
            if fwd_rate > 50:
                insights.append(f"Alto índice de mensagens encaminhadas ({fwd_rate}%) - conteúdo viral predominante")
        
        # Insights sobre qualidade
        if "quality_metrics" in statistics:
            empty_msgs = statistics["quality_metrics"]["data_quality_flags"].get("empty_messages", 0)
            if empty_msgs > 100:
                insights.append(f"{empty_msgs} mensagens vazias detectadas - verificar processo de limpeza")
        
        # Insights sobre padrões temporais
        if "temporal_distribution" in statistics:
            trends = statistics["temporal_distribution"].get("trends", {})
            if trends.get("std_messages_per_day", 0) > trends.get("average_messages_per_day", 1) * 2:
                insights.append("Alta variabilidade na atividade diária - possíveis eventos ou campanhas coordenadas")
        
        return insights
    
    def export_statistics(
        self, 
        statistics: Dict[str, Any], 
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Exporta estatísticas em diferentes formatos
        
        Args:
            statistics: Dicionário de estatísticas
            output_path: Caminho de saída
            format: Formato de exportação (json, csv, html)
            
        Returns:
            Caminho do arquivo exportado
        """
        output_file = None
        
        try:
            if format == "json":
                output_file = f"{output_path}/dataset_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(statistics, f, indent=2, ensure_ascii=False, default=str)
            
            elif format == "html":
                output_file = f"{output_path}/dataset_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                html_content = self._generate_html_report(statistics)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            logger.info(f"Estatísticas exportadas para: {output_file}")
            
        except Exception as e:
            logger.error(f"Erro ao exportar estatísticas: {e}")
            
        return output_file
    
    def _generate_html_report(self, statistics: Dict[str, Any]) -> str:
        """Gera relatório HTML com as estatísticas"""
        
        html = f"""
        <html>
        <head>
            <title>Dataset Statistics Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .warning {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Dataset Statistics Report</h1>
            <p>Generated at: {statistics['metadata']['generated_at']}</p>
            
            <h2>Overview</h2>
            <div class="metric">
                <p><strong>Total Messages:</strong> {statistics['metadata']['processed_size']:,}</p>
                <p><strong>Deduplication Rate:</strong> {statistics['metadata']['reduction_rate']}%</p>
            </div>
            
            <h2>Key Insights</h2>
            <div class="warning">
                <ul>
                    {''.join([f"<li>{insight}</li>" for insight in statistics.get('insights_and_recommendations', [])])}
                </ul>
            </div>
            
            <!-- Adicionar mais seções conforme necessário -->
            
        </body>
        </html>
        """
        
        return html


def create_dataset_statistics_generator(config: Dict[str, Any]) -> DatasetStatisticsGenerator:
    """
    Factory function to create DatasetStatisticsGenerator instance
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DatasetStatisticsGenerator instance
    """
    return DatasetStatisticsGenerator(config)