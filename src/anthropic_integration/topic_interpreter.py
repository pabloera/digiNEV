"""
Interpretador de Tópicos com API Anthropic
=========================================

Este módulo fornece interpretação contextualizada de tópicos LDA
usando a API Anthropic.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .base import AnthropicBase


class TopicInterpreter(AnthropicBase):
    """Classe para interpretação de tópicos com API Anthropic"""

    def __init__(self, config: dict):
        super().__init__(config)

        # Categorias de discurso político
        self.discourse_categories = [
            'negacionista_pandemia',
            'negacionista_ciencia',
            'negacionista_historia',
            'conspiratorio_globalismo',
            'conspiratorio_comunismo',
            'autoritario_antidemocratico',
            'autoritario_golpista',
            'institucional_stf',
            'institucional_tse',
            'mobilizacao_protestos',
            'religioso_conservador',
            'economico_liberal',
            'nacionalista'
        ]

    def interpret_topic(self, topic_words: List[Tuple[str, float]],
                       topic_id: int) -> Dict[str, any]:
        """Interpreta um tópico baseado em suas palavras principais"""

        # Preparar lista de palavras
        top_words = [word for word, _ in topic_words[:15]]

        prompt = f"""Analise este conjunto de palavras de um tópico LDA sobre discurso político brasileiro no Telegram:

Palavras principais: {', '.join(top_words)}

Forneça uma análise JSON com:
1. "nome": Nome conciso e descritivo para o tópico (máx 5 palavras)
2. "descricao": Descrição detalhada do tema central (2-3 frases)
3. "categoria_principal": Uma das categorias: {', '.join(self.discourse_categories)}
4. "categorias_secundarias": Lista de outras categorias relacionadas
5. "palavras_chave": 5-7 palavras mais representativas
6. "nivel_radicalizacao": Escala 0-10
"""

        try:
            response = self.create_message(
                prompt,
                stage="topic_interpretation",
                operation=f"interpret_topic_{topic_id}"
            )

            result = self.parse_claude_response_safe(response, ["nome", "descricao", "categoria_principal", "categorias_secundarias", "palavras_chave", "nivel_radicalizacao"])
            return result

        except Exception as e:
            self.logger.error(f"Erro na interpretação do tópico {topic_id}: {e}")
            return {
                "nome": f"Tópico {topic_id}",
                "descricao": "Interpretação não disponível",
                "categoria_principal": "institucional_stf",
                "categorias_secundarias": [],
                "palavras_chave": top_words[:5],
                "nivel_radicalizacao": 0
            }

    def extract_and_interpret_topics(self, df, text_column: str = "body_cleaned", n_topics: int = 10):
        """
        Extrai e interpreta tópicos usando API Anthropic

        Args:
            df: DataFrame com os dados
            text_column: Nome da coluna de texto
            n_topics: Número de tópicos a extrair

        Returns:
            DataFrame com tópicos adicionados
        """
        import json

        import pandas as pd

        logger = self.logger
        logger.info(f"Iniciando extração e interpretação de tópicos para {len(df)} registros")

        result_df = df.copy()
        texts = result_df[text_column].fillna('').astype(str).tolist()

        # Extrair tópicos em lotes
        batch_size = 50
        all_topics = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processando lote {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            topics = self.extract_topics_batch(batch_texts)
            all_topics.extend(topics)

        # Adicionar colunas de tópicos
        for i, topic_info in enumerate(all_topics):
            if i < len(result_df):
                result_df.loc[i, 'topic_category'] = topic_info.get('topic', 'general')
                result_df.loc[i, 'topic_confidence'] = topic_info.get('confidence', 0.5)
                result_df.loc[i, 'topic_keywords'] = json.dumps(topic_info.get('keywords', []))
                result_df.loc[i, 'discourse_type'] = topic_info.get('discourse_type', 'informativo')
                result_df.loc[i, 'political_alignment'] = topic_info.get('political_alignment', 'neutro')

        logger.info("Extração e interpretação de tópicos concluída")
        return result_df

    def extract_topics_batch(self, texts: List[str]) -> List[Dict]:
        """
        Extrai tópicos de um lote de textos usando API Anthropic

        Args:
            texts: Lista de textos para analisar

        Returns:
            Lista de informações de tópicos
        """

        # Preparar textos para análise
        texts_sample = "\n".join([f"{i+1}. {text[:200]}" for i, text in enumerate(texts)])

        prompt = f"""Analise os seguintes textos políticos brasileiros do Telegram e identifique tópicos principais:

TEXTOS:
{texts_sample}

Para cada texto, identifique:
1. Tópico principal (uma palavra-chave)
2. Confiança na classificação (0-1)
3. Palavras-chave relacionadas (3-5 palavras)
4. Tipo de discurso (informativo, mobilizador, conspiratório, negacionista, autoritário)
5. Alinhamento político (bolsonarista, anti-bolsonaro, neutro, indeterminado)

Responda em formato JSON:
{{
  "topic_analysis": [
    {{
      "text_id": 1,
      "topic": "economia",
      "confidence": 0.8,
      "keywords": ["inflação", "preços", "governo"],
      "discourse_type": "informativo",
      "political_alignment": "neutro"
    }}
  ]
}}

Contexto: Período Bolsonaro 2019-2023, polarização política, negacionismo, teorias conspiratórias.
"""

        try:
            response = self.create_message(
                prompt,
                stage="05_topic_modeling",
                operation="topic_extraction"
            )

            result = self.parse_claude_response_safe(response, ["topic_analysis"])
            return result.get("topic_analysis", [])

        except Exception as e:
            logger = self.logger
            logger.error(f"Erro na extração de tópicos via API: {e}")
            # Fallback: retornar tópicos genéricos
            return [{"topic": "general", "confidence": 0.1, "keywords": [], "discourse_type": "informativo", "political_alignment": "neutro"} for _ in texts]

    def generate_topic_report(self, df) -> Dict:
        """
        Gera relatório detalhado de análise de tópicos

        Args:
            df: DataFrame com análises de tópicos

        Returns:
            Relatório de tópicos
        """
        from collections import Counter

        import pandas as pd

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_texts": len(df),
            "topic_distribution": {},
            "discourse_analysis": {},
            "political_alignment_analysis": {},
            "quality_metrics": {}
        }

        # Distribuição de tópicos
        if 'topic_category' in df.columns:
            topic_counts = df['topic_category'].value_counts()
            report["topic_distribution"] = topic_counts.to_dict()

        # Análise de tipos de discurso
        if 'discourse_type' in df.columns:
            discourse_counts = df['discourse_type'].value_counts()
            report["discourse_analysis"] = discourse_counts.to_dict()

        # Análise de alinhamento político
        if 'political_alignment' in df.columns:
            alignment_counts = df['political_alignment'].value_counts()
            report["political_alignment_analysis"] = alignment_counts.to_dict()

        # Métricas de qualidade
        if 'topic_confidence' in df.columns:
            confidence_scores = df['topic_confidence'].fillna(0)
            report["quality_metrics"] = {
                "avg_confidence": float(confidence_scores.mean()),
                "high_confidence_ratio": float((confidence_scores > 0.7).sum() / len(df)),
                "low_confidence_count": int((confidence_scores < 0.3).sum())
            }

        return report

    def interpret_multiple_topics(self, topics: Dict[int, List[Tuple[str, float]]]) -> Dict[int, Dict]:
        """Interpreta múltiplos tópicos"""
        interpretations = {}

        for topic_id, words in topics.items():
            interpretation = self.interpret_topic(words, topic_id)
            interpretations[topic_id] = interpretation

        return interpretations

    def create_topic_hierarchy(self, interpretations: Dict[int, Dict]) -> Dict[str, List[int]]:
        """Cria hierarquia de tópicos baseada nas interpretações"""
        hierarchy = {category: [] for category in self.discourse_categories}
        hierarchy['outros'] = []

        for topic_id, interp in interpretations.items():
            categoria = interp.get('categoria_principal', 'outros')
            if categoria in hierarchy:
                hierarchy[categoria].append(topic_id)
            else:
                hierarchy['outros'].append(topic_id)

        # Remover categorias vazias
        hierarchy = {k: v for k, v in hierarchy.items() if v}

        return hierarchy
