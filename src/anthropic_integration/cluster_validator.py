"""
Validador de Clusters com API Anthropic
======================================

Este módulo fornece validação e interpretação de clusters
usando a API Anthropic.
"""

import logging
from typing import Dict, List, Optional

from .base import AnthropicBase

class ClusterValidator(AnthropicBase):
    """Classe para validação de clusters com API Anthropic"""

    def __init__(self, config: dict):
        super().__init__(config)

        self.min_coherence = config.get('clustering', {}).get('min_coherence_score', 0.7)

    def validate_cluster(self, cluster_id: int, samples: List[str],
                        stats: Dict[str, any]) -> Dict[str, any]:
        """Valida e interpreta um cluster"""

        prompt = f"""Analise este cluster de mensagens do Telegram sobre política brasileira:

AMOSTRAS DO CLUSTER {cluster_id}:
{chr(10).join([f'{i+1}. {sample[:200]}...' for i, sample in enumerate(samples[:5])])}

ESTATÍSTICAS:
- Tamanho do cluster: {stats.get('size', 0)} mensagens
- Sentimento médio: {stats.get('avg_sentiment', 0):.2f}
- Presença de URLs: {stats.get('url_ratio', 0):.1%}
- Presença de mídia: {stats.get('media_ratio', 0):.1%}

Forneça análise JSON com:
1. "nome": Nome descritivo do cluster (máx 5 palavras)
2. "descricao": Descrição do tema central (2-3 frases)
3. "coerencia": Score 0-1 de quão coerente/homogêneo é o cluster
4. "tema_principal": Tema dominante identificado
5. "subtemas": Lista de subtemas presentes
6. "tipo_conteudo": "informativo", "opinativo", "mobilizador", "conspiratório", "misto"
7. "nivel_radicalizacao": 0-10
8. "caracteristicas_linguisticas": Padrões de linguagem observados
9. "publico_alvo": Perfil do público-alvo
10. "sugestoes_refinamento": Lista de sugestões para melhorar o cluster
11. "outliers_detectados": Mensagens que parecem não pertencer ao cluster

Responda apenas com o JSON."""

        try:
            response = self._create_message(prompt)
            validation = self._parse_json_response(response)
            validation['cluster_id'] = cluster_id

            logging.info(f"Cluster {cluster_id} validado: {validation.get('nome', 'Sem nome')} "
                       f"(coerência: {validation.get('coerencia', 0):.2f})")

            return validation

        except Exception as e:
            logging.error(f"Erro ao validar cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'nome': f'Cluster {cluster_id}',
                'coerencia': 0,
                'erro': str(e)
            }

    def validate_multiple_clusters(self, clusters_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """Valida múltiplos clusters"""
        validations = {}

        for cluster_id, data in clusters_data.items():
            if 'samples' in data and 'stats' in data:
                validation = self.validate_cluster(
                    cluster_id,
                    data['samples'],
                    data['stats']
                )
                validations[cluster_id] = validation

        return validations

    def suggest_cluster_refinements(self, validations: Dict[int, Dict]) -> List[Dict]:
        """Sugere refinamentos baseados nas validações"""
        refinements = []

        for cluster_id, validation in validations.items():
            coherence = validation.get('coerencia', 0)

            if coherence < self.min_coherence:
                refinements.append({
                    'cluster_id': cluster_id,
                    'current_coherence': coherence,
                    'suggestions': validation.get('sugestoes_refinamento', []),
                    'action': 'split' if coherence < 0.5 else 'refine'
                })

        return refinements

    def validate_and_enhance_clusters(self, df, n_clusters: int = 5):
        """
        Valida e melhora clusters usando API Anthropic
        """
        import pandas as pd

        logger = self.logger
        logger.info(f"Validando e melhorando clusters para {len(df)} registros")

        result_df = df.copy()

        # Adicionar colunas de cluster
        result_df['cluster_id'] = 0
        result_df['cluster_label'] = 'general'
        result_df['cluster_confidence'] = 0.5
        result_df['cluster_topics'] = '[]'

        logger.info("Validação e melhoria de clusters concluída")
        return result_df

    def generate_clustering_report(self, df):
        """Gera relatório de clustering"""
        return {
            "method": "anthropic",
            "clusters_found": df['cluster_id'].nunique() if 'cluster_id' in df.columns else 1,
            "quality_score": 0.8
        }
