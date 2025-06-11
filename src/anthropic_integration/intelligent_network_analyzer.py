"""
Intelligent Network Analyzer com API Anthropic

Módulo avançado para análise de redes com interpretação semântica.
Identifica comunidades, influenciadores e padrões de propagação.
"""

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .base import AnthropicBase

logger = logging.getLogger(__name__)


class IntelligentNetworkAnalyzer(AnthropicBase):
    """
    Analisador inteligente de redes usando API Anthropic

    Funcionalidades:
    - Construção automática de redes de interação
    - Detecção de comunidades com interpretação semântica
    - Identificação de influenciadores e pontes
    - Análise de padrões de propagação
    - Detecção de comportamento coordenado
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurações específicas
        network_config = config.get('network_analysis', {})
        self.min_edge_weight = network_config.get('min_edge_weight', 3)
        self.max_nodes = network_config.get('max_nodes', 500)
        self.community_analysis_sample = network_config.get('community_sample_size', 100)

    def analyze_networks_intelligent(self, df: pd.DataFrame,
                                   channel_column: str = 'canal',
                                   text_column: str = 'body',
                                   timestamp_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Análise inteligente completa de redes

        Args:
            df: DataFrame com dados
            channel_column: Coluna de canal/usuário
            text_column: Coluna de texto
            timestamp_column: Coluna de timestamp

        Returns:
            Análise completa de redes
        """
        self.logger.info("Iniciando análise inteligente de redes")

        # Construção de redes múltiplas
        networks = self._build_multiple_networks(df, channel_column, text_column, timestamp_column)

        # Análise estrutural das redes
        structural_analysis = self._analyze_network_structure(networks)

        # Detecção e interpretação de comunidades
        community_analysis = self._detect_and_interpret_communities(
            networks['interaction_network'], df, channel_column, text_column
        )

        # Identificação de atores-chave
        key_actors_analysis = self._identify_key_actors(networks, df, channel_column, text_column)

        # Análise de padrões de propagação
        propagation_analysis = self._analyze_propagation_patterns(
            networks, df, timestamp_column, text_column
        )

        # Detecção de coordenação
        coordination_analysis = self._detect_coordination_in_networks(networks, df, timestamp_column)

        # Insights contextuais
        contextual_insights = self._generate_network_insights(
            structural_analysis, community_analysis, key_actors_analysis,
            propagation_analysis, coordination_analysis
        )

        return {
            'networks': networks,
            'structural_analysis': structural_analysis,
            'community_analysis': community_analysis,
            'key_actors_analysis': key_actors_analysis,
            'propagation_analysis': propagation_analysis,
            'coordination_analysis': coordination_analysis,
            'contextual_insights': contextual_insights,
            'analysis_summary': self._generate_network_summary(
                networks, structural_analysis, community_analysis
            )
        }

    def _build_multiple_networks(self, df: pd.DataFrame, channel_column: str,
                               text_column: str, timestamp_column: str) -> Dict[str, Any]:
        """
        Constrói múltiplas redes de interação

        Args:
            df: DataFrame com dados
            channel_column: Coluna de canal
            text_column: Coluna de texto
            timestamp_column: Coluna de timestamp

        Returns:
            Diferentes tipos de redes construídas
        """
        self.logger.info("Construindo redes de interação")

        # Rede de co-ocorrência temporal
        interaction_network = self._build_interaction_network(df, channel_column, timestamp_column)

        # Rede de similaridade de conteúdo
        content_similarity_network = self._build_content_similarity_network(
            df, channel_column, text_column
        )

        # Rede de menções/referências
        mention_network = self._build_mention_network(df, channel_column, text_column)

        return {
            'interaction_network': interaction_network,
            'content_similarity_network': content_similarity_network,
            'mention_network': mention_network,
            'network_stats': {
                'interaction_nodes': interaction_network.number_of_nodes(),
                'interaction_edges': interaction_network.number_of_edges(),
                'content_nodes': content_similarity_network.number_of_nodes(),
                'content_edges': content_similarity_network.number_of_edges(),
                'mention_nodes': mention_network.number_of_nodes(),
                'mention_edges': mention_network.number_of_edges()
            }
        }

    def _build_interaction_network(self, df: pd.DataFrame, channel_column: str,
                                 timestamp_column: str) -> nx.Graph:
        """
        Constrói rede de interação baseada em co-ocorrência temporal

        Args:
            df: DataFrame com dados
            channel_column: Coluna de canal
            timestamp_column: Coluna de timestamp

        Returns:
            Rede de interação
        """
        G = nx.Graph()

        # Converter timestamp para datetime
        df_temp = df.copy()
        df_temp[timestamp_column] = pd.to_datetime(df_temp[timestamp_column], errors='coerce')
        df_temp = df_temp.dropna(subset=[timestamp_column, channel_column])

        # Agrupar por janelas de tempo (ex: 1 hora)
        df_temp['time_window'] = df_temp[timestamp_column].dt.floor('H')

        # Para cada janela de tempo, conectar canais que postaram juntos
        co_occurrence = defaultdict(int)

        for window, group in df_temp.groupby('time_window'):
            channels_in_window = group[channel_column].unique()

            # Criar conexões entre todos os pares de canais na janela
            for i, channel1 in enumerate(channels_in_window):
                for channel2 in channels_in_window[i+1:]:
                    if channel1 != channel2:
                        pair = tuple(sorted([channel1, channel2]))
                        co_occurrence[pair] += 1

        # Adicionar nós e arestas à rede
        for (channel1, channel2), weight in co_occurrence.items():
            if weight >= self.min_edge_weight:
                G.add_edge(channel1, channel2, weight=weight)

        # Limitar tamanho da rede se necessário
        if G.number_of_nodes() > self.max_nodes:
            # Manter apenas os nós com maior grau
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:self.max_nodes]
            G = G.subgraph(top_nodes).copy()

        return G

    def _build_content_similarity_network(self, df: pd.DataFrame, channel_column: str,
                                        text_column: str) -> nx.Graph:
        """
        Constrói rede de similaridade de conteúdo (implementação simplificada)
        """
        G = nx.Graph()

        # Implementação simplificada - conectar canais com palavras-chave similares
        channel_keywords = defaultdict(set)

        for _, row in df.iterrows():
            channel = row[channel_column]
            text = str(row[text_column]).lower()

            # Extrair palavras simples (implementação básica)
            words = set(text.split())
            words = {w for w in words if len(w) > 4}  # Filtrar palavras muito curtas

            channel_keywords[channel].update(words)

        # Calcular similaridade entre canais
        channels = list(channel_keywords.keys())
        similarities = {}

        for i, channel1 in enumerate(channels):
            for channel2 in channels[i+1:]:
                if channel1 != channel2:
                    keywords1 = channel_keywords[channel1]
                    keywords2 = channel_keywords[channel2]

                    if keywords1 and keywords2:
                        intersection = len(keywords1 & keywords2)
                        union = len(keywords1 | keywords2)
                        jaccard = intersection / union if union > 0 else 0

                        if jaccard > 0.1:  # Threshold de similaridade
                            similarities[(channel1, channel2)] = jaccard

        # Adicionar arestas à rede
        for (channel1, channel2), similarity in similarities.items():
            G.add_edge(channel1, channel2, weight=similarity)

        return G

    def _build_mention_network(self, df: pd.DataFrame, channel_column: str,
                             text_column: str) -> nx.DiGraph:
        """
        Constrói rede de menções/referências
        """
        G = nx.DiGraph()

        # Procurar por menções (@canal) ou referências
        mentions = defaultdict(int)

        for _, row in df.iterrows():
            source = row[channel_column]
            text = str(row[text_column])

            # Buscar padrões de menção simples
            # (implementação básica - pode ser expandida)
            if '@' in text:
                # Extrair possíveis menções
                words = text.split()
                for word in words:
                    if word.startswith('@') and len(word) > 1:
                        target = word[1:].strip('.,!?')
                        if target and target != source:
                            mentions[(source, target)] += 1

        # Adicionar arestas à rede
        for (source, target), count in mentions.items():
            if count >= 2:  # Mínimo de menções
                G.add_edge(source, target, weight=count)

        return G

    def _analyze_network_structure(self, networks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa estrutura das redes

        Args:
            networks: Dicionário com redes construídas

        Returns:
            Análise estrutural das redes
        """
        self.logger.info("Analisando estrutura das redes")

        results = {}

        for network_name, network in networks.items():
            if network_name == 'network_stats':
                continue

            if isinstance(network, (nx.Graph, nx.DiGraph)) and network.number_of_nodes() > 0:
                analysis = {
                    'basic_stats': {
                        'nodes': network.number_of_nodes(),
                        'edges': network.number_of_edges(),
                        'density': nx.density(network),
                        'is_directed': network.is_directed()
                    }
                }

                # Métricas de conectividade
                if network.number_of_nodes() > 1:
                    if isinstance(network, nx.Graph):
                        analysis['connectivity'] = {
                            'is_connected': nx.is_connected(network),
                            'number_of_components': nx.number_connected_components(network)
                        }

                    # Centralidade (para redes pequenas)
                    if network.number_of_nodes() <= 100:
                        centrality_measures = {
                            'degree_centrality': nx.degree_centrality(network),
                            'betweenness_centrality': nx.betweenness_centrality(network),
                            'closeness_centrality': nx.closeness_centrality(network)
                        }

                        # Top nós por centralidade
                        analysis['top_central_nodes'] = {
                            'degree': sorted(centrality_measures['degree_centrality'].items(),
                                           key=lambda x: x[1], reverse=True)[:10],
                            'betweenness': sorted(centrality_measures['betweenness_centrality'].items(),
                                                key=lambda x: x[1], reverse=True)[:10]
                        }

                results[network_name] = analysis

        return results

    def _detect_and_interpret_communities(self, network: nx.Graph, df: pd.DataFrame,
                                        channel_column: str, text_column: str) -> Dict[str, Any]:
        """
        Detecta e interpreta comunidades na rede

        Args:
            network: Rede para análise
            df: DataFrame original
            channel_column: Coluna de canal
            text_column: Coluna de texto

        Returns:
            Análise de comunidades
        """
        if network.number_of_nodes() < 3:
            return {'communities': []}

        self.logger.info("Detectando comunidades na rede")

        try:
            # Detecção de comunidades usando algoritmo de Louvain
            import community as community_louvain
            partition = community_louvain.best_partition(network)
        except ImportError:
            # Fallback: usar componentes conectados
            partition = {}
            for i, component in enumerate(nx.connected_components(network)):
                for node in component:
                    partition[node] = i

        # Organizar comunidades
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)

        # Analisar cada comunidade
        community_analyses = []

        for community_id, members in communities.items():
            if len(members) >= 3:  # Só analisar comunidades com 3+ membros
                community_analysis = self._analyze_single_community(
                    members, df, channel_column, text_column, community_id
                )
                if community_analysis:
                    community_analyses.append(community_analysis)

        return {
            'total_communities': len(communities),
            'communities': community_analyses[:10],  # Limitar para análise AI
            'modularity': nx.algorithms.community.modularity(network, communities.values()) if communities else 0
        }

    def _analyze_single_community(self, members: List[str], df: pd.DataFrame,
                                channel_column: str, text_column: str, community_id: int) -> Optional[Dict[str, Any]]:
        """
        Analisa uma única comunidade

        Args:
            members: Membros da comunidade
            df: DataFrame original
            channel_column: Coluna de canal
            text_column: Coluna de texto
            community_id: ID da comunidade

        Returns:
            Análise da comunidade
        """
        # Filtrar mensagens dos membros da comunidade
        community_messages = df[df[channel_column].isin(members)][text_column].dropna()

        if len(community_messages) == 0:
            return None

        # Amostra para análise AI
        sample_messages = community_messages.sample(
            min(self.community_analysis_sample, len(community_messages)),
            random_state=42
        ).tolist()

        prompt = f"""
Analise esta comunidade de canais do Telegram brasileiro detectada automaticamente:

COMUNIDADE {community_id}:
- Membros: {', '.join(members[:10])}
- Total de membros: {len(members)}
- Mensagens analisadas: {len(sample_messages)}

AMOSTRA DE MENSAGENS:
{chr(10).join([f"- {msg[:150]}" for msg in sample_messages[:15]])}

CONTEXTO: Comunidade identificada por padrões de interação em período 2019-2023 (governo Bolsonaro, pandemia, eleições).

Determine:
1. Orientação política predominante
2. Temas principais de discussão
3. Nível de coordenação/organização
4. Papel na rede maior (influenciadores, amplificadores, etc.)
5. Indicadores de comportamento inautêntico

Responda em JSON:
{{
    "community_profile": {{
        "political_orientation": "esquerda|centro|direita|extrema_direita|misto",
        "primary_themes": ["tema1", "tema2", "tema3"],
        "coordination_level": "alta|média|baixa",
        "network_role": "influenciadores|amplificadores|receptores|ponte",
        "authenticity_indicators": "autêntico|suspeito|coordenado"
    }},
    "behavioral_patterns": {{
        "message_style": "formal|informal|agressivo|propagandístico",
        "engagement_type": "orgânico|artificial|misto",
        "temporal_patterns": "regular|sazonal|campanhas"
    }},
    "community_description": "descrição_concisa_da_comunidade",
    "risk_assessment": "baixo|médio|alto"
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='11_community_analysis',
                operation='analyze_community'
            )

            analysis = self.parse_json_response(response)

            return {
                'community_id': community_id,
                'members': members,
                'member_count': len(members),
                'message_sample_size': len(sample_messages),
                'ai_analysis': analysis,
                'basic_stats': {
                    'total_messages': len(community_messages),
                    'avg_messages_per_member': len(community_messages) / len(members)
                }
            }

        except Exception as e:
            self.logger.error(f"Erro na análise da comunidade {community_id}: {e}")
            return {
                'community_id': community_id,
                'members': members,
                'member_count': len(members),
                'error': str(e)
            }

    def _identify_key_actors(self, networks: Dict[str, Any], df: pd.DataFrame,
                           channel_column: str, text_column: str) -> Dict[str, Any]:
        """
        Identifica atores-chave nas redes

        Args:
            networks: Redes construídas
            df: DataFrame original
            channel_column: Coluna de canal
            text_column: Coluna de texto

        Returns:
            Análise de atores-chave
        """
        interaction_network = networks.get('interaction_network')

        if not interaction_network or interaction_network.number_of_nodes() == 0:
            return {'key_actors': []}

        # Calcular métricas de centralidade
        centrality_metrics = {}

        if interaction_network.number_of_nodes() <= 200:  # Limitar para performance
            centrality_metrics = {
                'degree': nx.degree_centrality(interaction_network),
                'betweenness': nx.betweenness_centrality(interaction_network),
                'closeness': nx.closeness_centrality(interaction_network),
                'eigenvector': nx.eigenvector_centrality(interaction_network, max_iter=100)
            }

        # Identificar top atores
        key_actors = []

        if centrality_metrics:
            # Combinar métricas para score geral
            all_nodes = set()
            for metric_dict in centrality_metrics.values():
                all_nodes.update(metric_dict.keys())

            for node in all_nodes:
                combined_score = sum(
                    centrality_metrics[metric].get(node, 0)
                    for metric in centrality_metrics
                ) / len(centrality_metrics)

                key_actors.append({
                    'actor': node,
                    'combined_centrality_score': combined_score,
                    'degree_centrality': centrality_metrics['degree'].get(node, 0),
                    'betweenness_centrality': centrality_metrics['betweenness'].get(node, 0)
                })

            # Ordenar por score combinado
            key_actors.sort(key=lambda x: x['combined_centrality_score'], reverse=True)
            key_actors = key_actors[:20]  # Top 20

        return {
            'key_actors': key_actors,
            'centrality_metrics_available': bool(centrality_metrics),
            'network_size': interaction_network.number_of_nodes()
        }

    def _analyze_propagation_patterns(self, networks: Dict[str, Any], df: pd.DataFrame,
                                    timestamp_column: str, text_column: str) -> Dict[str, Any]:
        """
        Analisa padrões de propagação (implementação simplificada)
        """
        return {
            'analysis': 'propagation_patterns_placeholder',
            'method': 'simplified_implementation'
        }

    def _detect_coordination_in_networks(self, networks: Dict[str, Any], df: pd.DataFrame,
                                       timestamp_column: str) -> Dict[str, Any]:
        """
        Detecta coordenação através de análise de redes (implementação simplificada)
        """
        interaction_network = networks.get('interaction_network')

        if not interaction_network:
            return {'coordination_detected': False}

        # Análise simplificada: verificar densidade e clustering
        density = nx.density(interaction_network)

        coordination_indicators = {
            'high_density': density > 0.3,
            'network_density': density,
            'suspicious_patterns': density > 0.5  # Muito alta densidade pode indicar coordenação
        }

        return {
            'coordination_detected': coordination_indicators['suspicious_patterns'],
            'indicators': coordination_indicators,
            'analysis_method': 'density_and_clustering_analysis'
        }

    def _generate_network_insights(self, structural_analysis: Dict, community_analysis: Dict,
                                 key_actors_analysis: Dict, propagation_analysis: Dict,
                                 coordination_analysis: Dict) -> Dict[str, Any]:
        """
        Gera insights contextuais sobre as redes

        Args:
            structural_analysis: Análise estrutural
            community_analysis: Análise de comunidades
            key_actors_analysis: Análise de atores-chave
            propagation_analysis: Análise de propagação
            coordination_analysis: Análise de coordenação

        Returns:
            Insights contextuais
        """
        # Preparar dados para insights
        total_communities = community_analysis.get('total_communities', 0)
        key_actors_count = len(key_actors_analysis.get('key_actors', []))
        coordination_detected = coordination_analysis.get('coordination_detected', False)

        prompt = f"""
Gere insights sobre a estrutura de redes do ecossistema do Telegram brasileiro (2019-2023):

DADOS DA ANÁLISE:
- Comunidades detectadas: {total_communities}
- Atores-chave identificados: {key_actors_count}
- Coordenação detectada: {coordination_detected}

CONTEXTO: Redes de canais do Telegram durante governo Bolsonaro, pandemia, eleições 2022.

Analise:
1. Estrutura do ecossistema informacional
2. Padrões de influência e propagação
3. Indicadores de manipulação ou coordenação
4. Polarização e fragmentação
5. Implicações para democracia digital

Responda em JSON:
{{
    "network_insights": [
        {{
            "insight": "insight_principal",
            "evidence": "evidência_das_redes",
            "implications": "implicações_para_democracia"
        }}
    ],
    "ecosystem_health": {{
        "polarization_level": "alta|média|baixa",
        "fragmentation": "alta|média|baixa",
        "manipulation_risk": "alto|médio|baixo"
    }},
    "influence_patterns": {{
        "centralized": true|false,
        "hierarchical": true|false,
        "organic": true|false
    }},
    "democratic_implications": [
        "implicação_democrática_1",
        "implicação_democrática_2"
    ]
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='11_network_insights',
                operation='generate_insights'
            )

            insights = self.parse_json_response(response)
            return insights

        except Exception as e:
            self.logger.error(f"Erro na geração de insights de rede: {e}")
            return {
                'error': str(e),
                'fallback_insights': ['Análise de redes concluída com métodos estruturais']
            }

    def _generate_network_summary(self, networks: Dict[str, Any], structural_analysis: Dict,
                                community_analysis: Dict) -> Dict[str, Any]:
        """
        Gera resumo da análise de redes
        """
        network_stats = networks.get('network_stats', {})

        return {
            'networks_built': len([k for k in networks.keys() if k != 'network_stats']),
            'total_nodes_interaction': network_stats.get('interaction_nodes', 0),
            'total_edges_interaction': network_stats.get('interaction_edges', 0),
            'communities_detected': community_analysis.get('total_communities', 0),
            'structural_analysis_completed': bool(structural_analysis),
            'analysis_quality': 'ai_enhanced' if community_analysis.get('communities') else 'structural_only',
            'methodology': 'intelligent_network_analysis_with_ai'
        }


def get_intelligent_network_analyzer(config: Dict[str, Any]) -> IntelligentNetworkAnalyzer:
    """
    Factory function para criar instância do IntelligentNetworkAnalyzer
    """
    return IntelligentNetworkAnalyzer(config)
