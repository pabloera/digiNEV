"""
Semantic TF-IDF Analyzer com API Anthropic

Módulo avançado para análise TF-IDF com interpretação semântica e contextual.
Combina análise estatística tradicional com insights semânticos da API.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import AnthropicBase

# Import Voyage embeddings for enhanced TF-IDF
try:
    from .voyage_embeddings import VoyageEmbeddingAnalyzer
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False
    VoyageEmbeddingAnalyzer = None

logger = logging.getLogger(__name__)


class SemanticTfidfAnalyzer(AnthropicBase):
    """
    Analisador TF-IDF com semântica usando API Anthropic

    Funcionalidades:
    - Extração TF-IDF tradicional
    - Interpretação semântica dos termos
    - Agrupamento temático de palavras-chave
    - Análise contextual de relevância
    - Identificação de termos emergentes
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurações específicas
        tfidf_config = config.get('tfidf', {})
        self.max_features = tfidf_config.get('max_features', 5000)
        self.ngram_range = tuple(tfidf_config.get('ngram_range', [1, 3]))
        self.min_df = tfidf_config.get('min_df', 2)
        self.max_df = tfidf_config.get('max_df', 0.95)

        # Parâmetros AI
        self.interpretation_batch_size = tfidf_config.get('interpretation_batch_size', 50)
        self.relevance_threshold = tfidf_config.get('relevance_threshold', 0.7)

        # Initialize Voyage embeddings if enabled
        self.voyage_analyzer = None
        self.use_voyage_embeddings = False

        # Check if Voyage is enabled for TF-IDF
        embeddings_config = config.get('embeddings', {})
        integration_config = embeddings_config.get('integration', {})

        if VOYAGE_AVAILABLE and integration_config.get('tfidf_analysis', False):
            try:
                self.voyage_analyzer = VoyageEmbeddingAnalyzer(config)
                self.use_voyage_embeddings = True
                self.logger.info("Voyage embeddings habilitado para TF-IDF análise")
            except Exception as e:
                self.logger.warning(f"Falha ao inicializar Voyage para TF-IDF: {e}")
                self.use_voyage_embeddings = False

    def extract_semantic_tfidf(self, df: pd.DataFrame, text_column: str = 'text_cleaned',
                              category_column: str = None) -> Dict[str, Any]:
        """
        Extrai TF-IDF com análise semântica

        Args:
            df: DataFrame com os dados
            text_column: Coluna de texto para análise
            category_column: Coluna de categoria para análise comparativa

        Returns:
            Resultado completo da análise TF-IDF semântica
        """
        self.logger.info("Iniciando análise TF-IDF semântica")

        if text_column not in df.columns:
            raise ValueError(f"Coluna {text_column} não encontrada")

        # Preparar textos
        texts = df[text_column].fillna('').astype(str)

        # Análise TF-IDF tradicional
        traditional_results = self._extract_traditional_tfidf(texts, category_column, df)

        # Enhanced semantic analysis with Voyage embeddings if available
        if self.use_voyage_embeddings:
            self.logger.info("Aplicando Voyage embeddings para análise TF-IDF aprimorada")
            semantic_analysis = self._analyze_terms_with_voyage(
                traditional_results['top_terms'],
                texts,
                traditional_results['category_terms'] if category_column else None
            )
        else:
            # Análise semântica dos top termos (método tradicional)
            semantic_analysis = self._analyze_terms_semantically(
                traditional_results['top_terms'],
                traditional_results['category_terms'] if category_column else None
            )

        # Análise contextual
        contextual_insights = self._generate_contextual_insights(
            traditional_results, semantic_analysis, df, text_column
        )

        return {
            'traditional_tfidf': traditional_results,
            'semantic_analysis': semantic_analysis,
            'contextual_insights': contextual_insights,
            'embeddings_enhanced': self.use_voyage_embeddings,
            'analysis_summary': self._generate_analysis_summary(
                traditional_results, semantic_analysis, contextual_insights
            )
        }

    def analyze_semantic_tfidf(self, texts: List[str], categories: List[str], context: str = "") -> List[Dict[str, Any]]:
        """
        Analisa lote de textos para extração TF-IDF semântica

        Args:
            texts: Lista de textos para analisar
            categories: Categorias de análise política
            context: Contexto da análise

        Returns:
            Lista de resultados TF-IDF para cada texto
        """
        try:
            # Preparar prompt para análise TF-IDF semântica
            texts_formatted = "\n".join([
                f"{i+1}. {text[:150]}..." if len(text) > 150 else f"{i+1}. {text}"
                for i, text in enumerate(texts)
            ])

            categories_str = ", ".join(categories)

            prompt = f"""
Analise os seguintes {len(texts)} textos brasileiros do Telegram (2019-2023) e extraia keywords TF-IDF com análise semântica:

{texts_formatted}

CATEGORIAS POLÍTICAS: {categories_str}

Para cada texto, identifique:
1. Palavras-chave mais relevantes (TF-IDF)
2. Grupo semântico principal
3. Keywords políticas específicas
4. Score de importância (0.0-1.0)
5. Relevância contextual
6. Alinhamento com tópicos
7. Marcadores discursivos

RESPOSTA (JSON por linha):
{{"tfidf_keywords": "palavra1,palavra2,palavra3", "tfidf_scores": "0.8,0.7,0.6", "semantic_group": "categoria", "political_keywords": "termo1,termo2", "importance_score": 0.85, "contextual_relevance": 0.9, "topic_alignment": "alinhamento", "discourse_markers": "marcador1,marcador2"}}
"""

            response = self.create_message(
                prompt,
                stage="06_tfidf_extraction",
                operation="semantic_analysis",
                temperature=0.2
            )

            return self._parse_tfidf_response(response, len(texts))

        except Exception as e:
            self.logger.error(f"Erro na análise semântica TF-IDF: {e}")
            # Fallback: retornar análise básica
            return [self._get_default_tfidf_result() for _ in texts]

    def _parse_tfidf_response(self, response: str, expected_count: int) -> List[Dict[str, Any]]:
        """Parse response da API para extrair análise TF-IDF"""
        results = []

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Procurar por padrão "número. {json}"
            import re
            match = re.match(r'^\d+\.\s*({.*})', line)
            if match:
                try:
                    tfidf_data = json.loads(match.group(1))
                    results.append(self._validate_tfidf_data(tfidf_data))
                except json.JSONDecodeError:
                    results.append(self._get_default_tfidf_result())
            elif len(results) == 0 and line.startswith('{'):
                try:
                    tfidf_data = json.loads(line)
                    results.append(self._validate_tfidf_data(tfidf_data))
                except json.JSONDecodeError:
                    results.append(self._get_default_tfidf_result())

        # Garantir que temos o número correto de resultados
        while len(results) < expected_count:
            results.append(self._get_default_tfidf_result())

        return results[:expected_count]

    def _validate_tfidf_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Valida e normaliza dados de TF-IDF"""
        return {
            'tfidf_keywords': str(data.get('tfidf_keywords', '')),
            'tfidf_scores': str(data.get('tfidf_scores', '')),
            'semantic_group': str(data.get('semantic_group', 'geral')),
            'political_keywords': str(data.get('political_keywords', '')),
            'importance_score': min(1.0, max(0.0, float(data.get('importance_score', 0.0)))),
            'contextual_relevance': min(1.0, max(0.0, float(data.get('contextual_relevance', 0.0)))),
            'topic_alignment': str(data.get('topic_alignment', '')),
            'discourse_markers': str(data.get('discourse_markers', ''))
        }

    def _get_default_tfidf_result(self) -> Dict[str, Any]:
        """Retorna resultado TF-IDF padrão em caso de erro"""
        return {
            'tfidf_keywords': '',
            'tfidf_scores': '',
            'semantic_group': 'geral',
            'political_keywords': '',
            'importance_score': 0.0,
            'contextual_relevance': 0.0,
            'topic_alignment': '',
            'discourse_markers': ''
        }

    def _extract_traditional_tfidf(self, texts: pd.Series, category_column: str = None,
                                 df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Extração TF-IDF tradicional

        Args:
            texts: Série de textos
            category_column: Coluna de categoria
            df: DataFrame original

        Returns:
            Resultados TF-IDF tradicionais
        """
        self.logger.info("Executando análise TF-IDF tradicional")

        # Configurar vectorizer
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=self._get_portuguese_stopwords()
        )

        # Fit e transform
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Top termos globais
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-100:][::-1]  # Top 100

        top_terms = [
            {
                'term': feature_names[idx],
                'score': float(mean_scores[idx]),
                'doc_frequency': int((tfidf_matrix[:, idx] > 0).sum())
            }
            for idx in top_indices
        ]

        result = {
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'feature_names': feature_names,
            'top_terms': top_terms,
            'total_features': len(feature_names),
            'total_documents': tfidf_matrix.shape[0]
        }

        # Análise por categoria se disponível
        if category_column and df is not None:
            category_analysis = self._analyze_by_category(
                df, texts, tfidf_matrix, feature_names, category_column
            )
            result['category_terms'] = category_analysis

        return result

    def _analyze_by_category(self, df: pd.DataFrame, texts: pd.Series,
                           tfidf_matrix, feature_names: np.ndarray,
                           category_column: str) -> Dict[str, List[Dict]]:
        """
        Análise TF-IDF por categoria

        Args:
            df: DataFrame original
            texts: Série de textos
            tfidf_matrix: Matriz TF-IDF
            feature_names: Nomes das features
            category_column: Coluna de categoria

        Returns:
            Termos por categoria
        """
        category_terms = {}

        for category in df[category_column].unique():
            if pd.isna(category):
                continue

            # Filtrar documentos da categoria
            category_mask = df[category_column] == category
            category_tfidf = tfidf_matrix[category_mask]

            if category_tfidf.shape[0] == 0:
                continue

            # Calcular scores médios para a categoria
            mean_scores = np.array(category_tfidf.mean(axis=0)).flatten()
            top_indices = mean_scores.argsort()[-50:][::-1]  # Top 50 por categoria

            category_terms[str(category)] = [
                {
                    'term': feature_names[idx],
                    'score': float(mean_scores[idx]),
                    'doc_frequency': int((category_tfidf[:, idx] > 0).sum()),
                    'category_docs': int(category_tfidf.shape[0])
                }
                for idx in top_indices if mean_scores[idx] > 0
            ]

        return category_terms

    def _analyze_terms_semantically(self, top_terms: List[Dict],
                                  category_terms: Dict[str, List[Dict]] = None) -> Dict[str, Any]:
        """
        Análise semântica dos termos usando AI

        Args:
            top_terms: Top termos globais
            category_terms: Termos por categoria

        Returns:
            Análise semântica dos termos
        """
        self.logger.info("Iniciando análise semântica dos termos")

        # Preparar lista de termos para análise
        terms_to_analyze = [term['term'] for term in top_terms[:self.interpretation_batch_size]]

        # Análise semântica dos termos principais
        semantic_groups = self._group_terms_semantically(terms_to_analyze)

        # Análise de relevância contextual
        relevance_analysis = self._analyze_term_relevance(terms_to_analyze)

        result = {
            'semantic_groups': semantic_groups,
            'relevance_analysis': relevance_analysis,
            'terms_analyzed': len(terms_to_analyze)
        }

        # Análise comparativa por categoria se disponível
        if category_terms:
            category_semantic_analysis = self._analyze_category_semantic_differences(category_terms)
            result['category_semantic_analysis'] = category_semantic_analysis

        return result

    def _group_terms_semantically(self, terms: List[str]) -> Dict[str, Any]:
        """
        Agrupa termos por similaridade semântica

        Args:
            terms: Lista de termos para agrupar

        Returns:
            Grupos semânticos de termos
        """
        if not terms:
            return {'groups': []}

        prompt = f"""
Analise os seguintes termos extraídos de mensagens do Telegram brasileiro sobre política e agrupe-os semanticamente:

TERMOS: {', '.join(terms)}

Contexto: Estes termos vêm de análise TF-IDF de mensagens políticas do período 2019-2023, incluindo discussões sobre governo Bolsonaro, pandemia, eleições.

Tarefas:
1. Agrupe termos semanticamente relacionados
2. Identifique temas principais
3. Classifique por relevância política/social
4. Identifique termos de alta carga emocional

Responda em JSON:
{{
    "semantic_groups": [
        {{
            "theme": "nome_do_tema",
            "terms": ["termo1", "termo2", "termo3"],
            "description": "descrição_do_tema",
            "political_relevance": "alta|média|baixa",
            "emotional_charge": "alta|média|baixa|neutra"
        }}
    ],
    "ungrouped_terms": ["termo_isolado1", "termo_isolado2"],
    "dominant_themes": ["tema1", "tema2", "tema3"],
    "analysis_notes": "observações_gerais"
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='06_semantic_grouping',
                operation='group_terms_semantically'
            )

            analysis = self.parse_json_response(response)

            self.logger.info(f"Agrupamento semântico concluído: {len(analysis.get('semantic_groups', []))} grupos identificados")

            return analysis

        except Exception as e:
            self.logger.error(f"Erro no agrupamento semântico: {e}")
            return {
                'semantic_groups': [],
                'ungrouped_terms': terms,
                'error': str(e)
            }

    def _analyze_term_relevance(self, terms: List[str]) -> Dict[str, Any]:
        """
        Analisa relevância contextual dos termos

        Args:
            terms: Lista de termos para analisar

        Returns:
            Análise de relevância
        """
        prompt = f"""
Avalie a relevância contextual destes termos para análise de discurso político brasileiro (2019-2023):

TERMOS: {', '.join(terms[:30])}

Contexto: Análise de mensagens do Telegram sobre movimento bolsonarista, incluindo temas como:
- Negacionismo (científico, pandêmico, histórico)
- Autoritarismo e democracia
- Desinformação e teorias conspiratórias
- Polarização política

Para cada termo, avalie:
1. Relevância para análise política (0-1)
2. Indicador de polarização
3. Potencial para desinformação
4. Significado no contexto brasileiro

Responda em JSON:
{{
    "term_relevance": [
        {{
            "term": "termo",
            "political_relevance": 0.95,
            "polarization_indicator": true,
            "misinformation_potential": "alto|médio|baixo",
            "brazilian_context_significance": "descrição_breve",
            "category": "negacionismo|autoritarismo|desinformação|neutro"
        }}
    ],
    "high_relevance_terms": ["termo1", "termo2"],
    "polarization_terms": ["termo_polarizador1", "termo_polarizador2"],
    "misinformation_indicators": ["termo_desinformação1", "termo_desinformação2"]
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='06_relevance_analysis',
                operation='analyze_term_relevance'
            )

            analysis = self.parse_json_response(response)

            return analysis

        except Exception as e:
            self.logger.error(f"Erro na análise de relevância: {e}")
            return {
                'term_relevance': [],
                'error': str(e)
            }

    def _analyze_category_semantic_differences(self, category_terms: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Analisa diferenças semânticas entre categorias

        Args:
            category_terms: Termos por categoria

        Returns:
            Análise de diferenças semânticas
        """
        if len(category_terms) < 2:
            return {'analysis': 'insufficient_categories'}

        # Preparar comparação entre categorias
        comparison_data = {}
        for category, terms in list(category_terms.items())[:5]:  # Limitar categorias
            top_terms = [term['term'] for term in terms[:20]]  # Top 20 por categoria
            comparison_data[category] = top_terms

        prompt = f"""
Compare os termos TF-IDF mais relevantes entre diferentes categorias de mensagens:

CATEGORIAS E TERMOS:
"""

        for category, terms in comparison_data.items():
            prompt += f"\n{category}: {', '.join(terms)}\n"

        prompt += """

Analise:
1. Diferenças semânticas entre categorias
2. Termos únicos e distintivos de cada categoria
3. Sobreposições e semelhanças
4. Perfis discursivos de cada categoria

Responda em JSON:
{
    "category_profiles": [
        {
            "category": "nome_categoria",
            "distinctive_terms": ["termo1", "termo2"],
            "discourse_profile": "descrição_do_perfil",
            "semantic_focus": "foco_principal"
        }
    ],
    "cross_category_analysis": {
        "common_terms": ["termo_comum1", "termo_comum2"],
        "unique_patterns": "padrões_únicos_identificados",
        "polarization_indicators": ["categoria_mais_polarizada"]
    },
    "recommendations": ["recomendação1", "recomendação2"]
}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='06_category_semantic_analysis',
                operation='analyze_category_differences'
            )

            analysis = self.parse_json_response(response)

            return analysis

        except Exception as e:
            self.logger.error(f"Erro na análise comparativa: {e}")
            return {
                'error': str(e),
                'categories_attempted': list(comparison_data.keys())
            }

    def _generate_contextual_insights(self, traditional_results: Dict, semantic_analysis: Dict,
                                    df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """
        Gera insights contextuais combinando análises

        Args:
            traditional_results: Resultados TF-IDF tradicionais
            semantic_analysis: Análise semântica
            df: DataFrame original
            text_column: Coluna de texto

        Returns:
            Insights contextuais
        """
        # Preparar dados para insights
        top_terms = traditional_results['top_terms'][:20]
        semantic_groups = semantic_analysis.get('semantic_groups', [])

        prompt = f"""
Gere insights contextuais combinando análise TF-IDF estatística com análise semântica:

DADOS ESTATÍSTICOS:
- Total de documentos: {traditional_results['total_documents']}
- Total de features: {traditional_results['total_features']}
- Top 10 termos: {', '.join([t['term'] for t in top_terms[:10]])}

GRUPOS SEMÂNTICOS IDENTIFICADOS:
{json.dumps([{'theme': g.get('theme', ''), 'terms': g.get('terms', [])} for g in semantic_groups[:5]], ensure_ascii=False, indent=2)}

Gere insights sobre:
1. Padrões discursivos dominantes
2. Evolução temática (se temporal)
3. Indicadores de polarização
4. Potenciais vieses ou tendências
5. Relevância para pesquisa acadêmica

Responda em JSON:
        {
            "key_insights": [
                {
                    "insight": "insight_principal",
                    "evidence": "evidência_estatística_ou_semântica",
                    "implications": "implicações_para_análise"
                }
            ],
            "discourse_patterns": {
                "dominant_themes": ["tema1", "tema2"],
                "polarization_level": "alta|média|baixa",
                "emotional_intensity": "alta|média|baixa"
            },
            "research_recommendations": [
                "recomendação_para_pesquisa1",
                "recomendação_para_pesquisa2"
            ],
            "methodological_notes": "observações_metodológicas"
}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='06_contextual_insights',
                operation='generate_insights'
            )

            insights = self.parse_json_response(response)

            return insights

        except Exception as e:
            self.logger.error(f"Erro na geração de insights: {e}")
            return {
                'error': str(e),
                'fallback_insights': ['Análise TF-IDF concluída com métodos tradicionais']
            }

    def _analyze_terms_with_voyage(self, top_terms: List[Dict], texts: pd.Series,
                                  category_terms: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, Any]:
        """
        Análise semântica aprimorada usando Voyage embeddings

        Args:
            top_terms: Top termos globais
            texts: Textos originais
            category_terms: Termos por categoria

        Returns:
            Análise semântica com embeddings
        """
        self.logger.info("🚀 Iniciando análise TF-IDF semântica com Voyage embeddings")

        try:
            # Extrair apenas os termos para embedding
            terms_to_embed = [term['term'] for term in top_terms[:self.interpretation_batch_size]]

            if not self.voyage_analyzer:
                raise ValueError("Voyage analyzer não inicializado")

            # Gerar embeddings para os termos com otimização de custo
            self.logger.info(f"📊 Gerando embeddings para {len(terms_to_embed)} termos TF-IDF")
            embedding_result = self.voyage_analyzer.generate_embeddings(
                terms_to_embed,
                input_type="query",  # Terms are queries for semantic analysis
                cache_key=f"tfidf_terms_{len(terms_to_embed)}"
            )

            if not embedding_result['embeddings']:
                raise ValueError("Nenhum embedding gerado")

            term_embeddings = np.array(embedding_result['embeddings'])

            # Calcular similaridade entre termos para agrupamento
            similarity_matrix = cosine_similarity(term_embeddings)

            # Agrupar termos semanticamente similares
            semantic_groups = self._cluster_terms_by_embeddings(
                terms_to_embed, term_embeddings, similarity_matrix
            )

            # Análise de relevância usando embeddings context
            relevance_analysis = self._analyze_relevance_with_embeddings_enhanced(
                terms_to_embed, term_embeddings, texts, top_terms
            )

            # Combinar com análise AI tradicional para insights mais ricos
            ai_enhanced_groups = self._enhance_groups_with_ai(semantic_groups)

            result = {
                'semantic_groups': ai_enhanced_groups,
                'relevance_analysis': relevance_analysis,
                'terms_analyzed': len(terms_to_embed),
                'embeddings_used': True,
                'embedding_model': self.voyage_analyzer.model_name if self.voyage_analyzer else None,
                'embedding_stats': embedding_result.get('processing_stats', {}),
                'cost_optimized': hasattr(self.voyage_analyzer, 'enable_sampling') and self.voyage_analyzer.enable_sampling
            }

            # Análise comparativa por categoria se disponível
            if category_terms:
                category_semantic_analysis = self._analyze_category_with_embeddings_enhanced(category_terms)
                result['category_semantic_analysis'] = category_semantic_analysis

            self.logger.info(f"✅ Análise TF-IDF semântica concluída com {len(ai_enhanced_groups)} grupos")
            return result

        except Exception as e:
            self.logger.error(f"❌ Erro na análise TF-IDF com Voyage embeddings: {e}")
            # Fallback para análise tradicional
            self.logger.info("🔄 Usando fallback para análise tradicional")
            if category_terms is not None:
                return self._analyze_terms_semantically(top_terms, category_terms)
            else:
                return self._analyze_terms_semantically(top_terms)

    def _cluster_terms_by_embeddings(self, terms: List[str], embeddings: np.ndarray,
                                    similarity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Agrupa termos usando embeddings e similaridade
        """
        # Determinar número ótimo de clusters
        n_clusters = min(len(terms) // 5, 10)  # Max 10 clusters

        if n_clusters < 2:
            return [{
                'theme': 'geral',
                'terms': terms,
                'description': 'Todos os termos',
                'cohesion_score': 1.0
            }]

        # Clustering com KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Organizar termos por cluster
        clusters = defaultdict(list)
        for term, label in zip(terms, cluster_labels):
            clusters[label].append(term)

        # Calcular coesão de cada cluster
        semantic_groups = []
        for label, cluster_terms in clusters.items():
            # Índices dos termos no cluster
            term_indices = [i for i, l in enumerate(cluster_labels) if l == label]

            # Calcular coesão média do cluster
            if len(term_indices) > 1:
                cluster_similarities = []
                for i in term_indices:
                    for j in term_indices:
                        if i != j:
                            cluster_similarities.append(similarity_matrix[i][j])
                cohesion_score = np.mean(cluster_similarities) if cluster_similarities else 0.0
            else:
                cohesion_score = 1.0

            semantic_groups.append({
                'cluster_id': int(label),
                'terms': cluster_terms,
                'cohesion_score': float(cohesion_score),
                'size': len(cluster_terms)
            })

        # Ordenar por coesão
        semantic_groups.sort(key=lambda x: x['cohesion_score'], reverse=True)

        return semantic_groups

    def _analyze_relevance_with_embeddings_enhanced(self, terms: List[str], embeddings: np.ndarray,
                                                  texts: pd.Series, top_terms: List[Dict]) -> Dict[str, Any]:
        """
        Análise aprimorada de relevância dos termos usando embeddings contextuais
        """
        try:
            if not self.voyage_analyzer:
                raise ValueError("Voyage analyzer não disponível")

            # Combinar embeddings com scores TF-IDF originais
            tfidf_scores = {term['term']: term['score'] for term in top_terms}

            # Calcular medidas de relevância mais sofisticadas
            term_relevances = []
            embedding_variances = np.var(embeddings, axis=1)
            embedding_norms = np.linalg.norm(embeddings, axis=1)

            for i, term in enumerate(terms):
                if i >= len(embeddings):
                    continue

                # Combinar múltiplas métricas
                tfidf_score = tfidf_scores.get(term, 0.0)
                embedding_variance = float(embedding_variances[i])
                embedding_norm = float(embedding_norms[i])

                # Score composto: TF-IDF + características semânticas
                composite_score = (
                    tfidf_score * 0.4 +                    # Importância estatística
                    embedding_variance * 0.3 +             # Diversidade semântica
                    (embedding_norm / np.max(embedding_norms)) * 0.3  # Magnitude semântica
                )

                term_relevances.append({
                    'term': term,
                    'tfidf_score': float(tfidf_score),
                    'semantic_variance': embedding_variance,
                    'semantic_magnitude': embedding_norm,
                    'composite_relevance': float(composite_score),
                    'embedding_dimension': len(embeddings[i]) if i < len(embeddings) else 0
                })

            # Identificar termos de alta relevância usando score composto
            sorted_terms = sorted(term_relevances, key=lambda x: x['composite_relevance'], reverse=True)
            composite_scores = [t['composite_relevance'] for t in term_relevances]
            high_relevance_threshold = float(np.percentile(composite_scores, 75)) if composite_scores else 0.5

            return {
                'term_relevance': term_relevances,
                'high_relevance_terms': [t['term'] for t in sorted_terms if t['composite_relevance'] >= high_relevance_threshold],
                'relevance_statistics': {
                    'mean_composite': float(np.mean(composite_scores)) if composite_scores else 0,
                    'std_composite': float(np.std(composite_scores)) if composite_scores else 0,
                    'threshold': high_relevance_threshold,
                    'total_terms': len(term_relevances)
                },
                'method': 'enhanced_embedding_composite',
                'features_combined': ['tfidf', 'semantic_variance', 'semantic_magnitude']
            }

        except Exception as e:
            self.logger.error(f"Erro na análise de relevância aprimorada: {e}")
            # Fallback para método simples
            return self._analyze_relevance_with_embeddings(terms, embeddings, texts)

    def _analyze_relevance_with_embeddings(self, terms: List[str], embeddings: np.ndarray,
                                          texts: pd.Series) -> Dict[str, Any]:
        """
        Analisa relevância dos termos usando embeddings - versão simplificada
        """
        try:
            if not self.voyage_analyzer:
                raise ValueError("Voyage analyzer não disponível")

            # Análise simplificada baseada nos próprios embeddings
            # Calcular variância dos embeddings como proxy para relevância
            embedding_variances = np.var(embeddings, axis=1)

            term_relevances = []
            for i, (term, variance) in enumerate(zip(terms, embedding_variances)):
                term_relevances.append({
                    'term': term,
                    'avg_relevance': float(variance),
                    'max_relevance': float(variance),
                    'relevance_variance': float(variance)
                })

            # Identificar termos de alta relevância
            sorted_terms = sorted(term_relevances, key=lambda x: x['avg_relevance'], reverse=True)
            high_relevance_threshold = float(np.percentile(embedding_variances, 75))

            return {
                'term_relevance': term_relevances,
                'high_relevance_terms': [t['term'] for t in sorted_terms if t['avg_relevance'] >= high_relevance_threshold],
                'relevance_statistics': {
                    'mean': float(np.mean(embedding_variances)),
                    'std': float(np.std(embedding_variances)),
                    'threshold': high_relevance_threshold
                },
                'method': 'embedding_variance'
            }

        except Exception as e:
            self.logger.error(f"Erro na análise de relevância com embeddings: {e}")
            # Fallback simples
            return {
                'term_relevance': [{'term': term, 'avg_relevance': 0.5} for term in terms],
                'high_relevance_terms': terms[:len(terms)//2],
                'relevance_statistics': {'mean': 0.5, 'std': 0.1, 'threshold': 0.5},
                'method': 'fallback'
            }

    def _enhance_groups_with_ai(self, embedding_groups: List[Dict]) -> List[Dict[str, Any]]:
        """
        Enriquece grupos de embeddings com análise AI
        """
        if not embedding_groups:
            return []

        # Preparar grupos para análise
        groups_text = "\n".join([
            f"Grupo {g['cluster_id']}: {', '.join(g['terms'][:10])} (coesão: {g['cohesion_score']:.2f})"
            for g in embedding_groups[:10]
        ])

        prompt = f"""
Analise os seguintes grupos de termos agrupados por similaridade semântica (embeddings):

{groups_text}

Para cada grupo, forneça:
1. Um tema/nome descritivo
2. Descrição do foco semântico
3. Relevância política/social
4. Interpretação no contexto brasileiro

Responda em JSON:
{{
    "enhanced_groups": [
        {{
            "cluster_id": 0,
            "theme": "nome_do_tema",
            "description": "descrição_do_grupo",
            "political_relevance": "alta|média|baixa",
            "brazilian_context": "interpretação_contextual"
        }}
    ]
}}
"""

        try:
            response = self.create_message(
                prompt=prompt,
                stage='06_embedding_enhancement',
                operation='enhance_semantic_groups'
            )

            enhancement = self.parse_json_response(response)

            # Mesclar com dados originais
            enhanced_groups = []
            for orig_group in embedding_groups:
                enhanced = next(
                    (eg for eg in enhancement.get('enhanced_groups', [])
                     if eg.get('cluster_id') == orig_group['cluster_id']),
                    {}
                )

                merged_group = {
                    **orig_group,
                    'theme': enhanced.get('theme', f"Grupo {orig_group['cluster_id']}"),
                    'description': enhanced.get('description', 'Grupo semântico'),
                    'political_relevance': enhanced.get('political_relevance', 'média'),
                    'brazilian_context': enhanced.get('brazilian_context', ''),
                    'embedding_based': True
                }
                enhanced_groups.append(merged_group)

            return enhanced_groups

        except Exception as e:
            self.logger.error(f"Erro ao enriquecer grupos com AI: {e}")
            # Retornar grupos básicos
            return [{
                **g,
                'theme': f"Grupo {g['cluster_id']}",
                'description': f"Grupo com {len(g['terms'])} termos",
                'embedding_based': True
            } for g in embedding_groups]

    def _analyze_category_with_embeddings_enhanced(self, category_terms: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Análise aprimorada de categorias usando embeddings contextuais
        """
        try:
            self.logger.info(f"🔍 Analisando {len(category_terms)} categorias com embeddings")
            category_profiles = []

            for category, terms in list(category_terms.items())[:5]:  # Processar até 5 categorias
                # Extrair top termos da categoria
                cat_terms = [t['term'] for t in terms[:15]]  # Mais termos para análise

                if not cat_terms:
                    continue

                try:
                    # Gerar embeddings para termos da categoria
                    cat_embedding_result = self.voyage_analyzer.generate_embeddings(
                        cat_terms,
                        input_type="query",
                        cache_key=f"category_{category}_{len(cat_terms)}"
                    )

                    if cat_embedding_result['embeddings']:
                        cat_embeddings = np.array(cat_embedding_result['embeddings'])

                        # Calcular coesão semântica da categoria
                        cat_similarity_matrix = cosine_similarity(cat_embeddings)
                        np.fill_diagonal(cat_similarity_matrix, 0)
                        category_cohesion = float(np.mean(cat_similarity_matrix))

                        # Identificar termos centrais da categoria
                        centrality_scores = np.mean(cat_similarity_matrix, axis=1)
                        central_term_indices = np.argsort(centrality_scores)[-3:][::-1]
                        central_terms = [cat_terms[i] for i in central_term_indices if i < len(cat_terms)]

                        category_profiles.append({
                            'category': category,
                            'term_count': len(cat_terms),
                            'top_terms': cat_terms[:5],
                            'central_terms': central_terms,
                            'semantic_cohesion': category_cohesion,
                            'embedding_dimension': len(cat_embeddings[0]) if len(cat_embeddings) > 0 else 0,
                            'embedding_analysis': 'enhanced',
                            'processing_stats': cat_embedding_result.get('processing_stats', {})
                        })

                    else:
                        # Fallback para análise simples
                        category_profiles.append({
                            'category': category,
                            'term_count': len(cat_terms),
                            'top_terms': cat_terms[:5],
                            'embedding_analysis': 'fallback_simple'
                        })

                except Exception as cat_error:
                    self.logger.warning(f"Erro ao analisar categoria {category}: {cat_error}")
                    continue

            # Análise comparativa entre categorias
            if len(category_profiles) > 1:
                comparative_analysis = self._analyze_category_relationships(category_profiles)
            else:
                comparative_analysis = {}

            return {
                'category_profiles': category_profiles,
                'comparative_analysis': comparative_analysis,
                'categories_processed': len(category_profiles),
                'embeddings_used': True,
                'method': 'enhanced_contextual'
            }

        except Exception as e:
            self.logger.error(f"Erro na análise aprimorada de categorias: {e}")
            # Fallback para método simples
            return self._analyze_category_with_embeddings(category_terms)

    def _analyze_category_relationships(self, category_profiles: List[Dict]) -> Dict[str, Any]:
        """
        Analisa relacionamentos semânticos entre categorias
        """
        try:
            # Identificar categorias com alta coesão
            high_cohesion_categories = [
                cat for cat in category_profiles
                if cat.get('semantic_cohesion', 0) > 0.5
            ]

            # Identificar categorias com baixa coesão (mais diversas)
            diverse_categories = [
                cat for cat in category_profiles
                if cat.get('semantic_cohesion', 0) < 0.3
            ]

            # Estatísticas comparativas
            cohesion_scores = [cat.get('semantic_cohesion', 0) for cat in category_profiles]

            return {
                'high_cohesion_categories': [cat['category'] for cat in high_cohesion_categories],
                'diverse_categories': [cat['category'] for cat in diverse_categories],
                'cohesion_statistics': {
                    'mean': float(np.mean(cohesion_scores)) if cohesion_scores else 0,
                    'std': float(np.std(cohesion_scores)) if cohesion_scores else 0,
                    'max': float(np.max(cohesion_scores)) if cohesion_scores else 0,
                    'min': float(np.min(cohesion_scores)) if cohesion_scores else 0
                },
                'most_cohesive_category': max(category_profiles, key=lambda x: x.get('semantic_cohesion', 0))['category'] if category_profiles else None,
                'most_diverse_category': min(category_profiles, key=lambda x: x.get('semantic_cohesion', 0))['category'] if category_profiles else None
            }

        except Exception as e:
            self.logger.error(f"Erro na análise de relacionamentos entre categorias: {e}")
            return {'error': str(e)}

    def _analyze_category_with_embeddings(self, category_terms: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Análise de categorias usando embeddings - versão simplificada
        """
        try:
            category_profiles = []

            for category, terms in list(category_terms.items())[:3]:  # Limitar a 3 categorias
                # Extrair top termos da categoria
                cat_terms = [t['term'] for t in terms[:10]]  # Limitar a 10 termos

                if not cat_terms:
                    continue

                # Análise simplificada sem embeddings complexos
                category_profiles.append({
                    'category': category,
                    'term_count': len(cat_terms),
                    'top_terms': cat_terms[:5],
                    'embedding_analysis': 'simplified'
                })

            return {
                'category_profiles': category_profiles,
                'embeddings_used': True,
                'method': 'simplified'
            }

        except Exception as e:
            self.logger.error(f"Erro na análise de categorias com embeddings: {e}")
            return {'error': str(e), 'embeddings_used': False}

    def _generate_analysis_summary(self, traditional_results: Dict, semantic_analysis: Dict,
                                 contextual_insights: Dict) -> Dict[str, Any]:
        """
        Gera resumo final da análise
        """
        return {
            'total_terms_extracted': traditional_results['total_features'],
            'documents_analyzed': traditional_results['total_documents'],
            'semantic_groups_found': len(semantic_analysis.get('semantic_groups', [])),
            'high_relevance_terms': len(semantic_analysis.get('relevance_analysis', {}).get('high_relevance_terms', [])),
            'key_insights_generated': len(contextual_insights.get('key_insights', [])),
            'analysis_quality': 'voyage_enhanced' if semantic_analysis.get('embeddings_used') else 'ai_enhanced' if not semantic_analysis.get('error') else 'traditional_fallback',
            'embeddings_model': semantic_analysis.get('embedding_model'),
            'recommendations_available': len(contextual_insights.get('research_recommendations', []))
        }

    def _get_portuguese_stopwords(self) -> List[str]:
        """Retorna lista de stopwords em português"""
        return [
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no',
            'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à',
            'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só',
            'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter',
            'seus', 'suas', 'nem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 'tinha', 'foram', 'essa',
            'num', 'numa', 'pelos', 'pelas', 'essa', 'este', 'del', 'te', 'lo', 'le', 'les'
        ]


def get_semantic_tfidf_analyzer(config: Dict[str, Any]) -> SemanticTfidfAnalyzer:
    """
    Factory function para criar instância do SemanticTfidfAnalyzer
    """
    return SemanticTfidfAnalyzer(config)
