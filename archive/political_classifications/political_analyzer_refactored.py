"""
Political Analyzer - Classificação política de texto
===================================================

Implementa análise política com abordagem híbrida (IA + heurísticas).
Segue princípios SOLID e padrão Strategy para diferentes métodos de análise.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import logging
import re
from pathlib import Path

from ...core.base_client import AnalysisClient, AnalysisRequest, AnalysisType, AnalysisResponse

logger = logging.getLogger(__name__)


class PoliticalCategory(Enum):
    """Categorias políticas brasileiras"""
    EXTREMA_DIREITA = "extrema_direita"
    DIREITA = "direita"
    CENTRO_DIREITA = "centro_direita"
    CENTRO = "centro"
    CENTRO_ESQUERDA = "centro_esquerda"
    ESQUERDA = "esquerda"


@dataclass
class PoliticalClassification:
    """Resultado da classificação política"""
    category: PoliticalCategory
    confidence: float
    keywords_found: List[str]
    reasoning: Optional[str] = None
    method_used: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PoliticalLexicon:
    """
    Carregador e gerenciador do léxico político hierárquico.

    Implementa padrão Repository para abstração do acesso aos dados.
    """

    def __init__(self, lexicon_path: Optional[Path] = None):
        self.lexicon_path = lexicon_path
        self._lexicon_cache: Optional[Dict[str, Any]] = None
        self._keywords_by_category: Optional[Dict[PoliticalCategory, List[str]]] = None

    def load_lexicon(self) -> Dict[str, Any]:
        """Carrega léxico político do arquivo ou usa padrão"""
        if self._lexicon_cache is not None:
            return self._lexicon_cache

        if self.lexicon_path and self.lexicon_path.exists():
            try:
                import json
                with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                    self._lexicon_cache = json.load(f)
                logger.debug(f"Loaded political lexicon from {self.lexicon_path}")
            except Exception as e:
                logger.warning(f"Failed to load lexicon from {self.lexicon_path}: {e}")
                self._lexicon_cache = self._get_default_lexicon()
        else:
            self._lexicon_cache = self._get_default_lexicon()

        return self._lexicon_cache

    def get_keywords_by_category(self, category: PoliticalCategory) -> List[str]:
        """Obtém palavras-chave para uma categoria específica"""
        if self._keywords_by_category is None:
            self._build_category_keywords()

        return self._keywords_by_category.get(category, [])

    def get_all_keywords(self) -> Dict[PoliticalCategory, List[str]]:
        """Obtém todas as palavras-chave organizadas por categoria"""
        if self._keywords_by_category is None:
            self._build_category_keywords()

        return self._keywords_by_category.copy()

    def _build_category_keywords(self) -> None:
        """Constrói cache de palavras-chave por categoria"""
        lexicon = self.load_lexicon()
        self._keywords_by_category = {}

        for category in PoliticalCategory:
            keywords = []
            category_data = lexicon.get(category.value, {})

            # Extrair palavras-chave de diferentes níveis
            if isinstance(category_data, dict):
                for level_name, level_data in category_data.items():
                    if isinstance(level_data, list):
                        keywords.extend(level_data)
                    elif isinstance(level_data, dict):
                        for sublist in level_data.values():
                            if isinstance(sublist, list):
                                keywords.extend(sublist)

            self._keywords_by_category[category] = list(set(keywords))

    def _get_default_lexicon(self) -> Dict[str, Any]:
        """Retorna léxico político padrão"""
        return {
            "extrema_direita": {
                "core": [
                    "militarismo", "nacionalismo extremo", "autoritarismo",
                    "supremacia", "tradicionalismo radical"
                ],
                "extended": [
                    "ordem e progresso", "valores tradicionais", "família tradicional",
                    "patriotismo", "conservadorismo", "anti-globalismo"
                ]
            },
            "direita": {
                "core": [
                    "livre mercado", "propriedade privada", "iniciativa privada",
                    "capitalismo", "desregulamentação"
                ],
                "extended": [
                    "economia de mercado", "empreendedorismo", "competitividade",
                    "privatização", "eficiência econômica"
                ]
            },
            "centro_direita": {
                "core": [
                    "economia mista", "regulação moderada", "reformas graduais",
                    "estabilidade econômica"
                ],
                "extended": [
                    "crescimento sustentável", "políticas fiscais responsáveis",
                    "modernização", "inovação"
                ]
            },
            "centro": {
                "core": [
                    "consenso", "moderação", "pragmatismo", "equilíbrio",
                    "diálogo político"
                ],
                "extended": [
                    "políticas públicas", "desenvolvimento", "cidadania",
                    "democracia", "instituições"
                ]
            },
            "centro_esquerda": {
                "core": [
                    "social-democracia", "estado de bem-estar", "regulação social",
                    "redistribuição moderada"
                ],
                "extended": [
                    "políticas sociais", "proteção trabalhista", "sustentabilidade",
                    "inclusão social", "direitos civis"
                ]
            },
            "esquerda": {
                "core": [
                    "igualdade social", "justiça social", "redistribuição",
                    "direitos trabalhistas", "movimentos sociais"
                ],
                "extended": [
                    "participação popular", "democracia participativa", "reforma agrária",
                    "direitos humanos", "anti-imperialismo"
                ]
            }
        }


class PoliticalAnalysisStrategy(ABC):
    """Interface abstrata para estratégias de análise política"""

    @abstractmethod
    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> PoliticalClassification:
        """Analisa texto e retorna classificação política"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Retorna nome da estratégia"""
        pass


class HeuristicPoliticalStrategy(PoliticalAnalysisStrategy):
    """
    Estratégia de análise política baseada em heurísticas.

    Usa léxico político hierárquico com scoring ponderado.
    """

    def __init__(self, lexicon: PoliticalLexicon):
        self.lexicon = lexicon
        self.category_weights = {
            PoliticalCategory.EXTREMA_DIREITA: 1.2,
            PoliticalCategory.DIREITA: 1.0,
            PoliticalCategory.CENTRO_DIREITA: 0.8,
            PoliticalCategory.CENTRO: 0.6,
            PoliticalCategory.CENTRO_ESQUERDA: 0.8,
            PoliticalCategory.ESQUERDA: 1.0
        }

    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> PoliticalClassification:
        """Análise heurística baseada em palavras-chave"""
        text_normalized = self._normalize_text(text)
        category_scores = self._calculate_category_scores(text_normalized)

        if not category_scores:
            return PoliticalClassification(
                category=PoliticalCategory.CENTRO,
                confidence=0.3,
                keywords_found=[],
                reasoning="Nenhuma palavra-chave política encontrada",
                method_used="heuristic_fallback"
            )

        # Encontrar categoria com maior score
        best_category = max(category_scores.keys(), key=lambda c: category_scores[c]['score'])
        best_score = category_scores[best_category]['score']
        keywords_found = category_scores[best_category]['keywords']

        # Calcular confidence baseado no score relativo
        confidence = min(0.9, best_score / 10.0)  # Normalizar para max 0.9

        return PoliticalClassification(
            category=best_category,
            confidence=confidence,
            keywords_found=keywords_found,
            reasoning=f"Score: {best_score:.2f}, baseado em {len(keywords_found)} palavras-chave",
            method_used="heuristic",
            metadata={
                "all_scores": {cat.value: data['score'] for cat, data in category_scores.items()},
                "total_keywords": sum(len(data['keywords']) for data in category_scores.values())
            }
        )

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para análise"""
        # Converter para lowercase
        normalized = text.lower()

        # Remover caracteres especiais mas manter espaços
        normalized = re.sub(r'[^\w\s]', ' ', normalized)

        # Normalizar espaços múltiplos
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _calculate_category_scores(self, text: str) -> Dict[PoliticalCategory, Dict[str, Any]]:
        """Calcula scores para cada categoria política"""
        category_scores = {}
        all_keywords = self.lexicon.get_all_keywords()

        for category, keywords in all_keywords.items():
            found_keywords = []
            score = 0.0

            for keyword in keywords:
                if keyword.lower() in text:
                    found_keywords.append(keyword)
                    # Score ponderado pelo peso da categoria e frequência
                    keyword_count = text.count(keyword.lower())
                    weight = self.category_weights.get(category, 1.0)
                    score += keyword_count * weight

            if found_keywords:
                category_scores[category] = {
                    'score': score,
                    'keywords': found_keywords
                }

        return category_scores

    def get_strategy_name(self) -> str:
        return "heuristic"


class AIPoliticalStrategy(PoliticalAnalysisStrategy):
    """
    Estratégia de análise política usando IA.

    Usa cliente Anthropic para análise contextual avançada.
    """

    def __init__(self, ai_client: AnalysisClient):
        self.ai_client = ai_client

    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> PoliticalClassification:
        """Análise usando IA"""
        request = AnalysisRequest(
            text=text,
            analysis_type=AnalysisType.POLITICAL_CLASSIFICATION,
            parameters={
                "categories": [cat.value for cat in PoliticalCategory],
                "context": context or {}
            },
            context=context
        )

        try:
            response = await self.ai_client.analyze_text(request)

            if response.success:
                result = response.result
                category = PoliticalCategory(result.get('category', 'centro'))
                confidence = result.get('confidence', 0.5)
                keywords = result.get('keywords', [])

                return PoliticalClassification(
                    category=category,
                    confidence=confidence,
                    keywords_found=keywords,
                    reasoning=result.get('reasoning', ''),
                    method_used="ai",
                    metadata={
                        "ai_response": result,
                        "processing_time": response.processing_time
                    }
                )
            else:
                raise Exception(response.error_message or "AI analysis failed")

        except Exception as e:
            logger.warning(f"AI political analysis failed: {e}")
            raise

    def get_strategy_name(self) -> str:
        return "ai"


class HybridPoliticalStrategy(PoliticalAnalysisStrategy):
    """
    Estratégia híbrida que combina heurística e IA.

    Usa heurística primeiro, IA para validação se confidence < threshold.
    """

    def __init__(self, heuristic_strategy: HeuristicPoliticalStrategy,
                 ai_strategy: AIPoliticalStrategy,
                 ai_threshold: float = 0.7):
        self.heuristic_strategy = heuristic_strategy
        self.ai_strategy = ai_strategy
        self.ai_threshold = ai_threshold

    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> PoliticalClassification:
        """Análise híbrida com fallback inteligente"""
        # Primeira passada: análise heurística
        heuristic_result = await self.heuristic_strategy.analyze(text, context)

        # Se confidence é alta, retornar resultado heurístico
        if heuristic_result.confidence >= self.ai_threshold:
            heuristic_result.method_used = "hybrid_heuristic"
            return heuristic_result

        # Confidence baixa: validar com IA
        try:
            ai_result = await self.ai_strategy.analyze(text, context)

            # Combinar resultados
            return self._combine_results(heuristic_result, ai_result, text)

        except Exception as e:
            logger.warning(f"AI validation failed, using heuristic result: {e}")
            heuristic_result.method_used = "hybrid_heuristic_fallback"
            return heuristic_result

    def _combine_results(self, heuristic: PoliticalClassification,
                        ai: PoliticalClassification, text: str) -> PoliticalClassification:
        """Combina resultados heurístico e IA"""
        # Se categorias concordam, usar IA result com confidence maior
        if heuristic.category == ai.category:
            combined_confidence = min(0.95, (heuristic.confidence + ai.confidence) / 2 + 0.1)
            combined_keywords = list(set(heuristic.keywords_found + ai.keywords_found))

            return PoliticalClassification(
                category=ai.category,
                confidence=combined_confidence,
                keywords_found=combined_keywords,
                reasoning=f"Concordância heurística-IA: {ai.reasoning}",
                method_used="hybrid_agreement",
                metadata={
                    "heuristic_result": heuristic,
                    "ai_result": ai
                }
            )

        # Categorias divergem: usar confidence para decidir
        if ai.confidence > heuristic.confidence:
            ai.method_used = "hybrid_ai_preferred"
            ai.metadata = ai.metadata or {}
            ai.metadata["heuristic_result"] = heuristic
            return ai
        else:
            heuristic.method_used = "hybrid_heuristic_preferred"
            heuristic.metadata = heuristic.metadata or {}
            heuristic.metadata["ai_result"] = ai
            return heuristic

    def get_strategy_name(self) -> str:
        return "hybrid"


class PoliticalAnalysisService:
    """
    Serviço principal de análise política.

    Responsabilidades:
    - Coordenação de estratégias de análise
    - Caching de resultados
    - Monitoramento de performance
    - Fallback robusto
    """

    def __init__(self, lexicon: PoliticalLexicon, ai_client: Optional[AnalysisClient] = None,
                 strategy: Optional[str] = None):
        self.lexicon = lexicon
        self.ai_client = ai_client

        # Configurar estratégias disponíveis
        self.strategies = {
            'heuristic': HeuristicPoliticalStrategy(lexicon),
            'ai': AIPoliticalStrategy(ai_client) if ai_client else None,
            'hybrid': None  # Será criado quando necessário
        }

        # Estratégia padrão
        self.default_strategy = strategy or self._determine_default_strategy()

    def _determine_default_strategy(self) -> str:
        """Determina estratégia padrão baseada na disponibilidade"""
        if self.ai_client and self.ai_client.is_available():
            return 'hybrid'
        else:
            return 'heuristic'

    async def analyze_text(self, text: str, strategy: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> PoliticalClassification:
        """
        Analisa texto político usando estratégia especificada.

        Args:
            text: Texto para análise
            strategy: Estratégia específica ('heuristic', 'ai', 'hybrid') ou None para padrão
            context: Contexto adicional para análise

        Returns:
            Classificação política
        """
        if not text or not text.strip():
            return PoliticalClassification(
                category=PoliticalCategory.CENTRO,
                confidence=0.0,
                keywords_found=[],
                reasoning="Texto vazio",
                method_used="empty_text"
            )

        strategy_name = strategy or self.default_strategy
        analysis_strategy = self._get_strategy(strategy_name)

        try:
            result = await analysis_strategy.analyze(text, context)

            # Adicionar metadados de serviço
            result.metadata = result.metadata or {}
            result.metadata.update({
                'service_version': '1.0',
                'text_length': len(text),
                'strategy_used': strategy_name
            })

            return result

        except Exception as e:
            logger.error(f"Political analysis failed with strategy {strategy_name}: {e}")

            # Fallback para heurística se não foi a estratégia usada
            if strategy_name != 'heuristic':
                logger.info("Falling back to heuristic strategy")
                fallback_strategy = self.strategies['heuristic']
                result = await fallback_strategy.analyze(text, context)
                result.method_used = f"{strategy_name}_fallback_heuristic"
                return result

            raise

    def _get_strategy(self, strategy_name: str) -> PoliticalAnalysisStrategy:
        """Obtém estratégia de análise"""
        if strategy_name == 'hybrid' and self.strategies['hybrid'] is None:
            # Criar estratégia híbrida sob demanda
            if self.strategies['ai'] is None:
                raise ValueError("AI client not available for hybrid strategy")

            self.strategies['hybrid'] = HybridPoliticalStrategy(
                self.strategies['heuristic'],
                self.strategies['ai']
            )

        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy not available: {strategy_name}")

        return strategy

    def get_available_strategies(self) -> List[str]:
        """Lista estratégias disponíveis"""
        available = ['heuristic']

        if self.ai_client and self.ai_client.is_available():
            available.extend(['ai', 'hybrid'])

        return available

    def get_service_info(self) -> Dict[str, Any]:
        """Informações sobre o serviço"""
        return {
            'available_strategies': self.get_available_strategies(),
            'default_strategy': self.default_strategy,
            'ai_client_available': self.ai_client is not None and self.ai_client.is_available(),
            'lexicon_categories': len(PoliticalCategory),
            'total_keywords': sum(len(self.lexicon.get_keywords_by_category(cat))
                                for cat in PoliticalCategory)
        }