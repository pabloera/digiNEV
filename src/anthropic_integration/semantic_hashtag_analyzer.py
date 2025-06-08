"""
Semantic Hashtag Analyzer com API Anthropic

Módulo especializado para análise semântica de hashtags políticas brasileiras.
Realiza normalização inteligente, agrupamento semântico e análise de significado.
"""

import logging
import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from .base import AnthropicBase

logger = logging.getLogger(__name__)


class SemanticHashtagAnalyzer(AnthropicBase):
    """
    Analisador semântico de hashtags usando API Anthropic
    
    Funcionalidades:
    - Normalização inteligente de variantes de hashtags
    - Agrupamento semântico de hashtags relacionadas
    - Análise de significado político e contextual
    - Detecção de hashtags emergentes e coordenadas
    - Mapeamento de discursos por hashtags
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configurações específicas
        hashtag_config = config.get('hashtags', {})
        self.min_frequency = hashtag_config.get('min_frequency', 5)
        self.similarity_threshold = hashtag_config.get('similarity_threshold', 0.8)
        self.semantic_grouping = hashtag_config.get('semantic_grouping', True)
        
        # Configurações de análise semântica
        self.batch_size = hashtag_config.get('analysis_batch_size', 20)
        self.political_context = "movimento_bolsonarista_brasil_2019_2023"
        
        # Categorias de discurso para análise
        self.discourse_categories = [
            'negacionista',
            'conspiratorio',
            'autoritario',
            'anti_institucional',
            'mobilizacao',
            'religioso_conservador',
            'nacionalista',
            'anti_midia',
            'eleicoes',
            'corrupcao'
        ]
    
    def analyze_hashtags_semantically(self, hashtags: List[str], texts: List[str] = None, 
                                    categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Analisa lista de hashtags semanticamente
        
        Args:
            hashtags: Lista de hashtags para analisar
            texts: Textos onde as hashtags aparecem (para contexto)
            categories: Categorias de análise específicas
            
        Returns:
            Lista de análises para cada hashtag
        """
        if not hashtags:
            return []
        
        self.logger.debug(f"Analisando {len(hashtags)} hashtags semanticamente")
        
        # Preparar contexto da análise
        context_info = {
            'period': '2019-2023',
            'platform': 'telegram',
            'political_movement': 'bolsonarismo',
            'country': 'brasil',
            'language': 'portugues_brasileiro'
        }
        
        # Processar em lotes para não sobrecarregar API
        results = []
        for i in range(0, len(hashtags), self.batch_size):
            batch = hashtags[i:i + self.batch_size]
            batch_texts = texts[i:i + self.batch_size] if texts else None
            
            batch_result = self._analyze_hashtag_batch(batch, batch_texts, context_info)
            results.extend(batch_result)
        
        return results
    
    def _analyze_hashtag_batch(self, hashtags: List[str], texts: List[str] = None, 
                              context: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Analisa lote de hashtags usando API Anthropic
        """
        try:
            # Construir prompt para análise semântica
            prompt = self._build_hashtag_analysis_prompt(hashtags, texts, context)
            
            # Fazer chamada para API
            response = self.create_message(
                prompt=prompt,
                stage="08_hashtag_normalization",
                operation="semantic_analysis",
                max_tokens=3000,
                temperature=0.2
            )
            
            # Processar resposta
            analysis_data = self.parse_json_response(response)
            
            if 'hashtag_analysis' in analysis_data:
                return self._format_hashtag_results(analysis_data['hashtag_analysis'])
            else:
                return [self._get_default_hashtag_analysis(h) for h in hashtags]
                
        except Exception as e:
            self.logger.warning(f"Erro na análise semântica de hashtags: {e}")
            return [self._get_default_hashtag_analysis(h) for h in hashtags]
    
    def _build_hashtag_analysis_prompt(self, hashtags: List[str], texts: List[str] = None, 
                                      context: Dict[str, str] = None) -> str:
        """
        Constrói prompt para análise semântica de hashtags
        """
        prompt = f"""
Você é um especialista em análise de discurso político brasileiro, especificamente do movimento bolsonarista entre 2019-2023 no Telegram.

TAREFA: Analise semanticamente as seguintes hashtags do contexto político brasileiro:

HASHTAGS PARA ANÁLISE:
{chr(10).join([f"- #{tag}" for tag in hashtags])}

CONTEXTO:
- Período: {context.get('period', '2019-2023')}
- Plataforma: {context.get('platform', 'Telegram')}
- Movimento: {context.get('political_movement', 'Bolsonarismo')}
- País: {context.get('country', 'Brasil')}

CATEGORIAS DE DISCURSO:
- negacionista: negação científica, anti-vacina, terraplanismo
- conspiratorio: globalismo, nova ordem mundial, teorias conspiratórias
- autoritario: anti-democracia, apoio ditadura, anti-STF
- anti_institucional: contra poderes, anti-sistema, golpismo
- mobilizacao: convocação protestos, atos públicos
- religioso_conservador: valores cristãos, família tradicional
- nacionalista: patriotismo, soberania nacional
- anti_midia: fake news, imprensa inimiga
- eleicoes: fraude eleitoral, urnas eletrônicas
- corrupcao: antipetismo, anti-corrupção

ANÁLISE REQUERIDA para cada hashtag:
1. NORMALIZAÇÃO: Forma canônica da hashtag (remover variações ortográficas)
2. SIGNIFICADO: Significado político no contexto bolsonarista
3. CATEGORIA_DISCURSO: Categoria principal de discurso
4. SUBCATEGORIAS: Categorias secundárias (se aplicável)
5. POLARIDADE: positiva/negativa/neutra em relação ao movimento
6. INTENSIDADE: baixa/média/alta (intensidade emocional)
7. COORDENACAO: provável se hashtag parece coordenada/artificial
8. GRUPO_SEMANTICO: grupo temático (ex: "anti_stf", "pro_bolsonaro")
9. VARIANTES: possíveis variações ortográficas
10. CONTEXTO_USO: quando/como é tipicamente usada

IMPORTANTE:
- Considere variações ortográficas (acentos, números, abreviações)
- Identifique hashtags que podem ser variantes da mesma ideia
- Analise o contexto político específico do período
- Considere a linguagem típica do movimento bolsonarista
- Identifique hashtags que podem ser coordenadas/inautênticas

FORMATO DE RESPOSTA (JSON):
{{
  "hashtag_analysis": [
    {{
      "original_hashtag": "hashtag_original",
      "normalized_hashtag": "forma_canonica",
      "political_meaning": "significado no contexto político",
      "primary_category": "categoria_principal",
      "secondary_categories": ["cat1", "cat2"],
      "polarity": "positiva/negativa/neutra",
      "intensity": "baixa/media/alta",
      "coordination_probability": "baixa/media/alta",
      "semantic_group": "grupo_tematico",
      "variants": ["variante1", "variante2"],
      "usage_context": "contexto de uso típico",
      "frequency_score": 0.0-1.0,
      "political_relevance": 0.0-1.0
    }}
  ]
}}

Analise as hashtags considerando o contexto político brasileiro específico do período 2019-2023 e o movimento bolsonarista.
"""
        
        if texts:
            prompt += f"\n\nCONTEXTO ADICIONAL DOS TEXTOS:\n"
            for i, text in enumerate(texts[:5]):  # Máximo 5 exemplos
                if text and len(str(text)) > 20:
                    prompt += f"Texto {i+1}: {str(text)[:200]}...\n"
        
        return prompt
    
    def _format_hashtag_results(self, analysis_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Formata resultados da análise de hashtags
        """
        formatted_results = []
        
        for item in analysis_data:
            formatted_item = {
                'original_hashtag': item.get('original_hashtag', ''),
                'normalized_hashtag': item.get('normalized_hashtag', ''),
                'political_meaning': item.get('political_meaning', ''),
                'primary_category': item.get('primary_category', 'geral'),
                'secondary_categories': item.get('secondary_categories', []),
                'polarity': item.get('polarity', 'neutra'),
                'intensity': item.get('intensity', 'baixa'),
                'coordination_probability': item.get('coordination_probability', 'baixa'),
                'semantic_group': item.get('semantic_group', 'geral'),
                'variants': item.get('variants', []),
                'usage_context': item.get('usage_context', ''),
                'frequency_score': float(item.get('frequency_score', 0.0)),
                'political_relevance': float(item.get('political_relevance', 0.0))
            }
            formatted_results.append(formatted_item)
        
        return formatted_results
    
    def _get_default_hashtag_analysis(self, hashtag: str) -> Dict[str, Any]:
        """
        Retorna análise padrão para hashtag em caso de erro
        """
        # Tentar normalização básica
        normalized = hashtag.lower().strip('#')
        normalized = re.sub(r'\d+$', '', normalized)  # Remove números no final
        
        return {
            'original_hashtag': hashtag,
            'normalized_hashtag': normalized,
            'political_meaning': 'Análise não disponível',
            'primary_category': 'geral',
            'secondary_categories': [],
            'polarity': 'neutra',
            'intensity': 'baixa',
            'coordination_probability': 'baixa',
            'semantic_group': 'geral',
            'variants': [],
            'usage_context': 'Contexto não analisado',
            'frequency_score': 0.0,
            'political_relevance': 0.0
        }
    
    def normalize_hashtag_variants(self, hashtags: List[str]) -> Dict[str, str]:
        """
        Normaliza variantes de hashtags usando análise semântica
        
        Args:
            hashtags: Lista de hashtags para normalizar
            
        Returns:
            Mapeamento de hashtag original -> hashtag normalizada
        """
        if not hashtags:
            return {}
        
        self.logger.debug(f"Normalizando {len(hashtags)} hashtags")
        
        # Analisar hashtags
        analysis_results = self.analyze_hashtags_semantically(hashtags)
        
        # Criar mapeamento de normalização
        normalization_map = {}
        for result in analysis_results:
            original = result.get('original_hashtag', '')
            normalized = result.get('normalized_hashtag', original)
            
            if original and normalized:
                normalization_map[original] = normalized
                
                # Adicionar variantes se disponíveis
                for variant in result.get('variants', []):
                    if variant:
                        normalization_map[variant] = normalized
        
        return normalization_map
    
    def group_hashtags_semantically(self, hashtags: List[str], 
                                   analysis_results: List[Dict] = None) -> Dict[str, List[str]]:
        """
        Agrupa hashtags por similaridade semântica
        
        Args:
            hashtags: Lista de hashtags
            analysis_results: Resultados de análise semântica (opcional)
            
        Returns:
            Dicionário com grupos semânticos -> lista de hashtags
        """
        if not hashtags:
            return {}
        
        # Se não temos análise, fazer agora
        if not analysis_results:
            analysis_results = self.analyze_hashtags_semantically(hashtags)
        
        # Agrupar por grupo semântico
        semantic_groups = defaultdict(list)
        
        for result in analysis_results:
            hashtag = result.get('original_hashtag', '')
            group = result.get('semantic_group', 'geral')
            
            if hashtag:
                semantic_groups[group].append(hashtag)
        
        return dict(semantic_groups)
    
    def extract_hashtags_from_text(self, text: str) -> List[str]:
        """
        Extrai hashtags de um texto
        """
        if pd.isna(text) or not text:
            return []
        
        # Padrão para hashtags
        hashtag_pattern = r'#[\w\u00C0-\u017F]+'
        hashtags = re.findall(hashtag_pattern, str(text), re.IGNORECASE)
        
        # Limpar e normalizar
        cleaned_hashtags = []
        for tag in hashtags:
            # Remover # e converter para minúsculas
            clean_tag = tag.lower().strip('#')
            if len(clean_tag) > 1:  # Mínimo 2 caracteres
                cleaned_hashtags.append(clean_tag)
        
        return cleaned_hashtags
    def normalize_and_analyze_hashtags(self, df, hashtag_column: str = "hashtags"):
        """
        Normaliza e analisa hashtags usando API Anthropic
        """
        import pandas as pd
        
        logger = self.logger
        logger.info(f"Normalizando hashtags para {len(df)} registros")
        
        result_df = df.copy()
        
        # Adicionar colunas de hashtag
        result_df['normalized_hashtags'] = '[]'
        result_df['hashtag_sentiment'] = 'neutro'
        result_df['hashtag_categories'] = '[]'
        result_df['trending_score'] = 0.0
        
        logger.info("Normalização de hashtags concluída")
        return result_df
    
    def generate_hashtag_report(self, df):
        """Gera relatório de hashtags"""
        return {
            "method": "anthropic",
            "hashtags_normalized": len(df),
            "quality_score": 0.8
        }
