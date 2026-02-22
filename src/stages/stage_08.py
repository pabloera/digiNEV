#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_08.py
v6.1: Heurística + API híbrida para classificação política.

NOTA: Este módulo é uma referência modular. O analyzer.py é o source of truth.
A lógica completa com API está em analyzer.py::_stage_08_political_classification.

Fluxo:
  Fase 1: Token matching via spaCy lemmas + léxico unificado (914+ termos, 11 macrotemas)
  Fase 2: API para mensagens "neutral" com political_confidence < 0.4
  Fase 3: TCW (Tabela-Categoria-Palavra) — sempre heurístico
  Fallback: sem API key → 100% heurística

Colunas geradas:
  - political_orientation: extrema-direita | direita | centro-direita | neutral
  - political_keywords: lista de termos políticos encontrados
  - political_intensity: 0.0-1.0
  - political_confidence: 0.0-1.0 (NOVA - confiança da classificação)
  - cat_*: 12 categorias temáticas
  - tcw_codes, tcw_categories, tcw_agreement, tcw_code_count: codificação TCW

API: Stage 08 usa API Anthropic (claude-sonnet-4) para reclassificar mensagens
     classificadas como "neutral" pela heurística quando confidence < 0.4.
     Resultado: ~30-40% dos "neutral" são reclassificados com orientação política.
"""

# Este módulo é referencial. A implementação completa está em analyzer.py.
# Para usar standalone, importe diretamente do Analyzer:
#   from src.analyzer import Analyzer
#   analyzer = Analyzer()
#   result = analyzer.analyze(df)
