#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_12.py
v6.1: Heurística + API híbrida para análise semântica e sentimento.

NOTA: Este módulo é uma referência modular. O analyzer.py é o source of truth.
A lógica completa com API está em analyzer.py::_stage_12_semantic_analysis.

Fluxo:
  Fase 1: Sentimento por léxico LIWC-PT + emoção por marcadores textuais (100% msgs)
  Fase 2: API para sentimento ambíguo (sentiment_confidence < 0.5)
           → sentimento contextual + emoções granulares + sarcasmo
  Fallback: sem API key → 100% heurística

Colunas geradas:
  - sentiment_polarity: -1.0 a 1.0
  - sentiment_label: positive | negative | neutral | mixed
  - sentiment_confidence: 0.0-1.0 (NOVA - confiança da classificação)
  - emotion_intensity: 0.0-1.0
  - has_aggressive_language: bool
  - semantic_diversity: 0.0-1.0
  - emotion_anger: 0.0-1.0 (NOVA - emoção granular)
  - emotion_fear: 0.0-1.0 (NOVA)
  - emotion_hope: 0.0-1.0 (NOVA)
  - emotion_disgust: 0.0-1.0 (NOVA)
  - emotion_sarcasm: bool (NOVA - detecção de sarcasmo/ironia)

API: Stage 12 usa API Anthropic (claude-sonnet-4) para reclassificar mensagens
     com sentimento ambíguo (polarity ≈ 0). Detecta sarcasmo, negação e contexto.
     Resultado: ~20-30% dos "neutral" reclassificados + emoções granulares.
"""

# Este módulo é referencial. A implementação completa está em analyzer.py.
# Para usar standalone, importe diretamente do Analyzer:
#   from src.analyzer import Analyzer
#   analyzer = Analyzer()
#   result = analyzer.analyze(df)
