#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_17.py
v6.1: Heurística + API híbrida para análise de canais.

NOTA: Este módulo é uma referência modular. O analyzer.py é o source of truth.
A lógica completa com API está em analyzer.py::_stage_17_channel_analysis.

Fluxo:
  Fase 1: Classificação por nome do canal (keyword matching, 100% msgs)
           - channel_type: news | political | entertainment | religious | general
           - channel_activity, is_active_channel
           - Análise de mídia, forwarding, influência
  Fase 2: API classifica canais 'general' usando amostra de conteúdo
           → envia nome + 5 msgs representativas de até 20 canais não-classificados
           Categorias: news, political, entertainment, religious, conspiracy, military, activism
  Fallback: sem API key → 100% heurística (keyword matching)

Colunas geradas:
  - channel_type: str (tipo do canal)
  - channel_activity: int (contagem de msgs no canal)
  - is_active_channel: bool
  - content_type: str (tipo de mídia)
  - has_media: bool
  - is_forwarded: bool
  - forwarding_context: float (ratio de forwarding)
  - sender_channel_influence: int
  - channel_confidence: 0.0-1.0 (NOVA - confiança na classificação do canal)
  - channel_theme: str (NOVA - tema principal do canal via API)

API: Stage 17 usa API Anthropic (claude-sonnet-4) para classificar canais
     com tipo 'general' baseado em amostra de conteúdo. Envia nome do canal
     + 5 mensagens representativas. Detecta conspiracy, military, activism.
     Resultado: ~100% dos "general" reclassificados com tipo e tema.
"""

# Este módulo é referencial. A implementação completa está em analyzer.py.
# Para usar standalone, importe diretamente do Analyzer:
#   from src.analyzer import Analyzer
#   analyzer = Analyzer()
#   result = analyzer.analyze(df)
