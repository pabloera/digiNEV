#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_16.py
v6.1: Heurística + API híbrida para análise de contexto de eventos.

NOTA: Este módulo é uma referência modular. O analyzer.py é o source of truth.
A lógica completa com API está em analyzer.py::_stage_16_event_context.

Fluxo:
  Fase 1: Heurística detecta contextos políticos, frames Entman, menções (100% msgs)
           - political_context: government | electoral | judicial | economic | ...
           - Frames: conflito, responsabilização, moralista, econômico (Entman 1993)
           - Menções: governo, oposição
  Fase 2: API para mensagens com contexto 'general' e event_confidence < 0.5
           → detecta referências indiretas a eventos específicos brasileiros
  Fallback: sem API key → 100% heurística

Colunas geradas:
  - political_context: str (tipo de contexto político)
  - mentions_government: bool
  - mentions_opposition: bool
  - election_context: bool
  - protest_context: bool
  - frame_conflito: 0.0-1.0
  - frame_responsabilizacao: 0.0-1.0
  - frame_moralista: 0.0-1.0
  - frame_economico: 0.0-1.0
  - is_weekend: bool
  - is_business_hours: bool
  - event_confidence: 0.0-1.0 (NOVA - confiança na classificação de contexto)
  - specific_event: str (NOVA - evento específico detectado via API)

API: Stage 16 usa API Anthropic (claude-sonnet-4) para detectar referências
     indiretas a eventos específicos como 8 de janeiro, CPI da COVID,
     eleição 2022, STF inquéritos, impeachment etc.
     Resultado: ~15-25% dos "general" reclassificados com evento específico.
"""

# Este módulo é referencial. A implementação completa está em analyzer.py.
# Para usar standalone, importe diretamente do Analyzer:
#   from src.analyzer import Analyzer
#   analyzer = Analyzer()
#   result = analyzer.analyze(df)
