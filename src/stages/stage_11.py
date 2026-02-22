#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_11.py
v6.1: Heurística + API híbrida para topic modeling.

NOTA: Este módulo é uma referência modular. O analyzer.py é o source of truth.
A lógica completa com API está em analyzer.py::_stage_11_topic_modeling.

Fluxo:
  Fase 1: LDA descobre clusters de tópicos com CountVectorizer (100% heurístico)
           - n_topics = min(5, n_docs // 20 + 1)
           - Stopwords PT customizadas (95+ termos)
           - Top 8 palavras-chave por tópico
  Fase 2: API nomeia os tópicos com rótulos descritivos (1 chamada)
           → envia palavras-chave + 3 msgs representativas por tópico
           API reclassifica mensagens com topic_confidence < 0.4
  Fallback: sem API key → tópicos nomeados por palavras-chave concatenadas

Colunas geradas:
  - dominant_topic: int (índice do tópico dominante)
  - topic_probability: 0.0-1.0 (probabilidade LDA do tópico dominante)
  - topic_keywords: lista de palavras-chave do tópico (top 3)
  - topic_label: str (NOVA - rótulo descritivo do tópico via API)
  - topic_confidence: 0.0-1.0 (NOVA - confiança na atribuição de tópico)

API: Stage 11 usa API Anthropic (claude-sonnet-4) para:
     1. Nomear clusters LDA com rótulos significativos em português
     2. Reclassificar mensagens com topic_confidence < 0.4
     Resultado: tópicos nomeados como "Notícias Políticas Lula" etc.
"""

# Este módulo é referencial. A implementação completa está em analyzer.py.
# Para usar standalone, importe diretamente do Analyzer:
#   from src.analyzer import Analyzer
#   analyzer = Analyzer()
#   result = analyzer.analyze(df)
