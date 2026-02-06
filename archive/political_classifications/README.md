# Classificacoes Politicas Preservadas - digiNEV

**Data de Preservacao**: 2025-10-03
**Origem**: Sistemas UnifiedAnthropicPipeline e RefactoredPipeline (removidos)
**Motivo**: Consolidacao para pipeline Analyzer (src/analyzer.py)
**Auditoria**: 2026-02-06

## Status de Integracao ao Pipeline Ativo

| Arquivo | Status | Notas |
|---------|--------|-------|
| `lexico_unified_system.json` | INTEGRADO | Copiado para `src/core/`. Recurso principal: 847 termos, 9 macrotemas, 41 subtemas |
| `political_keywords_dict.py` | INTEGRADO | Copiado para `src/core/`. 10 categorias tematicas, ~110 termos |
| `lexico_politico_hierarquizado.json` | NAO INTEGRADO | STUB: metadata diz 1.587 termos mas contem apenas ~16. Nao e usado por nenhum codigo |
| `lexicon_loader.py` | SUPERADO | Versao ativa em `src/lexicon_loader.py` com melhorias (auto-detect path, cache, expressoes) |
| `political_analyzer_unified.py` | NAO INTEGRADO | Depende de `core.base_client` (inexistente). Pipeline usa Stage 06 + Stage 08 |
| `political_analyzer_refactored.py` | NAO INTEGRADO | Depende de `core.base_client` (inexistente). Design SOLID preservado como referencia |
| `political_classification.md` | REFERENCIA | Lexico YAML em markdown. Termos consolidados no unified JSON |
| `refactoring_lexicon_task.md` | OBSOLETO | Paths referenciados nao existem mais (config/, batch_analyzer/) |

## Lexicons Politicos

### lexico_unified_system.json (RECURSO PRINCIPAL)
- 847 termos em 9 macrotemas: identidade_patriotica, inimigos_ideologicos, teorias_conspiracao, negacionismo, autoritarismo_violencia, mobilizacao_acao, desinformacao_verdade, estrategias_discursivas, eventos_simbolicos
- 41 subtemas com `palavras` e `expressoes`
- Escopo: discurso bolsonarista/direita brasileira (2019-2023)

### lexico_politico_hierarquizado.json (STUB - NAO USAR)
- Metadata afirma 1.587 palavras (v5.0.0)
- Conteudo real: apenas 2 macrotemas (extrema_direita, direita_tradicional) com ~16 termos
- Provavel truncamento em operacao anterior
- Taxonomia por espectro (diferente do unified que usa temas)

### political_keywords_dict.py
- 10 categorias tematicas: autoritarismo, pandemia, violencia, religiao, inimigos, identidade, meio_ambiente, moralidade, antissistema, polarizacao
- Numeracao original com gaps (cat0, cat2-4, cat6 duplicado, cat7-10)
- Versao ativa em `src/core/` corrigida: cat5_inimigos (antes cat6 duplicado)

## Analisadores Politicos (NAO INTEGRADOS)

### political_analyzer_unified.py (73K)
- Classificador API Anthropic com 4 niveis hierarquicos
- PoliticalLevel: politico/nao-politico
- PoliticalAlignment: bolsonarista/antibolsonarista/neutro/indefinido
- Batch processing (100 records, 5 concurrent)
- Requer: AnthropicBase, AnalysisClient, AnalysisType (classes inexistentes no pipeline atual)

### political_analyzer_refactored.py (19K)
- Strategy pattern (SOLID): Heuristic, AI, Hybrid strategies
- PoliticalCategory enum: EXTREMA_DIREITA, DIREITA, CENTRO_DIREITA, CENTRO, CENTRO_ESQUERDA, ESQUERDA
- Requer: core.base_client (inexistente)

## Orientacoes Politicas no Pipeline

O pipeline ativo (Stage 08) classifica:
- **extrema-direita**: radical_score >= 2 ou (conspiracao >= 2 e adversario >= 1)
- **direita**: adversario >= 2 ou conspiracao >= 2
- **centro-direita**: identidade_score >= 2
- **neutral**: sem matches significativos

**Nota**: Lexico unificado so contem termos de direita/extrema-direita. Classificacoes 'esquerda', 'centro-esquerda' e 'centro' NAO sao retornadas (fora do escopo do projeto: analise de bolsonarismo).

## Arquivos Preservados Para

1. Historico metodologico
2. Comparacoes futuras
3. Documentacao da evolucao das classificacoes
4. Auditorias academicas
5. Referencia de design patterns (Strategy, Repository, SOLID)
