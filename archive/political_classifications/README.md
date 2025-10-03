# Classificações Políticas Preservadas - digiNEV v.final

**Data de Preservação**: 2025-10-03
**Origem**: Sistemas UnifiedAnthropicPipeline e RefactoredPipeline (removidos)
**Motivo**: Consolidação para ScientificAnalyzer v.final

## Arquivos Preservados

### 1. Lexicons Políticos
- `lexico_politico_hierarquizado.json` - Lexicon principal (1587 palavras)
- `lexico_unified_system.json` - Cópia do sistema unified
- `political_keywords_dict.py` - Dicionário de categorias políticas

### 2. Analisadores Políticos
- `political_analyzer_unified.py` - Módulo do UnifiedAnthropicPipeline
- `political_analyzer_refactored.py` - Módulo do RefactoredPipeline
- `lexicon_loader.py` - Carregador de lexicons

### 3. Documentação Metodológica
- `political_classification.md` - Metodologia de classificação
- `refactoring_lexicon_task.md` - Documentação de refatoração

## Categorias Políticas Identificadas

### Spectrum Principal:
- extrema-direita
- direita
- centro-direita
- centro
- centro-esquerda
- esquerda

### Categorias Temáticas:
- cat0_autoritarismo_regime
- cat2_pandemia_covid
- cat3_violencia_seguranca
- cat4_religiao_moral
- cat6_inimigos_ideologicos
- cat6_identidade_politica

## Uso no ScientificAnalyzer v.final

O sistema ScientificAnalyzer v.final utiliza:
- `src/pipeline_stages/stage_05_political_analysis.py`
- Referências aos lexicons preservados nesta pasta

## ⚠️ IMPORTANTE

Estes arquivos foram preservados para:
1. Manter histórico metodológico
2. Permitir comparações futuras
3. Documentar evolução das classificações
4. Facilitar auditorias acadêmicas

**NÃO MODIFICAR** - Arquivos históricos preservados.