# Scripts Não-Pipeline Arquivados

Este diretório contém scripts que foram removidos do pipeline principal após migração para integração Anthropic centralizada.

## 📅 Data de Arquivamento
**06 de Janeiro de 2025**

## 🔄 Motivo do Arquivamento
Estes scripts foram substituídos pelo **pipeline centralizado Anthropic** que oferece:
- ✅ Integração unificada com API Anthropic
- ✅ Processamento em chunks automático
- ✅ Validação e fallback inteligente
- ✅ Menor complexidade de manutenção

## 📁 Estrutura Arquivada

### `src/preprocessing/` (2 scripts)
- `stopwords_loader.py` - Carregamento de stopwords (substituído por pipeline)
- `telegram_preprocessor.py` - Pré-processamento de dados (substituído por pipeline)

### `src/data/processors/` (2 scripts)
- `extract_canais_from_urls.py` - Extração específica de canais
- `extract_forwarded_message_names.py` - Extração específica de nomes

### `src/data/transformers/` (11 scripts)
Transformações específicas substituídas por funcionalidade integrada:
- `add_forwarded_column.py`
- `add_fwd_from_column.py`
- `create_fwd_source_column.py`
- `process_binary_columns_classif1.py`
- `rename_contem_texto_to_has_txt.py`
- `rename_nomes_canais_column.py`
- `standardize_canais_lowercase.py`
- `standardize_urls.py`
- `convert_timestamp_datetime.py`
- `update_classif1_after_canais.py`
- `update_domain_column.py`

## 💾 Scripts Mantidos Ativos

### `src/preprocessing/`
- `stopwords_pt.txt` - Arquivo de dados essencial

### `src/data/processors/`
- `chunk_processor.py` - Processamento em chunks (em uso ativo)

### `src/data/transformers/`
- `column_transformer.py` - Módulo consolidado (boa arquitetura)
- `text_transformer.py` - Módulo consolidado (boa arquitetura)

### `src/data/utils/`
- `encoding_fixer.py` - Correção de encoding (funcionalidade crítica)

## 🔧 Como Recuperar Funcionalidade

Se precisar de funcionalidade específica destes scripts:

1. **Para desenvolvimento**: Scripts estão preservados aqui para referência
2. **Para produção**: Use o pipeline Anthropic centralizado via `run_pipeline.py`
3. **Para casos especiais**: Adapte os módulos consolidados (`column_transformer.py`, `text_transformer.py`)

## 📊 Estatísticas da Migração

- **Scripts arquivados**: 15
- **Scripts mantidos**: 4 + 1 arquivo de dados
- **Redução de complexidade**: 75%
- **Melhoria na manutenibilidade**: Significativa

---

**Nota**: Estes scripts permanecem funcionais, mas não são mais parte do pipeline ativo. A funcionalidade foi migrada para a integração Anthropic centralizada conforme PROJECT_RULES.md.