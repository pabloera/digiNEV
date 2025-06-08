# 🔤 Guia de Instalação e Uso do Spacy pt-bt

## 📦 1. Instalação

### Instalar Spacy
```bash
# Instalar biblioteca Spacy
pip install spacy

# Baixar modelo português brasileiro (OBRIGATÓRIO)
python -m spacy download pt_core_news_lg

# Alternativa: modelo menor (fallback)
python -m spacy download pt_core_news_sm
```

### Verificar Instalação
```bash
# Testar se modelo foi instalado corretamente
python -c "import spacy; nlp = spacy.load('pt_core_news_lg'); print('✅ Spacy pt-bt instalado com sucesso!')"
```

## ⚙️ 2. Configuração no Projeto

### Atualizar pyproject.toml
```toml
# Adicionar ao dependencies
dependencies = [
    # ... dependências existentes ...
    "spacy>=3.6.1,<4.0.0",
]
```

### Atualizar processing.yaml
```yaml
# Configurar no config/processing.yaml
nlp:
  spacy_model: "pt_core_news_lg"
  batch_size: 100
  entity_recognition: true
  lemmatization: true
```

## 🚀 3. Execução

### Pipeline Completo
```bash
# Executar pipeline com Spacy (automaticamente inclui etapa 06b)
python run_pipeline.py
```

### Verificar Logs
```bash
# Verificar se Spacy está sendo usado
tail -f logs/pipeline.log | grep "🔤\|spacy"
```

## 📊 4. Features Adicionadas pelo Spacy

### Colunas Linguísticas (prefixo 'spacy_')
- `spacy_tokens_count` - Número de tokens
- `spacy_lemmas` - Palavras lematizadas
- `spacy_pos_tags` - Tags morfológicas
- `spacy_named_entities` - Entidades nomeadas
- `spacy_political_entities_found` - Entidades políticas
- `spacy_linguistic_complexity` - Score de complexidade
- `spacy_hashtag_segments` - Hashtags segmentadas
- `spacy_lexical_diversity` - Diversidade lexical (TTR)

### Colunas Agregadas
- `tokens_category` - Categoria por comprimento
- `complexity_category` - Categoria por complexidade 
- `lexical_richness` - Riqueza lexical
- `political_entity_density` - Densidade de entidades políticas

## 🔍 5. Análises Disponíveis

### Complexidade Linguística
```python
# Textos complexos (score > 0.7)
complex_messages = df[df['spacy_linguistic_complexity'] > 0.7]
```

### Entidades Políticas
```python
# Mensagens com alta densidade política
political_messages = df[df['political_entity_density'] > 0.5]
```

### Diversidade Lexical
```python
# Mensagens com alta diversidade vocabular
diverse_messages = df[df['spacy_lexical_diversity'] > 0.7]
```

## 🎯 6. Integração com Outras Etapas

### Sentiment Analysis (Etapa 07)
- **Input aprimorado**: Lemmas limpos para análise de sentimento
- **Feature adicional**: Intensidade linguística

### Topic Modeling (Etapa 08)
- **Input otimizado**: Lemmas sem stopwords
- **Qualidade superior**: Tópicos mais coerentes

### Political Analysis (Etapa 05)
- **NER político**: Detecção específica de entidades brasileiras
- **Confiança aumentada**: Score aprimorado com densidade política

### Hashtag Analysis (Etapa 11)
- **Segmentação**: #ForaBolsonaro → "Fora Bolsonaro"
- **Normalização**: Variações hashtag agrupadas

## ⚠️ 7. Troubleshooting

### Modelo não encontrado
```bash
# Erro: Can't find model 'pt_core_news_lg'
python -m spacy download pt_core_news_lg

# Verificar modelos instalados
python -m spacy info
```

### Performance Lenta
```yaml
# Reduzir batch_size no config
nlp:
  batch_size: 50  # Ao invés de 100
  dependency_parsing: false  # Desabilitar para performance
```

### Erro de Memória
```yaml
# Configurar limites
nlp:
  limits:
    max_text_length: 2000  # Reduzir de 5000
    memory_limit_mb: 512   # Reduzir de 1024
```

### Fallback Automático
- Sistema detecta se Spacy não está disponível
- Etapa 06b é pulada automaticamente  
- Pipeline continua sem interrupção
- Features Spacy ficam vazias/zero

## 📈 8. Benefícios Esperados

### Análise Política
- **+30%** precisão na detecção de entidades políticas
- **Contexto linguístico** aprimorado
- **Reconhecimento** de nomes políticos brasileiros

### Processamento de Texto
- **Lemmatização** profissional do português
- **Normalização** superior de variações
- **Qualidade** de features linguísticas

### Integração com Voyage.AI
- **Preprocessamento otimizado** para embeddings
- **Texto limpo** para análise semântica
- **Redução de ruído** nos embeddings

### Dashboard
- **Visualizações** de complexidade linguística
- **Métricas** de entidades políticas
- **Análises** de diversidade lexical

## 🏆 9. Exemplo de Uso Completo

```python
# Exemplo de análise após processamento Spacy
import pandas as pd

# Carregar dados processados
df = pd.read_csv('data/dashboard_results/dataset_06b_linguistically_processed.csv', sep=';')

# Análise de complexidade política
political_complex = df[
    (df['political_entity_density'] > 0.3) & 
    (df['spacy_linguistic_complexity'] > 0.6)
]

print(f"Mensagens politicamente complexas: {len(political_complex)}")

# Top entidades políticas
import json
all_entities = []
for entities_json in df['spacy_political_entities_found']:
    entities = json.loads(entities_json) if entities_json else []
    all_entities.extend([ent[0] for ent in entities])

from collections import Counter
top_entities = Counter(all_entities).most_common(10)
print("Top entidades políticas:", top_entities)
```

---

**Status**: ✅ Pronto para implementação
**Compatibilidade**: Pipeline v4.6+ 
**Dependências**: spacy>=3.6.1, pt_core_news_lg