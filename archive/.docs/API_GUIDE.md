# Guia de APIs - Batch Analyzer

## 🔌 Integração com APIs de IA

O Batch Analyzer integra com duas APIs principais para análise avançada, mas **funciona completamente sem elas** usando métodos heurísticos.

## Anthropic Claude API

### Configuração
```bash
# No arquivo .env
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

### Uso no Sistema
- **Estágio 5**: Classificação política
- **Estágio 6**: Análise de sentimento
- **Estágio 11**: Análise qualitativa

### Modelos Disponíveis
```yaml
# Em config/default.yaml
anthropic:
  model: claude-3-5-haiku-20241022  # Mais barato ($0.25/1M tokens)
  # Alternativas:
  # model: claude-3-5-sonnet-20241022  # Mais caro ($3/1M tokens)
```

### Otimização de Custos
```python
# Configuração econômica
config = BatchConfig(
    anthropic_model="claude-3-5-haiku-20241022",
    max_tokens=500,  # Limita resposta
    temperature=0.3,  # Mais determinístico
    sampling_rate=0.1  # Processa 10% com API
)
```

## Voyage.ai API

### Configuração
```bash
# No arquivo .env
VOYAGE_API_KEY=pa-YOUR-KEY-HERE
```

### Uso no Sistema
- **Estágio 9**: Embeddings para clustering
- **Estágio 10**: Modelagem de tópicos
- **Estágio 12**: Similaridade semântica

### Modelos Disponíveis
```yaml
voyage:
  model: voyage-3.5-lite  # Mais barato
  # Alternativas:
  # model: voyage-3  # Melhor qualidade
```

## Modo Fallback (Sem APIs)

O sistema funciona **100% sem APIs** usando:

### Classificação Política (Estágio 5)
```python
# Método heurístico baseado em palavras-chave
def classify_political_fallback(text):
    keywords = {
        'extrema-direita': ['bolsonaro', 'patriota', 'armamento'],
        'esquerda': ['lula', 'pt', 'trabalhador'],
        # ...
    }
    # Análise por frequência de termos
```

### Análise de Sentimento (Estágio 6)
```python
# Usando VADER adaptado para português
from sentiment_analyzer import PortugueseSentiment
analyzer = PortugueseSentiment()
score = analyzer.analyze(text)
```

### Embeddings (Estágios 9-10)
```python
# TF-IDF como alternativa a embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
embeddings = vectorizer.fit_transform(texts)
```

## Controle de Custos

### Monitoramento em Tempo Real
```python
# O sistema monitora custos automaticamente
analyzer = IntegratedBatchAnalyzer(config)
analyzer.analyze(df)

# Ver custos
print(f"Custo estimado: ${analyzer.api_stats['estimated_cost']:.2f}")
print(f"Chamadas API: {analyzer.api_stats['api_calls']}")
print(f"Cache hits: {analyzer.api_stats['cache_hits']}")
```

### Orçamento Mensal
```yaml
# config/academic.yaml
academic:
  monthly_budget: 50.0  # USD
  alert_threshold: 0.8  # Alerta em 80% do orçamento
```

### Estratégias de Economia

#### 1. Amostragem Inteligente
```python
config = BatchConfig(
    sampling_strategy="stratified",  # Mantém representatividade
    sampling_rate=0.1  # 10% do dataset
)
```

#### 2. Cache Semântico
```python
config = BatchConfig(
    semantic_cache=True,
    similarity_threshold=0.85  # Textos 85% similares usam cache
)
```

#### 3. Processamento em Lote
```python
config = BatchConfig(
    batch_size=100,  # Agrupa requisições
    wait_between_batches=1.0  # Evita rate limit
)
```

## Comparação de Custos

| Método | Custo/1000 msgs | Qualidade | Velocidade |
|--------|-----------------|-----------|------------|
| Sem APIs | $0.00 | ⭐⭐⭐ | ⚡⚡⚡⚡⚡ |
| Haiku + Amostragem | ~$0.50 | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| Haiku Completo | ~$5.00 | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| Sonnet Completo | ~$30.00 | ⭐⭐⭐⭐⭐ | ⚡⚡ |

## Exemplos de Uso

### Desenvolvimento (Sem APIs)
```bash
python batch_analysis.py --dev-mode
```

### Produção Acadêmica (Otimizado)
```bash
python batch_analysis.py --config config/academic.yaml
```

### Análise Completa (Alto Custo)
```bash
python batch_analysis.py --full-analysis --no-sampling
```

## Resolução de Problemas

### Erro: "Rate limit exceeded"
```python
# Aumentar delay entre requisições
config.wait_between_batches = 2.0
config.rate_limit = 10  # requisições/minuto
```

### Erro: "Invalid API key"
```bash
# Verificar chave
echo $ANTHROPIC_API_KEY

# Testar conexão
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"
```

### Erro: "Budget exceeded"
```python
# Usar modo econômico
config = BatchConfig(
    use_apis=False,  # Desabilita APIs temporariamente
    # ou
    sampling_rate=0.05,  # Reduz para 5%
    model="claude-3-5-haiku-20241022"  # Modelo mais barato
)
```