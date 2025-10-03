# 🚀 Como Usar o Batch Híbrido (Métodos Validados + API Anthropic)

## 📋 O que é o Batch Híbrido?

O **`batch_hybrid_anthropic.py`** combina:
1. **Métodos científicos validados** (locais, sem custo)
2. **API Batch Anthropic** (análise avançada com IA, 50% mais barato)

## 🔧 Configuração Inicial

### 1. Configurar API Key

```bash
# Opção 1: Variável de ambiente
export ANTHROPIC_API_KEY="sk-ant-api03-xxxxx"

# Opção 2: Arquivo .env
echo "ANTHROPIC_API_KEY=sk-ant-api03-xxxxx" > .env
```

### 2. Instalar Dependências (Opcional)

Se quiser usar métodos validados locais:

```bash
# Básico (sempre necessário)
pip install pandas numpy requests

# Para métodos validados (opcional)
pip install scikit-learn scipy
pip install spacy && python -m spacy download pt_core_news_lg
pip install bertopic hdbscan sentence-transformers  # Para análise avançada
```

### 3. Copiar Arquivos

```bash
# Copiar o script híbrido
cp batch_hybrid_anthropic.py /seu/projeto/

# Se quiser métodos validados locais:
cp files\ \(1\)/validated_methods_implementation.py /seu/projeto/
```

## 💻 Como Executar

### Uso Básico (API + Métodos Locais)

```bash
python batch_hybrid_anthropic.py data/mensagens.csv
```

### Com Amostra para Teste

```bash
# Testar com 100 mensagens primeiro
python batch_hybrid_anthropic.py data/mensagens.csv --sample 100
```

### Apenas API Anthropic (sem métodos locais)

```bash
python batch_hybrid_anthropic.py data/mensagens.csv --no-validated
```

### Apenas Métodos Locais (sem API)

```bash
python batch_hybrid_anthropic.py data/mensagens.csv --no-api
```

### Escolher Análises Específicas da API

```bash
python batch_hybrid_anthropic.py data/mensagens.csv \
  --analyses political_classification sentiment_advanced
```

## 📊 Tipos de Análise Disponíveis

### Via API Batch Anthropic (Custo)

1. **`political_classification`**: Classificação política (extrema-direita a esquerda)
2. **`sentiment_advanced`**: Sentimento + emoções + tom + intenção
3. **`semantic_interpretation`**: Tópicos + entidades + ideologia + frames
4. **`coordination_detection`**: Detecção de coordenação/bots/spam

### Via Métodos Validados Locais (Grátis)

- **Frame Analysis** (Entman, 1993): Frames de conflito, moralidade, economia
- **LIWC Portuguese** (Balage Filho et al., 2013): Análise psicológica
- **STM** (Roberts et al., 2014): Modelagem de tópicos estrutural
- **Mann-Kendall**: Análise de tendências temporais
- **HDBSCAN**: Clustering avançado

## 📈 Fluxo de Execução

```
1. Carrega dataset CSV
   ↓
2. Aplica métodos validados locais (se habilitado)
   ↓
3. Cria batches para API Anthropic
   ↓
4. Submete batches (até 10.000 msgs/batch)
   ↓
5. Aguarda processamento (até 24h, geralmente minutos)
   ↓
6. Baixa e integra resultados
   ↓
7. Salva CSV com todas as análises
```

## 💰 Custos Estimados (API Anthropic)

| Modelo | Preço Input | Preço Output | Batch (50% desconto) |
|--------|------------|--------------|---------------------|
| Claude 3.5 Haiku | $0.80/M tokens | $4/M tokens | $0.40 e $2/M |
| Claude 3.5 Sonnet | $3/M tokens | $15/M tokens | $1.50 e $7.50/M |

**Exemplo**: 10.000 mensagens ≈ 2M tokens ≈ **$0.80** com Haiku Batch

## 📁 Saída

O script gera um arquivo CSV em `outputs/` com:

### Colunas da API:
- `political_category_api`: Classificação política
- `political_confidence_api`: Confiança (0-1)
- `sentiment_score_api`: Score de sentimento (-1 a 1)
- `emotions_detected_api`: Lista de emoções

### Colunas dos Métodos Validados:
- `frame_conflito`, `frame_moralista`: Scores de frames
- `liwc_affect_positive`, `liwc_power`: Categorias LIWC
- Outras análises científicas

## 🎯 Exemplos de Uso Real

### 1. Análise Rápida de Teste

```bash
# Teste com 50 mensagens, apenas classificação política
python batch_hybrid_anthropic.py data/test.csv \
  --sample 50 \
  --analyses political_classification \
  --no-validated
```

### 2. Análise Completa de Produção

```bash
# Dataset completo, todos os métodos
python batch_hybrid_anthropic.py data/telegram_2023.csv \
  --analyses political_classification sentiment_advanced semantic_interpretation \
  --api-key $ANTHROPIC_API_KEY
```

### 3. Análise Científica Local (Sem Custos)

```bash
# Apenas métodos validados, sem API
python batch_hybrid_anthropic.py data/mensagens.csv \
  --no-api
```

## ⚙️ Configuração Avançada

### Modificar Prompts da API

Edite o dicionário `prompts` em `create_batch_requests()`:

```python
prompts = {
    'political_classification': """Seu prompt customizado aqui...""",
    'minha_analise': """Nova análise..."""
}
```

### Adicionar Novos Métodos Validados

Edite `analyze_with_validated_methods()` para incluir mais análises:

```python
# Adicionar BERTopic
from bertopic import BERTopic
model = BERTopic(language='portuguese')
topics, probs = model.fit_transform(texts)
df['topic_bertopic'] = topics
```

## 🐛 Resolução de Problemas

### "API key não encontrada"

```bash
# Verificar se a variável está configurada
echo $ANTHROPIC_API_KEY

# Ou passar diretamente
python batch_hybrid_anthropic.py data.csv --api-key sk-ant-xxx
```

### "Métodos validados não encontrados"

```bash
# Copiar o arquivo de implementação
cp files\ \(1\)/validated_methods_implementation.py ./
```

### "Batch timeout"

O processamento pode levar até 24h para datasets grandes. Para teste:

```bash
# Use amostra menor
python batch_hybrid_anthropic.py data.csv --sample 100
```

## 📊 Monitoramento

O script mostra progresso em tempo real:

```
⏳ Aguardando processamento do batch batch_xxx...
   Progresso: 450/1000 (45.0%)
   Progresso: 750/1000 (75.0%)
✅ Batch concluído: batch_xxx
```

## 💡 Dicas

1. **Teste primeiro**: Sempre rode com `--sample 100` antes do dataset completo
2. **Batch é mais barato**: 50% desconto vs chamadas individuais
3. **Combine métodos**: Use API para o essencial, métodos locais para o resto
4. **Salve resultados**: Output em `outputs/` com timestamp

## 🔗 Integração com Projeto Principal

Para integrar com o pipeline principal:

```python
from batch_hybrid_anthropic import HybridBatchAnalyzer

# No seu pipeline
analyzer = HybridBatchAnalyzer(api_key="sua_key")
df_analyzed = analyzer.run_hybrid_analysis("data.csv", sample_size=1000)
```

---

**Resumo**: O batch híbrido oferece o melhor dos dois mundos - análise científica validada localmente (grátis) + poder da IA Anthropic (com 50% desconto via batch). Ideal para pesquisa acadêmica com orçamento limitado.