# AMOSTRA TELEGRAM DATASET - 50 ENTRADAS

## 📁 Informações do Arquivo

- **Arquivo**: `telegram_sample_50_entries.csv`
- **Origem**: `data/DATASETS_FULL/telegram_chunk_001_compatible.csv`
- **Registros**: 50 entradas + header
- **Período**: 2019-07-02 01:10:00 a 2019-07-03 08:07:54
- **Mensagens com texto**: 18/50 (36.0%)

## 📊 Estrutura dos Dados

### Colunas Disponíveis:
- `message_id`: ID único da mensagem
- `datetime`: Data e hora da postagem
- `body`: Texto original da mensagem
- `url`: URL compartilhada (se houver)
- `hashtag`: Hashtags utilizadas
- `channel`: Canal de origem
- `is_fwrd`: Indica se é mensagem encaminhada
- `mentions`: Menções a usuários
- `sender`: Remetente da mensagem
- `media_type`: Tipo de mídia (text/url)
- `domain`: Domínio de URLs compartilhadas
- `body_cleaned`: Texto limpo e processado
- `source_dataset`: Dataset de origem
- `hash_id`: Hash único do registro

### 📝 Colunas de Texto para Análise:
- **Primária**: `body_cleaned` - Texto processado e limpo
- **Secundária**: `body` - Texto original

## 🎯 Uso da Amostra

### Para Testes do Pipeline:
```bash
# Executar pipeline completo
python run_pipeline.py

# Executar dashboard
python src/dashboard/start_dashboard.py
```

### Para Análises Específicas:
```python
import pandas as pd
df = pd.read_csv('data/uploads/telegram_sample_50_entries.csv')

# Análise política
from src.anthropic_integration.political_analyzer import PoliticalAnalyzer
analyzer = PoliticalAnalyzer()
result_df, report = analyzer.analyze_political_discourse(df)

# Análise de sentimentos
from src.anthropic_integration.sentiment_analyzer import AnthropicSentimentAnalyzer
sentiment = AnthropicSentimentAnalyzer()
sentiment_results = sentiment.analyze_political_sentiment(df['body_cleaned'].dropna().tolist())
```

## 📈 Características da Amostra

### Tipos de Conteúdo:
- **Mensagens políticas**: Apoio ao governo Bolsonaro
- **Compartilhamentos**: URLs de vídeos, artigos e redes sociais
- **Conteúdo ideológico**: Discussões sobre armas, conservadorismo
- **Período**: Início do governo Bolsonaro (julho 2019)

### Domínios Populares:
- youtube.com
- patriabook.com
- senado.leg.br

### ⚡ Otimização para Performance

Esta amostra é ideal para:
- ✅ **Testes rápidos** do pipeline (processamento em segundos)
- ✅ **Validação de componentes** Anthropic
- ✅ **Desenvolvimento** de novas funcionalidades
- ✅ **Demonstrações** do sistema

### 💰 Custos Estimados

Com 18 mensagens de texto válidas:
- **Análise política**: ~$0.05 USD
- **Análise completa**: ~$0.15 USD
- **Ideal para testes** sem impacto significativo nos custos

## 🚀 Pronto para Uso

A amostra está validada e pronta para execução com o **Pipeline Bolsonarismo v4.6** com parsing robusto Claude API implementado.