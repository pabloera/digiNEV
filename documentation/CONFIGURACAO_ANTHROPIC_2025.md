# Configuração Anthropic API - Pipeline Bolsonarismo 2025

## Visão Geral

Este documento detalha a configuração completa da integração Anthropic API no pipeline centralizado, incluindo setup, parâmetros, monitoramento de custos e diretrizes de uso.

## Configuração Básica

### 1. **Variáveis de Ambiente**

Criar arquivo `.env` na raiz do projeto:

```bash
# Configuração obrigatória
ANTHROPIC_API_KEY=sk-ant-api03-your-api-key-here

# Configurações opcionais
ANTHROPIC_MODEL=claude-3-haiku-20240307
ANTHROPIC_MAX_TOKENS=4000
ANTHROPIC_TEMPERATURE=0.3
```

### 2. **Arquivo de Configuração Principal (`config/settings.yaml`)**

```yaml
# Configuração Global Anthropic
anthropic:
  model: "claude-3-haiku-20240307"
  max_tokens: 4000
  temperature: 0.3
  cost_monitoring: true
  fallback_enabled: true
  
  # Configurações de rate limiting
  max_requests_per_minute: 50
  retry_attempts: 3
  retry_delay: 2
  
  # Configurações de contexto brasileiro
  context:
    language: "pt-BR"
    political_period: "2019-2023"
    focus_topics: ["bolsonarismo", "negacionismo", "autoritarismo"]
    
  # Monitoramento de custos
  cost_limits:
    daily_limit_usd: 10.0
    monthly_limit_usd: 100.0
    alert_threshold: 0.8
```

## Configuração por Stage

### **Stages com Anthropic Habilitado (12 de 13)**

```yaml
# Stage 02: Encoding Fix
encoding_fix:
  use_anthropic: true
  columns_to_fix: ["texto", "text_cleaned", "canal", "hashtags"]
  confidence_threshold: 0.8
  batch_size: 1000

# Stage 02b: Deduplication  
deduplication:
  use_anthropic: true
  text_column: "texto"
  similarity_threshold: 0.9
  semantic_analysis: true
  max_chunk_size: 5000

# Stage 01b: Feature Extraction
feature_extraction:
  use_anthropic: true
  extract_political_features: true
  extract_emotional_features: true
  extract_narrative_features: true

# Stage 03: Text Cleaning
text_cleaning:
  use_anthropic: true
  text_column: "texto"
  preserve_context: true
  preserve_political_terms: true
  remove_urls: false
  remove_emojis: false

# Stage 04: Sentiment Analysis
sentiment:
  use_anthropic: true
  text_column: "text_cleaned"
  method: "hybrid"
  language: "pt"
  political_context: true
  dimensions: ["polarity", "emotion", "political_stance"]

# Stage 05: Topic Modeling
lda:
  use_anthropic_interpretation: true
  n_topics: 15
  iterations: 1000
  alpha: 0.01
  beta: 0.01
  interpret_themes: true

# Stage 06: TF-IDF Extraction
tfidf:
  use_anthropic: true
  max_features: 5000
  ngram_range: [1, 3]
  semantic_grouping: true
  political_relevance_filter: true

# Stage 07: Clustering
clustering:
  use_anthropic_validation: true
  method: "kmeans"
  n_clusters: 10
  validation_method: "semantic"
  interpret_clusters: true

# Stage 08: Hashtag Normalization
hashtag_normalization:
  use_anthropic: true
  min_frequency: 5
  similarity_threshold: 0.8
  semantic_clustering: true
  political_hashtag_focus: true

# Stage 09: Domain Analysis
domain_analysis:
  use_anthropic: true
  min_frequency: 5
  batch_size: 30
  credibility_analysis: true
  misinformation_detection: true

# Stage 10: Temporal Analysis
temporal_analysis:
  use_anthropic: true
  analysis_window_days: 7
  significance_threshold: 2.0
  event_sensitivity: 0.8
  historical_context: true

# Stage 11: Network Analysis
network_analysis:
  use_anthropic: true
  min_edge_weight: 3
  max_nodes: 500
  community_sample_size: 100
  influence_analysis: true

# Stage 12: Qualitative Analysis
qualitative:
  use_anthropic_classification: true
  confidence_threshold: 0.8
  conspiracy_detection: true
  negacionism_detection: true
  authoritarianism_detection: true

# Stage 13: Pipeline Review
pipeline_review:
  use_anthropic: true
  quality_threshold: 0.8
  detail_level: "comprehensive"
  generate_recommendations: true
  scientific_rigor_assessment: true
```

### **Stage Sem Anthropic (1 de 13)**

```yaml
# Stage 01: Data Validation - Sem AI por performance
data_validation:
  use_anthropic: false  # Mantido tradicional por eficiência
  validation_level: "comprehensive"
  structural_checks: true
  basic_stats: true
```

## Estrutura de Prompts Padronizada

### **Template Base para Contexto Brasileiro**

```python
BRAZILIAN_CONTEXT_TEMPLATE = """
CONTEXTO ESPECÍFICO BRASILEIRO (2019-2023):

PERÍODO HISTÓRICO:
- Governo Jair Bolsonaro (2019-2022)
- Pandemia COVID-19 (2020-2022)
- Eleições presidenciais 2022
- Transição governamental 2022-2023

MOVIMENTOS POLÍTICOS:
- Bolsonarismo e extrema-direita
- Negacionismo científico
- Autoritarismo digital
- Polarização política

PLATAFORMAS ANALISADAS:
- Telegram (canais políticos)
- Mensagens em português brasileiro
- Conteúdo político-ideológico

OBJETIVOS DA PESQUISA:
- Análise de discurso político
- Detecção de desinformação
- Padrões de comunicação autoritária
- Impacto na democracia digital
"""

def create_contextual_prompt(task_description: str, data_sample: str) -> str:
    """Cria prompt contextualizado para análise brasileira"""
    return f"""
{BRAZILIAN_CONTEXT_TEMPLATE}

TAREFA ESPECÍFICA:
{task_description}

DADOS PARA ANÁLISE:
{data_sample}

INSTRUÇÕES:
- Analise considerando o contexto político brasileiro
- Identifique padrões específicos do bolsonarismo
- Detecte elementos autoritários e negacionistas
- Responda em JSON estruturado
- Use terminologia acadêmica apropriada
"""
```

### **Prompts Específicos por Stage**

#### **Stage 04: Sentiment Analysis**
```python
SENTIMENT_PROMPT = """
Analise o sentimento político nas mensagens do Telegram brasileiro:

{brazilian_context}

DIMENSÕES DE ANÁLISE:
1. Polaridade: positivo/negativo/neutro
2. Emoção: raiva/medo/esperança/orgulho/desprezo
3. Postura política: pró-governo/oposição/neutro
4. Intensidade: baixa/média/alta

DADOS: {message_sample}

Responda em JSON:
{
  "sentiment_analysis": [
    {
      "message_id": "id",
      "polarity": "positivo|negativo|neutro",
      "emotion": "raiva|medo|esperança|orgulho|desprezo|neutro",
      "political_stance": "pró-governo|oposição|neutro",
      "intensity": "baixa|média|alta",
      "confidence": 0.85
    }
  ],
  "summary": {
    "dominant_sentiment": "descrição",
    "political_patterns": ["padrão1", "padrão2"]
  }
}
"""
```

#### **Stage 12: Qualitative Analysis**
```python
QUALITATIVE_PROMPT = """
Classifique o conteúdo quanto a teorias conspiratórias e negacionismo:

{brazilian_context}

CATEGORIAS DE ANÁLISE:
1. Conspiração: teoria_conspiratória/suspeita/neutro
2. Negacionismo: científico/histórico/institucional/ausente
3. Autoritarismo: autoritário/democrático/neutro
4. Desinformação: falsa/enganosa/verdadeira/não_verificável

DADOS: {message_sample}

Responda em JSON:
{
  "qualitative_classification": [
    {
      "message_id": "id",
      "conspiracy_level": "alta|média|baixa|ausente",
      "negationism_type": "científico|histórico|institucional|ausente",
      "authoritarianism": "autoritário|democrático|neutro",
      "misinformation": "falsa|enganosa|verdadeira|não_verificável",
      "evidence": ["evidência1", "evidência2"],
      "confidence": 0.9
    }
  ],
  "patterns": {
    "conspiracy_themes": ["tema1", "tema2"],
    "negationist_narratives": ["narrativa1", "narrativa2"],
    "authoritarian_indicators": ["indicador1", "indicador2"]
  }
}
"""
```

## Monitoramento de Custos

### **Sistema de Monitoramento Integrado**

```python
class AnthropicCostMonitor:
    """Monitor de custos integrado ao pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.daily_limit = config.get('anthropic', {}).get('cost_limits', {}).get('daily_limit_usd', 10.0)
        self.monthly_limit = config.get('anthropic', {}).get('cost_limits', {}).get('monthly_limit_usd', 100.0)
        self.alert_threshold = config.get('anthropic', {}).get('cost_limits', {}).get('alert_threshold', 0.8)
        
        self.cost_log_file = Path('logs/anthropic/anthropic_costs.json')
        self.cost_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estima custo antes da chamada"""
        # Claude 3 Haiku: $0.25/1M input tokens, $1.25/1M output tokens
        input_tokens = len(prompt) // 4  # Aproximação
        input_cost = (input_tokens / 1_000_000) * 0.25
        output_cost = (max_tokens / 1_000_000) * 1.25
        return input_cost + output_cost
    
    def track_usage(self, stage: str, operation: str, cost: float):
        """Registra uso da API"""
        usage_data = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'operation': operation,
            'cost_usd': cost,
            'daily_total': self.get_daily_total() + cost,
            'monthly_total': self.get_monthly_total() + cost
        }
        
        self._save_usage(usage_data)
        self._check_limits(usage_data)
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Gera relatório de custos"""
        return {
            'daily_usage': self.get_daily_total(),
            'monthly_usage': self.get_monthly_total(),
            'daily_limit': self.daily_limit,
            'monthly_limit': self.monthly_limit,
            'usage_by_stage': self.get_usage_by_stage(),
            'projected_monthly': self.project_monthly_cost()
        }
```

### **Logs de Custo Estruturados**

```json
{
  "usage_log": [
    {
      "timestamp": "2025-01-26T10:30:00",
      "stage": "04_sentiment_analysis",
      "operation": "analyze_sentiment",
      "input_tokens": 1500,
      "output_tokens": 800,
      "cost_usd": 0.001375,
      "daily_total": 0.045,
      "monthly_total": 2.3
    }
  ],
  "daily_summary": {
    "date": "2025-01-26",
    "total_calls": 45,
    "total_cost": 0.045,
    "cost_by_stage": {
      "04_sentiment_analysis": 0.015,
      "12_qualitative_analysis": 0.020,
      "others": 0.010
    }
  }
}
```

## Configuração de Rate Limiting

### **Implementação de Throttling**

```python
class AnthropicRateLimiter:
    """Controle de rate limiting para API"""
    
    def __init__(self, max_requests_per_minute: int = 50):
        self.max_requests = max_requests_per_minute
        self.request_times = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Aguarda se necessário para respeitar rate limit"""
        with self.lock:
            now = time.time()
            # Remove requests antigas (> 1 minuto)
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.max_requests:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit atingido. Aguardando {sleep_time:.1f}s")
                    time.sleep(sleep_time)
            
            self.request_times.append(now)
```

## Configuração de Fallback

### **Estratégia de Fallback Hierárquica**

```yaml
# Configuração de fallback em settings.yaml
anthropic:
  fallback_strategy:
    # Nível 1: Retry com exponential backoff
    retry_attempts: 3
    retry_delays: [1, 2, 4]  # segundos
    
    # Nível 2: Modelo alternativo
    fallback_model: "claude-3-sonnet-20240229"
    
    # Nível 3: Processamento tradicional
    traditional_fallback: true
    fallback_threshold: 0.5  # Após 50% de falhas
    
    # Nível 4: Modo de emergência
    emergency_mode: true
    emergency_sample_size: 1000  # Processar apenas amostra
```

### **Implementação de Fallback Inteligente**

```python
class IntelligentFallback:
    """Sistema de fallback inteligente"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('anthropic', {}).get('fallback_strategy', {})
        self.failure_count = 0
        self.total_attempts = 0
        
    def should_use_fallback(self) -> bool:
        """Decide se deve usar fallback"""
        if self.total_attempts == 0:
            return False
            
        failure_rate = self.failure_count / self.total_attempts
        threshold = self.config.get('fallback_threshold', 0.5)
        
        return failure_rate > threshold
    
    def execute_with_fallback(self, ai_function: Callable, traditional_function: Callable, *args, **kwargs):
        """Executa com fallback inteligente"""
        self.total_attempts += 1
        
        if self.should_use_fallback():
            logger.warning("Alto índice de falhas. Usando método tradicional.")
            return traditional_function(*args, **kwargs)
        
        try:
            return ai_function(*args, **kwargs)
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Falha na AI: {e}. Usando fallback.")
            return traditional_function(*args, **kwargs)
```

## Comando de Configuração

### **Script de Setup Automatizado**

```bash
# Criar script setup_anthropic.py
python setup_anthropic.py --configure

# Opções disponíveis:
python setup_anthropic.py --configure --model claude-3-haiku-20240307
python setup_anthropic.py --test-connection
python setup_anthropic.py --cost-report
python setup_anthropic.py --reset-limits
```

### **Validação de Configuração**

```python
def validate_anthropic_config():
    """Valida configuração completa"""
    checks = {
        'api_key': check_api_key(),
        'model_access': check_model_access(),
        'cost_limits': check_cost_limits(),
        'rate_limits': check_rate_limits(),
        'fallback_config': check_fallback_config()
    }
    
    all_passed = all(checks.values())
    
    print("🔍 VALIDAÇÃO DA CONFIGURAÇÃO ANTHROPIC")
    print("=" * 50)
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}: {'OK' if status else 'ERRO'}")
    
    if all_passed:
        print("\n🎉 Configuração válida! Pipeline pronto para executar.")
    else:
        print("\n⚠️ Problemas encontrados. Verifique a configuração.")
    
    return all_passed
```

## Resumo da Configuração

### **Checklist de Setup**

- [ ] **API Key configurada** no arquivo `.env`
- [ ] **Modelo selecionado** em `settings.yaml`
- [ ] **Limites de custo definidos** e monitoramento ativo
- [ ] **Rate limiting configurado** apropriadamente
- [ ] **Fallback habilitado** para todos os stages
- [ ] **Context brasileiro** nos prompts
- [ ] **Logging detalhado** de uso e custos
- [ ] **Validação da configuração** executada com sucesso

### **Comandos de Verificação**

```bash
# Verificar configuração completa
python run_centralized_pipeline.py --list

# Testar conexão Anthropic
python -c "from src.anthropic_integration.base import AnthropicBase; AnthropicBase({}).test_connection()"

# Ver relatório de custos
python -c "from src.anthropic_integration.cost_monitor import get_cost_report; print(get_cost_report())"
```

Esta configuração garante uso otimizado da API Anthropic com controle rigoroso de custos, fallbacks robustos e contexto específico para análise política brasileira.