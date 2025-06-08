# Guia de Execução do Pipeline Centralizado Bolsonarismo 2025

## Visão Geral

Este guia fornece instruções completas para executar o pipeline centralizado com integração Anthropic API. O sistema elimina a necessidade de scripts separados e centraliza toda a execução através de um único comando.

## Pré-requisitos

### 1. **Setup do Ambiente**

```bash
# 1. Ativar ambiente virtual
source activate.sh

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar API Anthropic
echo "ANTHROPIC_API_KEY=sk-ant-api03-your-key-here" > .env

# 4. Validar configuração
python run_centralized_pipeline.py --list
```

### 2. **Estrutura de Dados**

```
data/
├── raw/
│   └── telegram_combined_full.csv    # Dataset principal
├── interim/                          # Checkpoints automáticos
└── processed/                        # Resultados finais
```

### 3. **Verificação de Configuração**

```bash
# Verificar se tudo está configurado
python -c "
from src.pipeline.stage_factory import get_stage_factory
from pathlib import Path
import yaml

config = yaml.safe_load(open('config/settings.yaml'))
factory = get_stage_factory(config, Path('.'))
info = factory.list_all_stages()

print(f'✅ API Anthropic: {\"Disponível\" if info[\"anthropic_available\"] else \"Indisponível\"}')
print(f'✅ Total Stages: {info[\"total_stages\"]}')

ai_enabled = sum(1 for s in info['stages'].values() if s['will_use_ai'])
print(f'✅ Stages com AI: {ai_enabled}/{info[\"total_stages\"]}')
"
```

## Comandos de Execução

### **Execução Completa (Recomendado)**

```bash
# Pipeline completo com todos os 13 stages
python run_centralized_pipeline.py

# Com nível de log detalhado
python run_centralized_pipeline.py --log-level DEBUG

# Forçar reinício (sem checkpoint)
python run_centralized_pipeline.py --no-resume
```

**Saída Esperada:**
```
🚀 Iniciando execução do pipeline completo
📂 Carregando dados de: data/raw/telegram_combined_full.csv
✅ Dados carregados: 1,234,567 linhas, 15 colunas
🤖 Pipeline executará com integração Anthropic CENTRALIZADA

============================
Executando etapa: 01_validate_data
============================
🔧 Stage 01: Usando validação tradicional
✅ Etapa 01_validate_data concluída com sucesso

============================
Executando etapa: 02_fix_encoding
============================
🤖 Stage 02: Usando correção inteligente de encoding
✅ Etapa 02_fix_encoding concluída com sucesso

[... outros stages ...]

🎉 Pipeline concluído! Dataset final: 1,234,567 linhas
```

### **Execução de Stages Específicos**

```bash
# Apenas stages de processamento de texto
python run_centralized_pipeline.py --stages 02_fix_encoding 03_clean_text 04_sentiment_analysis

# Apenas stages de análise avançada
python run_centralized_pipeline.py --stages 10_temporal_analysis 11_network_structure 12_qualitative_analysis

# Apenas revisão final
python run_centralized_pipeline.py --stages 13_review_reproducibility
```

### **Execução de Stage Individual**

```bash
# Executar apenas análise de sentimentos
python run_centralized_pipeline.py --single 04_sentiment_analysis

# Executar apenas análise qualitativa
python run_centralized_pipeline.py --single 12_qualitative_analysis

# Executar apenas revisão do pipeline
python run_centralized_pipeline.py --single 13_review_reproducibility
```

### **Execução Sem Anthropic API**

```bash
# Forçar uso de métodos tradicionais (apenas para operações simples)
python run_centralized_pipeline.py --no-anthropic

# NOTA: Muitos stages falharão pois requerem AI para análise complexa
```

### **Execução com Dados de Amostra**

```bash
# Processar apenas 10.000 linhas (para testes)
python run_centralized_pipeline.py --sample 10000

# Processar apenas 1.000 linhas (desenvolvimento)
python run_centralized_pipeline.py --sample 1000
```

## Monitoramento da Execução

### **Logs em Tempo Real**

```bash
# Terminal 1: Executar pipeline
python run_centralized_pipeline.py

# Terminal 2: Monitorar logs
tail -f logs/pipeline/pipeline_$(date +%Y%m%d)*.log

# Terminal 3: Monitorar custos Anthropic
watch -n 10 'python -c "
from src.anthropic_integration.cost_monitor import get_cost_report
import json
print(json.dumps(get_cost_report(), indent=2))
"'
```

### **Checkpoints Automáticos**

O pipeline salva checkpoints automaticamente após cada stage:

```
data/interim/
├── checkpoint_01_validate_data.csv
├── checkpoint_02_fix_encoding.csv
├── checkpoint_02b_deduplication.csv
├── checkpoint_01b_feature_extraction.csv
├── checkpoint_03_clean_text.csv
├── checkpoint_04_sentiment_analysis.csv
└── ...
```

Para retomar execução:
```bash
# Retoma automaticamente do último checkpoint
python run_centralized_pipeline.py

# Forçar reinício completo
python run_centralized_pipeline.py --no-resume
```

## Informações e Diagnósticos

### **Listar Todos os Stages**

```bash
python run_centralized_pipeline.py --list
```

**Saída Esperada:**
```
📋 ETAPAS DO PIPELINE
============================================================

01_validate_data - ✅ Habilitada
  Módulo: stage_01_validate_data
  Dependências: 
  Parâmetros: ['skip_structure_check']

02_fix_encoding - ✅ Habilitada
  Módulo: stage_02_fix_encoding
  Dependências: 01_validate_data
  Parâmetros: ['columns_to_fix']

[... outros stages ...]

🤖 STATUS ANTHROPIC:
API Disponível: ✅
Total de Stages: 13
Stages com AI: 12/13
```

### **Simulação (Dry Run)**

```bash
# Simular execução sem processar dados
python run_centralized_pipeline.py --dry-run

# Simular stages específicos
python run_centralized_pipeline.py --stages 04_sentiment_analysis 12_qualitative_analysis --dry-run
```

### **Análise de Configuração**

```bash
# Verificar configuração de cada stage
python -c "
import yaml
from pathlib import Path

config = yaml.safe_load(open('config/settings.yaml'))
anthropic_config = config.get('anthropic', {})

print('🤖 CONFIGURAÇÃO ANTHROPIC')
print(f'Modelo: {anthropic_config.get(\"model\", \"Não configurado\")}')
print(f'Max Tokens: {anthropic_config.get(\"max_tokens\", \"Não configurado\")}')
print(f'Temperature: {anthropic_config.get(\"temperature\", \"Não configurado\")}')

print('\n📊 STAGES COM AI HABILITADA:')
ai_stages = []
for key, value in config.items():
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            if 'use_anthropic' in subkey and subvalue:
                ai_stages.append(f'{key}.{subkey}')

for stage in ai_stages:
    print(f'  ✅ {stage}')
"
```

## Fluxos de Trabalho Comuns

### **1. Desenvolvimento e Testes**

```bash
# 1. Testar com amostra pequena
python run_centralized_pipeline.py --sample 1000 --single 04_sentiment_analysis

# 2. Executar stages de análise específicos
python run_centralized_pipeline.py --sample 5000 --stages 04_sentiment_analysis 12_qualitative_analysis

# 3. Pipeline completo com amostra
python run_centralized_pipeline.py --sample 10000
```

### **2. Análise de Produção**

```bash
# 1. Validar configuração
python run_centralized_pipeline.py --list

# 2. Executar pipeline completo
python run_centralized_pipeline.py

# 3. Verificar resultados
ls -la data/processed/
cat logs/pipeline/pipeline_report_*.json | jq .
```

### **3. Re-processamento Seletivo**

```bash
# 1. Re-executar apenas análise de texto
python run_centralized_pipeline.py --stages 03_clean_text 04_sentiment_analysis --no-resume

# 2. Re-executar apenas análises avançadas
python run_centralized_pipeline.py --stages 10_temporal_analysis 11_network_structure 12_qualitative_analysis --no-resume

# 3. Re-executar apenas revisão
python run_centralized_pipeline.py --single 13_review_reproducibility --no-resume
```

### **4. Análise de Custos**

```bash
# Monitorar custos durante execução
python -c "
from src.anthropic_integration.cost_monitor import AnthropicCostMonitor
import yaml

config = yaml.safe_load(open('config/settings.yaml'))
monitor = AnthropicCostMonitor(config)
report = monitor.get_cost_report()

print('💰 RELATÓRIO DE CUSTOS ANTHROPIC')
print(f'Uso Diário: ${report[\"daily_usage\"]:.4f} / ${report[\"daily_limit\"]:.2f}')
print(f'Uso Mensal: ${report[\"monthly_usage\"]:.4f} / ${report[\"monthly_limit\"]:.2f}')
print(f'Projeção Mensal: ${report[\"projected_monthly\"]:.2f}')

print('\n📊 USO POR STAGE:')
for stage, cost in report['usage_by_stage'].items():
    print(f'  {stage}: ${cost:.4f}')
"
```

## Tratamento de Erros

### **Erros Comuns e Soluções**

#### **1. API Key Não Configurada**

```
❌ Erro: ANTHROPIC_API_KEY não configurada

Solução:
echo "ANTHROPIC_API_KEY=sk-ant-api03-your-key" > .env
```

#### **2. Arquivo de Dados Não Encontrado**

```
❌ Erro: Arquivo de entrada não encontrado: data/raw/telegram_combined_full.csv

Solução:
# Verificar se arquivo existe
ls -la data/raw/

# Especificar arquivo alternativo
python run_centralized_pipeline.py --input data/raw/sample_dataset.csv
```

#### **3. Limite de Custo Atingido**

```
❌ Erro: Limite diário de custo atingido ($10.00)

Solução:
# Aumentar limite em config/settings.yaml
anthropic:
  cost_limits:
    daily_limit_usd: 20.0

# Ou resetar contador
python -c "
from src.anthropic_integration.cost_monitor import AnthropicCostMonitor
monitor = AnthropicCostMonitor({})
monitor.reset_daily_usage()
"
```

#### **4. Stage Requer Anthropic Mas API Indisponível**

```
❌ Erro: Stage 04_sentiment_analysis requer Anthropic API para análise complexa

Solução:
# Verificar API key
cat .env | grep ANTHROPIC

# Testar conexão
python -c "from src.anthropic_integration.base import AnthropicBase; AnthropicBase({}).test_connection()"

# Usar fallback apenas para stages simples (não recomendado para análise)
python run_centralized_pipeline.py --no-anthropic
```

### **Recovery de Execução Interrompida**

```bash
# Pipeline interrompido? Retomar automaticamente
python run_centralized_pipeline.py

# Verificar último checkpoint
ls -lat data/interim/checkpoint_*.csv | head -5

# Retomar de stage específico
python run_centralized_pipeline.py --stages 05_topic_modeling 06_tfidf_extraction 07_clustering
```

## Resultados Esperados

### **Arquivos de Saída**

```
data/processed/
├── final_dataset.csv                 # Dataset final processado
└── final_dataset_metadata.json       # Metadados e estatísticas

results/
├── text_analysis/                    # Análises de texto
├── visualizations/                   # Gráficos e visualizações
└── final_report/                     # Relatório final

logs/pipeline/
├── pipeline_20250126_143022.log      # Log detalhado da execução
└── pipeline_report_20250126_143022.json  # Relatório estruturado
```

### **Métricas de Sucesso**

```json
{
  "pipeline_execution": {
    "total_stages": 13,
    "successful_stages": 13,
    "failed_stages": 0,
    "success_rate": 100.0,
    "total_duration": 3642.5,
    "anthropic_enhanced_stages": 12
  },
  "data_processing": {
    "input_rows": 1234567,
    "output_rows": 1230045,
    "data_quality_score": 0.94,
    "processing_efficiency": 0.89
  },
  "anthropic_usage": {
    "total_api_calls": 247,
    "total_cost_usd": 8.45,
    "average_cost_per_stage": 0.70,
    "ai_enhancement_quality": 0.92
  }
}
```

### **Indicadores de Qualidade**

- **Taxa de Sucesso**: 100% dos stages executados
- **Qualidade dos Dados**: Score > 0.90
- **Eficiência de Processamento**: > 85%
- **Custo por Análise**: < $10 por execução completa
- **Reprodutibilidade**: Resultados consistentes entre execuções

## Troubleshooting Avançado

### **Debug Detalhado**

```bash
# Ativar logging máximo
python run_centralized_pipeline.py --log-level DEBUG --single 04_sentiment_analysis

# Verificar imports dos módulos
python -c "
import sys
sys.path.insert(0, 'src')

try:
    from anthropic_integration.sentiment_analyzer import AnthropicSentimentAnalyzer
    print('✅ AnthropicSentimentAnalyzer importado com sucesso')
except Exception as e:
    print(f'❌ Erro no import: {e}')

try:
    from pipeline.stage_factory import get_stage_factory
    print('✅ StageFactory importado com sucesso')
except Exception as e:
    print(f'❌ Erro no import: {e}')
"

# Verificar configuração YAML
python -c "
import yaml
try:
    config = yaml.safe_load(open('config/settings.yaml'))
    print('✅ Configuração YAML válida')
    print(f'Anthropic configurado: {\"anthropic\" in config}')
except Exception as e:
    print(f'❌ Erro na configuração: {e}')
"
```

### **Performance Monitoring**

```bash
# Monitorar uso de memória
python run_centralized_pipeline.py --sample 5000 &
PID=$!
while kill -0 $PID 2>/dev/null; do
    ps -p $PID -o %cpu,%mem,rss,vsz,comm
    sleep 30
done

# Monitorar uso de disco
df -h data/
du -sh data/interim/
```

Este guia fornece todas as informações necessárias para executar o pipeline centralizado de forma eficiente e monitorar sua execução de perto.