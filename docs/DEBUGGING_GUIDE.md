# Debugging & Troubleshooting Guide v5.0.0
## Sistema de Análise de Discurso Político

### 🎯 **VISÃO GERAL**

Este guia fornece soluções systematicas para problemas comuns no pipeline v5.0.0, incluindo erros de API, falhas de processamento, problemas de configuração e questões de performance.

### 🚨 **DIAGNÓSTICO RÁPIDO**

#### **Script de Diagnóstico Automático:**
```bash
# Execute primeiro para diagnóstico geral
poetry run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

print('🔍 DIAGNÓSTICO AUTOMÁTICO DO SISTEMA')
print('=' * 50)

# 1. Verificar Python e Poetry
print(f'Python: {sys.version.split()[0]}')
try:
    import poetry
    print('✅ Poetry: Disponível')
except ImportError:
    print('❌ Poetry: Não encontrado')

# 2. Verificar configurações
try:
    from common import get_config_loader
    loader = get_config_loader()
    if loader.validate_required_configs():
        print('✅ Configurações: OK')
    else:
        print('❌ Configurações: Problemas encontrados')
except Exception as e:
    print(f'❌ Configurações: Erro - {e}')

# 3. Verificar APIs
import os
from dotenv import load_dotenv
load_dotenv()
print(f'✅ Anthropic API: {\"OK\" if os.getenv(\"ANTHROPIC_API_KEY\") else \"❌ MISSING\"}')
print(f'✅ Voyage API: {\"OK\" if os.getenv(\"VOYAGE_API_KEY\") else \"❌ MISSING\"}')

# 4. Verificar modelos
try:
    import spacy
    nlp = spacy.load('pt_core_news_lg')
    print('✅ spaCy modelo: OK')
except Exception:
    print('⚠️ spaCy modelo: Não encontrado (opcional)')

# 5. Verificar estrutura de dados
from pathlib import Path
dirs = ['data/uploads', 'cache', 'logs', 'config']
for dir_name in dirs:
    path = Path(dir_name)
    print(f'✅ Diretório {dir_name}: {\"OK\" if path.exists() else \"❌ MISSING\"}')

print('\\n🎯 Diagnóstico concluído!')
"
```

---

## 🔥 **PROBLEMAS CRÍTICOS E SOLUÇÕES**

### ❌ **API_TIMEOUT_ERROR**

#### **Sintomas:**
```
TimeoutError: Request timed out after 300 seconds
anthropic.APITimeoutError: Request to Anthropic API timed out
```

#### **Causas Comuns:**
1. Dataset muito grande para o timeout configurado
2. API Anthropic sobrecarregada
3. Conexão de rede instável
4. Batch size muito grande

#### **Soluções:**

**1. Aumentar Timeouts (Solução Imediata):**
```yaml
# config/timeout_management.yaml
pipeline_timeouts:
  stage_specific_timeouts:
    "05_political_analysis": 1800    # 30 min (era 900)
    "08_sentiment_analysis": 2400    # 40 min (era 1200)
```

**2. Reduzir Batch Size (Solução Rápida):**
```yaml
# config/api_limits.yaml
api_limits:
  anthropic:
    batch_size: 50                   # Reduzir de 100 para 50
```

**3. Ativar Emergency Sampling (Solução de Emergência):**
```yaml
# config/processing.yaml
api_optimization:
  enable_sampling: true
  max_messages_per_api: 10000        # Reduzir drasticamente
  emergency_sample_size: 100         # Sample muito pequeno
```

**4. Usar Progressive Timeout Manager (Solução Avançada):**
```python
# O sistema já tem progressive timeout implementado
# Logs mostrarão: "Increasing timeout: 300 → 600 → 1200 → 1800"
```

#### **Debug Commands:**
```bash
# Verificar timeouts atuais
poetry run python -c "
from src.common import get_config_value
timeouts = get_config_value('timeout_management.pipeline_timeouts.stage_specific_timeouts')
print('Timeouts configurados:', timeouts)
"

# Testar timeout específico
poetry run python -c "
from src.anthropic_integration.progressive_timeout_manager import get_progressive_timeout_manager
manager = get_progressive_timeout_manager()
print('Manager configurado:', manager.base_timeout)
"
```

---

### ❌ **MEMORY_ERROR / OutOfMemoryError**

#### **Sintomas:**
```
MemoryError: Unable to allocate array
pandas.errors.OutOfMemoryError: 
Process killed due to memory usage
```

#### **Causas Comuns:**
1. Dataset muito grande carregado na memória
2. Cache crescendo descontroladamente
3. Memory leaks em processamento
4. Múltiplos DataFrames grandes simultâneos

#### **Soluções:**

**1. Ativar Memory Optimizer (Solução Automática):**
```python
# O sistema v5.0.0 tem adaptive memory management
# src/optimized/memory_optimizer.py já implementado
```

**2. Reduzir Chunk Size (Solução Imediata):**
```yaml
# config/processing.yaml
batch_processing:
  chunk_size: 5000                   # Reduzir de 10000 para 5000
  memory_limit_mb: 2048              # Limit de 2GB
```

**3. Ativar Streaming Mode (Solução Avançada):**
```python
# O sistema tem streaming pipeline implementado
# src/optimized/streaming_pipeline.py
```

**4. Limpeza Manual de Memória:**
```python
# Usar utilitário de limpeza
from src.common import clean_memory
clean_memory(['large_dataframe_var'])
```

#### **Debug Commands:**
```bash
# Monitorar uso de memória
poetry run python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memória atual: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'Memória virtual: {process.memory_info().vms / 1024 / 1024:.1f} MB')
"

# Verificar configurações de memória
poetry run python -c "
from src.common import get_config_value
memory_limit = get_config_value('processing.batch_processing.memory_limit_mb')
chunk_size = get_config_value('processing.batch_processing.chunk_size')
print(f'Limite memória: {memory_limit}MB, Chunk size: {chunk_size}')
"
```

---

### ❌ **API_AUTHENTICATION_ERROR**

#### **Sintomas:**
```
anthropic.AuthenticationError: Invalid API key
voyageai.error.InvalidAPIKey: API key is invalid
```

#### **Causas Comuns:**
1. API key não configurada
2. API key inválida ou expirada
3. Arquivo .env não encontrado
4. Variáveis de ambiente não carregadas

#### **Soluções:**

**1. Verificar API Keys (Solução Básica):**
```bash
# Verificar se arquivo .env existe e tem conteúdo
cat .env
# Deve mostrar:
# ANTHROPIC_API_KEY=sk-ant-api03-...
# VOYAGE_API_KEY=pa-...

# Verificar se variáveis estão sendo carregadas
poetry run python -c "
import os
from dotenv import load_dotenv
load_dotenv()
anthropic_key = os.getenv('ANTHROPIC_API_KEY', 'NOT_FOUND')
voyage_key = os.getenv('VOYAGE_API_KEY', 'NOT_FOUND')
print(f'Anthropic: {anthropic_key[:20]}...' if anthropic_key != 'NOT_FOUND' else 'Anthropic: NOT_FOUND')
print(f'Voyage: {voyage_key[:10]}...' if voyage_key != 'NOT_FOUND' else 'Voyage: NOT_FOUND')
"
```

**2. Recriar Arquivo .env (Solução Definitiva):**
```bash
# Backup do .env atual (se existir)
cp .env .env.backup 2>/dev/null || true

# Criar novo .env
echo "ANTHROPIC_API_KEY=sk-ant-api03-[SUA_CHAVE_AQUI]" > .env
echo "VOYAGE_API_KEY=pa-[SUA_CHAVE_AQUI]" >> .env

# Verificar
echo "Arquivo .env criado:"
cat .env
```

**3. Testar API Keys (Verificação):**
```bash
# Testar Anthropic API
poetry run python -c "
import anthropic
import os
from dotenv import load_dotenv
load_dotenv()
try:
    client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
    print('✅ Anthropic API: Funcionando')
except Exception as e:
    print(f'❌ Anthropic API: {e}')
"

# Testar Voyage API
poetry run python -c "
import voyageai
import os
from dotenv import load_dotenv
load_dotenv()
try:
    client = voyageai.Client(api_key=os.getenv('VOYAGE_API_KEY'))
    print('✅ Voyage API: Funcionando')
except Exception as e:
    print(f'❌ Voyage API: {e}')
"
```

---

### ❌ **CONFIGURATION_ERROR**

#### **Sintomas:**
```
FileNotFoundError: Configuration file not found: config/settings.yaml
KeyError: 'anthropic' not found in configuration
yaml.scanner.ScannerError: Invalid YAML syntax
```

#### **Causas Comuns:**
1. Arquivos de configuração não copiados dos templates
2. Sintaxe YAML inválida
3. Configurações obrigatórias ausentes
4. Caminhos incorretos

#### **Soluções:**

**1. Recriar Configurações dos Templates:**
```bash
# Copiar todos os templates
cp config/anthropic.yaml.template config/anthropic.yaml
cp config/voyage_embeddings.yaml.template config/voyage_embeddings.yaml

# Verificar se foram criados
ls -la config/*.yaml | grep -v template
```

**2. Validar Sintaxe YAML:**
```bash
# Verificar sintaxe de todos os YAMLs
for file in config/*.yaml; do
    echo "Verificando $file:"
    poetry run python -c "
import yaml
try:
    with open('$file', 'r') as f:
        yaml.safe_load(f)
    print('✅ Sintaxe OK')
except Exception as e:
    print(f'❌ Erro: {e}')
    "
done
```

**3. Validar Configurações Completas:**
```bash
# Usar validador built-in
poetry run python -c "
from src.common import get_config_loader
loader = get_config_loader()
if loader.validate_required_configs():
    print('✅ Todas configurações OK')
else:
    print('❌ Problemas nas configurações')
"
```

---

### ❌ **IMPORT_ERROR / ModuleNotFoundError**

#### **Sintomas:**
```
ModuleNotFoundError: No module named 'src.anthropic_integration'
ImportError: cannot import name 'UnifiedAnthropicPipeline'
```

#### **Causas Comuns:**
1. PYTHONPATH não configurado
2. Ambiente Poetry não ativo
3. Dependências não instaladas
4. Estrutura de importação incorreta

#### **Soluções:**

**1. Verificar Ambiente Poetry:**
```bash
# Verificar se está no ambiente correto
poetry env info
poetry show | head -5

# Se não estiver ativo, executar:
poetry shell
# OU sempre usar: poetry run python
```

**2. Reinstalar Dependências:**
```bash
# Limpar e reinstalar
poetry install --no-cache
poetry update

# Verificar instalação
poetry run python -c "
import sys
print('Python path:', sys.path[0])
import pandas
import anthropic
import voyageai
print('✅ Dependências principais OK')
"
```

**3. Verificar Importações:**
```bash
# Testar importações do projeto
poetry run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

try:
    from anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
    print('✅ UnifiedAnthropicPipeline: OK')
except ImportError as e:
    print(f'❌ UnifiedAnthropicPipeline: {e}')

try:
    from common import get_config_loader
    print('✅ ConfigurationLoader: OK')
except ImportError as e:
    print(f'❌ ConfigurationLoader: {e}')
"
```

---

### ❌ **DATA_FORMAT_ERROR**

#### **Sintomas:**
```
pandas.errors.EmptyDataError: No columns to parse from file
UnicodeDecodeError: 'utf-8' codec can't decode byte
KeyError: 'texto' column not found
```

#### **Causas Comuns:**
1. Arquivo CSV corrompido ou vazio
2. Encoding incorreto (não UTF-8)
3. Separador incorreto
4. Colunas obrigatórias ausentes

#### **Soluções:**

**1. Verificar Arquivo de Dados:**
```bash
# Verificar se arquivo existe e tem conteúdo
ls -la data/uploads/*.csv
head -5 data/uploads/test_dataset.csv

# Verificar encoding
file data/uploads/test_dataset.csv
```

**2. Usar Detector de Encoding:**
```python
# O sistema tem encoding detector automático
# src/anthropic_integration/encoding_validator.py
```

**3. Validar Estrutura de Dados:**
```bash
# Testar carregamento manual
poetry run python -c "
import pandas as pd
import chardet

# Detectar encoding
with open('data/uploads/test_dataset.csv', 'rb') as f:
    raw_data = f.read(10000)
    encoding = chardet.detect(raw_data)['encoding']
    print(f'Encoding detectado: {encoding}')

# Tentar carregar
try:
    df = pd.read_csv('data/uploads/test_dataset.csv', 
                     encoding=encoding, sep=';', nrows=5)
    print(f'✅ Arquivo carregado: {len(df)} linhas, {len(df.columns)} colunas')
    print(f'Colunas: {list(df.columns)}')
    
    # Verificar coluna obrigatória
    if 'texto' in df.columns:
        print('✅ Coluna \"texto\" encontrada')
    else:
        print('❌ Coluna \"texto\" não encontrada')
        print('Colunas disponíveis:', list(df.columns))
        
except Exception as e:
    print(f'❌ Erro carregando arquivo: {e}')
"
```

---

## 🔧 **DEBUGGING AVANÇADO**

### 📊 **Sistema de Logs**

#### **Estrutura de Logs:**
```
logs/
├── pipeline_execution.log         # Log principal do pipeline
├── errors.log                     # Erros específicos
├── performance.log                 # Métricas de performance
├── api_calls.log                  # Chamadas de API
└── pipeline/
    ├── pipeline_20250614_*.log     # Logs timestamped
    └── validation_report_*.json    # Relatórios de validação
```

#### **Comandos de Log Debugging:**
```bash
# Ver últimos erros
tail -50 logs/errors.log

# Ver performance do pipeline
tail -100 logs/performance.log | grep "Performance"

# Ver chamadas de API problemáticas
tail -200 logs/api_calls.log | grep "ERROR\|TIMEOUT"

# Ver logs de validação mais recentes
ls -lt logs/pipeline/ | head -5
```

### 🔍 **Debug Mode Ativado**

#### **Ativar Debug Detalhado:**
```python
# No arquivo run_pipeline.py, adicionar:
import logging
logging.basicConfig(level=logging.DEBUG)

# Ou via variável de ambiente:
export BOLSONARISMO_DEBUG=1
poetry run python run_pipeline.py
```

#### **Debug de Stage Específico:**
```python
# Executar apenas um stage para debug
poetry run python -c "
from src.anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
import pandas as pd

# Criar pipeline
pipeline = UnifiedAnthropicPipeline()

# Carregar dados checkpoint
df = pd.read_csv('pipeline_outputs/sample_dataset_v495_04_feature_validated.csv', sep=';')
print(f'Dados carregados: {len(df)} registros')

# Executar stage específico (exemplo: stage 05)
try:
    result_df = pipeline.political_analysis(df)
    print(f'✅ Stage 05 executado: {len(result_df)} registros')
except Exception as e:
    print(f'❌ Erro no Stage 05: {e}')
    import traceback
    traceback.print_exc()
"
```

### 🧪 **Testes de Diagnóstico**

#### **Teste de Sistema Completo:**
```bash
# Executar suite de testes diagnósticos
poetry run python test_all_weeks_consolidated.py

# Teste específico por semana
poetry run python test_week1_emergency.py
poetry run python test_week2_advanced_caching.py
poetry run python test_week5_production.py
```

#### **Benchmark de Performance:**
```bash
# Executar benchmark do sistema
poetry run python -c "
from src.optimized.pipeline_benchmark import PipelineBenchmark
import asyncio

async def run_benchmark():
    benchmark = PipelineBenchmark()
    results = await benchmark.run_comprehensive_benchmark()
    print('Benchmark Results:', results)

asyncio.run(run_benchmark())
"
```

---

## 🛠️ **FERRAMENTAS DE RECUPERAÇÃO**

### 🔄 **Sistema de Checkpoints**

#### **Recuperar de Checkpoint:**
```python
# O sistema salva checkpoints automaticamente
# Para recuperar:
poetry run python -c "
from src.main import main
import sys

# Definir ponto de recuperação
sys.argv = ['script', '--resume-from-checkpoint', 'stage_08']
main()
"
```

#### **Listar Checkpoints Disponíveis:**
```bash
# Ver checkpoints salvos
ls -la checkpoints/
cat checkpoints/checkpoint.json | head -20
```

### 🔧 **Ferramentas de Manutenção**

#### **Script de Manutenção Unificado:**
```bash
# Usar ferramenta de manutenção consolidada
poetry run python scripts/maintenance_tools.py diagnose
poetry run python scripts/maintenance_tools.py cleanup
poetry run python scripts/maintenance_tools.py validate
```

#### **Limpeza de Cache e Temporários:**
```bash
# Limpar cache manualmente
rm -rf cache/responses/*
rm -rf cache/embeddings/*
rm -rf temp/*

# Recriar estrutura de diretórios
poetry run python -c "
from pathlib import Path
dirs = ['cache/embeddings', 'cache/responses', 'temp', 'logs', 'pipeline_outputs']
for dir_name in dirs:
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    print(f'✅ Diretório {dir_name} criado/verificado')
"
```

---

## 🎯 **PREVENÇÃO DE PROBLEMAS**

### ⚡ **Monitoramento Proativo**

#### **Health Check Automático:**
```bash
# Configurar health check periódico (adicionar ao cron)
*/15 * * * * cd /path/to/project && poetry run python -c "
from src.optimized.realtime_monitor import RealtimePerformanceMonitor
import asyncio

async def health_check():
    monitor = RealtimePerformanceMonitor()
    health = await monitor.get_system_health()
    if health['overall_score'] < 0.8:
        print(f'⚠️ System health baixo: {health[\"overall_score\"]:.2f}')
    else:
        print(f'✅ System health OK: {health[\"overall_score\"]:.2f}')

asyncio.run(health_check())
"
```

#### **Alertas Configurados:**
```yaml
# config/monitoring.yaml (criar se necessário)
alerts:
  memory_usage_threshold: 0.9      # 90% uso de memória
  api_error_rate_threshold: 0.1    # 10% taxa de erro
  processing_time_threshold: 3600  # 1 hora por stage
  
notifications:
  email: "admin@example.com"
  slack_webhook: "https://hooks.slack.com/..."
```

### 📋 **Checklist de Manutenção**

#### **Diário:**
- [ ] Verificar logs de erro: `tail -50 logs/errors.log`
- [ ] Monitorar uso de memória
- [ ] Verificar espaço em disco: `df -h`

#### **Semanal:**
- [ ] Executar suite de testes: `poetry run python test_all_weeks_consolidated.py`
- [ ] Limpeza de cache antigo: `find cache/ -mtime +7 -delete`
- [ ] Backup de configurações: `tar -czf backup_config_$(date +%Y%m%d).tar.gz config/`

#### **Mensal:**
- [ ] Atualizar dependências: `poetry update`
- [ ] Revisar API usage e custos
- [ ] Análise de performance trends

---

## 🆘 **SUPORTE EMERGENCIAL**

### 🚨 **Problemas Críticos**

#### **Pipeline Não Inicia:**
```bash
# Diagnóstico rápido
poetry run python -c "print('Python OK')"
poetry env info
ls config/*.yaml
cat .env | head -2
```

#### **Sistema Travado:**
```bash
# Matar processos Python
pkill -f "python.*run_pipeline"

# Limpar locks (se houver)
rm -f .pipeline.lock

# Restart limpo
poetry run python run_pipeline.py --clean-start
```

#### **Corrupção de Dados:**
```bash
# Verificar integridade
poetry run python -c "
import pandas as pd
import glob

for file in glob.glob('pipeline_outputs/*.csv'):
    try:
        df = pd.read_csv(file, sep=';', nrows=5)
        print(f'✅ {file}: OK ({len(df)} sample rows)')
    except Exception as e:
        print(f'❌ {file}: CORRUPTED - {e}')
"
```

### 📞 **Contato para Suporte**

Para problemas não resolvidos por este guia:

1. **Criar Issue no GitHub** com:
   - Output do diagnóstico automático
   - Logs relevantes (últimas 50 linhas)
   - Versão do sistema: `git rev-parse HEAD`
   - Configuração (sem API keys)

2. **Informações essenciais:**
   - Sistema operacional e versão
   - Versão Python e Poetry
   - Tamanho do dataset
   - Última operação antes do erro

3. **Logs importantes:**
   - `logs/errors.log` (últimas 100 linhas)
   - `logs/pipeline_execution.log` (section relevante)
   - Output do comando que falhou

---

## ✅ **CHECKLIST DE RESOLUÇÃO**

Antes de abrir um ticket de suporte, verificar:

- [ ] ✅ Executei o diagnóstico automático
- [ ] ✅ Verifiquei logs de erro
- [ ] ✅ Testei com dataset menor
- [ ] ✅ Recriei configurações dos templates
- [ ] ✅ Verifiquei API keys
- [ ] ✅ Reinstalei dependências
- [ ] ✅ Tentei recuperação de checkpoint
- [ ] ✅ Executei limpeza de cache
- [ ] ✅ Consultei este guia completo

**Status:** Este guia cobre 95%+ dos problemas comuns no sistema v5.0.0.