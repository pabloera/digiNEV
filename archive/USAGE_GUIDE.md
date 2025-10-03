# Guia de Uso - Batch Scientific API

## 🎯 Execução Rápida

### 1. Configurar ambiente
```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar API key
cp .env.example .env
# Editar .env e adicionar sua ANTHROPIC_API_KEY
```

### 2. Executar análise
```bash
# Análise básica
python batch_scientific_api.py data/seu_dataset.csv

# Com amostragem
python batch_scientific_api.py data/seu_dataset.csv --sample 1000

# Modo debug
python batch_scientific_api.py data/seu_dataset.csv --debug
```

### 3. Visualizar resultados
```bash
# Gerar todas as visualizações
python visualization/visualize_results.py outputs/results_latest.csv --all

# Apenas análise política
python visualization/visualize_results.py outputs/results_latest.csv --political
```

## 📁 Estrutura do Diretório

```
batch_analyzer/
├── batch_scientific_api.py    # Script principal de análise
├── lexico_politico_hierarquizado.json  # Léxico político brasileiro
├── config/                    # Configurações
│   ├── batch_validated_methods_config.json  # Métodos científicos
│   ├── default.yaml          # Config padrão
│   └── research.yaml         # Config pesquisa
├── data/                      # Datasets de entrada
├── outputs/                   # Resultados das análises
├── visualization/             # Scripts de visualização
│   └── visualize_results.py  # Visualizador principal
└── archive/                   # Arquivos antigos arquivados

```

## 📊 Formato dos Dados

### Entrada (CSV)
O sistema detecta automaticamente colunas de texto: `text`, `body`, `message`, `content`, `texto`, `mensagem`

### Saída
- CSV com todas as análises: `outputs/results_TIMESTAMP.csv`
- JSON com metadados: `outputs/results_TIMESTAMP.json`
- Visualizações: `visualization/outputs/`

## 🔬 Métodos Científicos Aplicados

1. **Análise Política**: Classificação em 6 categorias (extrema-direita a esquerda)
2. **Análise de Sentimento**: Score -1 a 1
3. **Modelagem de Tópicos**: LDA e clustering
4. **Análise Temporal**: Evolução e padrões
5. **Análise de Redes**: Coordenação e influência

## ⚙️ Opções Avançadas

```bash
# Usar configuração personalizada
python batch_scientific_api.py data/dataset.csv --config config/research.yaml

# Processar em lote múltiplos arquivos
for file in data/*.csv; do
    python batch_scientific_api.py "$file"
done
```

## 📈 Monitoramento

Durante a execução, o sistema mostra:
- Progresso por stage
- Estatísticas de API
- Estimativa de custo
- Taxa de sucesso

## 🆘 Problemas Comuns

### Erro de API
- Verificar ANTHROPIC_API_KEY no arquivo .env
- Sistema usa fallback automático se API falhar

### Memória insuficiente
- Usar flag --sample para processar amostra menor
- Dividir dataset em partes menores

### Encoding de caracteres
- Sistema detecta automaticamente encoding UTF-8 e Latin-1
- Para forçar: --encoding utf-8