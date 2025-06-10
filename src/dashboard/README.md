# Dashboard Pipeline Bolsonarismo v4.9.1 📊

> Interface web para análise de datasets com 22 etapas do pipeline

## 🚀 Início Rápido

```bash
# 1. Instalar dependências
pip install streamlit plotly

# 2. Iniciar dashboard
cd src/dashboard
python start_dashboard.py

# 3. Acessar no navegador
# http://localhost:8501
```

## 📊 Funcionalidades

### **Upload e Processamento**
- Upload múltiplo de CSV via drag-and-drop
- Validação automática de estrutura
- Processamento paralelo de datasets
- Monitoramento em tempo real

### **Visualizações por Etapa**
- **Stages 01-04**: Validação, encoding, deduplicação, features
- **Stages 05-08**: Política, limpeza, linguística, sentimentos  
- **Stages 09-11**: Topic modeling, TF-IDF, clustering (Voyage.ai)
- **Stages 12-16**: Hashtags, domínios, temporal, redes, qualitativa
- **Stages 17-20**: Review, interpretação, busca, validação

### **Análises Disponíveis**
- Distribuições estatísticas por etapa
- Análise temporal de padrões
- Redes de influência e coordenação
- Classificação de discursos políticos
- Métricas de qualidade do pipeline

## 🔧 Configuração

### **Dependências**
Listadas em `requirements.txt`:
```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
```

### **Estrutura**
- `app.py` - Interface principal Streamlit
- `csv_parser.py` - Parser integrado de CSV
- `start_dashboard.py` - Script de inicialização

## 📈 Uso

1. **Upload**: Arrastar CSV para área de upload
2. **Processamento**: Pipeline executa automaticamente 22 etapas
3. **Visualização**: Gráficos interativos por etapa
4. **Download**: Resultados processados em CSV/JSON

## 🚨 Troubleshooting

### **Problemas Comuns**
```bash
# Erro de porta ocupada
streamlit run app.py --server.port 8502

# Erro de memória com datasets grandes
# Usar chunking automático (já implementado)

# Verificar logs
tail -f ../../logs/pipeline_execution.log
```

---
**Referência**: Documentação completa no [README.md principal](../../README.md)