# 📁 Estrutura de Dados - Monitor do Discurso Digital

## 🎯 **Organização dos Diretórios**

Esta pasta contém toda a estrutura de dados do projeto, organizada por etapas de processamento.

```
data/
├── DATASETS_FULL/     # 📊 Datasets originais (CSV de entrada) - 896MB
├── interim/           # 🔄 Dados intermediários (por etapa do pipeline)
└── dashboard_results/ # 📈 Resultados finais deduplicados - 153MB
```

**📊 ESTRUTURA OTIMIZADA (v4.4 - 07/06/2025)**
- ✅ **Remoção de uploads/**: 638MB de dados intermediários redundantes eliminados
- ✅ **Estrutura limpa**: Source data (DATASETS_FULL) → Working data (dashboard_results)
- ✅ **40% economia**: De 1.6GB para 1.0GB de dados essenciais

## 📊 **DATASETS_FULL/**

### **Propósito**
- **Datasets originais** para análise
- **Arquivos CSV** com dados do Telegram (2019-2023)
- **Ponto de entrada** do pipeline

### **Formato Esperado**
```csv
texto;data_hora;canal;url;hashtags
"Mensagem exemplo";2022-01-01 12:00:00;canal_exemplo;https://...;#brasil
```

### **Como Usar**
```bash
# Adicionar seu dataset
cp seu_arquivo.csv data/DATASETS_FULL/

# Executar pipeline
python run_pipeline.py
```

## 🔄 **interim/**

### **Propósito**
- **Dados intermediários** de cada etapa do pipeline
- **Checkpoints automáticos** para recuperação
- **Arquivos temporários** de processamento

### **Estrutura Automática**
```
interim/
├── dataset_01b_features_extracted.csv    # Após extração de features
├── dataset_02_encoding_fixed.csv         # Após correção de encoding
├── dataset_02b_deduplicated.csv          # Após deduplicação
├── dataset_03_text_cleaned.csv           # Após limpeza de texto
├── dataset_04_sentiment_analyzed.csv     # Após análise de sentimento
├── dataset_05_topic_modeled.csv          # Após modelagem de tópicos
├── dataset_06_tfidf_extracted.csv        # Após TF-IDF
├── dataset_07_clustered.csv              # Após clustering
├── dataset_08_hashtags_normalized.csv    # Após normalização hashtags
├── dataset_09_domains_analyzed.csv       # Após análise de domínios
├── dataset_10_temporal_analyzed.csv      # Após análise temporal
├── dataset_11_network_analyzed.csv       # Após análise de rede
├── dataset_12_qualitative_analyzed.csv   # Após análise qualitativa
└── dataset_13_final_processed.csv        # Dataset final processado
```

### **Características**
- ✅ **Criação automática** pelo pipeline
- ✅ **Recuperação de checkpoints** em caso de erro
- ✅ **Backup incremental** de cada etapa

## 📈 **dashboard_results/**

### **Propósito**
- **Dados otimizados** para visualização
- **Cache de embeddings** para busca semântica
- **Resultados agregados** para dashboard

### **Conteúdo Automático**
```
dashboard_results/
├── embeddings_cache/        # Cache de embeddings Voyage.ai
├── aggregated_data.json     # Dados agregados para visualização
├── network_data.json        # Dados de rede social
├── temporal_data.json       # Dados de análise temporal
└── sentiment_summary.json   # Resumo de sentimentos
```

### **Uso**
- 🎯 **Acesso via dashboard**: http://localhost:8501
- 🔄 **Atualização automática** após processamento
- 📊 **Visualizações interativas** dos resultados

## ❌ **uploads/ (REMOVIDO v4.4)**

### **Status: DIRETÓRIO REMOVIDO**
- **Data de remoção**: 07/06/2025
- **Motivo**: Dados intermediários redundantes (638MB economia)
- **Backup**: `data_uploads_backup_20250607_063748.tar.gz`
- **Funcionalidade**: Mantida via pipeline direto DATASETS_FULL → dashboard_results

## 🔄 **Fluxo Completo de Dados**

### **Entrada**
```
DATASETS_FULL/ → Pipeline → interim/ (13 etapas) → dashboard_results/
                     ↓
                 Visualização no Dashboard
```

### **Processamento**
1. **📊 Validação** - Verificação estrutural
2. **🔧 Encoding** - Correção UTF-8
3. **🔄 Deduplicação** - Remoção de duplicatas
4. **🎯 Features** - Extração de características
5. **🧹 Limpeza** - Normalização de texto
6. **😊 Sentimento** - Análise multi-dimensional
7. **🏷️ Tópicos** - Modelagem semântica
8. **📊 TF-IDF** - Análise de frequência
9. **🎯 Clustering** - Agrupamento inteligente
10. **#️⃣ Hashtags** - Normalização semântica
11. **🌐 Domínios** - Análise de credibilidade
12. **⏰ Temporal** - Padrões temporais
13. **🕸️ Rede** - Estrutura social
14. **🎭 Qualitativa** - Classificação de conteúdo

## ⚙️ **Comandos Úteis**

### **Limpeza Manual**
```bash
# Limpar dados intermediários
rm -rf data/interim/*

# Limpar cache do dashboard (regenerável)
rm -rf data/dashboard_results/*cache*

# ⚠️  uploads/ removido permanentemente (v4.4)
# Backup disponível: data_uploads_backup_20250607_063748.tar.gz
```

### **Verificação de Espaço**
```bash
# Verificar tamanho por diretório
du -sh data/*/

# Verificar arquivos grandes
find data -size +100M -ls
```

### **Backup de Resultados**
```bash
# Backup dos resultados finais
cp -r data/interim/*_final_processed.csv backup/
cp -r data/dashboard_results/ backup/
```

## 📋 **Manutenção Automática**

### **Limpeza Automática**
- ✅ **Arquivos temporários** removidos após sucesso
- ✅ **Cache otimizado** para performance
- ✅ **Logs rotativos** para controle de espaço

### **Monitoramento**
- 📊 **Tamanho dos arquivos** monitorado
- 🔄 **Progresso das etapas** registrado
- ⚠️ **Alertas de espaço** em disco baixo

---

**📁 Estrutura de dados completamente organizada e automatizada**

**Para mais detalhes, consulte: [CLAUDE.md](../CLAUDE.md) ou [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)**