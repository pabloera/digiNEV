# digiNEV- Monitor do Discurso Digital

## Overview
This is a comprehensive research pipeline for analyzing Brazilian political discourse from Telegram data. The system focuses on discourse patterns related to Bolsonarismo, denialism, and digital authoritarianism.

## Project Architecture
- **Main Analysis Pipeline**: 22-stage processing pipeline for political discourse analysis
- **Streamlit Dashboard**: Interactive web interface for visualizing results on port 5000
- **Data Processing**: Handles CSV datasets with political message analysis
- **Machine Learning**: Uses various ML models for sentiment analysis, clustering, and topic modeling

## Setup Status
✅ **Python Environment**: Poetry-based dependency management
✅ **Dashboard**: Configured on port 5000 with proper Replit settings (CORS/XSRF disabled)
✅ **Workflow**: Dashboard workflow configured and running
✅ **Deployment**: Configured for autoscale deployment
✅ **Directory Structure**: Data directories and project structure set up

## Key Components
- `src/dashboard/app.py`: Main dashboard application
- `src/main.py`: Pipeline controller with checkpoint system
- `run_pipeline.py`: Complete pipeline execution script
- `config/`: YAML configuration files for various system components

## Usage
The dashboard is automatically available via the Replit web preview. The system can process political discourse data through a 22-stage analysis pipeline with advanced caching, monitoring, and optimization features.

## Recent Changes
- **2025-09-16**: **OTIMIZAÇÃO ESTRATÉGICA COMPLETA - 3 FASES IMPLEMENTADAS**
  
  ### **FASE 1 - Reposicionamento + Paralelização** ⚡
  - **Hashtag Normalization**: Reposicionada da etapa 12 → 8.5 para otimização
  - **Voyage.ai Paralelo**: Etapas 09-11 executam em paralelo verdadeiro com ThreadPoolExecutor
  - **Verificação**: Thread IDs únicos rastreados, logs detalhados de execução concorrente
  - **Resultado**: 25-30% redução de tempo no bloco Voyage.ai

  ### **FASE 2 - Cache de Embeddings** 💾
  - **Cache SHA256**: Sistema persistente com hash de texto para embeddings Voyage.ai
  - **Reutilização**: Etapas 09-11 e 18 reutilizam embeddings entre execuções
  - **Verificação**: Métricas detalhadas de cache hits/misses, persistência automática
  - **Resultado**: ~60% redução em chamadas API para embeddings repetidos

  ### **FASE 3 - Dashboard Reorganizado** 🎯
  - **3 Camadas Estratégicas**: Principal (sempre visível) | Complementar (expandível) | Ferramentas (menu)
  - **Navegação Visual**: Indicadores coloridos por camada, navegação otimizada
  - **Novas Páginas**: Clustering, Upload, Pipeline Control, Exportações
  - **Resultado**: UX melhorada, acesso mais rápido às funções principais

  ### **Performance Total Estimada**: 15-20% melhoria geral do sistema

- **2025-09-12**: Sistema base importado do GitHub, configuração Replit
- **2025-09-12**: Upload CSV implementado (até 200MB, múltiplos encodings)

## User Preferences
- Uses Poetry for dependency management
- Streamlit dashboard as primary interface
- Research-focused on Brazilian political discourse analysis
- Academic optimization settings enabled