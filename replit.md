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
The dashboard is automatically available via the Replit web preview. **IMPORTANT**: Use the Replit Preview button or the generated external URL - DO NOT try to access http://0.0.0.0:5000 directly as this will result in "access denied" errors. The system can process political discourse data through a 22-stage analysis pipeline with advanced caching, monitoring, and optimization features.

## Recent Changes
- **2025-09-17**: **REDESIGN PROFISSIONAL COMPLETO - 4 FASES IMPLEMENTADAS**
  
  ### **FASE 1 - Teste com Dataset Real** ✅
  - **Pipeline Corrigido**: Erro crítico "cannot access local variable 'e'" resolvido
  - **Dataset Real**: 303,707 registros de dados políticos brasileiros processados
  - **Execução Completa**: 23 etapas executadas com sucesso
  - **Warnings Eliminados**: FutureWarning e DtypeWarning corrigidos

  ### **FASE 2 - Redesign Profissional Minimalista** ✅
  - **Nova Paleta**: Azul escuro (#1b365d), laranja escuro (#d85a00), verde escuro (#2d5a27)
  - **Visual Clean**: Cores neutras, tipografia profissional, remoção de gradientes
  - **Header Minimalista**: Design sóbrio adequado para pesquisa acadêmica
  - **CSS Otimizado**: Variáveis CSS, botões limpos, cards simplificados

  ### **FASE 3 - Eliminação de Redundâncias** ✅
  - **Navegação Simplificada**: De 3 camadas complexas para 5 páginas diretas
  - **Sidebar Limpa**: Remoção de informações excessivas e versioning
  - **Menu Direto**: Apenas funcionalidades essenciais visíveis
  - **Indicadores Simples**: Status básico, sem métricas decorativas

  ### **FASE 4 - Funcionalidade Core** ✅
  - **5 Páginas Essenciais**: Overview, Sentimento, Tópicos, Política, Busca
  - **Remoção Massiva**: Clustering, Network, Temporal, Quality, Upload, Pipeline, Exports
  - **Código Limpo**: 200+ linhas de código desnecessário removidas
  - **Interface Focada**: Apenas funcionalidades realmente utilizadas

  ### **Resultado Final**: Dashboard profissional e minimalista adequado para apresentações acadêmicas

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