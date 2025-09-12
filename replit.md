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
- **2025-09-12**: Imported from GitHub and configured for Replit environment
- Dashboard configured for port 5000 with proper host settings
- Added `src/dashboard/__init__.py` to fix import issues
- Set up workflow for dashboard auto-restart
- Configured deployment settings for production use
- **2025-09-12**: Implementado sistema de upload de arquivos CSV na página inicial
- Suporte para arquivos grandes (até 200MB) com múltiplos encodings
- Análise rápida automática dos dados carregados
- Integração com sistema de dados existente

## User Preferences
- Uses Poetry for dependency management
- Streamlit dashboard as primary interface
- Research-focused on Brazilian political discourse analysis
- Academic optimization settings enabled