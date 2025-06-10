# Dashboard Pipeline Bolsonarismo v4.9.1

## 🎯 Visão Geral

Dashboard completo para monitoramento em tempo real das 22 etapas do pipeline de análise política, com gráficos de controle de qualidade, métricas de performance e visualizações específicas por etapa.

## ✨ Principais Funcionalidades

### 📋 **Visão Geral**
- **Métricas Resumo**: Progresso geral, tempo restante, registros processados
- **Timeline Interativa**: Gráfico Gantt das 22 etapas com status em tempo real
- **Progresso por Categoria**: Visualização do avanço em cada categoria de processamento
- **Tabela de Status**: Overview detalhado de todas as etapas

### 🔄 **Monitor do Pipeline**
- **Alertas em Tempo Real**: Notificações de falhas e conclusões
- **Performance das Etapas**: Tempo real vs esperado
- **Uso de Recursos**: Monitoramento de CPU e memória
- **Etapa Atual**: Detalhes da execução em andamento

### 🔍 **Detalhes das Etapas**
- **Seletor Interativo**: Escolha qualquer das 22 etapas
- **Métricas Específicas**: Duração, qualidade, taxa de sucesso
- **Informações por Categoria**: 
  - IA: Custos API, chamadas realizadas
  - Qualidade: Registros entrada/saída
  - NLP: Taxa de processamento, uso de memória

### 📊 **Controle de Qualidade**
- **Gráficos de Controle**: Limites estatísticos 3-sigma
- **Análise de Capacidade**: Índices Cp, Cpk
- **Pareto de Problemas**: Identificação dos gargalos principais
- **Alertas Automáticos**: Notificações de valores fora de controle

### ⚡ **Análise de Performance**
- **Eficiência por Categoria**: Comparação tempo real vs esperado
- **Recomendações**: Sugestões automáticas de otimização
- **Benchmarks**: Comparação com execuções anteriores

### 💰 **Análise de Custos API**
- **Custos em Tempo Real**: Tracking de gastos por etapa
- **Projeções**: Cenários com/sem sampling
- **Economia Achieved**: Percentual de redução de custos

### 🏥 **Saúde do Sistema**
- **Status dos Componentes**: Pipeline, APIs, NLP, Database
- **Métricas do Sistema**: Uptime, disponibilidade, latência
- **Logs Recentes**: Eventos importantes do sistema

## 🏗️ Arquitetura

### **Estrutura de Módulos**

```
src/dashboard/
├── app.py                          # Dashboard principal integrado
├── pipeline_monitor.py             # Monitor das 22 etapas
├── pipeline_visualizations.py      # Visualizações por etapa
├── quality_control_charts.py       # Gráficos de controle
├── start_dashboard.py             # Script de inicialização
└── README_DASHBOARD_v4.9.1.md    # Esta documentação
```

### **Classes Principais**

#### `PipelineMonitor`
- **Responsabilidade**: Carregar e processar dados das 22 etapas
- **Principais Métodos**:
  - `get_pipeline_overview()`: Métricas gerais
  - `get_stage_details(stage_id)`: Detalhes específicos
  - `get_timeline_data()`: Dados para timeline

#### `PipelineVisualizations`
- **Responsabilidade**: Gerar todos os gráficos e visualizações
- **Principais Métodos**:
  - `create_pipeline_progress_chart()`: Timeline Gantt
  - `create_stage_performance_chart()`: Performance das etapas
  - `create_stage_details_panel()`: Painel detalhado

#### `QualityControlCharts`
- **Responsabilidade**: Gráficos de controle estatístico
- **Principais Métodos**:
  - `create_control_chart()`: Gráfico de controle 3-sigma
  - `create_capability_analysis()`: Análise Cp/Cpk
  - `create_pareto_chart()`: Pareto de problemas

## 🚀 Como Usar

### **1. Executar o Pipeline**
```bash
# Primeiro, execute o pipeline para gerar dados
python run_pipeline.py
```

### **2. Iniciar o Dashboard**
```bash
# Inicie o dashboard integrado
python src/dashboard/start_dashboard.py
```

### **3. Acessar a Interface**
- **URL**: http://localhost:8501
- **Auto-refresh**: Configurável (10s - 5min)
- **Navegação**: Menu lateral com 7 seções

## 📊 Navegação e Recursos

### **Menu Lateral**
- 🧭 **Navegação**: Acesso rápido às 7 seções
- ⚙️ **Configurações**: Auto-refresh customizável
- 📈 **Status Rápido**: Métricas em tempo real

### **Funcionalidades Interativas**
- **Auto-refresh**: Atualização automática configurável
- **Seletores Dinâmicos**: Filtros por etapa, métrica, categoria
- **Hover Tooltips**: Informações detalhadas nos gráficos
- **Status Colors**: Código de cores por status das etapas

## 🎨 Design e UX

### **Paleta de Cores**
- ✅ **Verde**: Etapas concluídas
- 🟡 **Amarelo**: Etapas em execução
- 🔴 **Vermelho**: Etapas falhadas
- ⚪ **Cinza**: Etapas pendentes

### **Responsividade**
- **Layout Wide**: Otimizado para monitores grandes
- **Colunas Flexíveis**: Adaptação automática
- **Sidebar Expansível**: Menu lateral colapsável

## 📈 Métricas Monitoradas

### **Por Etapa**
- ⏱️ **Duração**: Tempo real vs esperado
- 📊 **Qualidade**: Score 0-1
- ✅ **Taxa de Sucesso**: Percentual de sucesso
- 🔢 **Registros**: Entrada/Saída processados
- 💾 **Recursos**: Uso de CPU/Memória
- 💰 **Custos**: APIs utilizadas

### **Globais**
- 📈 **Progresso**: Percentual de conclusão
- ⏰ **Tempo Restante**: Estimativa baseada na performance
- 🎯 **Eficiência**: Média geral do pipeline
- 🚨 **Alertas**: Problemas identificados

## 🔧 Configurações Avançadas

### **Auto-refresh**
```python
# Intervalos disponíveis
intervals = [10, 30, 60, 120, 300]  # segundos
```

### **Limites de Controle**
```python
# Configurações por métrica
control_configs = {
    'success_rate': {'target': 0.95, 'upper_spec': 1.0, 'lower_spec': 0.8},
    'quality_score': {'target': 0.85, 'upper_spec': 1.0, 'lower_spec': 0.7},
    'processing_time': {'target': 1.0, 'upper_spec': 2.0, 'lower_spec': 0.2}
}
```

## 🐛 Solução de Problemas

### **Módulos Não Disponíveis**
```bash
# Verificar arquivos
ls src/dashboard/pipeline_*.py

# Reiniciar dashboard
python src/dashboard/start_dashboard.py
```

### **Sem Dados**
```bash
# Executar pipeline primeiro
python run_pipeline.py

# Verificar checkpoints
ls checkpoints/
```

### **Performance Lenta**
- Reduzir intervalo de auto-refresh
- Filtrar por etapas específicas
- Verificar uso de memória do sistema

## 🔮 Próximas Funcionalidades

### **v5.0 (Planejado)**
- 📱 **Mobile Dashboard**: Interface responsiva para dispositivos móveis
- 📧 **Alertas por Email**: Notificações automáticas
- 🔄 **Histórico Completo**: Base de dados de execuções anteriores
- 🤖 **ML Predictions**: Previsões de tempo e recursos
- 📊 **Custom Dashboards**: Painéis personalizáveis

### **Integrações Futuras**
- 🐳 **Docker Monitoring**: Métricas de containers
- ☁️ **Cloud Metrics**: AWS/GCP/Azure integration
- 📱 **Slack/Teams**: Notificações em tempo real
- 📈 **Grafana**: Dashboards externos

## 📄 Changelog

### **v4.9.1 (Atual)**
- ✅ Dashboard completo implementado
- ✅ Monitoramento das 22 etapas
- ✅ Gráficos de controle de qualidade
- ✅ Auto-refresh configurável
- ✅ Interface moderna e responsiva

### **v4.9.0**
- ✅ Base do sistema de monitoramento
- ✅ Integração com pipeline unificado
- ✅ Visualizações básicas

## 👥 Contribuições

Para contribuir com melhorias:

1. **Issues**: Reporte bugs ou sugira funcionalidades
2. **Pull Requests**: Implemente novas features
3. **Documentação**: Aprimore esta documentação
4. **Testes**: Adicione testes automatizados

---

## 📞 Suporte

**Desenvolvido por**: Pablo Emanuel Romero Almada, Ph.D.
**Versão**: 4.9.1 - Junho 2025
**Status**: ✅ Produção - Totalmente Funcional