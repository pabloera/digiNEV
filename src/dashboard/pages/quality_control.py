"""
Página de Controle de Qualidade do Dashboard digiNEV
Análise de validação e métricas de qualidade do pipeline
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Optional, Dict, Any

def render_quality_page(data_loader):
    """Renderiza a página de controle de qualidade"""
    
    st.markdown('<div class="page-header"><h2>🔬 Controle de Qualidade</h2></div>', unsafe_allow_html=True)
    
    if not data_loader:
        st.error("Sistema de dados não disponível")
        return
    
    # Carregar dados de validação
    validation_data = data_loader.load_data('validation_report')
    
    if validation_data is None:
        st.warning("📊 Relatório de validação não disponível")
        st.info("Execute o pipeline principal para gerar o relatório de validação (Stage 20)")
        return
    
    # Status geral do pipeline
    st.subheader("🎯 Status Geral do Pipeline")
    
    # Extrair informações do relatório de validação
    if not validation_data.empty:
        try:
            # Se for JSON normalizado, pode ter estrutura aninhada
            if 'overall_assessment.overall_score' in validation_data.columns:
                overall_score = validation_data['overall_assessment.overall_score'].iloc[0]
            elif 'overall_score' in validation_data.columns:
                overall_score = validation_data['overall_score'].iloc[0]
            else:
                overall_score = 0.0
                
            quality_level = "Excelente" if overall_score > 0.9 else "Boa" if overall_score > 0.7 else "Adequada" if overall_score > 0.5 else "Precisa Melhorar"
            
        except Exception as e:
            overall_score = 0.0
            quality_level = "Desconhecido"
    else:
        overall_score = 0.0
        quality_level = "Sem dados"
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Score Geral de Qualidade", f"{overall_score:.3f}", delta=None)
    
    with col2:
        st.metric("Nível de Qualidade", quality_level)
    
    with col3:
        status = data_loader.get_data_status()
        completion_rate = (status['available_files'] / status['total_files']) * 100
        st.metric("Taxa de Completude", f"{completion_rate:.1f}%")
    
    with col4:
        st.metric("Última Validação", status.get('last_execution', 'N/A'))
    
    # Gauge de qualidade geral
    st.subheader("📊 Indicador de Qualidade Geral")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = overall_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score de Qualidade (0-1)"},
        delta = {'reference': 0.8},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightgray"},
                {'range': [0.5, 0.7], 'color': "yellow"},
                {'range': [0.7, 0.9], 'color': "lightgreen"},
                {'range': [0.9, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Análise por dimensão de qualidade
    st.subheader("📋 Análise por Dimensão de Qualidade")
    
    # Carregar todas as análises para verificar qualidade
    all_data_types = ['dataset_stats', 'political_analysis', 'sentiment_analysis', 'topic_modeling', 'clustering_results']
    all_datasets = data_loader.load_multiple_data(all_data_types)
    
    # Calcular métricas de qualidade por análise
    quality_metrics = calculate_quality_metrics(all_datasets)
    
    if quality_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras com scores por análise
            analyses = list(quality_metrics.keys())
            scores = [quality_metrics[analysis]['quality_score'] for analysis in analyses]
            
            fig_quality_bars = go.Figure()
            fig_quality_bars.add_trace(go.Bar(
                x=analyses,
                y=scores,
                marker_color=['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in scores],
                text=[f"{s:.2f}" for s in scores],
                textposition='auto'
            ))
            
            fig_quality_bars.update_layout(
                title="Score de Qualidade por Análise",
                xaxis_title="Tipo de Análise",
                yaxis_title="Score de Qualidade",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_quality_bars, use_container_width=True)
        
        with col2:
            # Radar chart com diferentes dimensões
            categories = ['Completude', 'Consistência', 'Confiabilidade', 'Cobertura', 'Precisão']
            
            # Calcular médias por categoria
            avg_scores = []
            for category in categories:
                scores = [quality_metrics[analysis].get(category.lower(), 0.5) for analysis in analyses]
                avg_scores.append(sum(scores) / len(scores) if scores else 0.5)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_scores,
                theta=categories,
                fill='toself',
                name='Qualidade Geral'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Dimensões de Qualidade",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # Detalhes por análise
    st.subheader("🔍 Detalhes por Análise")
    
    if quality_metrics:
        # Seletor de análise
        selected_analysis = st.selectbox(
            "Selecionar análise para detalhes:",
            list(quality_metrics.keys()),
            key="quality_analysis_detail"
        )
        
        if selected_analysis and selected_analysis in quality_metrics:
            analysis_quality = quality_metrics[selected_analysis]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score de Qualidade", f"{analysis_quality['quality_score']:.3f}")
                st.metric("Registros Válidos", f"{analysis_quality.get('valid_records', 0):,}")
            
            with col2:
                st.metric("Taxa de Completude", f"{analysis_quality.get('completude', 0):.1%}")
                st.metric("Taxa de Confiança", f"{analysis_quality.get('confiabilidade', 0):.1%}")
            
            with col3:
                st.metric("Consistência", f"{analysis_quality.get('consistência', 0):.1%}")
                st.metric("Cobertura", f"{analysis_quality.get('cobertura', 0):.1%}")
            
            # Problemas identificados
            if 'issues' in analysis_quality:
                issues = analysis_quality['issues']
                if issues:
                    st.subheader("⚠️ Problemas Identificados")
                    for issue in issues:
                        severity = issue.get('severity', 'medium')
                        icon = "🔴" if severity == 'high' else "🟡" if severity == 'medium' else "🟢"
                        st.write(f"{icon} **{issue.get('type', 'Problema')}**: {issue.get('description', 'Sem descrição')}")
    
    # Análise de integridade dos dados
    st.subheader("🔒 Integridade dos Dados")
    
    integrity_checks = perform_integrity_checks(all_datasets)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Verificações Aprovadas")
        for check in integrity_checks['passed']:
            st.success(f"✓ {check}")
    
    with col2:
        st.subheader("❌ Verificações Falharam")
        for check in integrity_checks['failed']:
            st.error(f"✗ {check}")
        
        if integrity_checks['warnings']:
            st.subheader("⚠️ Avisos")
            for warning in integrity_checks['warnings']:
                st.warning(f"⚠ {warning}")
    
    # Análise temporal da qualidade
    st.subheader("📈 Evolução da Qualidade")
    
    # Simular dados históricos de qualidade (em produção viria de logs)
    historical_quality = generate_historical_quality_data()
    
    if historical_quality:
        fig_temporal = px.line(
            historical_quality,
            x='date',
            y='quality_score',
            title="Evolução do Score de Qualidade",
            labels={'quality_score': 'Score de Qualidade', 'date': 'Data'}
        )
        
        # Adicionar linha de meta
        fig_temporal.add_hline(y=0.8, line_dash="dash", line_color="red", 
                              annotation_text="Meta de Qualidade (0.8)")
        
        fig_temporal.update_layout(height=400)
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    # Relatório de validação detalhado
    if validation_data is not None and not validation_data.empty:
        st.subheader("📋 Relatório de Validação Completo")
        
        # Expandir para mostrar dados brutos
        with st.expander("📄 Dados Brutos do Relatório"):
            st.json(validation_data.to_dict('records')[0] if len(validation_data) > 0 else {})
        
        # Resumo estruturado
        st.subheader("📊 Resumo Executivo")
        
        # Extrair informações principais
        summary_info = extract_validation_summary(validation_data)
        
        for section, info in summary_info.items():
            st.subheader(f"🔸 {section}")
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    if key.endswith('_score') or key.endswith('_rate'):
                        st.metric(key.replace('_', ' ').title(), f"{value:.3f}")
                    else:
                        st.metric(key.replace('_', ' ').title(), f"{value:,}")
                else:
                    st.write(f"**{key.replace('_', ' ').title()}**: {value}")
    
    # Ferramentas de controle
    st.subheader("🛠️ Ferramentas de Controle")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Atualizar Métricas", key="refresh_quality"):
            data_loader.clear_cache()
            st.rerun()
    
    with col2:
        if st.button("📊 Exportar Relatório", key="export_quality"):
            if validation_data is not None:
                export_path = data_loader.export_data('validation_report', 'json')
                if export_path:
                    st.success(f"Relatório exportado: {export_path.name}")
            else:
                st.warning("Dados de validação não disponíveis")
    
    with col3:
        if st.button("🔍 Diagnóstico Detalhado", key="detailed_diagnosis"):
            run_detailed_diagnosis(all_datasets)
    
    with col4:
        if st.button("📈 Gerar Insight", key="generate_insight"):
            insights = generate_quality_insights(quality_metrics, integrity_checks)
            for insight in insights:
                st.info(f"💡 {insight}")
    
    # Recomendações de melhoria
    st.subheader("💡 Recomendações de Melhoria")
    
    recommendations = generate_recommendations(quality_metrics, integrity_checks, overall_score)
    
    for i, rec in enumerate(recommendations, 1):
        priority = rec.get('priority', 'medium')
        icon = "🔴" if priority == 'high' else "🟡" if priority == 'medium' else "🟢"
        
        with st.expander(f"{icon} {rec['title']} (Prioridade: {priority.title()})"):
            st.write(f"**Descrição**: {rec['description']}")
            st.write(f"**Impacto**: {rec['impact']}")
            if 'steps' in rec:
                st.write("**Passos para implementação**:")
                for step in rec['steps']:
                    st.write(f"• {step}")

def calculate_quality_metrics(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Calcula métricas de qualidade para cada dataset"""
    metrics = {}
    
    for data_type, df in datasets.items():
        if df is None:
            metrics[data_type] = {
                'quality_score': 0.0,
                'completude': 0.0,
                'consistência': 0.0,
                'confiabilidade': 0.0,
                'cobertura': 0.0,
                'precisão': 0.0,
                'valid_records': 0,
                'issues': [{'type': 'Dados Ausentes', 'description': 'Dataset não disponível', 'severity': 'high'}]
            }
            continue
        
        # Calcular métricas básicas
        total_rows = len(df)
        non_null_ratio = df.notna().mean().mean()
        
        # Confiabilidade baseada em colunas de confiança
        confidence_columns = [col for col in df.columns if 'confidence' in col.lower() or 'score' in col.lower()]
        if confidence_columns:
            confidence_avg = df[confidence_columns].mean().mean()
        else:
            confidence_avg = 0.7  # Valor padrão
        
        # Consistência baseada em valores únicos vs total
        uniqueness_ratio = df.nunique().mean() / len(df) if len(df) > 0 else 0
        
        # Score geral
        quality_score = (non_null_ratio * 0.3 + confidence_avg * 0.4 + min(uniqueness_ratio, 1.0) * 0.3)
        
        # Identificar problemas
        issues = []
        if non_null_ratio < 0.8:
            issues.append({
                'type': 'Dados Incompletos',
                'description': f'Apenas {non_null_ratio:.1%} dos dados estão completos',
                'severity': 'high' if non_null_ratio < 0.5 else 'medium'
            })
        
        if confidence_avg < 0.6:
            issues.append({
                'type': 'Baixa Confiança',
                'description': f'Confiança média de apenas {confidence_avg:.1%}',
                'severity': 'medium'
            })
        
        metrics[data_type] = {
            'quality_score': quality_score,
            'completude': non_null_ratio,
            'consistência': min(uniqueness_ratio, 1.0),
            'confiabilidade': confidence_avg,
            'cobertura': 1.0 if total_rows > 0 else 0.0,
            'precisão': confidence_avg,  # Aproximação
            'valid_records': total_rows,
            'issues': issues
        }
    
    return metrics

def perform_integrity_checks(datasets: Dict[str, pd.DataFrame]) -> Dict[str, list]:
    """Realiza verificações de integridade dos dados"""
    checks = {
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    # Verificar se datasets essenciais existem
    essential_datasets = ['dataset_stats', 'political_analysis', 'sentiment_analysis']
    
    for dataset in essential_datasets:
        if datasets.get(dataset) is not None:
            checks['passed'].append(f"Dataset {dataset} disponível")
        else:
            checks['failed'].append(f"Dataset {dataset} ausente")
    
    # Verificar consistência entre datasets
    if datasets.get('political_analysis') is not None and datasets.get('sentiment_analysis') is not None:
        pol_rows = len(datasets['political_analysis'])
        sent_rows = len(datasets['sentiment_analysis'])
        
        if abs(pol_rows - sent_rows) / max(pol_rows, sent_rows) < 0.1:
            checks['passed'].append("Consistência entre análise política e sentimento")
        else:
            checks['warnings'].append(f"Diferença significativa entre datasets: {pol_rows} vs {sent_rows} registros")
    
    # Verificar qualidade dos dados
    for data_type, df in datasets.items():
        if df is not None:
            # Verificar se há duplicatas excessivas
            if len(df) > 0:
                duplicate_ratio = df.duplicated().sum() / len(df)
                if duplicate_ratio < 0.05:
                    checks['passed'].append(f"Baixa taxa de duplicatas em {data_type}")
                elif duplicate_ratio > 0.2:
                    checks['failed'].append(f"Alta taxa de duplicatas em {data_type}: {duplicate_ratio:.1%}")
                else:
                    checks['warnings'].append(f"Taxa moderada de duplicatas em {data_type}: {duplicate_ratio:.1%}")
    
    return checks

def generate_historical_quality_data() -> pd.DataFrame:
    """Gera dados históricos simulados de qualidade"""
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    
    # Gerar 30 dias de dados
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    
    # Simular evolução da qualidade com tendência crescente
    base_quality = 0.6
    trend = 0.01
    noise = 0.05
    
    quality_scores = []
    for i, date in enumerate(dates):
        score = base_quality + (trend * i) + np.random.normal(0, noise)
        score = max(0.3, min(1.0, score))  # Limitar entre 0.3 e 1.0
        quality_scores.append(score)
    
    return pd.DataFrame({
        'date': dates,
        'quality_score': quality_scores
    })

def extract_validation_summary(validation_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Extrai resumo estruturado do relatório de validação"""
    if validation_data.empty:
        return {}
    
    summary = {}
    row = validation_data.iloc[0]
    
    # Análise geral
    summary['Análise Geral'] = {}
    for col in row.index:
        if 'overall' in col.lower() or 'general' in col.lower():
            summary['Análise Geral'][col] = row[col]
    
    # Métricas de qualidade
    summary['Métricas de Qualidade'] = {}
    for col in row.index:
        if any(term in col.lower() for term in ['score', 'rate', 'quality', 'accuracy']):
            summary['Métricas de Qualidade'][col] = row[col]
    
    # Estatísticas
    summary['Estatísticas'] = {}
    for col in row.index:
        if any(term in col.lower() for term in ['count', 'total', 'number', 'records']):
            summary['Estatísticas'][col] = row[col]
    
    return summary

def run_detailed_diagnosis(datasets: Dict[str, pd.DataFrame]):
    """Executa diagnóstico detalhado dos dados"""
    st.subheader("🔍 Diagnóstico Detalhado")
    
    total_issues = 0
    
    for data_type, df in datasets.items():
        if df is None:
            st.error(f"❌ {data_type}: Dataset não disponível")
            total_issues += 1
            continue
        
        with st.expander(f"📊 {data_type.replace('_', ' ').title()}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Registros**: {len(df):,}")
                st.write(f"**Colunas**: {len(df.columns)}")
                st.write(f"**Memória**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            with col2:
                null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
                st.write(f"**Valores nulos**: {null_ratio:.1%}")
                
                duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
                st.write(f"**Duplicatas**: {duplicate_ratio:.1%}")
                
                if null_ratio > 0.1 or duplicate_ratio > 0.1:
                    total_issues += 1
    
    if total_issues == 0:
        st.success("✅ Nenhum problema crítico identificado!")
    else:
        st.warning(f"⚠️ {total_issues} problema(s) identificado(s)")

def generate_quality_insights(quality_metrics: Dict, integrity_checks: Dict) -> List[str]:
    """Gera insights automáticos sobre a qualidade"""
    insights = []
    
    if quality_metrics:
        # Melhor e pior análise
        scores = {k: v['quality_score'] for k, v in quality_metrics.items()}
        best_analysis = max(scores, key=scores.get)
        worst_analysis = min(scores, key=scores.get)
        
        insights.append(f"Melhor qualidade: {best_analysis} ({scores[best_analysis]:.3f})")
        insights.append(f"Precisa atenção: {worst_analysis} ({scores[worst_analysis]:.3f})")
        
        # Média geral
        avg_quality = sum(scores.values()) / len(scores)
        insights.append(f"Qualidade média do pipeline: {avg_quality:.3f}")
    
    # Status das verificações
    total_checks = len(integrity_checks['passed']) + len(integrity_checks['failed'])
    if total_checks > 0:
        success_rate = len(integrity_checks['passed']) / total_checks
        insights.append(f"Taxa de sucesso das verificações: {success_rate:.1%}")
    
    return insights

def generate_recommendations(quality_metrics: Dict, integrity_checks: Dict, overall_score: float) -> List[Dict]:
    """Gera recomendações de melhoria"""
    recommendations = []
    
    # Recomendações baseadas no score geral
    if overall_score < 0.7:
        recommendations.append({
            'title': 'Melhorar Qualidade Geral',
            'description': 'Score geral de qualidade abaixo do recomendado',
            'impact': 'Alto - afeta confiabilidade de todas as análises',
            'priority': 'high',
            'steps': [
                'Revisar parâmetros de configuração do pipeline',
                'Verificar qualidade dos dados de entrada',
                'Aumentar thresholds de confiança das análises'
            ]
        })
    
    # Recomendações baseadas em verificações falhadas
    if integrity_checks['failed']:
        recommendations.append({
            'title': 'Corrigir Problemas de Integridade',
            'description': f'{len(integrity_checks["failed"])} verificações de integridade falharam',
            'impact': 'Médio - pode afetar consistência dos resultados',
            'priority': 'medium',
            'steps': [
                'Investigar causas das falhas de verificação',
                'Reexecutar etapas problemáticas do pipeline',
                'Validar dados de entrada'
            ]
        })
    
    # Recomendações baseadas em métricas específicas
    if quality_metrics:
        low_quality_analyses = [k for k, v in quality_metrics.items() if v['quality_score'] < 0.6]
        
        if low_quality_analyses:
            recommendations.append({
                'title': f'Melhorar Análises Específicas',
                'description': f'Análises com baixa qualidade: {", ".join(low_quality_analyses)}',
                'impact': 'Médio - afeta análises específicas',
                'priority': 'medium',
                'steps': [
                    'Revisar configurações das análises problemáticas',
                    'Verificar qualidade dos dados de entrada específicos',
                    'Considerar retreinamento de modelos se aplicável'
                ]
            })
    
    # Recomendação geral se tudo estiver bem
    if overall_score > 0.8 and not integrity_checks['failed']:
        recommendations.append({
            'title': 'Manter Qualidade Atual',
            'description': 'Sistema operando com alta qualidade',
            'impact': 'Baixo - manutenção preventiva',
            'priority': 'low',
            'steps': [
                'Continuar monitoramento regular',
                'Fazer backup dos dados de qualidade',
                'Documentar configurações atuais'
            ]
        })
    
    return recommendations