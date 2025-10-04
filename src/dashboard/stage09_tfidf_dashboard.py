"""
Stage 09 TF-IDF Vectorization Dashboard
Análise de Termos Mais Relevantes via TF-IDF

Foco: Visualização de termos mais importantes no discurso político brasileiro
- Bar chart: Top 20 termos mais relevantes com scores TF-IDF
- Treemap: Hierarquia de termos por importância e frequência (até 50 termos)
- Difference analysis: Termos únicos vs compartilhados entre períodos
- Ranking evolution: Mudanças no ranking de termos importantes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import re
import logging
from datetime import datetime, timedelta
import ast

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar disponibilidade de bibliotecas
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn não disponível")

class TFIDFAnalyzer:
    """Analisador de TF-IDF com foco em termos políticos brasileiros."""

    def __init__(self):
        self.stop_words_pt = {
            'a', 'o', 'e', 'é', 'de', 'do', 'da', 'que', 'em', 'um', 'uma',
            'para', 'com', 'não', 'se', 'na', 'no', 'por', 'mais', 'as',
            'os', 'ao', 'ser', 'ter', 'dos', 'das', 'seu', 'sua', 'seus',
            'suas', 'mas', 'ou', 'foi', 'são', 'pelo', 'pela', 'pelos',
            'pelas', 'isso', 'essa', 'esse', 'esta', 'este', 'estas', 'estes',
            'apenas', 'sobre', 'até', 'após', 'antes', 'durante', 'desde'
        }

        # Termos políticos relevantes para priorizar
        self.political_terms = {
            'governo', 'presidente', 'política', 'brasil', 'país', 'estado',
            'direitos', 'democracia', 'constituição', 'congresso', 'senado',
            'deputado', 'ministro', 'partido', 'eleição', 'voto', 'cidadão',
            'público', 'nacional', 'federal', 'municipal', 'lei', 'justiça',
            'economia', 'educação', 'saúde', 'segurança', 'social', 'trabalho',
            'desenvolvimento', 'reforma', 'projeto', 'proposta', 'decisão'
        }

    def extract_terms_from_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extrai e agrega termos TF-IDF do dataframe."""
        try:
            if 'tfidf_top_terms' not in df.columns:
                return {}

            # Agregar todos os termos com seus scores
            term_scores = defaultdict(list)

            for idx, row in df.iterrows():
                try:
                    # Parse da lista de termos
                    tfidf_terms = row['tfidf_top_terms']

                    # Verificar se não é nulo/vazio
                    if tfidf_terms is None or (isinstance(tfidf_terms, str) and not tfidf_terms.strip()):
                        continue

                    if isinstance(tfidf_terms, str):
                        # Tentar fazer parse da string
                        if tfidf_terms.startswith('['):
                            terms = ast.literal_eval(tfidf_terms)
                        else:
                            terms = [term.strip() for term in tfidf_terms.split(',') if term.strip()]
                    else:
                        terms = tfidf_terms

                        # Usar score médio como peso
                        score = row.get('tfidf_score_mean', 0.1) if pd.notna(row.get('tfidf_score_mean')) else 0.1

                        if isinstance(terms, list):
                            for term in terms:
                                if term and isinstance(term, str) and len(term) > 2:
                                    term_clean = term.lower().strip()
                                    if term_clean not in self.stop_words_pt:
                                        term_scores[term_clean].append(score)
                        elif isinstance(terms, str) and terms:
                            # Lidar com strings diretas também
                            term_clean = terms.lower().strip()
                            if len(term_clean) > 2 and term_clean not in self.stop_words_pt:
                                term_scores[term_clean].append(score)

                except Exception as e:
                    logger.debug(f"Erro ao processar linha {idx}: {e}")
                    continue

            # Calcular scores finais (média ponderada)
            final_scores = {}
            for term, scores in term_scores.items():
                final_scores[term] = {
                    'mean_score': np.mean(scores),
                    'max_score': np.max(scores),
                    'frequency': len(scores),
                    'total_score': np.sum(scores)
                }

            return final_scores

        except Exception as e:
            logger.error(f"Erro ao extrair termos TF-IDF: {e}")
            return {}

    def prepare_terms_for_visualization(self, term_scores: Dict, limit: int = 50) -> pd.DataFrame:
        """Prepara dados de termos para visualização."""
        if not term_scores:
            return pd.DataFrame()

        # Converter para DataFrame
        data = []
        for term, scores in term_scores.items():
            data.append({
                'term': term,
                'mean_score': scores['mean_score'],
                'max_score': scores['max_score'],
                'frequency': scores['frequency'],
                'total_score': scores['total_score'],
                'is_political': term in self.political_terms,
                'relevance_score': scores['mean_score'] * np.log1p(scores['frequency'])
            })

        df_terms = pd.DataFrame(data)

        if df_terms.empty:
            return df_terms

        # Ordenar por relevância
        df_terms = df_terms.sort_values('relevance_score', ascending=False)

        # Filtrar termos de qualidade
        df_terms = df_terms[
            (df_terms['term'].str.len() > 2) &
            (df_terms['frequency'] >= 1) &
            (~df_terms['term'].str.contains(r'^\d+$', regex=True))
        ]

        return df_terms.head(limit)

    def analyze_temporal_evolution(self, df: pd.DataFrame) -> Dict:
        """Analisa evolução temporal dos termos."""
        if 'datetime' not in df.columns:
            return {}

        try:
            # Converter datetime
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime'])

            if len(df) == 0:
                return {}

            # Dividir em períodos
            df = df.sort_values('datetime')
            total_days = (df['datetime'].max() - df['datetime'].min()).days

            if total_days < 30:
                # Para períodos curtos, dividir pela metade
                mid_date = df['datetime'].min() + timedelta(days=total_days//2)
                period1 = df[df['datetime'] <= mid_date]
                period2 = df[df['datetime'] > mid_date]
                period_names = ['Período Inicial', 'Período Final']
            else:
                # Para períodos longos, dividir em meses ou trimestres
                months = df['datetime'].dt.to_period('M').unique()
                if len(months) >= 3:
                    period1 = df[df['datetime'].dt.to_period('M') == months[0]]
                    period2 = df[df['datetime'].dt.to_period('M') == months[-1]]
                    period_names = [f'{months[0]}', f'{months[-1]}']
                else:
                    mid_date = df['datetime'].min() + timedelta(days=total_days//2)
                    period1 = df[df['datetime'] <= mid_date]
                    period2 = df[df['datetime'] > mid_date]
                    period_names = ['Primeira Metade', 'Segunda Metade']

            # Extrair termos para cada período
            terms_p1 = self.extract_terms_from_data(period1)
            terms_p2 = self.extract_terms_from_data(period2)

            # Preparar dados
            df_p1 = self.prepare_terms_for_visualization(terms_p1, limit=30)
            df_p2 = self.prepare_terms_for_visualization(terms_p2, limit=30)

            if df_p1.empty or df_p2.empty:
                return {}

            # Análise comparativa
            set_p1 = set(df_p1['term'].tolist())
            set_p2 = set(df_p2['term'].tolist())

            shared_terms = set_p1.intersection(set_p2)
            unique_p1 = set_p1 - set_p2
            unique_p2 = set_p2 - set_p1

            return {
                'period_names': period_names,
                'period1_data': df_p1,
                'period2_data': df_p2,
                'shared_terms': list(shared_terms),
                'unique_period1': list(unique_p1),
                'unique_period2': list(unique_p2),
                'terms_p1': terms_p1,
                'terms_p2': terms_p2
            }

        except Exception as e:
            logger.error(f"Erro na análise temporal: {e}")
            return {}

def create_top_terms_bar_chart(df_terms: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    """Cria gráfico de barras dos top termos TF-IDF."""
    if df_terms.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Dados insuficientes para análise TF-IDF",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 16}
        )
        return fig

    # Top 20 termos
    top_terms = df_terms.head(20).copy()

    # Destacar termos políticos
    colors = ['#1f77b4' if not political else '#d62728'
              for political in top_terms['is_political']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_terms['relevance_score'],
        y=top_terms['term'],
        orientation='h',
        marker=dict(color=colors, opacity=0.8),
        text=[f"{score:.3f}" for score in top_terms['relevance_score']],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Score de Relevância: %{x:.3f}<br>" +
            "Frequência: %{customdata[0]}<br>" +
            "Score Médio TF-IDF: %{customdata[1]:.3f}<br>" +
            "<extra></extra>"
        ),
        customdata=np.column_stack((
            top_terms['frequency'],
            top_terms['mean_score']
        ))
    ))

    fig.update_layout(
        title=f"Top 20 Termos Mais Relevantes (TF-IDF){title_suffix}",
        xaxis_title="Score de Relevância",
        yaxis_title="Termos",
        height=600,
        yaxis=dict(categoryorder='total ascending'),
        showlegend=False,
        template="plotly_white",
        margin=dict(l=120, r=50, t=80, b=50)
    )

    # Adicionar anotação sobre cores
    fig.add_annotation(
        text="🔴 Termos Políticos | 🔵 Termos Gerais",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)"
    )

    return fig

def create_terms_treemap(df_terms: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    """Cria treemap hierárquico dos termos."""
    if df_terms.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Dados insuficientes para treemap",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 16}
        )
        return fig

    # Top 50 termos para treemap
    top_terms = df_terms.head(50).copy()

    # Categorizar termos
    def categorize_term(term):
        if term in ['governo', 'presidente', 'política', 'partido', 'eleição', 'democracia']:
            return 'Política Institucional'
        elif term in ['economia', 'trabalho', 'desenvolvimento', 'dinheiro', 'emprego']:
            return 'Economia'
        elif term in ['educação', 'saúde', 'segurança', 'social']:
            return 'Políticas Sociais'
        elif term in ['brasil', 'país', 'estado', 'nacional', 'público']:
            return 'Estado e Nação'
        else:
            return 'Outros Temas'

    top_terms['category'] = top_terms['term'].apply(categorize_term)

    # Preparar dados para treemap
    fig = go.Figure(go.Treemap(
        labels=top_terms['term'],
        parents=top_terms['category'],
        values=top_terms['relevance_score'],
        texttemplate="<b>%{label}</b><br>%{value:.3f}",
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Categoria: %{parent}<br>" +
            "Score: %{value:.3f}<br>" +
            "Frequência: %{customdata[0]}<br>" +
            "<extra></extra>"
        ),
        customdata=np.column_stack((top_terms['frequency'],)),
        maxdepth=2,
        pathbar_visible=False
    ))

    fig.update_layout(
        title=f"Hierarquia de Termos por Relevância TF-IDF{title_suffix}",
        height=600,
        template="plotly_white",
        margin=dict(l=10, r=10, t=80, b=10)
    )

    return fig

def create_temporal_difference_analysis(temporal_data: Dict) -> go.Figure:
    """Cria análise de diferenças temporais entre termos."""
    if not temporal_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Dados temporais insuficientes",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 16}
        )
        return fig

    period_names = temporal_data['period_names']
    shared = temporal_data['shared_terms'][:15]  # Top 15 compartilhados
    unique_p1 = temporal_data['unique_period1'][:10]  # Top 10 únicos
    unique_p2 = temporal_data['unique_period2'][:10]  # Top 10 únicos

    # Criar subplot
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"Únicos: {period_names[0]}",
            "Termos Compartilhados",
            f"Únicos: {period_names[1]}"
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.05
    )

    # Únicos período 1
    if unique_p1:
        terms_p1 = temporal_data['terms_p1']
        scores_p1 = [terms_p1.get(term, {}).get('relevance_score', 0.1) for term in unique_p1]

        fig.add_trace(
            go.Bar(
                x=scores_p1,
                y=unique_p1,
                orientation='h',
                marker_color='#ff7f0e',
                name=f"Únicos {period_names[0]}",
                showlegend=False
            ),
            row=1, col=1
        )

    # Compartilhados
    if shared:
        terms_p1 = temporal_data['terms_p1']
        scores_shared = [terms_p1.get(term, {}).get('relevance_score', 0.1) for term in shared]

        fig.add_trace(
            go.Bar(
                x=scores_shared,
                y=shared,
                orientation='h',
                marker_color='#2ca02c',
                name="Compartilhados",
                showlegend=False
            ),
            row=1, col=2
        )

    # Únicos período 2
    if unique_p2:
        terms_p2 = temporal_data['terms_p2']
        scores_p2 = [terms_p2.get(term, {}).get('relevance_score', 0.1) for term in unique_p2]

        fig.add_trace(
            go.Bar(
                x=scores_p2,
                y=unique_p2,
                orientation='h',
                marker_color='#d62728',
                name=f"Únicos {period_names[1]}",
                showlegend=False
            ),
            row=1, col=3
        )

    fig.update_layout(
        title="Análise de Diferenças Temporais nos Termos TF-IDF",
        height=500,
        template="plotly_white",
        margin=dict(l=10, r=10, t=100, b=50)
    )

    # Atualizar eixos
    for i in range(1, 4):
        fig.update_xaxes(title_text="Score", row=1, col=i)
        fig.update_yaxes(categoryorder='total ascending', row=1, col=i)

    return fig

def create_ranking_evolution_chart(temporal_data: Dict) -> go.Figure:
    """Cria gráfico de evolução do ranking dos termos."""
    if not temporal_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Dados temporais insuficientes para evolução",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 16}
        )
        return fig

    shared_terms = temporal_data['shared_terms'][:20]  # Top 20 compartilhados
    period_names = temporal_data['period_names']

    if not shared_terms:
        fig = go.Figure()
        fig.add_annotation(
            text="Nenhum termo compartilhado encontrado",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 16}
        )
        return fig

    # Obter rankings
    df_p1 = temporal_data['period1_data']
    df_p2 = temporal_data['period2_data']

    # Criar mapping de posições
    rank_p1 = {term: idx + 1 for idx, term in enumerate(df_p1['term'])}
    rank_p2 = {term: idx + 1 for idx, term in enumerate(df_p2['term'])}

    # Preparar dados para visualização
    evolution_data = []
    for term in shared_terms:
        pos_p1 = rank_p1.get(term, len(df_p1) + 1)
        pos_p2 = rank_p2.get(term, len(df_p2) + 1)
        change = pos_p1 - pos_p2  # Positivo = subiu no ranking

        evolution_data.append({
            'term': term,
            'position_p1': pos_p1,
            'position_p2': pos_p2,
            'change': change,
            'status': 'Subiu' if change > 0 else ('Desceu' if change < 0 else 'Estável')
        })

    df_evolution = pd.DataFrame(evolution_data)

    # Criar gráfico de evolução
    fig = go.Figure()

    # Linhas conectando posições
    for _, row in df_evolution.iterrows():
        color = '#2ca02c' if row['change'] > 0 else ('#d62728' if row['change'] < 0 else '#1f77b4')
        opacity = 0.7

        fig.add_trace(go.Scatter(
            x=[1, 2],
            y=[row['position_p1'], row['position_p2']],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
            name=row['term'],
            hovertemplate=(
                f"<b>{row['term']}</b><br>" +
                f"{period_names[0]}: Posição {row['position_p1']}<br>" +
                f"{period_names[1]}: Posição {row['position_p2']}<br>" +
                f"Mudança: {'+' if row['change'] > 0 else ''}{row['change']}<br>" +
                "<extra></extra>"
            ),
            showlegend=False
        ))

    # Adicionar labels dos termos
    for _, row in df_evolution.iterrows():
        fig.add_annotation(
            x=2.05, y=row['position_p2'],
            text=row['term'],
            showarrow=False,
            xanchor='left',
            font=dict(size=9)
        )

    fig.update_layout(
        title=f"Evolução do Ranking dos Termos: {period_names[0]} → {period_names[1]}",
        xaxis=dict(
            tickvals=[1, 2],
            ticktext=period_names,
            range=[0.8, 2.8]
        ),
        yaxis=dict(
            title="Posição no Ranking",
            autorange='reversed',  # Posição 1 no topo
            gridcolor='lightgray'
        ),
        height=600,
        template="plotly_white",
        margin=dict(l=50, r=150, t=80, b=50)
    )

    # Adicionar legenda manual
    fig.add_annotation(
        text="🟢 Subiu | 🔴 Desceu | 🔵 Estável",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)"
    )

    return fig

def main_dashboard(df: pd.DataFrame):
    """Dashboard principal do Stage 09 - TF-IDF Vectorization."""

    # Verificar se há dados TF-IDF
    tfidf_columns = ['tfidf_score_mean', 'tfidf_score_max', 'tfidf_top_terms']
    missing_columns = [col for col in tfidf_columns if col not in df.columns]

    if missing_columns:
        st.error(f"❌ Colunas TF-IDF não encontradas: {missing_columns}")
        st.info("""
        O Stage 09 (TF-IDF Vectorization) não foi executado neste dataset.
        Execute o pipeline completo para gerar os dados TF-IDF.
        """)
        return

    # Verificar se há dados válidos
    valid_tfidf = df[
        (df['tfidf_top_terms'].notna()) &
        (df['tfidf_top_terms'] != '[]') &
        (df['tfidf_top_terms'] != '')
    ]

    if len(valid_tfidf) == 0:
        st.warning("⚠️ Nenhum dado TF-IDF válido encontrado no dataset")
        return

    st.header("📊 Stage 09 - Análise TF-IDF Vectorization")

    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Documentos com TF-IDF",
            f"{len(valid_tfidf):,}",
            f"{(len(valid_tfidf)/len(df)*100):.1f}% do total"
        )

    with col2:
        avg_score = valid_tfidf['tfidf_score_mean'].mean()
        st.metric(
            "Score Médio TF-IDF",
            f"{avg_score:.3f}",
            f"σ: {valid_tfidf['tfidf_score_mean'].std():.3f}"
        )

    with col3:
        max_score = valid_tfidf['tfidf_score_max'].max()
        st.metric(
            "Score Máximo",
            f"{max_score:.3f}",
            f"Média: {valid_tfidf['tfidf_score_max'].mean():.3f}"
        )

    with col4:
        # Estimar número total de termos únicos
        analyzer = TFIDFAnalyzer()
        term_scores = analyzer.extract_terms_from_data(valid_tfidf)
        st.metric(
            "Termos Únicos",
            f"{len(term_scores):,}",
            f"Vocabulário extraído"
        )

    # Extrair e preparar dados de termos
    with st.spinner("Processando termos TF-IDF..."):
        df_terms = analyzer.prepare_terms_for_visualization(term_scores, limit=50)
        temporal_data = analyzer.analyze_temporal_evolution(valid_tfidf)

    if df_terms.empty:
        st.error("❌ Não foi possível extrair termos válidos dos dados TF-IDF")
        return

    # Tabs para diferentes visualizações
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Top Termos",
        "🗺️ Hierarquia",
        "⏰ Evolução Temporal",
        "📈 Ranking"
    ])

    with tab1:
        st.subheader("Top 20 Termos Mais Relevantes")
        st.markdown("""
        **Score de Relevância = TF-IDF Médio × log(1 + Frequência)**

        Combina importância individual (TF-IDF) com frequência de aparição.
        Termos políticos são destacados em vermelho.
        """)

        fig_bar = create_top_terms_bar_chart(df_terms)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Tabela detalhada
        with st.expander("📋 Tabela Detalhada dos Top Termos"):
            display_df = df_terms.head(20)[['term', 'relevance_score', 'mean_score', 'frequency', 'is_political']].copy()
            display_df.columns = ['Termo', 'Score Relevância', 'TF-IDF Médio', 'Frequência', 'Político']
            st.dataframe(display_df, use_container_width=True)

    with tab2:
        st.subheader("Hierarquia de Termos (Treemap)")
        st.markdown("""
        **Visualização hierárquica dos 50 termos mais relevantes**

        Agrupados por categorias temáticas:
        - 🏛️ Política Institucional
        - 💰 Economia
        - 🏥 Políticas Sociais
        - 🇧🇷 Estado e Nação
        """)

        fig_treemap = create_terms_treemap(df_terms)
        st.plotly_chart(fig_treemap, use_container_width=True)

    with tab3:
        st.subheader("Análise de Diferenças Temporais")

        if temporal_data:
            st.markdown(f"""
            **Comparação entre: {temporal_data['period_names'][0]} vs {temporal_data['period_names'][1]}**

            - 🟠 **Termos Únicos do Primeiro Período**: Temas que perderam relevância
            - 🟢 **Termos Compartilhados**: Temas consistentes
            - 🔴 **Termos Únicos do Segundo Período**: Temas emergentes
            """)

            fig_diff = create_temporal_difference_analysis(temporal_data)
            st.plotly_chart(fig_diff, use_container_width=True)

            # Estatísticas temporais
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Termos Únicos (P1)",
                    len(temporal_data['unique_period1']),
                    "Temas que saíram"
                )
            with col2:
                st.metric(
                    "Termos Compartilhados",
                    len(temporal_data['shared_terms']),
                    "Temas persistentes"
                )
            with col3:
                st.metric(
                    "Termos Únicos (P2)",
                    len(temporal_data['unique_period2']),
                    "Temas emergentes"
                )
        else:
            st.info("📅 Dados temporais insuficientes para análise de evolução")

    with tab4:
        st.subheader("Evolução do Ranking")

        if temporal_data:
            st.markdown("""
            **Mudanças na posição dos termos entre períodos**

            Mostra como os termos mais importantes mudaram de posição no ranking TF-IDF.
            Linhas conectam as posições do mesmo termo em períodos diferentes.
            """)

            fig_ranking = create_ranking_evolution_chart(temporal_data)
            st.plotly_chart(fig_ranking, use_container_width=True)

            # Análise de mudanças
            if 'shared_terms' in temporal_data and temporal_data['shared_terms']:
                st.markdown("### 📊 Principais Mudanças no Ranking")

                # Calcular mudanças significativas
                df_p1 = temporal_data['period1_data']
                df_p2 = temporal_data['period2_data']

                rank_p1 = {term: idx + 1 for idx, term in enumerate(df_p1['term'])}
                rank_p2 = {term: idx + 1 for idx, term in enumerate(df_p2['term'])}

                changes = []
                for term in temporal_data['shared_terms'][:15]:
                    change = rank_p1.get(term, 99) - rank_p2.get(term, 99)
                    if abs(change) >= 2:  # Mudanças significativas
                        changes.append({
                            'term': term,
                            'change': change,
                            'direction': 'Subiu' if change > 0 else 'Desceu'
                        })

                if changes:
                    changes.sort(key=lambda x: abs(x['change']), reverse=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🔼 Maiores Altas**")
                        ups = [c for c in changes if c['change'] > 0][:5]
                        for change in ups:
                            st.write(f"• **{change['term']}**: +{change['change']} posições")

                    with col2:
                        st.markdown("**🔽 Maiores Quedas**")
                        downs = [c for c in changes if c['change'] < 0][:5]
                        for change in downs:
                            st.write(f"• **{change['term']}**: {change['change']} posições")
                else:
                    st.info("📊 Ranking relativamente estável entre períodos")
        else:
            st.info("📅 Dados temporais insuficientes para análise de ranking")

    # Seção de análise avançada
    st.markdown("---")
    st.subheader("🔍 Análise Técnica Avançada")

    with st.expander("📊 Estatísticas Detalhadas do TF-IDF"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Distribuição de Scores TF-IDF**")
            fig_hist = px.histogram(
                valid_tfidf,
                x='tfidf_score_mean',
                nbins=20,
                title='Distribuição dos Scores Médios TF-IDF'
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.markdown("**Top Termos por Frequência**")
            freq_terms = df_terms.nlargest(10, 'frequency')[['term', 'frequency']]
            fig_freq = px.bar(
                freq_terms,
                x='frequency',
                y='term',
                orientation='h',
                title='Termos Mais Frequentes'
            )
            fig_freq.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_freq, use_container_width=True)

    with st.expander("⚙️ Informações Técnicas"):
        st.markdown(f"""
        **Configuração TF-IDF:**
        - **Total de documentos processados**: {len(valid_tfidf):,}
        - **Vocabulário extraído**: {len(term_scores):,} termos únicos
        - **Score médio**: {avg_score:.4f}
        - **Filtragem**: Remoção de stopwords em português
        - **Peso**: Score TF-IDF × log(1 + frequência)

        **Categorização Política:**
        - Termos políticos brasileiros priorizados
        - Filtragem por relevância mínima
        - Análise temporal quando disponível

        **Qualidade dos Dados:**
        - Documentos válidos: {(len(valid_tfidf)/len(df)*100):.1f}%
        - Range de scores: {valid_tfidf['tfidf_score_mean'].min():.3f} - {valid_tfidf['tfidf_score_mean'].max():.3f}
        """)

if __name__ == "__main__":
    # Teste básico
    st.title("🔬 Stage 09 - TF-IDF Dashboard (Teste)")
    st.write("Execute através da página principal do dashboard")