"""
Stage 07 Linguistic Processing Dashboard
Análise Linguística com Named Entity Recognition (NER)

Foco: Visualização de entidades nomeadas em discurso político brasileiro
- Word cloud de entidades por tipo (PERSON, ORG, GPE)
- Rede de conexões entre entidades políticas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import re
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar disponibilidade de bibliotecas
try:
    import spacy
    SPACY_AVAILABLE = True

    # Carregar modelo spaCy
    try:
        nlp = spacy.load("pt_core_news_lg")
        SPACY_MODEL_AVAILABLE = True
    except OSError:
        try:
            nlp = spacy.load("pt_core_news_sm")
            SPACY_MODEL_AVAILABLE = True
        except OSError:
            SPACY_MODEL_AVAILABLE = False
            nlp = None
            logger.warning("Nenhum modelo spaCy português encontrado")

except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL_AVAILABLE = False
    nlp = None
    logger.warning("spaCy não disponível")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None
    logger.warning("WordCloud não disponível")

class LinguisticAnalyzer:
    """Analisador linguístico com foco em entidades nomeadas."""

    def __init__(self):
        self.nlp = nlp if SPACY_MODEL_AVAILABLE else None

        # Categorias de entidades relevantes para análise política brasileira
        # Usando labels do modelo português: PER, ORG, LOC
        self.entity_types = {
            'PER': 'Pessoas',
            'ORG': 'Organizações',
            'LOC': 'Locais Geopolíticos'
        }

        # Mapeamento para compatibilidade com labels internacionais
        self.label_mapping = {
            'PER': 'PERSON',
            'ORG': 'ORG',
            'LOC': 'GPE'
        }

        # Léxico de entidades políticas brasileiras importantes
        # Usando as mesmas categorias, mas com mapeamento interno para PER/ORG/LOC
        self.political_entities = {
            'PER': {
                'bolsonaro', 'lula', 'dilma', 'temer', 'haddad', 'marina silva',
                'ciro gomes', 'geraldo alckmin', 'moro', 'mandetta', 'guedes',
                'mourão', 'gleisi', 'doria', 'flávio bolsonaro', 'eduardo bolsonaro',
                'carlos bolsonaro', 'damares', 'weintraub', 'sergio moro',
                'jair bolsonaro', 'luiz inácio lula da silva'
            },
            'ORG': {
                'pt', 'psdb', 'psl', 'pl', 'mdb', 'dem', 'psol', 'pco',
                'stf', 'tse', 'tcu', 'pf', 'mpf', 'agu', 'congresso',
                'senado', 'câmara', 'planalto', 'anvisa', 'ibge', 'fiocruz',
                'petrobras', 'caixa', 'banco do brasil', 'bndes', 'supremo tribunal federal',
                'tribunal superior eleitoral', 'tribunal de contas da união'
            },
            'LOC': {
                'brasil', 'brasília', 'são paulo', 'rio de janeiro', 'minas gerais',
                'bahia', 'paraná', 'rio grande do sul', 'pernambuco', 'ceará',
                'pará', 'santa catarina', 'goiás', 'maranhão', 'amazonas',
                'espírito santo', 'paraíba', 'mato grosso', 'rio grande do norte',
                'alagoas', 'piauí', 'distrito federal', 'mato grosso do sul',
                'sergipe', 'rondônia', 'acre', 'amapá', 'roraima', 'tocantins'
            }
        }

    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extrair entidades nomeadas de um texto usando spaCy."""
        if not self.nlp or pd.isna(text):
            return {entity_type: [] for entity_type in self.entity_types.keys()}

        try:
            # Processar texto com limite para performance
            doc = self.nlp(str(text)[:2000])

            entities = {entity_type: [] for entity_type in self.entity_types.keys()}

            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    # Normalizar texto da entidade
                    entity_text = ent.text.lower().strip()

                    # Filtrar entidades muito curtas ou irrelevantes
                    if len(entity_text) > 2:
                        entities[ent.label_].append(entity_text)

            # Adicionar extração heurística baseada no léxico político
            text_lower = str(text).lower()
            for entity_type, political_entities in self.political_entities.items():
                for political_entity in political_entities:
                    if political_entity in text_lower:
                        if political_entity not in entities[entity_type]:
                            entities[entity_type].append(political_entity)

            return entities

        except Exception as e:
            logger.warning(f"Erro ao extrair entidades: {e}")
            return {entity_type: [] for entity_type in self.entity_types.keys()}

    def extract_entities_from_dataframe(self, df: pd.DataFrame, text_column: str = 'body') -> pd.DataFrame:
        """Extrair entidades de todo o dataframe."""
        if text_column not in df.columns:
            logger.warning(f"Coluna '{text_column}' não encontrada")
            return df

        logger.info("Extraindo entidades nomeadas...")

        # Extrair entidades para cada linha
        entities_data = []
        for idx, row in df.iterrows():
            entities = self.extract_entities_from_text(row[text_column])
            entities_data.append(entities)

        # Adicionar colunas de entidades ao dataframe
        for entity_type in self.entity_types.keys():
            df[f'entities_{entity_type.lower()}'] = [
                data[entity_type] for data in entities_data
            ]
            df[f'entities_{entity_type.lower()}_count'] = [
                len(data[entity_type]) for data in entities_data
            ]

        return df

    def get_entity_frequencies(self, df: pd.DataFrame) -> Dict[str, Counter]:
        """Calcular frequências de entidades por tipo."""
        frequencies = {}

        for entity_type in self.entity_types.keys():
            column_name = f'entities_{entity_type.lower()}'
            if column_name in df.columns:
                all_entities = []
                for entities_list in df[column_name]:
                    if isinstance(entities_list, list):
                        all_entities.extend(entities_list)

                frequencies[entity_type] = Counter(all_entities)
            else:
                frequencies[entity_type] = Counter()

        return frequencies

    def filter_political_entities(self, frequencies: Dict[str, Counter]) -> Dict[str, Counter]:
        """Filtrar apenas entidades politicamente relevantes."""
        filtered = {}

        for entity_type, counter in frequencies.items():
            filtered_counter = Counter()
            political_set = self.political_entities.get(entity_type, set())

            for entity, count in counter.items():
                # Incluir se está no léxico político ou tem alta frequência
                if entity in political_set or count >= 3:
                    filtered_counter[entity] = count

            filtered[entity_type] = filtered_counter

        return filtered

    def build_entity_network(self, df: pd.DataFrame, min_cooccurrence: int = 2) -> nx.Graph:
        """Construir rede de co-ocorrência de entidades."""
        G = nx.Graph()

        # Coletar co-ocorrências
        cooccurrences = defaultdict(int)

        for _, row in df.iterrows():
            # Coletar todas as entidades de uma mensagem
            message_entities = set()

            for entity_type in self.entity_types.keys():
                column_name = f'entities_{entity_type.lower()}'
                if column_name in df.columns and isinstance(row[column_name], list):
                    for entity in row[column_name]:
                        # Adicionar tipo à entidade para diferenciação
                        message_entities.add(f"{entity} ({entity_type})")

            # Calcular co-ocorrências entre entidades na mesma mensagem
            entities_list = list(message_entities)
            for i in range(len(entities_list)):
                for j in range(i + 1, len(entities_list)):
                    pair = tuple(sorted([entities_list[i], entities_list[j]]))
                    cooccurrences[pair] += 1

        # Adicionar nós e arestas ao grafo
        for (entity1, entity2), weight in cooccurrences.items():
            if weight >= min_cooccurrence:
                G.add_edge(entity1, entity2, weight=weight)

        return G

def create_entity_wordcloud(frequencies: Dict[str, Counter], entity_type: str):
    """Criar word cloud para um tipo específico de entidade."""
    if not WORDCLOUD_AVAILABLE:
        return None

    entity_counter = frequencies.get(entity_type, Counter())

    if not entity_counter:
        return None

    # Configuração do WordCloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=50,
        relative_scaling=0.5,
        random_state=42
    ).generate_from_frequencies(entity_counter)

    return wordcloud

def create_network_visualization(G: nx.Graph, max_nodes: int = 50) -> go.Figure:
    """Criar visualização de rede com Plotly."""
    if len(G.nodes()) == 0:
        # Retornar gráfico vazio se não há dados
        fig = go.Figure()
        fig.update_layout(
            title="Nenhuma conexão entre entidades encontrada",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # Limitar número de nós para performance
    if len(G.nodes()) > max_nodes:
        # Manter apenas os nós com maior centralidade
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([node for node, _ in top_nodes])

    # Calcular layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

    # Preparar dados para Plotly
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2].get('weight', 1))

    # Criar trace para arestas
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='lightgray'),
        hoverinfo='none',
        mode='lines'
    )

    # Preparar dados dos nós
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Extrair tipo de entidade da string
        if '(PERSON)' in node:
            color = 'red'
            size = 15
        elif '(ORG)' in node:
            color = 'blue'
            size = 12
        elif '(GPE)' in node:
            color = 'green'
            size = 10
        else:
            color = 'gray'
            size = 8

        node_colors.append(color)
        node_sizes.append(size)

        # Informações do nó
        adjacencies = list(G.neighbors(node))
        node_info = f"{node}<br>Conexões: {len(adjacencies)}"
        node_text.append(node_info)

    # Criar trace para nós
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node.split(' (')[0] for node in G.nodes()],  # Mostrar apenas o nome
        textposition="middle center",
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='white')
        )
    )

    # Criar figura
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                        title='Rede de Entidades Políticas',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Vermelho: Pessoas | Azul: Organizações | Verde: Locais",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(size=10, color="gray")
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    return fig

def main_dashboard(df: pd.DataFrame):
    """Dashboard principal de análise linguística."""
    st.title("🔤 Stage 07 - Análise Linguística")
    st.markdown("**Processamento linguístico com Named Entity Recognition (NER)**")

    # Verificar disponibilidade de recursos
    if not SPACY_MODEL_AVAILABLE:
        st.error("⚠️ Modelo spaCy português não encontrado. Instale com: python -m spacy download pt_core_news_lg")
        return

    # Sidebar com configurações
    st.sidebar.header("Configurações")

    # Seleção de coluna de texto
    text_columns = [col for col in df.columns if df[col].dtype == 'object' and 'text' in col.lower()]
    if 'body' in df.columns:
        text_columns.insert(0, 'body')
    if 'normalized_text' in df.columns:
        text_columns.insert(0, 'normalized_text')

    text_column = st.sidebar.selectbox(
        "Coluna de texto para análise:",
        text_columns,
        index=0 if text_columns else None
    )

    if not text_column:
        st.error("Nenhuma coluna de texto encontrada no dataset")
        return

    # Filtros
    min_entity_freq = st.sidebar.slider(
        "Frequência mínima de entidades:",
        min_value=1, max_value=10, value=2
    )

    max_entities_wordcloud = st.sidebar.slider(
        "Máximo de entidades no word cloud:",
        min_value=20, max_value=100, value=50
    )

    min_cooccurrence = st.sidebar.slider(
        "Mínimo de co-ocorrências na rede:",
        min_value=1, max_value=5, value=2
    )

    only_political = st.sidebar.checkbox(
        "Mostrar apenas entidades políticas relevantes",
        value=True
    )

    # Inicializar analisador
    analyzer = LinguisticAnalyzer()

    # Cache para evitar reprocessamento
    @st.cache_data
    def process_entities(df_sample, text_col):
        return analyzer.extract_entities_from_dataframe(df_sample, text_col)

    # Amostra de dados para performance
    sample_size = min(1000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    with st.spinner("Extraindo entidades nomeadas..."):
        df_with_entities = process_entities(df_sample, text_column)

    # Calcular frequências
    frequencies = analyzer.get_entity_frequencies(df_with_entities)

    if only_political:
        frequencies = analyzer.filter_political_entities(frequencies)

    # Métricas gerais
    st.subheader("📊 Visão Geral")

    col1, col2, col3, col4 = st.columns(4)

    total_entities = sum(sum(counter.values()) for counter in frequencies.values())
    unique_entities = sum(len(counter) for counter in frequencies.values())

    with col1:
        st.metric("Total de Entidades", total_entities)

    with col2:
        st.metric("Entidades Únicas", unique_entities)

    with col3:
        avg_entities = df_with_entities[[col for col in df_with_entities.columns if col.endswith('_count')]].sum(axis=1).mean()
        st.metric("Média por Mensagem", f"{avg_entities:.1f}")

    with col4:
        st.metric("Mensagens Analisadas", len(df_sample))

    # Distribuição por tipo
    st.subheader("📈 Distribuição por Tipo de Entidade")

    entity_type_data = []
    for entity_type, type_name in analyzer.entity_types.items():
        count = sum(frequencies[entity_type].values())
        entity_type_data.append({
            'Tipo': type_name,
            'Quantidade': count,
            'Código': entity_type
        })

    if entity_type_data:
        df_types = pd.DataFrame(entity_type_data)

        fig_bar = px.bar(
            df_types,
            x='Tipo',
            y='Quantidade',
            title='Quantidade de Entidades por Tipo',
            color='Tipo',
            color_discrete_map={
                'Pessoas': '#ff6b6b',
                'Organizações': '#4ecdc4',
                'Locais Geopolíticos': '#45b7d1'
            }
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Word clouds por tipo
    st.subheader("☁️ Nuvem de Palavras por Tipo")

    tabs = st.tabs([analyzer.entity_types[et] for et in analyzer.entity_types.keys()])

    for i, (entity_type, type_name) in enumerate(analyzer.entity_types.items()):
        with tabs[i]:
            entity_counter = frequencies[entity_type]

            if entity_counter:
                # Filtrar por frequência mínima
                filtered_counter = Counter({
                    entity: count for entity, count in entity_counter.items()
                    if count >= min_entity_freq
                })

                if filtered_counter and WORDCLOUD_AVAILABLE:
                    wordcloud = create_entity_wordcloud({entity_type: filtered_counter}, entity_type)
                    if wordcloud:
                        st.image(wordcloud.to_array(), use_column_width=True)
                    else:
                        st.info(f"Sem dados suficientes para word cloud de {type_name}")

                # Tabela das entidades mais frequentes
                st.subheader(f"Top {type_name}")
                if filtered_counter:
                    top_entities = filtered_counter.most_common(10)
                    df_top = pd.DataFrame(top_entities, columns=['Entidade', 'Frequência'])
                    st.dataframe(df_top, use_container_width=True)
                else:
                    st.info(f"Nenhuma entidade de {type_name} encontrada com frequência >= {min_entity_freq}")
            else:
                st.info(f"Nenhuma entidade de {type_name} encontrada")

    # Rede de entidades
    st.subheader("🕸️ Rede de Conexões entre Entidades")

    with st.spinner("Construindo rede de entidades..."):
        G = analyzer.build_entity_network(df_with_entities, min_cooccurrence)

    if len(G.nodes()) > 0:
        st.info(f"Rede com {len(G.nodes())} entidades e {len(G.edges())} conexões")

        fig_network = create_network_visualization(G, max_nodes=50)
        st.plotly_chart(fig_network, use_container_width=True)

        # Estatísticas da rede
        st.subheader("📊 Estatísticas da Rede")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Número de Nós", len(G.nodes()))

        with col2:
            st.metric("Número de Arestas", len(G.edges()))

        with col3:
            if len(G.nodes()) > 0:
                density = nx.density(G)
                st.metric("Densidade", f"{density:.3f}")

        # Top entidades por centralidade
        if len(G.nodes()) > 0:
            centrality = nx.degree_centrality(G)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

            st.subheader("🎯 Entidades Mais Centrais")
            df_central = pd.DataFrame(top_central, columns=['Entidade', 'Centralidade'])
            df_central['Entidade'] = df_central['Entidade'].str.replace(r' \([A-Z]+\)', '', regex=True)
            st.dataframe(df_central, use_container_width=True)

    else:
        st.info(f"Não há co-ocorrências suficientes (mínimo: {min_cooccurrence}) para construir a rede")

    # Análise temporal (se disponível)
    if 'datetime' in df_with_entities.columns or 'date' in df_with_entities.columns:
        st.subheader("📅 Análise Temporal de Entidades")

        date_col = 'datetime' if 'datetime' in df_with_entities.columns else 'date'

        try:
            df_with_entities[date_col] = pd.to_datetime(df_with_entities[date_col])
            df_with_entities['date_only'] = df_with_entities[date_col].dt.date

            # Evolução temporal das entidades
            temporal_data = []
            for date in df_with_entities['date_only'].unique():
                date_df = df_with_entities[df_with_entities['date_only'] == date]
                date_frequencies = analyzer.get_entity_frequencies(date_df)

                for entity_type, counter in date_frequencies.items():
                    total_count = sum(counter.values())
                    temporal_data.append({
                        'Data': date,
                        'Tipo': analyzer.entity_types[entity_type],
                        'Quantidade': total_count
                    })

            if temporal_data:
                df_temporal = pd.DataFrame(temporal_data)

                fig_temporal = px.line(
                    df_temporal,
                    x='Data',
                    y='Quantidade',
                    color='Tipo',
                    title='Evolução Temporal das Entidades',
                    color_discrete_map={
                        'Pessoas': '#ff6b6b',
                        'Organizações': '#4ecdc4',
                        'Locais Geopolíticos': '#45b7d1'
                    }
                )
                st.plotly_chart(fig_temporal, use_container_width=True)

        except Exception as e:
            st.warning(f"Erro na análise temporal: {e}")

if __name__ == "__main__":
    # Teste básico
    st.set_page_config(
        page_title="Stage 07 - Análise Linguística",
        page_icon="🔤",
        layout="wide"
    )

    # Carregar dados de teste
    try:
        df = pd.read_csv("data/controlled_test_100.csv")
        main_dashboard(df)
    except FileNotFoundError:
        st.error("Arquivo de dados não encontrado. Execute o dashboard através do sistema principal.")