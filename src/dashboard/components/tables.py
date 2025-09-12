"""
Componentes de tabelas interativas para o dashboard digiNEV
Tabelas configuráveis e reutilizáveis para exibição de dados
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np

def create_data_table(data: pd.DataFrame, title: str = "", 
                     columns: Optional[List[str]] = None,
                     max_rows: int = 100, sortable: bool = True,
                     searchable: bool = True) -> None:
    """
    Cria tabela de dados interativa
    
    Args:
        data: DataFrame com os dados
        title: Título da tabela
        columns: Colunas específicas para exibir
        max_rows: Número máximo de linhas
        sortable: Permitir ordenação
        searchable: Permitir busca
    """
    if data.empty:
        st.info("Nenhum dado disponível para exibição")
        return
    
    if title:
        st.subheader(title)
    
    # Seleção de colunas
    if columns:
        available_columns = [col for col in columns if col in data.columns]
        display_data = data[available_columns] if available_columns else data
    else:
        display_data = data
    
    # Controles da tabela
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if searchable and len(display_data) > 10:
            search_term = st.text_input("🔍 Buscar na tabela:", key=f"search_{title}")
            if search_term:
                # Buscar em colunas de texto
                text_columns = display_data.select_dtypes(include=['object']).columns
                mask = pd.Series([False] * len(display_data))
                
                for col in text_columns:
                    mask |= display_data[col].astype(str).str.contains(
                        search_term, case=False, na=False
                    )
                
                display_data = display_data[mask]
    
    with col2:
        if sortable and len(display_data.columns) > 1:
            sort_column = st.selectbox(
                "📊 Ordenar por:",
                options=list(display_data.columns),
                key=f"sort_{title}"
            )
        else:
            sort_column = None
    
    with col3:
        if len(display_data) > max_rows:
            show_rows = st.selectbox(
                "📋 Mostrar linhas:",
                options=[25, 50, 100, 200],
                index=2,
                key=f"rows_{title}"
            )
        else:
            show_rows = len(display_data)
    
    # Aplicar ordenação
    if sort_column and sort_column in display_data.columns:
        try:
            if display_data[sort_column].dtype in ['object', 'string']:
                display_data = display_data.sort_values(sort_column, ascending=True)
            else:
                display_data = display_data.sort_values(sort_column, ascending=False)
        except:
            pass  # Manter ordem original se houver erro
    
    # Exibir tabela
    st.dataframe(
        display_data.head(show_rows),
        use_container_width=True,
        height=min(len(display_data.head(show_rows)) * 35 + 38, 600)
    )
    
    # Informações da tabela
    if len(display_data) > show_rows:
        st.info(f"Mostrando {show_rows} de {len(display_data)} registros")

def create_summary_table(data: pd.DataFrame, title: str = "Resumo Estatístico",
                        columns: Optional[List[str]] = None) -> None:
    """
    Cria tabela de resumo estatístico
    
    Args:
        data: DataFrame com os dados
        title: Título da tabela
        columns: Colunas específicas para resumir
    """
    if data.empty:
        st.info("Nenhum dado disponível para resumo")
        return
    
    st.subheader(title)
    
    # Selecionar colunas numéricas
    if columns:
        numeric_data = data[columns].select_dtypes(include=[np.number])
    else:
        numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        st.info("Nenhuma coluna numérica encontrada para resumo")
        return
    
    # Calcular estatísticas
    summary_stats = pd.DataFrame({
        'Coluna': numeric_data.columns,
        'Contagem': numeric_data.count().values,
        'Média': numeric_data.mean().round(3).values,
        'Mediana': numeric_data.median().round(3).values,
        'Desvio Padrão': numeric_data.std().round(3).values,
        'Mínimo': numeric_data.min().round(3).values,
        'Máximo': numeric_data.max().round(3).values,
        'Valores Únicos': numeric_data.nunique().values
    })
    
    st.dataframe(summary_stats, use_container_width=True)

def create_frequency_table(data: pd.DataFrame, column: str, 
                         title: str = "", top_n: int = 20) -> None:
    """
    Cria tabela de frequência
    
    Args:
        data: DataFrame com os dados
        column: Coluna para análise de frequência
        title: Título da tabela
        top_n: Número de valores mais frequentes
    """
    if column not in data.columns:
        st.error(f"Coluna {column} não encontrada")
        return
    
    if not title:
        title = f"Frequência - {column.replace('_', ' ').title()}"
    
    st.subheader(title)
    
    # Calcular frequências
    value_counts = data[column].value_counts().head(top_n)
    
    # Criar DataFrame para exibição
    freq_table = pd.DataFrame({
        'Valor': value_counts.index,
        'Frequência': value_counts.values,
        'Porcentagem': (value_counts.values / len(data) * 100).round(2)
    })
    
    # Adicionar coluna de porcentagem acumulada
    freq_table['% Acumulada'] = freq_table['Porcentagem'].cumsum().round(2)
    
    st.dataframe(freq_table, use_container_width=True)
    
    # Mostrar total de valores únicos se truncado
    total_unique = data[column].nunique()
    if total_unique > top_n:
        st.info(f"Mostrando top {top_n} de {total_unique} valores únicos")

def create_cross_table(data: pd.DataFrame, row_column: str, col_column: str,
                      title: str = "", normalize: str = None) -> None:
    """
    Cria tabela cruzada (crosstab)
    
    Args:
        data: DataFrame com os dados
        row_column: Coluna para linhas
        col_column: Coluna para colunas
        title: Título da tabela
        normalize: Normalização ('index', 'columns', 'all', None)
    """
    if row_column not in data.columns or col_column not in data.columns:
        st.error("Colunas não encontradas para tabela cruzada")
        return
    
    if not title:
        title = f"Tabela Cruzada: {row_column} vs {col_column}"
    
    st.subheader(title)
    
    try:
        # Criar tabela cruzada
        cross_tab = pd.crosstab(
            data[row_column], 
            data[col_column],
            normalize=normalize,
            margins=True
        )
        
        # Formatar valores se normalizado
        if normalize:
            cross_tab = cross_tab.round(3)
        
        st.dataframe(cross_tab, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao criar tabela cruzada: {e}")

def create_pivot_table(data: pd.DataFrame, values: str, index: str,
                      columns: Optional[str] = None, aggfunc: str = 'mean',
                      title: str = "") -> None:
    """
    Cria tabela dinâmica (pivot table)
    
    Args:
        data: DataFrame com os dados
        values: Coluna de valores para agregação
        index: Coluna para índice
        columns: Coluna para colunas (opcional)
        aggfunc: Função de agregação
        title: Título da tabela
    """
    required_columns = [values, index]
    if columns:
        required_columns.append(columns)
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Colunas não encontradas: {missing_columns}")
        return
    
    if not title:
        title = f"Tabela Dinâmica: {values} por {index}"
        if columns:
            title += f" e {columns}"
    
    st.subheader(title)
    
    try:
        # Criar tabela dinâmica
        pivot = pd.pivot_table(
            data=data,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=0
        )
        
        # Formatar valores
        if aggfunc in ['mean', 'std']:
            pivot = pivot.round(3)
        
        st.dataframe(pivot, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao criar tabela dinâmica: {e}")

def create_comparison_table(data_dict: Dict[str, pd.DataFrame], 
                          metric_column: str, title: str = "Comparação") -> None:
    """
    Cria tabela de comparação entre diferentes datasets
    
    Args:
        data_dict: Dicionário com {nome: DataFrame}
        metric_column: Coluna para comparar
        title: Título da tabela
    """
    st.subheader(title)
    
    comparison_data = []
    
    for name, df in data_dict.items():
        if metric_column in df.columns:
            stats = {
                'Dataset': name,
                'Registros': len(df),
                'Média': df[metric_column].mean().round(3),
                'Mediana': df[metric_column].median().round(3),
                'Desvio Padrão': df[metric_column].std().round(3),
                'Mínimo': df[metric_column].min().round(3),
                'Máximo': df[metric_column].max().round(3)
            }
            comparison_data.append(stats)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info(f"Coluna {metric_column} não encontrada nos datasets")

def create_correlation_table(data: pd.DataFrame, title: str = "Matriz de Correlação",
                           columns: Optional[List[str]] = None,
                           threshold: float = 0.5) -> None:
    """
    Cria tabela de correlação
    
    Args:
        data: DataFrame com os dados
        title: Título da tabela
        columns: Colunas específicas para correlação
        threshold: Threshold para destacar correlações
    """
    st.subheader(title)
    
    # Selecionar colunas numéricas
    if columns:
        numeric_data = data[columns].select_dtypes(include=[np.number])
    else:
        numeric_data = data.select_dtypes(include=[np.number])
    
    if len(numeric_data.columns) < 2:
        st.info("Pelo menos 2 colunas numéricas são necessárias para correlação")
        return
    
    # Calcular correlação
    correlation_matrix = numeric_data.corr().round(3)
    
    # Destacar correlações altas
    def highlight_correlation(val):
        if pd.isna(val) or val == 1.0:
            return ''
        elif abs(val) >= threshold:
            return 'background-color: #ffcccc' if val > 0 else 'background-color: #ccccff'
        else:
            return ''
    
    styled_corr = correlation_matrix.style.applymap(highlight_correlation)
    st.dataframe(styled_corr, use_container_width=True)
    
    # Mostrar correlações mais altas
    if threshold < 1.0:
        # Encontrar pares com alta correlação
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'Variável 1': correlation_matrix.columns[i],
                        'Variável 2': correlation_matrix.columns[j],
                        'Correlação': corr_val
                    })
        
        if high_corr_pairs:
            st.write(f"**Correlações ≥ {threshold}:**")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('Correlação', key=abs, ascending=False)
            st.dataframe(high_corr_df, use_container_width=True)

def create_missing_data_table(data: pd.DataFrame, title: str = "Análise de Dados Ausentes") -> None:
    """
    Cria tabela de análise de dados ausentes
    
    Args:
        data: DataFrame com os dados
        title: Título da tabela
    """
    st.subheader(title)
    
    # Calcular dados ausentes
    missing_data = pd.DataFrame({
        'Coluna': data.columns,
        'Valores Ausentes': data.isnull().sum().values,
        'Porcentagem Ausente': (data.isnull().sum().values / len(data) * 100).round(2),
        'Valores Presentes': data.notnull().sum().values,
        'Tipo de Dados': data.dtypes.values
    })
    
    # Ordenar por porcentagem de dados ausentes
    missing_data = missing_data.sort_values('Porcentagem Ausente', ascending=False)
    
    # Destacar colunas com muitos dados ausentes
    def highlight_missing(row):
        if row['Porcentagem Ausente'] > 50:
            return ['background-color: #ffcccc'] * len(row)
        elif row['Porcentagem Ausente'] > 20:
            return ['background-color: #fff2cc'] * len(row)
        else:
            return [''] * len(row)
    
    styled_missing = missing_data.style.apply(highlight_missing, axis=1)
    st.dataframe(styled_missing, use_container_width=True)
    
    # Resumo
    total_missing = missing_data['Valores Ausentes'].sum()
    total_cells = len(data) * len(data.columns)
    overall_missing_pct = (total_missing / total_cells * 100).round(2)
    
    st.info(f"**Resumo**: {overall_missing_pct}% de dados ausentes no total ({total_missing:,} de {total_cells:,} células)")

def create_data_quality_table(data: pd.DataFrame, title: str = "Qualidade dos Dados") -> None:
    """
    Cria tabela de qualidade dos dados
    
    Args:
        data: DataFrame com os dados
        title: Título da tabela
    """
    st.subheader(title)
    
    quality_metrics = []
    
    for column in data.columns:
        # Métricas básicas
        total_values = len(data)
        non_null_values = data[column].notnull().sum()
        null_values = data[column].isnull().sum()
        unique_values = data[column].nunique()
        
        # Calcular score de qualidade
        completeness = non_null_values / total_values
        uniqueness = unique_values / non_null_values if non_null_values > 0 else 0
        
        # Score geral (média ponderada)
        quality_score = (completeness * 0.7 + min(uniqueness, 1.0) * 0.3)
        
        quality_metrics.append({
            'Coluna': column,
            'Tipo': str(data[column].dtype),
            'Completude': f"{completeness:.1%}",
            'Valores Únicos': unique_values,
            'Taxa de Unicidade': f"{uniqueness:.1%}",
            'Score de Qualidade': f"{quality_score:.3f}",
            'Status': '✅' if quality_score > 0.8 else '⚠️' if quality_score > 0.5 else '❌'
        })
    
    quality_df = pd.DataFrame(quality_metrics)
    st.dataframe(quality_df, use_container_width=True)
    
    # Resumo geral
    avg_quality = pd.to_numeric(quality_df['Score de Qualidade']).mean()
    st.metric("Score Médio de Qualidade", f"{avg_quality:.3f}")

def create_exportable_table(data: pd.DataFrame, filename_prefix: str = "dados",
                           title: str = "Exportar Dados") -> None:
    """
    Cria tabela com opções de exportação
    
    Args:
        data: DataFrame com os dados
        filename_prefix: Prefixo do arquivo
        title: Título da seção
    """
    st.subheader(title)
    
    # Mostrar preview dos dados
    st.write("**Preview dos dados:**")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Opções de exportação
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Exportar CSV", key=f"export_csv_{filename_prefix}"):
            csv = data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"{filename_prefix}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Exportar Excel", key=f"export_excel_{filename_prefix}"):
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data.to_excel(writer, index=False, sheet_name='Dados')
            
            st.download_button(
                label="⬇️ Download Excel",
                data=output.getvalue(),
                file_name=f"{filename_prefix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("📋 Copiar JSON", key=f"copy_json_{filename_prefix}"):
            json_data = data.to_json(orient='records', indent=2)
            st.code(json_data[:1000] + "..." if len(json_data) > 1000 else json_data)
            st.info("Dados copiados para área de transferência (preview mostrado acima)")