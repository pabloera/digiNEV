"""
Componentes de exportação para o dashboard digiNEV
Funcionalidades de exportação de dados e visualizações
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from datetime import datetime
import json
from io import BytesIO

def create_csv_export(data: pd.DataFrame, filename: str = "dados") -> None:
    """
    Cria botão de exportação CSV
    
    Args:
        data: DataFrame para exportar
        filename: Nome base do arquivo
    """
    if data.empty:
        st.warning("Nenhum dado disponível para exportação")
        return
    
    csv = data.to_csv(index=False)
    
    st.download_button(
        label="📄 Baixar CSV",
        data=csv,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def create_excel_export(data: pd.DataFrame, filename: str = "dados") -> None:
    """
    Cria botão de exportação Excel
    
    Args:
        data: DataFrame para exportar
        filename: Nome base do arquivo
    """
    if data.empty:
        st.warning("Nenhum dado disponível para exportação")
        return
    
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Dados')
        
        st.download_button(
            label="📊 Baixar Excel",
            data=output.getvalue(),
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Erro ao criar arquivo Excel: {e}")

def create_json_export(data: pd.DataFrame, filename: str = "dados") -> None:
    """
    Cria botão de exportação JSON
    
    Args:
        data: DataFrame para exportar
        filename: Nome base do arquivo
    """
    if data.empty:
        st.warning("Nenhum dado disponível para exportação")
        return
    
    json_data = data.to_json(orient='records', indent=2)
    
    st.download_button(
        label="📋 Baixar JSON",
        data=json_data,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

def create_chart_export(fig: go.Figure, filename: str = "grafico") -> None:
    """
    Cria botão de exportação de gráfico
    
    Args:
        fig: Figura do Plotly
        filename: Nome base do arquivo
    """
    try:
        # Exportar como HTML interativo
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        st.download_button(
            label="📈 Baixar Gráfico (HTML)",
            data=html_str,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Erro ao exportar gráfico: {e}")

def create_full_export_section(data: pd.DataFrame, filename: str = "analise") -> None:
    """
    Cria seção completa de exportação com múltiplas opções
    
    Args:
        data: DataFrame para exportar
        filename: Nome base dos arquivos
    """
    st.subheader("📥 Exportar Dados")
    
    if data.empty:
        st.info("Nenhum dado disponível para exportação")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_csv_export(data, filename)
    
    with col2:
        create_excel_export(data, filename)
    
    with col3:
        create_json_export(data, filename)
    
    # Informações sobre os dados
    st.info(f"📊 **Dados para exportação**: {len(data):,} registros, {len(data.columns)} colunas")

def create_filtered_export(original_data: pd.DataFrame, filtered_data: pd.DataFrame, 
                         filename: str = "dados_filtrados") -> None:
    """
    Cria exportação com dados originais e filtrados
    
    Args:
        original_data: DataFrame original
        filtered_data: DataFrame filtrado
        filename: Nome base do arquivo
    """
    st.subheader("📥 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dados Filtrados**")
        create_csv_export(filtered_data, f"{filename}_filtrado")
        st.info(f"📊 {len(filtered_data):,} registros filtrados")
    
    with col2:
        st.write("**Dados Originais**")
        create_csv_export(original_data, f"{filename}_completo")
        st.info(f"📊 {len(original_data):,} registros totais")

def create_summary_export(summary_data: dict, filename: str = "resumo") -> None:
    """
    Cria exportação de dados de resumo
    
    Args:
        summary_data: Dicionário com dados de resumo
        filename: Nome base do arquivo
    """
    st.subheader("📥 Exportar Resumo")
    
    # Converter resumo para JSON
    json_summary = json.dumps(summary_data, indent=2, ensure_ascii=False, default=str)
    
    st.download_button(
        label="📋 Baixar Resumo (JSON)",
        data=json_summary,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
    
    # Mostrar preview
    with st.expander("👀 Preview do Resumo"):
        st.json(summary_data)

def create_batch_export(data_dict: dict, base_filename: str = "analise_completa") -> None:
    """
    Cria exportação em lote de múltiplos datasets
    
    Args:
        data_dict: Dicionário com {nome: DataFrame}
        base_filename: Nome base dos arquivos
    """
    st.subheader("📥 Exportação em Lote")
    
    if not data_dict:
        st.info("Nenhum dataset disponível para exportação")
        return
    
    # Criar Excel com múltiplas abas
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for name, df in data_dict.items():
                if not df.empty:
                    # Limitar nome da aba a 31 caracteres (limite do Excel)
                    sheet_name = name[:31] if len(name) > 31 else name
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        st.download_button(
            label="📊 Baixar Análise Completa (Excel)",
            data=output.getvalue(),
            file_name=f"{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # Mostrar resumo
        total_records = sum(len(df) for df in data_dict.values() if not df.empty)
        st.info(f"📊 **Total**: {len(data_dict)} datasets, {total_records:,} registros")
        
    except Exception as e:
        st.error(f"Erro ao criar arquivo Excel: {e}")

def create_report_export(title: str, summary: dict, data: pd.DataFrame, 
                        insights: list, filename: str = "relatorio") -> None:
    """
    Cria exportação de relatório completo
    
    Args:
        title: Título do relatório
        summary: Resumo executivo
        data: Dados principais
        insights: Lista de insights
        filename: Nome base do arquivo
    """
    st.subheader("📋 Exportar Relatório Completo")
    
    # Gerar relatório em texto
    report_content = f"""
# {title}
Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resumo Executivo
"""
    
    for key, value in summary.items():
        report_content += f"- **{key}**: {value}\n"
    
    report_content += "\n## Insights Principais\n"
    for i, insight in enumerate(insights, 1):
        report_content += f"{i}. {insight}\n"
    
    report_content += f"\n## Dados\n"
    report_content += f"Total de registros: {len(data):,}\n"
    report_content += f"Colunas: {', '.join(data.columns)}\n"
    
    # Exportar como texto
    st.download_button(
        label="📄 Baixar Relatório (TXT)",
        data=report_content,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # Também exportar dados em CSV
    col1, col2 = st.columns(2)
    
    with col1:
        create_csv_export(data, f"{filename}_dados")
    
    with col2:
        # Exportar resumo como JSON
        create_summary_export({"resumo": summary, "insights": insights}, f"{filename}_resumo")