"""
Stage 06 Affordances Classification Page
=======================================

Streamlit page for Brazilian political discourse affordance analysis.
Part of the digiNEV v.final dashboard system.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from dashboard.stage06_affordances_dashboard import Stage06AffordancesDashboard

# Page configuration
st.set_page_config(
    page_title="Stage 06 - Affordances Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para análise de affordances
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
    }

    .affordance-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .stage-info {
        background-color: #e8f4fd;
        border-left: 4px solid #1976d2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .category-badge {
        display: inline-block;
        background-color: #667eea;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }

    .visualization-container {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 Stage 06: Classificação de Affordances</div>', unsafe_allow_html=True)

# Informações sobre o Stage 06
st.markdown("""
<div class="stage-info">
<h4>📋 Sobre o Stage 06 - Affordances Classification</h4>
<p><strong>Objetivo:</strong> Classificação inteligente de mensagens em categorias de affordances comunicativas usando IA avançada e métodos heurísticos.</p>
<p><strong>Metodologia:</strong> API Anthropic Claude 3.5 Haiku para análise zero-shot + fallback heurístico baseado em padrões linguísticos.</p>
<p><strong>Categorias:</strong> 8 tipos de affordances - notícia, mídia social, multimídia, opinião, mobilização, ataque, interação, encaminhamento.</p>
<p><strong>Saída:</strong> Classificação múltipla com scores de confiança e análise de co-ocorrência.</p>
</div>
""", unsafe_allow_html=True)

# Categorias de Affordances
st.markdown("### 🏷️ Categorias de Affordances")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Categorias Informativas:**
    <div class="category-badge">📰 Notícia</div> - Conteúdo informativo, reportagens, fatos
    <div class="category-badge">📱 Mídia Social</div> - Posts típicos de redes sociais
    <div class="category-badge">🎬 Multimídia</div> - Vídeos, áudios, GIFs
    <div class="category-badge">💭 Opinião</div> - Opiniões pessoais, análises subjetivas
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    **Categorias Interativas:**
    <div class="category-badge">📢 Mobilização</div> - Chamadas para ação política
    <div class="category-badge">⚔️ Ataque</div> - Ataques pessoais, insultos
    <div class="category-badge">💬 Interação</div> - Respostas, menções, conversações
    <div class="category-badge">↗️ Encaminhado</div> - Conteúdo repassado/compartilhado
    """, unsafe_allow_html=True)

# Metodologia técnica
st.markdown("### 🔬 Metodologia Técnica")

method_col1, method_col2 = st.columns(2)

with method_col1:
    st.markdown("""
    **🤖 Classificação por IA:**
    - **Modelo:** Claude 3.5 Haiku (econômico)
    - **Abordagem:** Zero-shot learning
    - **Tokens:** Máximo 150 por resposta
    - **Rate Limiting:** 100ms entre requests
    - **Confiança:** Score 0.0-1.0 por classificação
    """)

with method_col2:
    st.markdown("""
    **🔍 Fallback Heurístico:**
    - **Padrões:** Expressões regulares otimizadas
    - **Léxico:** Termos específicos por categoria
    - **Ativação:** Quando API indisponível
    - **Confiança:** 0.5 (fixo para heurística)
    - **Cobertura:** 100% das mensagens
    """)

# Informações dos datasets
st.markdown("### 📊 Pipeline de Processamento")
st.markdown("""
| Etapa | Processo | Resultado |
|-------|----------|-----------|
| **1. Carregamento** | Leitura dos dados do Stage 05 | Texto normalizado disponível |
| **2. Classificação IA** | API Anthropic para análise zero-shot | 8 categorias + confiança |
| **3. Fallback** | Heurística baseada em padrões | Cobertura completa |
| **4. Pós-processamento** | Colunas binárias + estatísticas | Dados prontos para análise |
| **5. Validação** | Verificação de qualidade | 102+ colunas geradas |
""")

# Sidebar com informações técnicas
with st.sidebar:
    st.markdown("### ℹ️ Informações Técnicas")

    st.markdown("**🔍 Campos Gerados:**")
    st.markdown("""
    - `affordance_categories`: Lista de categorias
    - `affordance_confidence`: Score de confiança
    - `aff_noticia`: Binário para notícia
    - `aff_midia_social`: Binário para mídia social
    - `aff_video_audio_gif`: Binário para multimídia
    - `aff_opiniao`: Binário para opinião
    - `aff_mobilizacao`: Binário para mobilização
    - `aff_ataque`: Binário para ataque
    - `aff_interacao`: Binário para interação
    - `aff_is_forwarded`: Binário para encaminhado
    """)

    st.markdown("**📈 Métricas Disponíveis:**")
    st.markdown("""
    - Distribuição por categoria
    - Co-ocorrência de affordances
    - Confiança da classificação
    - Evolução temporal
    - Padrões de fluxo comunicativo
    """)

    st.markdown("**🎯 Aplicações Acadêmicas:**")
    st.markdown("""
    - Análise de estratégias comunicativas
    - Padrões de mobilização política
    - Evolução do discurso autoritário
    - Dinâmicas de interação em rede
    - Propagação de desinformação
    """)

    st.markdown("**⚙️ Configurações:**")
    st.markdown("""
    - **API Key:** ANTHROPIC_API_KEY (env)
    - **Modelo:** claude-3-5-haiku-20241022
    - **Timeout:** 30s por request
    - **Batch Size:** 50 mensagens
    - **Temperatura:** 0.1 (determinística)
    """)

# Renderizar dashboard principal
try:
    dashboard = Stage06AffordancesDashboard()
    dashboard.render_dashboard()

except Exception as e:
    st.error(f"Erro ao carregar dashboard de affordances: {e}")

    with st.expander("🔧 Soluções Possíveis"):
        st.markdown("""
        ### Diagnóstico e Soluções:

        **1. Dados não processados:**
        ```bash
        # Executar pipeline completo
        python run_pipeline.py

        # Verificar logs
        tail -f logs/analyzer.log
        ```

        **2. API Anthropic não configurada:**
        ```bash
        # Configurar chave da API
        export ANTHROPIC_API_KEY="your-api-key"

        # Verificar configuração
        echo $ANTHROPIC_API_KEY
        ```

        **3. Dependências missing:**
        ```bash
        # Instalar dependências
        pip install anthropic requests networkx

        # Verificar instalação
        python -c "import anthropic, networkx; print('OK')"
        ```

        **4. Problemas de memória:**
        ```bash
        # Executar com dataset pequeno
        python test_clean_analyzer.py

        # Verificar uso de memória
        python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
        ```

        **5. Logs detalhados:**
        ```bash
        # Habilitar debug
        export LOG_LEVEL=DEBUG

        # Executar com logs verbosos
        python run_pipeline.py --verbose
        ```
        """)

    # Informações adicionais sobre o Stage 06
    st.markdown("### 📚 Sobre a Classificação de Affordances")

    st.markdown("""
    **Conceito de Affordances:**

    O conceito de "affordances" refere-se às possibilidades de ação que um objeto ou ambiente oferece
    a um agente. No contexto de comunicação política digital, affordances representam os diferentes
    tipos de ação comunicativa que uma mensagem pode realizar:

    - **Informar** (notícia): Transmitir informações factuais
    - **Socializar** (mídia social): Criar vínculos sociais
    - **Entreter** (multimídia): Engajar através de conteúdo audiovisual
    - **Persuadir** (opinião): Influenciar pontos de vista
    - **Mobilizar** (mobilização): Incitar ação política
    - **Atacar** (ataque): Deslegitimar adversários
    - **Interagir** (interação): Estabelecer diálogo
    - **Propagar** (encaminhamento): Amplificar mensagens

    **Importância para Pesquisa:**

    A classificação de affordances permite entender como diferentes estratégias comunicativas
    são empregadas no discurso político digital, revelando padrões de:
    - Estratégias de legitimação e deslegitimação
    - Dinâmicas de polarização política
    - Mecanismos de coordenação em rede
    - Evolução de narrativas autoritárias
    """)

# Footer com informações do projeto
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>digiNEV v.final</strong> - Análise de Discurso Político Brasileiro</p>
<p>Stage 06: Affordances Classification | Projeto de Pesquisa Acadêmica | 2025</p>
<p><em>Classificação inteligente de affordances comunicativas em mensagens políticas</em></p>
</div>
""", unsafe_allow_html=True)