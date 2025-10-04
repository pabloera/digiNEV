"""
Stage 10: Clustering Analysis Page
==================================

Professional Streamlit page for K-Means clustering analysis of Brazilian political discourse.
Part of the digiNEV v.final dashboard system.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from dashboard.stage10_clustering_dashboard import ClusteringAnalysisDashboard

# Page configuration
st.set_page_config(
    page_title="Stage 10: Clustering Analysis - digiNEV",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Run the clustering analysis page."""
    # Create and run dashboard
    dashboard = ClusteringAnalysisDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
else:
    main()