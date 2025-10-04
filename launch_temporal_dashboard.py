#!/usr/bin/env python3
"""
Stage 13 Temporal Analysis Dashboard Launcher
Launch script for the temporal analysis dashboard

Usage:
    python launch_temporal_dashboard.py

Requirements:
    - Processed data in /data/processed/
    - Stage 13 temporal analysis completed
    - Streamlit installed
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if requirements are met."""
    print("🔍 Checking requirements...")

    # Check if processed data exists
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        print("❌ Processed data directory not found")
        return False

    csv_files = list(processed_dir.glob("*.csv"))
    if not csv_files:
        print("❌ No processed CSV files found")
        return False

    print(f"✅ Found {len(csv_files)} processed datasets")

    # Check if temporal dashboard exists
    dashboard_file = Path("src/dashboard/stage13_temporal_dashboard.py")
    if not dashboard_file.exists():
        print("❌ Temporal dashboard file not found")
        return False

    print("✅ Temporal dashboard file found")

    # Check if Streamlit page exists
    page_file = Path("src/dashboard/pages/13_⏰_Temporal.py")
    if not page_file.exists():
        print("❌ Temporal analysis page not found")
        return False

    print("✅ Temporal analysis page found")

    return True

def launch_dashboard():
    """Launch the temporal analysis dashboard."""
    if not check_requirements():
        print("\n❌ Requirements not met. Please:")
        print("1. Run the pipeline: python run_pipeline.py")
        print("2. Ensure Stage 13 temporal analysis is completed")
        print("3. Install Streamlit: pip install streamlit")
        return False

    print("\n🚀 Launching Stage 13 Temporal Analysis Dashboard...")
    print("📊 Available visualizations:")
    print("   1. 📈 Volume de mensagens ao longo do tempo")
    print("   2. 🎯 Correlação com eventos políticos brasileiros")
    print("   3. 🔥 Heatmap de coordenação temporal")
    print("   4. 🕸️ Rede de atividade sincronizada")
    print("   5. ⏱️ Timeline de períodos de alta coordenação")
    print("   6. 🌊 Fluxo temporal → sentimento → affordances")

    try:
        # Launch the main dashboard with temporal page
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/start_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ]

        print(f"\n🌐 Dashboard will open at: http://localhost:8501")
        print("📄 Navigate to: '13 ⏰ Temporal' in the sidebar")
        print("\n⏹️  Press Ctrl+C to stop the dashboard")

        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("\nAlternative launch method:")
        print("streamlit run src/dashboard/start_dashboard.py")

if __name__ == "__main__":
    print("⏰ Stage 13 - Temporal Analysis Dashboard Launcher")
    print("=" * 50)
    launch_dashboard()