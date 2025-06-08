#!/usr/bin/env python3
"""
Script para instalar bibliotecas de visualização avançada do dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Instala um pacote usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar {package}: {e}")
        return False

def main():
    """Instala todas as bibliotecas necessárias para visualizações avançadas"""
    
    print("🚀 Instalando bibliotecas para visualizações avançadas do dashboard...")
    print("=" * 60)
    
    # Lista de pacotes necessários
    packages = [
        "networkx>=3.0",
        "scipy>=1.9.0", 
        "wordcloud>=1.9.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0"
    ]
    
    success_count = 0
    
    for package in packages:
        print(f"\n📦 Instalando {package}...")
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Resultado: {success_count}/{len(packages)} pacotes instalados")
    
    if success_count == len(packages):
        print("\n🎉 Todas as bibliotecas foram instaladas com sucesso!")
        print("✅ O dashboard agora tem acesso a todas as visualizações avançadas:")
        print("   - 🕸️  Visualizações de rede (NetworkX)")
        print("   - 🌳 Dendrogramas hierárquicos (Scipy)")
        print("   - ☁️  Nuvens de palavras (WordCloud)")
        print("   - 📈 Gráficos estatísticos avançados (Seaborn)")
        print("   - 🤖 Análise de machine learning (Scikit-learn)")
    else:
        print("\n⚠️  Algumas bibliotecas não foram instaladas.")
        print("💡 Tente executar manualmente:")
        print("   pip install networkx scipy wordcloud matplotlib seaborn scikit-learn")
    
    print("\n🔧 Para iniciar o dashboard:")
    print("   python start_dashboard.py")
    print("   ou")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()