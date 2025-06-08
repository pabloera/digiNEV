#!/usr/bin/env python3
"""
Script para verificar as colunas dos datasets e diagnosticar o erro
"""

import pandas as pd
import sys
import os
from pathlib import Path

def check_dataset_columns(file_path):
    """Verifica as colunas de um dataset CSV"""
    print(f"\n=== Verificando: {file_path} ===")
    
    try:
        # Tentar diferentes separadores
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(file_path, sep=sep, nrows=5)
                print(f"✅ Leitura bem-sucedida com separador: '{sep}'")
                print(f"📊 Shape: {df.shape}")
                print(f"📋 Colunas encontradas: {list(df.columns)}")
                
                # Verificar se tem coluna 'body'
                if 'body' in df.columns:
                    print("✅ Coluna 'body' encontrada!")
                else:
                    print("❌ Coluna 'body' NÃO encontrada!")
                    
                # Procurar colunas de texto
                text_columns = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        avg_len = df[col].dropna().astype(str).str.len().mean()
                        if avg_len > 30:
                            text_columns.append((col, avg_len))
                            
                if text_columns:
                    print(f"📝 Colunas de texto encontradas:")
                    for col, avg_len in text_columns:
                        print(f"   - {col}: média de {avg_len:.0f} caracteres")
                
                # Mostrar amostra das primeiras linhas
                print("\n📄 Primeiras 3 linhas:")
                print(df.head(3))
                
                return True
                
            except Exception as e:
                continue
                
        print("❌ Não foi possível ler o arquivo com nenhum separador comum")
        return False
        
    except Exception as e:
        print(f"❌ Erro ao processar arquivo: {e}")
        return False

def main():
    """Função principal"""
    # Diretórios para verificar
    data_dirs = [
        "data/DATASETS_FULL",
        "data/interim",
        "data/uploads"
    ]
    
    print("🔍 Verificando estrutura dos datasets...")
    
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            print(f"\n📁 Verificando diretório: {data_dir}")
            
            # Procurar arquivos CSV
            csv_files = list(dir_path.glob("*.csv"))
            
            if csv_files:
                for csv_file in csv_files[:3]:  # Verificar até 3 arquivos por diretório
                    check_dataset_columns(csv_file)
            else:
                print(f"   ⚠️ Nenhum arquivo CSV encontrado em {data_dir}")
    
    # Verificar arquivo específico se fornecido
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            check_dataset_columns(file_path)
        else:
            print(f"\n❌ Arquivo não encontrado: {file_path}")

if __name__ == "__main__":
    main()