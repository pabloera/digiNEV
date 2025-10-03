#!/usr/bin/env python3
"""
Script simplificado de execução do Batch Analyzer
Análise de discurso político brasileiro
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add batch_analyzer to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_analysis import IntegratedBatchAnalyzer, BatchConfig

def main():
    """Executa análise batch"""

    # Configura para uso sem APIs (método heurístico)
    config = BatchConfig()
    config.USE_APIS = False
    config.DEBUG = False

    # Inicializa analisador
    analyzer = IntegratedBatchAnalyzer(config)

    # Detecta arquivo de entrada
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Usa arquivo de exemplo se nenhum for fornecido
        input_file = "data/sample_messages.csv"

    if not Path(input_file).exists():
        print(f"❌ Arquivo não encontrado: {input_file}")
        sys.exit(1)

    print(f"📊 Analisando: {input_file}")

    # Executa análise
    try:
        results = analyzer.run_analysis(input_file)

        # Exibe resumo
        if results and 'stats' in results:
            stats = results['stats']
            print(f"\n✅ Análise Concluída!")
            print(f"📈 Registros processados: {stats.get('total_records', 0)}")
            print(f"⏱️ Tempo total: {stats.get('total_time', 0):.2f}s")

            # Salva resultados
            if 'data' in results and results['data'] is not None:
                output_file = f"outputs/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                Path("outputs").mkdir(exist_ok=True)
                results['data'].to_csv(output_file, index=False)
                print(f"💾 Resultados salvos: {output_file}")

    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()