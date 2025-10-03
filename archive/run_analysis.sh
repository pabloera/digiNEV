#!/bin/bash
# Script de execução do Batch Analyzer independente

echo "🚀 Batch Analyzer - Sistema Independente de Análise"
echo "=================================================="

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado!"
    exit 1
fi

# Ativa ambiente virtual se existir
if [ -d "venv" ]; then
    echo "📦 Ativando ambiente virtual..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "📦 Ativando ambiente virtual do projeto principal..."
    source ../venv/bin/activate
fi

# Verifica argumentos
if [ "$1" == "--test" ]; then
    echo "🧪 Executando teste..."
    python3 test_batch.py
elif [ "$1" == "--dev" ]; then
    echo "🛠️ Modo desenvolvimento (sem APIs)..."
    python3 batch_analysis.py --dev-mode "${@:2}"
elif [ "$1" == "--academic" ]; then
    echo "🎓 Modo acadêmico (otimizado)..."
    python3 batch_analysis.py --config config/academic.yaml "${@:2}"
elif [ "$1" == "--help" ]; then
    echo ""
    echo "Uso: ./run_analysis.sh [opção] [arquivo.csv]"
    echo ""
    echo "Opções:"
    echo "  --test        Executa teste básico"
    echo "  --dev         Modo desenvolvimento (sem APIs)"
    echo "  --academic    Modo acadêmico (com otimizações)"
    echo "  --help        Mostra esta ajuda"
    echo ""
    echo "Exemplo:"
    echo "  ./run_analysis.sh --dev data/sample.csv"
    echo "  ./run_analysis.sh --academic data/telegram_messages.csv"
else
    # Execução padrão
    echo "💼 Executando análise padrão..."
    python3 batch_analysis.py "$@"
fi

echo ""
echo "✅ Processamento concluído!"