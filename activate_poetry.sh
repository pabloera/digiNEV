#!/bin/bash

# Script de verificação e ativação Poetry - dataanalysis-bolsonarismo
# Uso: ./activate_poetry.sh

echo "🚀 Verificando ambiente Poetry..."

# Verificar Poetry instalado
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry não encontrado. Instale: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Verificar diretório correto
if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ Execute no diretório raiz do projeto (onde está pyproject.toml)"
    exit 1
fi

# Instalar/verificar dependências
echo "📦 Verificando dependências..."
poetry install --quiet

# Status do ambiente
if poetry env info --path &> /dev/null; then
    VENV_PATH=$(poetry env info --path)
    PYTHON_VERSION=$(poetry run python --version)
    echo "✅ Ambiente virtual: $VENV_PATH"
    echo "✅ $PYTHON_VERSION"
    echo "✅ Poetry pronto!"
    echo ""
    echo "🔧 Comandos disponíveis:"
    echo "   poetry run pipeline          # Executa pipeline completo"
    echo "   poetry run dashboard         # Inicia dashboard"
    echo "   poetry run python --version  # Testa ambiente"
    echo "   poetry shell                 # Ativa shell interativo"
else
    echo "❌ Erro na configuração do ambiente virtual"
    exit 1
fi