#!/usr/bin/env python3
"""
Módulo de inicialização para o pacote data, garantindo que subpacotes sejam importáveis.
"""

# Importar componentes principais para facilitar acesso
from src.data.processors.chunk_processor import ChunkProcessor
from src.data.utils.encoding_fixer import EncodingFixer

# Versão do módulo
__version__ = "0.2.0"