#!/usr/bin/env python3
"""
Lexicon Loader - Single Responsibility
Responsável apenas por carregar e gerenciar o léxico político
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger('BatchScientific')

class LexiconLoader:
    """Carregador de léxico político brasileiro"""

    def __init__(self, lexicon_path: str = 'lexico_politico_hierarquizado.json'):
        """
        Inicializa carregador de léxico.

        Args:
            lexicon_path (str): Caminho para arquivo do léxico
        """
        self.lexicon_path = Path(lexicon_path)
        self._lexicon = None

    @property
    def lexicon(self) -> Optional[Dict]:
        """Lazy loading do léxico"""
        if self._lexicon is None:
            self._load_lexicon()
        return self._lexicon

    def _load_lexicon(self) -> None:
        """
        Carrega léxico político do arquivo JSON.
        """
        try:
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                lexicon_data = json.load(f)
                self._lexicon = lexicon_data

            terms_count = len(lexicon_data.get('lexico', {}))
            logger.info(f"✅ Léxico político carregado: {terms_count} termos")

        except FileNotFoundError:
            logger.warning(f"⚠️ Léxico político não encontrado: {self.lexicon_path}")
            self._lexicon = None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao decodificar léxico: {e}")
            self._lexicon = None
        except Exception as e:
            logger.warning(f"⚠️ Erro inesperado ao carregar léxico: {e}")
            self._lexicon = None

    def get_political_terms(self, category: Optional[str] = None) -> List[str]:
        """
        Obtém termos políticos por categoria (adaptado para léxico hierárquico).

        Args:
            category (str, opcional): Categoria específica

        Returns:
            List[str]: Lista de termos
        """
        if not self.lexicon:
            return []

        try:
            lexico = self.lexicon.get('lexico', {})
            all_terms = []

            if category:
                category_data = lexico.get(category, {})
                if isinstance(category_data, dict):
                    # Extrair termos dos subtemas
                    subtemas = category_data.get('subtemas', {})
                    for subtema_data in subtemas.values():
                        if isinstance(subtema_data, dict):
                            palavras = subtema_data.get('palavras', [])
                            all_terms.extend(palavras)
                return all_terms
            else:
                # Retorna todos os termos de todas as categorias
                for category_data in lexico.values():
                    if isinstance(category_data, dict):
                        subtemas = category_data.get('subtemas', {})
                        for subtema_data in subtemas.values():
                            if isinstance(subtema_data, dict):
                                palavras = subtema_data.get('palavras', [])
                                all_terms.extend(palavras)
                return all_terms

        except Exception as e:
            logger.error(f"❌ Erro ao obter termos políticos: {e}")
            return ['política', 'governo', 'eleição', 'democracia']  # Fallback terms

    def is_political_term(self, term: str) -> bool:
        """
        Verifica se um termo é político.

        Args:
            term (str): Termo a verificar

        Returns:
            bool: True se é termo político
        """
        if not self.lexicon:
            return False

        term_lower = term.lower()
        all_terms = self.get_political_terms()
        return term_lower in [t.lower() for t in all_terms]

    def get_term_category(self, term: str) -> Optional[str]:
        """
        Obtém categoria de um termo político (adaptado para léxico hierárquico).

        Args:
            term (str): Termo a classificar

        Returns:
            str ou None: Categoria do termo
        """
        if not self.lexicon:
            return None

        term_lower = term.lower()

        try:
            lexico = self.lexicon.get('lexico', {})

            for category, category_data in lexico.items():
                if isinstance(category_data, dict):
                    subtemas = category_data.get('subtemas', {})
                    for subtema_data in subtemas.values():
                        if isinstance(subtema_data, dict):
                            palavras = subtema_data.get('palavras', [])
                            if term_lower in [t.lower() for t in palavras]:
                                return category

        except Exception as e:
            logger.error(f"❌ Erro ao obter categoria do termo: {e}")

        return None

    def validate_lexicon(self) -> bool:
        """
        Valida estrutura do léxico.

        Returns:
            bool: True se válido
        """
        if not self.lexicon:
            return False

        try:
            if 'lexico' not in self.lexicon:
                logger.error("❌ Campo 'lexico' ausente")
                return False

            if not isinstance(self.lexicon['lexico'], dict):
                logger.error("❌ 'lexico' deve ser um dicionário")
                return False

            logger.info("✅ Léxico validado com sucesso")
            return True

        except Exception as e:
            logger.error(f"❌ Erro na validação do léxico: {e}")
            return False