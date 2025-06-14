"""
Validador de Encoding via API Anthropic
Verifica correção de encoding, detecta caracteres problemáticos e identifica palavras que necessitam correção.
"""

import csv
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chardet
import pandas as pd

from .api_error_handler import APIErrorHandler, APIQualityChecker
from .base import AnthropicBase

logger = logging.getLogger(__name__)

class EncodingValidator(AnthropicBase):
    """
    Validador avançado de encoding usando API Anthropic

    Capacidades:
    - Detecção de problemas de encoding em texto português
    - Identificação de caracteres problemáticos específicos
    - Sugestões de correção contextual
    - Validação de qualidade pós-correção
    - Detecção de padrões de corrupção específicos do Brasil
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.error_handler = APIErrorHandler()
        self.quality_checker = APIQualityChecker(config)

        # Padrões comuns de problemas de encoding em português
        self.encoding_patterns = {
            "latin1_to_utf8": {
                "Ã¡": "á", "Ã ": "à", "Ã£": "ã", "Ã¢": "â", "Ã¤": "ä",
                "Ã©": "é", "Ãª": "ê", "Ã¨": "è", "Ã«": "ë",
                "Ã­": "í", "Ã®": "î", "Ã¬": "ì", "Ã¯": "ï",
                "Ã³": "ó", "Ã´": "ô", "Ã²": "ò", "Ãµ": "õ", "Ã¶": "ö",
                "Ãº": "ú", "Ã»": "û", "Ã¹": "ù", "Ã¼": "ü",
                "Ã§": "ç", "ÃŸ": "ß", "Ã±": "ñ",
                "Ã_A": "Á", "Ã€": "À", "Ãƒ": "Ã", "Ã‚": "Â", "Ã„": "Ä",
                "Ã‰": "É", "ÃŠ": "Ê", "Ãˆ": "È", "Ã‹": "Ë",
                "Ã_I": "Í", "ÃŽ": "Î", "ÃŒ": "Ì", "Ã_I2": "Ï",
                "ÃO": "Ó", "ÃO2": "Ô", "ÃO3": "Ò", "Ã•": "Õ", "Ã–": "Ö",
                "Ãš": "Ú", "Ã›": "Û", "Ã™": "Ù", "Ãœ": "Ü",
                "Ã‡": "Ç"
            },
            "windows1252_artifacts": {
                # Common Windows-1252 encoding artifacts
                "â€™": "'", "â€œ": '"', "â€": '"',
                "Â": " "
            },
            "html_entities": {
                "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                "&apos;": "'", "&nbsp;": " ", "&#39;": "'", "&#x27;": "'"
            }
        }

        # Caracteres problemáticos específicos
        self.problematic_chars = {
            "null_chars": ["\x00", "\ufffd", "�"],
            "control_chars": [chr(i) for i in range(32) if i not in [9, 10, 13]],
            "invisible_chars": ["\u200b", "\u200c", "\u200d", "\ufeff"],
            "confusables": ["⁇", "؟", "？", "¿"]  # Caracteres que se parecem com outros
        }

        # Palavras comuns em português para validação contextual
        self.portuguese_common_words = {
            "articles": ["o", "a", "os", "as", "um", "uma", "uns", "umas"],
            "prepositions": ["de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "para", "por", "com", "sem"],
            "pronouns": ["eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas", "que", "quem", "qual"],
            "verbs": ["ser", "estar", "ter", "haver", "fazer", "ir", "vir", "ver", "dar", "saber", "poder", "dizer"],
            "conjunctions": ["e", "ou", "mas", "porém", "porque", "que", "se", "como", "quando", "onde"]
        }

    def validate_encoding_quality(
        self,
        df: pd.DataFrame,
        text_columns: List[str] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Valida qualidade do encoding do dataset

        Args:
            df: DataFrame para validar
            text_columns: Colunas de texto para analisar
            sample_size: Tamanho da amostra para análise detalhada

        Returns:
            Relatório de validação de encoding
        """
        logger.info(f"Iniciando validação de encoding para {len(df)} registros")

        if text_columns is None:
            text_columns = self._detect_text_columns(df)

        # Análise rápida de todo o dataset
        quick_analysis = self._quick_encoding_analysis(df, text_columns)

        # Análise detalhada via API em amostra
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        detailed_analysis = self._detailed_encoding_analysis_api(sample_df, text_columns)

        # Combinar resultados
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "text_columns_analyzed": text_columns,
            "quick_analysis": quick_analysis,
            "detailed_analysis": detailed_analysis,
            "overall_quality_score": self._calculate_encoding_quality_score(quick_analysis, detailed_analysis),
            "recommendations": self._generate_encoding_recommendations(quick_analysis, detailed_analysis)
        }

        logger.info(f"Validação concluída. Score de qualidade: {validation_report['overall_quality_score']}")

        return validation_report

    def detect_and_fix_encoding_issues(
        self,
        df: pd.DataFrame,
        text_columns: List[str] = None,
        fix_mode: str = "conservative"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detecta e corrige problemas de encoding

        Args:
            df: DataFrame para corrigir
            text_columns: Colunas de texto para processar
            fix_mode: "conservative", "aggressive", "api_assisted"

        Returns:
            Tuple com DataFrame corrigido e relatório de correções
        """
        logger.info(f"Iniciando correção de encoding (modo: {fix_mode})")

        if text_columns is None:
            text_columns = self._detect_text_columns(df)

        # Fazer backup
        backup_file = f"data/interim/encoding_fix_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False, sep=';', encoding='utf-8')
        logger.info(f"Backup criado: {backup_file}")

        corrected_df = df.copy()
        correction_report = {
            "mode": fix_mode,
            "columns_processed": text_columns,
            "corrections_applied": [],
            "statistics": {}
        }

        for column in text_columns:
            if column in corrected_df.columns:
                column_corrections = self._fix_column_encoding(
                    corrected_df, column, fix_mode
                )
                correction_report["corrections_applied"].extend(column_corrections)

        # Validação pós-correção
        post_fix_validation = self.validate_encoding_quality(corrected_df, text_columns, sample_size=500)
        correction_report["post_fix_validation"] = post_fix_validation

        return corrected_df, correction_report

    def _detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Detecta automaticamente colunas de texto"""
        text_columns = []

        for col in df.columns:
            if df[col].dtype == 'object':
                # Verificar se contém texto significativo
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    avg_length = sample_values.astype(str).str.len().mean()
                    if avg_length > 10:  # Assumir que texto tem mais de 10 caracteres
                        text_columns.append(col)

        return text_columns

    def _quick_encoding_analysis(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
        """Análise rápida de problemas de encoding"""
        analysis = {}

        for column in text_columns:
            if column in df.columns:
                column_analysis = {
                    "total_values": len(df),
                    "non_null_values": df[column].notna().sum(),
                    "problematic_chars_count": 0,
                    "encoding_artifacts_count": 0,
                    "problematic_patterns": []
                }

                # Converter para string e analisar
                text_series = df[column].fillna("").astype(str)

                # Detectar caracteres problemáticos
                for char_type, chars in self.problematic_chars.items():
                    count = sum(text_series.str.contains(f"[{''.join(re.escape(c) for c in chars)}]", regex=True, na=False))
                    if count > 0:
                        column_analysis["problematic_chars_count"] += count
                        column_analysis["problematic_patterns"].append(f"{char_type}: {count}")

                # Detectar artefatos de encoding
                for pattern_type, patterns in self.encoding_patterns.items():
                    for bad_char, good_char in patterns.items():
                        count = text_series.str.contains(re.escape(bad_char), na=False).sum()
                        if count > 0:
                            column_analysis["encoding_artifacts_count"] += count
                            column_analysis["problematic_patterns"].append(f"{pattern_type} - {bad_char}: {count}")

                analysis[column] = column_analysis

        return analysis

    def _detailed_encoding_analysis_api(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
        """Análise detalhada via API"""

        analysis = {}

        for column in text_columns:
            if column in df.columns:
                # Pegar amostra de textos problemáticos
                sample_texts = self._get_problematic_text_samples(df[column])

                if sample_texts:
                    # Usar error handler para análise com retry
                    result = self.error_handler.execute_with_retry(
                        self._analyze_encoding_with_api,
                        stage="02_encoding_validation",
                        operation=f"analyze_{column}",
                        texts=sample_texts,
                        column_name=column
                    )

                    if result.success:
                        analysis[column] = result.data
                    else:
                        logger.warning(f"Falha na análise API para {column}: {result.error.error_message}")
                        analysis[column] = {"error": result.error.error_message}

        return analysis

    def _get_problematic_text_samples(self, series: pd.Series, max_samples: int = 20) -> List[str]:
        """Obtém amostras de textos com possíveis problemas de encoding"""

        text_series = series.fillna("").astype(str)
        problematic_texts = []

        # Encontrar textos com artefatos de encoding
        for pattern_type, patterns in self.encoding_patterns.items():
            for bad_char in patterns.keys():
                matches = text_series[text_series.str.contains(re.escape(bad_char), na=False)]
                if len(matches) > 0:
                    problematic_texts.extend(matches.head(5).tolist())

        # Encontrar textos com caracteres problemáticos
        for char_type, chars in self.problematic_chars.items():
            if chars:
                pattern = f"[{''.join(re.escape(c) for c in chars)}]"
                matches = text_series[text_series.str.contains(pattern, regex=True, na=False)]
                if len(matches) > 0:
                    problematic_texts.extend(matches.head(3).tolist())

        # Remover duplicatas e limitar
        unique_texts = list(set(problematic_texts))[:max_samples]

        return unique_texts

    def _analyze_encoding_with_api(self, texts: List[str], column_name: str) -> Dict[str, Any]:
        """Usa API para análise detalhada de problemas de encoding"""

        texts_sample = "\n".join([f"{i+1}. {text[:150]}..." for i, text in enumerate(texts)])

        prompt = f"""
Analise os seguintes textos em português brasileiro para problemas de encoding/codificação:

COLUNA: {column_name}
TEXTOS:
{texts_sample}

Identifique:
1. Problemas específicos de encoding (ex: caracteres corrompidos)
2. Palavras ou frases que parecem estar mal codificadas
3. Padrões de corrupção de caracteres
4. Sugestões de correção para cada problema encontrado
5. Avalie se o texto faz sentido em português

Responda em formato JSON:
{{
  "encoding_issues": [
    {{
      "text_id": 1,
      "issues_found": ["problema1", "problema2"],
      "corrupted_chars": {{"char_ruim": "char_correto"}},
      "corrupted_words": {{"palavra_ruim": "palavra_correta"}},
      "severity": "baixo|medio|alto",
      "makes_sense_in_portuguese": true/false,
      "suggested_fixes": ["fix1", "fix2"]
    }}
  ],
  "overall_assessment": {{
    "total_texts_analyzed": 0,
    "texts_with_issues": 0,
    "most_common_issues": ["issue1", "issue2"],
    "recommended_actions": ["action1", "action2"]
  }}
}}

IMPORTANTE: Foque em problemas reais de encoding, não em erros ortográficos normais.
"""

        try:
            response = self.create_message(
                prompt,
                stage="02_encoding_validation",
                operation=f"detailed_analysis_{column_name}"
            )

            # Validar qualidade da resposta
            validation = self.quality_checker.validate_output_quality(
                response,
                expected_format="json",
                context={"column": column_name, "texts_count": len(texts)},
                stage="02_encoding_validation"
            )

            if not validation["valid"]:
                logger.warning(f"Qualidade da resposta baixa para {column_name}: {validation['issues']}")

            return self.parse_json_response(response)

        except Exception as e:
            logger.error(f"Erro na análise de encoding via API: {e}")
            return {"error": str(e)}

    def _fix_column_encoding(self, df: pd.DataFrame, column: str, fix_mode: str) -> List[str]:
        """Corrige problemas de encoding em uma coluna"""

        corrections = []
        original_values = df[column].copy()

        # Aplicar correções baseadas em padrões conhecidos
        if fix_mode in ["conservative", "aggressive", "api_assisted"]:
            for pattern_type, patterns in self.encoding_patterns.items():
                for bad_char, good_char in patterns.items():
                    mask = df[column].fillna("").astype(str).str.contains(re.escape(bad_char), na=False)
                    if mask.any():
                        df[column] = df[column].fillna("").astype(str).str.replace(bad_char, good_char, regex=False)
                        count = mask.sum()
                        corrections.append(f"{column}: {bad_char} -> {good_char} ({count} ocorrências)")

        # Remover caracteres problemáticos
        if fix_mode in ["aggressive", "api_assisted"]:
            for char_type, chars in self.problematic_chars.items():
                for char in chars:
                    mask = df[column].fillna("").astype(str).str.contains(re.escape(char), na=False)
                    if mask.any():
                        df[column] = df[column].fillna("").astype(str).str.replace(char, "", regex=False)
                        count = mask.sum()
                        corrections.append(f"{column}: removido {char_type} '{char}' ({count} ocorrências)")

        # Usar API para correções contextuais
        if fix_mode == "api_assisted":
            api_corrections = self._apply_api_assisted_corrections(df, column)
            corrections.extend(api_corrections)

        return corrections

    def _apply_api_assisted_corrections(self, df: pd.DataFrame, column: str) -> List[str]:
        """Aplica correções assistidas por API"""

        corrections = []

        # Encontrar textos que ainda parecem problemáticos
        problematic_indices = self._find_remaining_problematic_texts(df[column])

        if problematic_indices:
            # Processar em pequenos lotes
            batch_size = 10
            for i in range(0, len(problematic_indices), batch_size):
                batch_indices = problematic_indices[i:i+batch_size]
                batch_texts = df.loc[batch_indices, column].tolist()

                # Usar API para sugerir correções
                api_result = self.error_handler.execute_with_retry(
                    self._get_api_corrections,
                    stage="02_encoding_validation",
                    operation=f"api_corrections_{column}",
                    texts=batch_texts
                )

                if api_result.success and api_result.data:
                    # Aplicar correções sugeridas
                    applied = self._apply_suggested_corrections(df, column, batch_indices, api_result.data)
                    corrections.extend(applied)

        return corrections

    def _find_remaining_problematic_texts(self, series: pd.Series) -> List[int]:
        """Encontra índices de textos que ainda parecem problemáticos"""

        problematic_indices = []
        text_series = series.fillna("").astype(str)

        for idx, text in text_series.items():
            # Heurísticas para detectar texto ainda problemático
            if len(text) > 10:
                # Muitos caracteres não-ASCII consecutivos
                non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
                if non_ascii_ratio > 0.3:
                    problematic_indices.append(idx)

                # Sequências suspeitas de caracteres
                if re.search(r'[Ã]{2,}|[â€]{2,}|[�]{2,}', text):
                    problematic_indices.append(idx)

                # Palavras muito curtas ou fragmentadas demais
                words = text.split()
                if len(words) > 5:
                    short_words_ratio = sum(1 for w in words if len(w) <= 2) / len(words)
                    if short_words_ratio > 0.5:
                        problematic_indices.append(idx)

        return list(set(problematic_indices))

    def _get_api_corrections(self, texts: List[str]) -> Dict[str, Any]:
        """Obtém sugestões de correção da API"""

        texts_sample = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])

        prompt = f"""
Os seguintes textos em português brasileiro podem ter problemas de encoding.
Para cada texto, forneça a versão corrigida se houver problemas evidentes:

{texts_sample}

Responda APENAS com JSON no formato:
{{
  "corrections": [
    {{
      "text_id": 1,
      "original": "texto original",
      "corrected": "texto corrigido",
      "changes_made": ["mudança1", "mudança2"],
      "confidence": "alto|medio|baixo"
    }}
  ]
}}

REGRAS:
- Só sugira correções para problemas EVIDENTES de encoding
- Mantenha o significado original
- Não corrija erros ortográficos normais
- Se não há problemas evidentes, mantenha "corrected" igual a "original"
"""

        try:
            response = self.create_message(
                prompt,
                stage="02_encoding_validation",
                operation="api_corrections"
            )

            return self.parse_json_response(response)

        except Exception as e:
            logger.error(f"Erro ao obter correções da API: {e}")
            return {}

    def _apply_suggested_corrections(
        self,
        df: pd.DataFrame,
        column: str,
        indices: List[int],
        api_corrections: Dict[str, Any]
    ) -> List[str]:
        """Aplica correções sugeridas pela API"""

        corrections_applied = []

        if "corrections" in api_corrections:
            for correction in api_corrections["corrections"]:
                text_id = correction.get("text_id", 1) - 1  # Converter para índice base 0

                if text_id < len(indices):
                    actual_index = indices[text_id]
                    original = correction.get("original", "")
                    corrected = correction.get("corrected", "")
                    confidence = correction.get("confidence", "baixo")

                    # Aplicar apenas correções de alta confiança
                    if confidence in ["alto", "medio"] and corrected != original:
                        df.loc[actual_index, column] = corrected
                        changes = correction.get("changes_made", [])
                        corrections_applied.append(
                            f"{column}[{actual_index}]: API correction ({confidence} confidence) - {changes}"
                        )

        return corrections_applied

    def _calculate_encoding_quality_score(
        self,
        quick_analysis: Dict[str, Any],
        detailed_analysis: Dict[str, Any]
    ) -> float:
        """Calcula score de qualidade do encoding (0-1)"""

        total_score = 0.0
        columns_analyzed = 0

        for column, analysis in quick_analysis.items():
            if "error" not in analysis:
                columns_analyzed += 1

                # Score baseado em problemas encontrados
                total_values = analysis.get("non_null_values", 1)
                problematic_chars = analysis.get("problematic_chars_count", 0)
                encoding_artifacts = analysis.get("encoding_artifacts_count", 0)

                # Penalizar problemas
                problems_ratio = (problematic_chars + encoding_artifacts) / total_values
                column_score = max(0.0, 1.0 - problems_ratio)

                total_score += column_score

        return total_score / columns_analyzed if columns_analyzed > 0 else 0.0

    def _generate_encoding_recommendations(
        self,
        quick_analysis: Dict[str, Any],
        detailed_analysis: Dict[str, Any]
    ) -> List[str]:
        """Gera recomendações baseadas na análise"""

        recommendations = []

        # Analisar problemas encontrados
        total_problems = 0
        for column, analysis in quick_analysis.items():
            if "error" not in analysis:
                total_problems += analysis.get("problematic_chars_count", 0)
                total_problems += analysis.get("encoding_artifacts_count", 0)

        if total_problems > 0:
            recommendations.append(f"Encontrados {total_problems} problemas de encoding - recomenda-se correção")

            if total_problems > 1000:
                recommendations.append("Grande quantidade de problemas - considere re-codificação completa do dataset")

            recommendations.append("Execute correção com modo 'api_assisted' para melhor precisão")
        else:
            recommendations.append("Qualidade de encoding satisfatória - nenhuma ação necessária")

        # Recomendações específicas baseadas na análise detalhada
        for column, analysis in detailed_analysis.items():
            if "overall_assessment" in analysis:
                recommended_actions = analysis["overall_assessment"].get("recommended_actions", [])
                recommendations.extend([f"{column}: {action}" for action in recommended_actions])

        return recommendations

    def detect_encoding_with_chardet(self, file_path: str) -> Dict[str, Any]:
        """
        Detecta encoding de arquivo usando chardet com múltiplas estratégias

        Args:
            file_path: Caminho para o arquivo

        Returns:
            Dicionário com informações de encoding detectado
        """
        logger.info(f"Detectando encoding para: {file_path}")

        detection_report = {
            "file_path": file_path,
            "file_size": 0,
            "chardet_detection": None,
            "manual_detection": None,
            "recommended_encoding": None,
            "confidence_score": 0.0,
            "encoding_issues": []
        }

        try:
            # Obter tamanho do arquivo
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                detection_report["file_size"] = file_path_obj.stat().st_size

            # Estratégia 1: Chardet detection
            with open(file_path, 'rb') as f:
                # Ler amostra maior para melhor detecção
                sample_size = min(1024 * 1024, detection_report["file_size"])  # 1MB ou tamanho total
                raw_data = f.read(sample_size)

                chardet_result = chardet.detect(raw_data)
                detection_report["chardet_detection"] = chardet_result

                if chardet_result and chardet_result['confidence'] > 0.7:
                    detection_report["recommended_encoding"] = chardet_result['encoding']
                    detection_report["confidence_score"] = chardet_result['confidence']

            # Estratégia 2: Teste manual com encodings comuns
            manual_results = self._test_common_encodings(file_path)
            detection_report["manual_detection"] = manual_results

            # Estratégia 3: Fallback baseado em heurísticas
            if detection_report["confidence_score"] < 0.8:
                fallback_encoding = self._fallback_encoding_detection(file_path)
                if fallback_encoding:
                    detection_report["recommended_encoding"] = fallback_encoding
                    detection_report["encoding_issues"].append("Low confidence - using fallback detection")

            logger.info(f"Encoding detectado: {detection_report['recommended_encoding']} (confiança: {detection_report['confidence_score']:.2f})")

        except Exception as e:
            logger.error(f"Erro na detecção de encoding: {e}")
            detection_report["encoding_issues"].append(f"Detection error: {str(e)}")
            detection_report["recommended_encoding"] = "utf-8"  # Fallback seguro

        return detection_report

    def enhance_csv_loading_with_fallbacks(self, file_path: str) -> pd.DataFrame:
        """
        Carrega CSV com detecção automática de encoding e separadores

        Args:
            file_path: Caminho para o arquivo CSV

        Returns:
            DataFrame carregado com encoding correto
        """
        logger.info(f"Carregando CSV com fallbacks: {file_path}")

        # Detectar encoding primeiro
        encoding_info = self.detect_encoding_with_chardet(file_path)
        recommended_encoding = encoding_info["recommended_encoding"]

        # Configurações de fallback
        encoding_fallbacks = [recommended_encoding, "utf-8", "latin-1", "iso-8859-1", "cp1252"]
        separator_fallbacks = [",", ";", "\t", "|"]

        loading_report = {
            "file_path": file_path,
            "attempts": [],
            "successful_config": None,
            "df_shape": None,
            "issues_found": []
        }

        # Tentar diferentes combinações
        for encoding in encoding_fallbacks:
            if encoding is None:
                continue

            for separator in separator_fallbacks:
                try:
                    # Configuração de carregamento robusta
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        sep=separator,
                        dtype=str,  # Preservar tudo como string inicialmente
                        na_values=['', 'NA', 'null', 'NULL', 'None'],
                        keep_default_na=True,
                        on_bad_lines='skip',  # Pular linhas problemáticas
                        quoting=csv.QUOTE_MINIMAL,
                        engine='python'  # Mais robusto para CSVs malformados
                    )

                    # Validar se o carregamento faz sentido
                    if self._validate_csv_loading(df):
                        loading_report["successful_config"] = {
                            "encoding": encoding,
                            "separator": separator
                        }
                        loading_report["df_shape"] = df.shape

                        logger.info(f"CSV carregado com sucesso: {encoding} + '{separator}' -> {df.shape}")

                        # Aplicar limpeza básica pós-carregamento
                        df = self._post_loading_cleanup(df)

                        return df

                except Exception as e:
                    loading_report["attempts"].append({
                        "encoding": encoding,
                        "separator": separator,
                        "error": str(e)
                    })
                    continue

        # Se chegou aqui, nenhuma configuração funcionou
        logger.error(f"Falha ao carregar CSV com todas as configurações testadas")
        raise ValueError(f"Não foi possível carregar o arquivo {file_path} com nenhuma configuração testada")

    def _test_common_encodings(self, file_path: str) -> Dict[str, Any]:
        """Testa encodings comuns manualmente"""

        common_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        results = {}

        for encoding in common_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    sample = f.read(1000)  # Ler amostra

                # Verificar qualidade do texto decodificado
                quality_score = self._assess_text_quality(sample)
                results[encoding] = {
                    "readable": True,
                    "quality_score": quality_score,
                    "sample_preview": sample[:100]
                }

            except (UnicodeDecodeError, UnicodeError):
                results[encoding] = {
                    "readable": False,
                    "quality_score": 0.0,
                    "error": "UnicodeDecodeError"
                }

        return results

    def _fallback_encoding_detection(self, file_path: str) -> Optional[str]:
        """Detecção de encoding baseada em heurísticas"""

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(2048)

            # Heurística 1: Presença de BOM
            if raw_data.startswith(b'\xff\xfe'):
                return 'utf-16-le'
            elif raw_data.startswith(b'\xfe\xff'):
                return 'utf-16-be'
            elif raw_data.startswith(b'\xef\xbb\xbf'):
                return 'utf-8-sig'

            # Heurística 2: Caracteres típicos do português
            try:
                # Testar UTF-8
                text = raw_data.decode('utf-8')
                if self._contains_portuguese_chars(text):
                    return 'utf-8'
            except UnicodeDecodeError:
                pass

            # Heurística 3: Fallback para latin-1 (sempre funciona)
            return 'latin-1'

        except Exception:
            return 'utf-8'  # Fallback final

    def _assess_text_quality(self, text: str) -> float:
        """Avalia qualidade do texto decodificado (0-1)"""

        if not text:
            return 0.0

        quality_score = 1.0

        # Penalizar caracteres de substituição
        replacement_chars = text.count('�')
        if replacement_chars > 0:
            quality_score -= (replacement_chars / len(text)) * 2

        # Penalizar sequências suspeitas
        suspicious_patterns = [
            r'Ã[²³¹°]',  # Encoding artifacts
            r'â€[œ™"]',  # Windows-1252 artifacts
            r'[Â]{2,}',   # Repeated artifacts
        ]

        for pattern in suspicious_patterns:
            matches = len(re.findall(pattern, text))
            if matches > 0:
                quality_score -= (matches / len(text.split())) * 0.5

        # Bonus para caracteres portugueses
        portuguese_chars = 'ãçáéíóúàèìòùâêîôûäëïöü'
        portuguese_count = sum(1 for c in text.lower() if c in portuguese_chars)
        if portuguese_count > 0:
            quality_score += min(0.2, portuguese_count / len(text))

        return max(0.0, min(1.0, quality_score))

    def _contains_portuguese_chars(self, text: str) -> bool:
        """Verifica se texto contém caracteres típicos do português"""

        portuguese_indicators = [
            'ção', 'são', 'não', 'também', 'português', 'brasil',
            'á', 'é', 'í', 'ó', 'ú', 'ã', 'õ', 'ç'
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in portuguese_indicators)

    def _validate_csv_loading(self, df: pd.DataFrame) -> bool:
        """Valida se o carregamento do CSV fez sentido"""

        # Verificações básicas
        if df.empty:
            return False

        # Deve ter pelo menos 2 colunas (dados estruturados)
        if len(df.columns) < 2:
            return False

        # Não deve ter linhas demais como NaN
        nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if nan_ratio > 0.8:
            return False

        # Colunas não devem ser todas números (indica separador errado)
        numeric_columns = 0
        for col in df.columns:
            if str(col).replace('.', '').replace(',', '').isdigit():
                numeric_columns += 1

        if numeric_columns == len(df.columns):
            return False

        return True

    def _post_loading_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpeza básica pós-carregamento"""

        # Remover colunas completamente vazias
        df = df.dropna(axis=1, how='all')

        # Remover linhas completamente vazias
        df = df.dropna(axis=0, how='all')

        # Limpar nomes de colunas
        df.columns = [str(col).strip() for col in df.columns]

        # Reset index
        df = df.reset_index(drop=True)

        return df
