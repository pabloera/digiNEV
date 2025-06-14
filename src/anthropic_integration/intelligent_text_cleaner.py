"""
Limpeza de Texto Inteligente via API Anthropic - Substituto Completo

Este módulo substitui completamente o processamento Python tradicional,
utilizando a API Anthropic para limpeza contextual de texto em português brasileiro.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .api_error_handler import APIErrorHandler, APIQualityChecker
from .base import AnthropicBase

logger = logging.getLogger(__name__)

class IntelligentTextCleaner(AnthropicBase):
    """
    Limpeza de texto contextual usando API Anthropic

    Este limpador substitui métodos tradicionais de regex/string
    por análise inteligente que compreende contexto, preserva
    significado e identifica nuances do português brasileiro.

    Capacidades Avançadas:
    - Análise contextual completa
    - Preservação de termos políticos importantes
    - Detecção e correção de problemas específicos
    - Validação de qualidade em tempo real
    - Auto-correção de erros detectados
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.error_handler = APIErrorHandler()
        self.quality_checker = APIQualityChecker(config)

        # Configurações específicas para limpeza
        self.cleaning_config = {
            "preserve_political_terms": True,
            "preserve_hashtags": True,
            "preserve_mentions": True,
            "normalize_whitespace": True,
            "remove_system_messages": True,
            "preserve_emphasis": True,  # CAPS, repetitions for emphasis
            "context": "political_telegram_brazil_2019_2023",
            "quality_validation": True,
            "auto_correction": True
        }

        # Termos críticos que devem ser preservados
        self.critical_terms = {
            "political_figures": [
                "bolsonaro", "lula", "temer", "dilma", "jair", "messias",
                "moro", "dallagnol", "glenn", "moro", "mandetta"
            ],
            "political_expressions": [
                "mito", "lula livre", "fora bolsonaro", "brasil acima de tudo",
                "deus acima de todos", "9 dedos", "bozo", "pocket", "barba"
            ],
            "institutions": [
                "stf", "tse", "pf", "mpf", "congresso", "senado",
                "câmara", "planalto", "brasília", "onu", "oms"
            ]
        }

    def clean_text_intelligent(
        self,
        df: pd.DataFrame,
        text_column: str = "body",
        output_column: str = "text_cleaned",
        batch_size: int = 20
    ) -> pd.DataFrame:
        """
        Realiza limpeza inteligente de texto via API com validação e correção

        Args:
            df: DataFrame com textos para limpar
            text_column: Coluna com texto original
            output_column: Coluna para texto limpo
            batch_size: Tamanho do lote para processamento

        Returns:
            DataFrame com textos limpos e validados
        """
        logger.info(f"Iniciando limpeza inteligente de {len(df)} textos")

        # Fazer backup
        backup_file = f"data/interim/text_cleaning_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False, sep=';', encoding='utf-8')
        logger.info(f"Backup criado: {backup_file}")

        result_df = df.copy()
        result_df[output_column] = ""

        # Processar em lotes
        total_batches = (len(df) + batch_size - 1) // batch_size

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Processando lote {batch_num}/{total_batches}")

            # Usar error handler para processamento com retry
            result = self.error_handler.execute_with_retry(
                self._process_cleaning_batch,
                stage="03_text_cleaning",
                operation=f"batch_{batch_num}",
                batch_df=batch_df,
                text_column=text_column,
                start_index=i
            )

            if result.success:
                cleaned_data = result.data
                # Aplicar textos limpos
                for j, cleaned_text in enumerate(cleaned_data):
                    actual_index = i + j
                    if actual_index < len(result_df):
                        result_df.at[actual_index, output_column] = cleaned_text
            else:
                logger.error(f"Falha no lote {batch_num}: {result.error.error_message}")
                # Fallback: manter textos originais
                for j in range(len(batch_df)):
                    actual_index = i + j
                    if actual_index < len(result_df):
                        result_df.at[actual_index, output_column] = batch_df.iloc[j][text_column]

        # Validação e correção final
        final_validation = self.validate_and_correct_cleaning(result_df, text_column, output_column)

        logger.info("Limpeza inteligente concluída com validação")
        return result_df

    def _process_cleaning_batch(
        self,
        batch_df: pd.DataFrame,
        text_column: str,
        start_index: int
    ) -> List[str]:
        """Processa um lote de textos para limpeza com validação"""

        texts = batch_df[text_column].fillna("").astype(str).tolist()

        # Primeira passada: limpeza inicial
        cleaned_texts = self._clean_batch_via_api(texts)

        # Validação de qualidade
        quality_issues = self._validate_batch_quality(texts, cleaned_texts)

        # Correção se necessário
        if quality_issues and self.cleaning_config.get("auto_correction", True):
            logger.info(f"Detectados {len(quality_issues)} problemas de qualidade, aplicando correções")
            corrected_texts = self._correct_cleaning_issues(texts, cleaned_texts, quality_issues)
            return corrected_texts

        return cleaned_texts

    def _clean_batch_via_api(self, texts: List[str]) -> List[str]:
        """Limpa lote de textos via API com prompt aprimorado"""

        # Preparar textos para prompt
        texts_formatted = "\n".join([
            f"{i+1}. {text[:400]}..." if len(text) > 400 else f"{i+1}. {text}"
            for i, text in enumerate(texts)
        ])

        prompt = f"""
Limpe os seguintes textos de mensagens do Telegram brasileiro (contexto político 2019-2023).

TEXTOS ORIGINAIS:
{texts_formatted}

INSTRUÇÕES CRÍTICAS DE LIMPEZA:
1. PRESERVE ABSOLUTAMENTE: termos políticos, nomes, hashtags, menções
2. PRESERVE: emojis relevantes, ênfase (CAPS), gírias políticas
3. REMOVA APENAS: spam evidente, caracteres corrompidos, mensagens de sistema
4. NORMALIZE: espaços múltiplos, quebras de linha excessivas
5. MANTENHA: tom, significado e contexto original

TERMOS QUE DEVEM SER PRESERVADOS:
- Figuras políticas: Bolsonaro, Lula, Moro, etc.
- Expressões: "Mito", "9 dedos", "Fora", "Brasil acima de tudo"
- Instituições: STF, TSE, PF, MPF, Congresso
- Hashtags políticas: #ForaBolsonaro, #LulaLivre, etc.

CONTEXTO ESPECÍFICO:
- Mensagens de canais pró/anti governo
- Linguagem política brasileira (formal e informal)
- Período: 2019-2023 (pandemia, eleições)

FORMATO DE RESPOSTA:
Responda APENAS com os textos limpos, numerados:

1. [texto limpo preservando contexto político]
2. [texto limpo preservando contexto político]
...

CRÍTICO: Não altere significado político ou remova informação relevante.
"""

        try:
            response = self.create_message(
                prompt,
                stage="03_text_cleaning",
                operation="intelligent_cleaning",
                model=self.model,
                temperature=0.1  # Baixa temperatura para consistência
            )

            # Validar qualidade da resposta
            validation = self.quality_checker.validate_output_quality(
                response,
                expected_format="text",
                context={"texts_count": len(texts), "operation": "text_cleaning"},
                stage="03_text_cleaning"
            )

            if not validation["valid"]:
                logger.warning(f"Qualidade da resposta baixa: {validation['issues']}")

            # Extrair textos limpos da resposta
            cleaned_texts = self._parse_cleaned_texts(response, len(texts))

            return cleaned_texts

        except Exception as e:
            logger.error(f"Erro na limpeza via API: {e}")
            # Fallback: retornar textos originais
            return texts

    def _parse_cleaned_texts(self, response: str, expected_count: int) -> List[str]:
        """Extrai textos limpos da resposta da API com validação"""

        cleaned_texts = []

        # Dividir resposta em linhas
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Procurar por padrão "número. texto"
            match = re.match(r'^\d+\.\s*(.*)', line)
            if match:
                cleaned_text = match.group(1).strip()
                if cleaned_text:  # Não adicionar textos vazios
                    cleaned_texts.append(cleaned_text)
            elif len(cleaned_texts) == 0 and line:
                # Se não há numeração, pode ser resposta simples
                cleaned_texts.append(line)

        # Garantir que temos o número correto de textos
        while len(cleaned_texts) < expected_count:
            cleaned_texts.append("")  # Texto vazio para casos sem resposta

        return cleaned_texts[:expected_count]  # Limitar ao número esperado

    def _validate_batch_quality(self, original_texts: List[str], cleaned_texts: List[str]) -> List[Dict[str, Any]]:
        """Valida qualidade da limpeza em tempo real"""

        quality_issues = []

        for i, (original, cleaned) in enumerate(zip(original_texts, cleaned_texts)):
            issues = []

            # Verificar se texto ficou vazio inadequadamente
            if len(original.strip()) > 20 and len(cleaned.strip()) == 0:
                issues.append("texto_vazio_inadequado")

            # Verificar perda excessiva de conteúdo
            if len(original) > 50 and len(cleaned) < len(original) * 0.3:
                issues.append("perda_conteudo_excessiva")

            # Verificar preservação de termos críticos
            critical_lost = self._check_critical_terms_lost(original, cleaned)
            if critical_lost:
                issues.extend([f"termo_critico_perdido_{term}" for term in critical_lost])

            # Verificar se hashtags importantes foram preservadas
            original_hashtags = re.findall(r'#\w+', original.lower())
            cleaned_hashtags = re.findall(r'#\w+', cleaned.lower())
            lost_hashtags = set(original_hashtags) - set(cleaned_hashtags)
            if len(lost_hashtags) > 0:
                issues.append(f"hashtags_perdidas_{len(lost_hashtags)}")

            if issues:
                quality_issues.append({
                    "index": i,
                    "issues": issues,
                    "original": original,
                    "cleaned": cleaned
                })

        return quality_issues

    def _check_critical_terms_lost(self, original: str, cleaned: str) -> List[str]:
        """Verifica se termos críticos foram perdidos na limpeza"""

        lost_terms = []
        original_lower = original.lower()
        cleaned_lower = cleaned.lower()

        for category, terms in self.critical_terms.items():
            for term in terms:
                if term in original_lower and term not in cleaned_lower:
                    lost_terms.append(term)

        return lost_terms

    def _correct_cleaning_issues(
        self,
        original_texts: List[str],
        cleaned_texts: List[str],
        quality_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Corrige problemas detectados na limpeza"""

        corrected_texts = cleaned_texts.copy()

        # Corrigir cada problema identificado
        for issue_info in quality_issues:
            index = issue_info["index"]
            issues = issue_info["issues"]
            original = issue_info["original"]

            # Estratégias de correção baseadas no tipo de problema
            if "texto_vazio_inadequado" in issues:
                # Usar API para re-limpeza mais conservadora
                corrected = self._conservative_cleaning_api(original)
                if corrected:
                    corrected_texts[index] = corrected
                else:
                    corrected_texts[index] = original  # Manter original se falhar

            elif any("termo_critico_perdido" in issue for issue in issues):
                # Tentar preservar termos críticos
                corrected = self._preserve_critical_terms_api(original, corrected_texts[index])
                if corrected:
                    corrected_texts[index] = corrected

            elif "perda_conteudo_excessiva" in issues:
                # Re-limpar com instruções mais conservadoras
                corrected = self._conservative_cleaning_api(original)
                if corrected and len(corrected) > len(corrected_texts[index]):
                    corrected_texts[index] = corrected

        return corrected_texts

    def _conservative_cleaning_api(self, text: str) -> str:
        """Limpeza mais conservadora via API"""

        prompt = f"""
Limpe CONSERVADORAMENTE o seguinte texto, removendo apenas ruído óbvio:

TEXTO: {text}

INSTRUÇÕES:
- PRESERVE todo conteúdo político relevante
- REMOVA apenas: caracteres corrompidos óbvios, spam evidente
- NORMALIZE apenas espaços em excesso
- MANTENHA: termos políticos, hashtags, menções, emojis relevantes

Responda APENAS com o texto limpo, sem numeração ou explicações:
"""

        try:
            response = self.create_message(
                prompt,
                stage="03_text_cleaning",
                operation="conservative_correction",
                temperature=0.0
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Erro na limpeza conservadora: {e}")
            return ""

    def _preserve_critical_terms_api(self, original: str, cleaned: str) -> str:
        """Preserva termos críticos usando API"""

        prompt = f"""
O texto limpo perdeu termos políticos importantes. Corrija preservando termos críticos:

ORIGINAL: {original}
LIMPO: {cleaned}

INSTRUÇÕES:
- Identifique termos políticos perdidos na limpeza
- Reintegre termos importantes mantendo a limpeza
- PRESERVE: nomes políticos, expressões políticas, hashtags importantes

Responda APENAS com o texto corrigido:
"""

        try:
            response = self.create_message(
                prompt,
                stage="03_text_cleaning",
                operation="term_preservation",
                temperature=0.0
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Erro na preservação de termos: {e}")
            return cleaned  # Retornar texto limpo original se falhar

    def validate_and_correct_cleaning(
        self,
        df: pd.DataFrame,
        original_column: str,
        cleaned_column: str
    ) -> Dict[str, Any]:
        """Validação e correção final da limpeza"""

        logger.info("Executando validação final da limpeza")

        validation_report = self.validate_cleaning_quality(df, df, original_column, cleaned_column)

        # Aplicar correções se necessário
        if validation_report.get("issues_detected"):
            logger.info(f"Aplicando correções finais para {len(validation_report['issues_detected'])} problemas")

            # Identificar textos problemáticos
            problematic_indices = self._identify_problematic_texts(df, original_column, cleaned_column)

            # Corrigir textos problemáticos
            if problematic_indices:
                corrections_applied = self._apply_final_corrections(df, problematic_indices, original_column, cleaned_column)
                validation_report["corrections_applied"] = corrections_applied

        return validation_report

    def _identify_problematic_texts(
        self,
        df: pd.DataFrame,
        original_column: str,
        cleaned_column: str
    ) -> List[int]:
        """Identifica índices de textos com problemas"""

        problematic_indices = []

        for idx in df.index:
            original = str(df.loc[idx, original_column])
            cleaned = str(df.loc[idx, cleaned_column])

            # Critérios para textos problemáticos
            if len(original.strip()) > 20 and len(cleaned.strip()) == 0:
                problematic_indices.append(idx)
            elif len(original) > 50 and len(cleaned) < len(original) * 0.2:
                problematic_indices.append(idx)
            elif self._check_critical_terms_lost(original, cleaned):
                problematic_indices.append(idx)

        return problematic_indices[:50]  # Limitar para performance

    def _apply_final_corrections(
        self,
        df: pd.DataFrame,
        problematic_indices: List[int],
        original_column: str,
        cleaned_column: str
    ) -> List[str]:
        """Aplica correções finais a textos problemáticos"""

        corrections_applied = []

        for idx in problematic_indices:
            original = str(df.loc[idx, original_column])

            # Tentar limpeza conservadora
            corrected = self._conservative_cleaning_api(original)

            if corrected and len(corrected.strip()) > 0:
                df.loc[idx, cleaned_column] = corrected
                corrections_applied.append(f"Índice {idx}: limpeza conservadora aplicada")
            else:
                # Último recurso: manter original
                df.loc[idx, cleaned_column] = original
                corrections_applied.append(f"Índice {idx}: texto original mantido")

        return corrections_applied

    def validate_cleaning_quality(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        original_column: str = "body",
        cleaned_column: str = "text_cleaned"
    ) -> Dict[str, Any]:
        """
        Valida qualidade da limpeza realizada com métricas avançadas

        Args:
            original_df: DataFrame original
            cleaned_df: DataFrame com textos limpos
            original_column: Coluna com texto original
            cleaned_column: Coluna com texto limpo

        Returns:
            Relatório detalhado de qualidade da limpeza
        """

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_texts": len(cleaned_df),
            "cleaning_statistics": {},
            "quality_metrics": {},
            "preservation_analysis": {},
            "issues_detected": [],
            "recommendations": []
        }

        if cleaned_column in cleaned_df.columns and original_column in original_df.columns:
            original_texts = original_df[original_column].fillna("").astype(str)
            cleaned_texts = cleaned_df[cleaned_column].fillna("").astype(str)

            # Estatísticas básicas
            report["cleaning_statistics"] = {
                "avg_original_length": round(original_texts.str.len().mean(), 2),
                "avg_cleaned_length": round(cleaned_texts.str.len().mean(), 2),
                "texts_unchanged": int((original_texts == cleaned_texts).sum()),
                "texts_modified": int((original_texts != cleaned_texts).sum()),
                "empty_after_cleaning": int((cleaned_texts.str.strip() == "").sum())
            }

            # Métricas de qualidade avançadas
            modification_rate = report["cleaning_statistics"]["texts_modified"] / len(cleaned_df)
            empty_rate = report["cleaning_statistics"]["empty_after_cleaning"] / len(cleaned_df)

            report["quality_metrics"] = {
                "modification_rate": round(modification_rate * 100, 2),
                "empty_rate": round(empty_rate * 100, 2),
                "avg_length_reduction": round(
                    (report["cleaning_statistics"]["avg_original_length"] -
                     report["cleaning_statistics"]["avg_cleaned_length"]) /
                    report["cleaning_statistics"]["avg_original_length"] * 100, 2
                ) if report["cleaning_statistics"]["avg_original_length"] > 0 else 0
            }

            # Análise de preservação de elementos importantes
            report["preservation_analysis"] = self._analyze_preservation(original_texts, cleaned_texts)

            # Detectar problemas
            if empty_rate > 0.05:  # Mais de 5% vazios
                report["issues_detected"].append(f"Taxa alta de textos vazios: {empty_rate:.1%}")

            if modification_rate < 0.1:  # Menos de 10% modificados
                report["issues_detected"].append("Taxa muito baixa de modificação - limpeza pode estar inadequada")

            if modification_rate > 0.8:  # Mais de 80% modificados
                report["issues_detected"].append("Taxa muito alta de modificação - limpeza pode estar agressiva demais")

            if report["preservation_analysis"]["critical_terms_lost"] > 10:
                report["issues_detected"].append("Muitos termos críticos perdidos na limpeza")

            # Gerar recomendações
            if report["quality_metrics"]["avg_length_reduction"] > 50:
                report["recommendations"].append("Redução de comprimento muito alta - verificar se informação importante não foi perdida")

            if empty_rate > 0.1:
                report["recommendations"].append("Muitos textos ficaram vazios - ajustar critérios de limpeza")

            if report["preservation_analysis"]["hashtag_preservation_rate"] < 80:
                report["recommendations"].append("Taxa baixa de preservação de hashtags - melhorar instruções de limpeza")

        return report

    def _analyze_preservation(self, original_texts: pd.Series, cleaned_texts: pd.Series) -> Dict[str, Any]:
        """Analisa preservação de elementos importantes"""

        analysis = {
            "hashtag_preservation_rate": 0,
            "mention_preservation_rate": 0,
            "critical_terms_lost": 0,
            "emoji_preservation_rate": 0
        }

        total_samples = min(1000, len(original_texts))  # Amostra para performance
        sample_indices = range(total_samples)

        hashtag_preserved = 0
        mention_preserved = 0
        emoji_preserved = 0
        critical_lost_count = 0

        for i in sample_indices:
            original = original_texts.iloc[i]
            cleaned = cleaned_texts.iloc[i]

            # Hashtags
            orig_hashtags = set(re.findall(r'#\w+', original.lower()))
            clean_hashtags = set(re.findall(r'#\w+', cleaned.lower()))
            if orig_hashtags:
                preservation_rate = len(orig_hashtags.intersection(clean_hashtags)) / len(orig_hashtags)
                hashtag_preserved += preservation_rate

            # Mentions
            orig_mentions = set(re.findall(r'@\w+', original.lower()))
            clean_mentions = set(re.findall(r'@\w+', cleaned.lower()))
            if orig_mentions:
                preservation_rate = len(orig_mentions.intersection(clean_mentions)) / len(orig_mentions)
                mention_preserved += preservation_rate

            # Emojis
            orig_emojis = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', original))
            clean_emojis = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', cleaned))
            if orig_emojis > 0:
                emoji_preserved += min(clean_emojis / orig_emojis, 1.0)

            # Termos críticos
            lost_terms = self._check_critical_terms_lost(original, cleaned)
            critical_lost_count += len(lost_terms)

        # Calcular médias
        if total_samples > 0:
            analysis["hashtag_preservation_rate"] = round((hashtag_preserved / total_samples) * 100, 2)
            analysis["mention_preservation_rate"] = round((mention_preserved / total_samples) * 100, 2)
            analysis["emoji_preservation_rate"] = round((emoji_preserved / total_samples) * 100, 2)

        analysis["critical_terms_lost"] = critical_lost_count

        return analysis

    def enhance_text_cleaning_with_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Limpeza de texto aprimorada com validação robusta

        Args:
            df: DataFrame para limpar

        Returns:
            Tuple com DataFrame limpo e relatório de limpeza
        """
        logger.info(f"Iniciando limpeza de texto aprimorada para {len(df)} registros")

        cleaning_report = {
            "timestamp": datetime.now().isoformat(),
            "original_count": len(df),
            "cleaning_strategy": "enhanced_with_validation",
            "pre_cleaning_analysis": {},
            "cleaning_results": {},
            "post_cleaning_analysis": {},
            "validation_results": {},
            "quality_score": 0.0,
            "recommendations": []
        }

        # Fazer backup
        backup_file = f"data/interim/enhanced_cleaning_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False, sep=';', encoding='utf-8')
        logger.info(f"Backup criado: {backup_file}")

        result_df = df.copy()

        # Detectar coluna de texto
        text_column = self._detect_text_column(result_df)
        cleaning_report["text_column_used"] = text_column

        # Análise pré-limpeza
        pre_analysis = self._pre_cleaning_analysis(result_df, text_column)
        cleaning_report["pre_cleaning_analysis"] = pre_analysis

        # Aplicar estratégias de limpeza graduais
        result_df = self._apply_graduated_cleaning(result_df, text_column)

        # Análise pós-limpeza
        post_analysis = self._post_cleaning_analysis(result_df, text_column)
        cleaning_report["post_cleaning_analysis"] = post_analysis

        # Validação robusta
        validation_results = self._robust_cleaning_validation(df, result_df, text_column)
        cleaning_report["validation_results"] = validation_results

        # Correção de problemas detectados
        if validation_results.get("critical_issues"):
            result_df = self._fix_critical_cleaning_issues(df, result_df, text_column, validation_results)
            cleaning_report["corrections_applied"] = True

        # Calcular score de qualidade final
        cleaning_report["quality_score"] = self._calculate_cleaning_quality_score(cleaning_report)

        # Gerar recomendações
        cleaning_report["recommendations"] = self._generate_cleaning_recommendations(cleaning_report)

        logger.info(f"Limpeza aprimorada concluída. Score de qualidade: {cleaning_report['quality_score']:.2f}")

        return result_df, cleaning_report

    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """Detecta automaticamente a melhor coluna de texto"""

        # Prioridade: body_cleaned > body > texto > content
        text_candidates = ['body_cleaned', 'body', 'texto', 'content', 'text', 'message']

        for candidate in text_candidates:
            if candidate in df.columns:
                # Verificar se tem conteúdo útil
                non_empty = df[candidate].dropna().astype(str).str.len().gt(10).sum()
                if non_empty > len(df) * 0.1:  # Pelo menos 10% com conteúdo
                    logger.info(f"Coluna de texto detectada: {candidate}")
                    return candidate

        # Fallback para primeira coluna string
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.warning(f"Usando fallback para coluna de texto: {col}")
                return col

        logger.error("Nenhuma coluna de texto encontrada")
        return 'body'  # Fallback final

    def _pre_cleaning_analysis(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Análise detalhada antes da limpeza"""

        if text_column not in df.columns:
            return {"error": f"Coluna '{text_column}' não encontrada"}

        text_series = df[text_column].fillna("").astype(str)

        analysis = {
            "total_texts": len(text_series),
            "empty_texts": (text_series.str.strip() == "").sum(),
            "very_short_texts": (text_series.str.len() < 10).sum(),
            "very_long_texts": (text_series.str.len() > 1000).sum(),
            "avg_length": round(text_series.str.len().mean(), 2),
            "encoding_issues": self._detect_encoding_issues(text_series),
            "formatting_issues": self._detect_formatting_issues(text_series),
            "content_diversity": text_series.nunique(),
            "special_characters": self._analyze_special_characters(text_series)
        }

        return analysis

    def _detect_encoding_issues(self, text_series: pd.Series) -> Dict[str, int]:
        """Detecta problemas de encoding"""

        issues = {
            "replacement_chars": text_series.str.contains('�').sum(),
            "control_chars": text_series.str.contains(r'[\x00-\x1f]', regex=True).sum(),
            "suspicious_sequences": text_series.str.contains(r'Ã[²³¹°]|â€[œ™"]', regex=True).sum(),
            "mixed_encodings": text_series.str.contains(r'[À-ÿ]{3,}', regex=True).sum()
        }

        return issues

    def _detect_formatting_issues(self, text_series: pd.Series) -> Dict[str, int]:
        """Detecta problemas de formatação"""

        issues = {
            "multiple_spaces": text_series.str.contains(r'  +').sum(),
            "multiple_newlines": text_series.str.contains(r'\n{2,}').sum(),
            "leading_trailing_spaces": (text_series != text_series.str.strip()).sum(),
            "html_tags": text_series.str.contains(r'<[^>]+>').sum(),
            "url_fragments": text_series.str.contains(r'https?://').sum()
        }

        return issues

    def _analyze_special_characters(self, text_series: pd.Series) -> Dict[str, int]:
        """Analisa caracteres especiais"""

        analysis = {
            "hashtags": text_series.str.count(r'#\w+').sum(),
            "mentions": text_series.str.count(r'@\w+').sum(),
            "emojis": text_series.str.count(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]').sum(),
            "punctuation_clusters": text_series.str.count(r'[.!?]{2,}').sum()
        }

        return analysis

    def _apply_graduated_cleaning(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Aplica limpeza em etapas graduais com validação"""

        result_df = df.copy()
        cleaned_column = f"{text_column}_cleaned"

        # Etapa 1: Normalização Unicode (NFKC)
        logger.info("Aplicando normalização Unicode...")
        result_df[cleaned_column] = result_df[text_column].fillna("").astype(str)
        result_df[cleaned_column] = result_df[cleaned_column].apply(
            lambda x: unicodedata.normalize('NFKC', x) if x else x
        )

        # Etapa 2: Limpeza básica de caracteres de controle
        logger.info("Removendo caracteres de controle...")
        result_df[cleaned_column] = result_df[cleaned_column].str.replace(
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', regex=True
        )

        # Etapa 3: Normalização de espaços
        logger.info("Normalizando espaços em branco...")
        result_df[cleaned_column] = result_df[cleaned_column].str.replace(
            r'\s+', ' ', regex=True
        ).str.strip()

        # Etapa 4: Remoção de artefatos do Telegram
        logger.info("Removendo artefatos do Telegram...")
        result_df[cleaned_column] = self._remove_telegram_artifacts(result_df[cleaned_column])

        # Etapa 5: Limpeza conservadora com fallback
        logger.info("Aplicando limpeza conservadora...")
        result_df[cleaned_column] = self._conservative_cleaning_with_fallback(
            result_df[text_column], result_df[cleaned_column]
        )

        return result_df

    def _remove_telegram_artifacts(self, text_series: pd.Series) -> pd.Series:
        """Remove artefatos específicos do Telegram"""

        # Padrões de artefatos do Telegram
        telegram_patterns = [
            r'\[.*?\]',  # [foto], [vídeo], etc.
            r'Mensagem encaminhada.*?\n',
            r'Forwarded from.*?\n',
            r'\n\n+',  # Múltiplas quebras de linha
            r'^\s*-\s*',  # Hífens iniciais
            r'\s*Sent via.*$',  # Assinaturas de apps
        ]

        cleaned_series = text_series.copy()

        for pattern in telegram_patterns:
            cleaned_series = cleaned_series.str.replace(pattern, ' ', regex=True)

        # Normalizar espaços novamente
        cleaned_series = cleaned_series.str.replace(r'\s+', ' ', regex=True).str.strip()

        return cleaned_series

    def _conservative_cleaning_with_fallback(
        self,
        original_series: pd.Series,
        partially_cleaned_series: pd.Series
    ) -> pd.Series:
        """Aplica limpeza conservadora com fallback para original"""

        result_series = partially_cleaned_series.copy()

        for i in range(len(result_series)):
            original = str(original_series.iloc[i])
            cleaned = str(result_series.iloc[i])

            # Validação por item
            if len(original.strip()) > 20 and len(cleaned.strip()) == 0:
                # Fallback: manter original se limpeza resultou em vazio
                result_series.iloc[i] = original
            elif len(original) > 50 and len(cleaned) < len(original) * 0.3:
                # Fallback: re-aplicar limpeza mais conservadora
                conservative_clean = self._single_text_conservative_clean(original)
                result_series.iloc[i] = conservative_clean if conservative_clean else original

        return result_series

    def _single_text_conservative_clean(self, text: str) -> str:
        """Limpeza conservadora para um único texto"""

        if not text or len(text.strip()) == 0:
            return text

        # Aplicar apenas limpezas seguras
        cleaned = text

        # Normalizar Unicode
        cleaned = unicodedata.normalize('NFKC', cleaned)

        # Remover apenas caracteres de controle óbvios
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)

        # Normalizar espaços (conservador)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def _post_cleaning_analysis(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Análise após a limpeza"""

        cleaned_column = f"{text_column}_cleaned"

        if cleaned_column not in df.columns:
            return {"error": f"Coluna '{cleaned_column}' não encontrada"}

        text_series = df[cleaned_column].fillna("").astype(str)

        analysis = {
            "total_texts": len(text_series),
            "empty_texts": (text_series.str.strip() == "").sum(),
            "avg_length": round(text_series.str.len().mean(), 2),
            "content_diversity": text_series.nunique(),
            "preserved_elements": self._count_preserved_elements(text_series),
            "cleaning_artifacts": self._detect_cleaning_artifacts(text_series)
        }

        return analysis

    def _count_preserved_elements(self, text_series: pd.Series) -> Dict[str, int]:
        """Conta elementos preservados após limpeza"""

        preserved = {
            "hashtags": text_series.str.count(r'#\w+').sum(),
            "mentions": text_series.str.count(r'@\w+').sum(),
            "urls": text_series.str.count(r'https?://\S+').sum(),
            "emojis": text_series.str.count(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]').sum()
        }

        return preserved

    def _detect_cleaning_artifacts(self, text_series: pd.Series) -> Dict[str, int]:
        """Detecta artefatos deixados pela limpeza"""

        artifacts = {
            "double_spaces": text_series.str.contains(r'  +').sum(),
            "orphaned_punctuation": text_series.str.count(r'\s[.!?]\s').sum(),
            "incomplete_sentences": text_series.str.count(r'\w\s*$').sum(),
            "empty_lines": text_series.str.count(r'\n\s*\n').sum()
        }

        return artifacts

    def _robust_cleaning_validation(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        text_column: str
    ) -> Dict[str, Any]:
        """Validação robusta do processo de limpeza"""

        cleaned_column = f"{text_column}_cleaned"

        validation = {
            "validation_passed": True,
            "critical_issues": [],
            "warnings": [],
            "statistics": {},
            "quality_metrics": {}
        }

        if text_column in original_df.columns and cleaned_column in cleaned_df.columns:
            original_texts = original_df[text_column].fillna("").astype(str)
            cleaned_texts = cleaned_df[cleaned_column].fillna("").astype(str)

            # Estatísticas de transformação
            validation["statistics"] = {
                "texts_processed": len(cleaned_texts),
                "texts_unchanged": (original_texts == cleaned_texts).sum(),
                "texts_empty_after": (cleaned_texts.str.strip() == "").sum(),
                "avg_length_change": round(
                    cleaned_texts.str.len().mean() - original_texts.str.len().mean(), 2
                ),
                "length_reduction_ratio": round(
                    1 - (cleaned_texts.str.len().mean() / original_texts.str.len().mean()), 4
                ) if original_texts.str.len().mean() > 0 else 0
            }

            # Verificações críticas
            empty_rate = validation["statistics"]["texts_empty_after"] / len(cleaned_texts)
            if empty_rate > 0.1:  # Mais de 10% vazios
                validation["critical_issues"].append(f"Taxa alta de textos vazios: {empty_rate:.1%}")
                validation["validation_passed"] = False

            length_reduction = validation["statistics"]["length_reduction_ratio"]
            if length_reduction > 0.5:  # Mais de 50% de redução
                validation["critical_issues"].append(f"Redução excessiva de conteúdo: {length_reduction:.1%}")
                validation["validation_passed"] = False

            # Análise de preservação
            preservation_analysis = self._detailed_preservation_analysis(original_texts, cleaned_texts)
            validation["preservation_analysis"] = preservation_analysis

            if preservation_analysis["critical_terms_lost"] > 50:
                validation["critical_issues"].append("Muitos termos críticos perdidos")
                validation["validation_passed"] = False

            if preservation_analysis["hashtag_preservation_rate"] < 70:
                validation["warnings"].append("Taxa baixa de preservação de hashtags")

            # Métricas de qualidade
            validation["quality_metrics"] = {
                "content_preservation": 100 - (length_reduction * 100),
                "structure_preservation": preservation_analysis["structure_preservation_rate"],
                "element_preservation": preservation_analysis["element_preservation_rate"],
                "overall_quality": self._calculate_validation_quality_score(validation)
            }

        return validation

    def _detailed_preservation_analysis(self, original_texts: pd.Series, cleaned_texts: pd.Series) -> Dict[str, Any]:
        """Análise detalhada de preservação de elementos"""

        analysis = {
            "hashtag_preservation_rate": 0,
            "mention_preservation_rate": 0,
            "url_preservation_rate": 0,
            "emoji_preservation_rate": 0,
            "critical_terms_lost": 0,
            "structure_preservation_rate": 0,
            "element_preservation_rate": 0
        }

        total_samples = min(500, len(original_texts))  # Amostra para performance
        preserved_elements = 0
        total_elements = 0
        structure_preserved = 0

        for i in range(total_samples):
            original = original_texts.iloc[i]
            cleaned = cleaned_texts.iloc[i]

            # Hashtags
            orig_hashtags = set(re.findall(r'#\w+', original.lower()))
            clean_hashtags = set(re.findall(r'#\w+', cleaned.lower()))
            if orig_hashtags:
                preserved = len(orig_hashtags.intersection(clean_hashtags))
                analysis["hashtag_preservation_rate"] += preserved / len(orig_hashtags)
                preserved_elements += preserved
                total_elements += len(orig_hashtags)

            # Mentions
            orig_mentions = set(re.findall(r'@\w+', original.lower()))
            clean_mentions = set(re.findall(r'@\w+', cleaned.lower()))
            if orig_mentions:
                preserved = len(orig_mentions.intersection(clean_mentions))
                analysis["mention_preservation_rate"] += preserved / len(orig_mentions)
                preserved_elements += preserved
                total_elements += len(orig_mentions)

            # URLs
            orig_urls = set(re.findall(r'https?://\S+', original))
            clean_urls = set(re.findall(r'https?://\S+', cleaned))
            if orig_urls:
                preserved = len(orig_urls.intersection(clean_urls))
                analysis["url_preservation_rate"] += preserved / len(orig_urls)
                preserved_elements += preserved
                total_elements += len(orig_urls)

            # Termos críticos
            critical_lost = self._check_critical_terms_lost(original, cleaned)
            analysis["critical_terms_lost"] += len(critical_lost)

            # Estrutura (frases básicas)
            orig_sentences = len(re.findall(r'[.!?]+', original))
            clean_sentences = len(re.findall(r'[.!?]+', cleaned))
            if orig_sentences > 0:
                structure_preserved += min(clean_sentences / orig_sentences, 1.0)

        # Calcular médias
        if total_samples > 0:
            analysis["hashtag_preservation_rate"] = round((analysis["hashtag_preservation_rate"] / total_samples) * 100, 2)
            analysis["mention_preservation_rate"] = round((analysis["mention_preservation_rate"] / total_samples) * 100, 2)
            analysis["url_preservation_rate"] = round((analysis["url_preservation_rate"] / total_samples) * 100, 2)
            analysis["structure_preservation_rate"] = round((structure_preserved / total_samples) * 100, 2)

        if total_elements > 0:
            analysis["element_preservation_rate"] = round((preserved_elements / total_elements) * 100, 2)

        return analysis

    def _fix_critical_cleaning_issues(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        text_column: str,
        validation_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Corrige problemas críticos detectados na validação"""

        logger.info("Aplicando correções para problemas críticos de limpeza")

        cleaned_column = f"{text_column}_cleaned"
        result_df = cleaned_df.copy()

        # Identificar textos problemáticos
        problematic_indices = []

        for i in range(len(result_df)):
            original = str(original_df.iloc[i][text_column])
            cleaned = str(result_df.iloc[i][cleaned_column])

            # Critérios para problemas críticos
            if len(original.strip()) > 20 and len(cleaned.strip()) == 0:
                problematic_indices.append(i)
            elif len(original) > 100 and len(cleaned) < len(original) * 0.2:
                problematic_indices.append(i)
            elif self._check_critical_terms_lost(original, cleaned):
                problematic_indices.append(i)

        # Aplicar correções
        for idx in problematic_indices[:100]:  # Limitar para performance
            original = str(original_df.iloc[idx][text_column])

            # Estratégia 1: Limpeza ultra-conservadora
            ultra_conservative = self._ultra_conservative_clean(original)

            if ultra_conservative and len(ultra_conservative.strip()) > len(original) * 0.5:
                result_df.iloc[idx, result_df.columns.get_loc(cleaned_column)] = ultra_conservative
            else:
                # Estratégia 2: Manter original se tudo falhar
                result_df.iloc[idx, result_df.columns.get_loc(cleaned_column)] = original

        logger.info(f"Correções aplicadas a {len(problematic_indices)} textos problemáticos")

        return result_df

    def _ultra_conservative_clean(self, text: str) -> str:
        """Limpeza ultra-conservadora que apenas remove ruído óbvio"""

        if not text or len(text.strip()) == 0:
            return text

        cleaned = text

        # Apenas remoções muito seguras
        cleaned = unicodedata.normalize('NFKC', cleaned)  # Normalização Unicode
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)  # Caracteres de controle
        cleaned = re.sub(r'  +', ' ', cleaned)  # Espaços duplos
        cleaned = cleaned.strip()  # Espaços nas bordas

        return cleaned

    def _calculate_validation_quality_score(self, validation: Dict[str, Any]) -> float:
        """Calcula score de qualidade da validação"""

        score = 1.0

        # Penalizar problemas críticos
        critical_issues = len(validation.get("critical_issues", []))
        score -= critical_issues * 0.3

        # Penalizar avisos
        warnings = len(validation.get("warnings", []))
        score -= warnings * 0.1

        # Considerar estatísticas
        stats = validation.get("statistics", {})
        empty_rate = stats.get("texts_empty_after", 0) / max(stats.get("texts_processed", 1), 1)
        score -= empty_rate * 0.5

        return max(0.0, min(1.0, score))

    def _calculate_cleaning_quality_score(self, cleaning_report: Dict[str, Any]) -> float:
        """Calcula score geral de qualidade da limpeza"""

        validation_results = cleaning_report.get("validation_results", {})
        quality_metrics = validation_results.get("quality_metrics", {})

        # Fatores de qualidade
        content_preservation = quality_metrics.get("content_preservation", 0) / 100
        structure_preservation = quality_metrics.get("structure_preservation", 0) / 100
        element_preservation = quality_metrics.get("element_preservation", 0) / 100

        # Penalizar problemas críticos
        critical_penalty = len(validation_results.get("critical_issues", [])) * 0.2

        # Calcular score ponderado
        score = (
            content_preservation * 0.4 +
            structure_preservation * 0.3 +
            element_preservation * 0.3
        ) - critical_penalty

        return max(0.0, min(1.0, score))

    def _generate_cleaning_recommendations(self, cleaning_report: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas no relatório de limpeza"""

        recommendations = []
        quality_score = cleaning_report.get("quality_score", 0)
        validation_results = cleaning_report.get("validation_results", {})

        # Baseado no score de qualidade
        if quality_score < 0.5:
            recommendations.append("Qualidade da limpeza insatisfatória - revisar parâmetros e estratégia")
        elif quality_score < 0.7:
            recommendations.append("Qualidade da limpeza razoável - considerar ajustes menores")
        else:
            recommendations.append("Qualidade da limpeza satisfatória")

        # Baseado em problemas críticos
        critical_issues = validation_results.get("critical_issues", [])
        if critical_issues:
            recommendations.extend([f"Corrigir problema crítico: {issue}" for issue in critical_issues])

        # Baseado em avisos
        warnings = validation_results.get("warnings", [])
        if warnings:
            recommendations.extend([f"Atenção: {warning}" for warning in warnings])

        # Baseado em estatísticas
        stats = validation_results.get("statistics", {})
        empty_rate = stats.get("texts_empty_after", 0) / max(stats.get("texts_processed", 1), 1)
        if empty_rate > 0.05:
            recommendations.append(f"Taxa de textos vazios alta ({empty_rate:.1%}) - ajustar critérios")

        length_reduction = stats.get("length_reduction_ratio", 0)
        if length_reduction > 0.4:
            recommendations.append(f"Redução de conteúdo excessiva ({length_reduction:.1%}) - usar limpeza mais conservadora")

        if not recommendations:
            recommendations.append("Processo de limpeza executado com êxito - nenhuma ação adicional necessária")

        return recommendations
