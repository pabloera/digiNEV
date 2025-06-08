"""
Limpeza de Texto Inteligente via API Anthropic - Substituto Completo

Este módulo substitui completamente o processamento Python tradicional,
utilizando a API Anthropic para limpeza contextual de texto em português brasileiro.
"""

import pandas as pd
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .base import AnthropicBase
from .api_error_handler import APIErrorHandler, APIQualityChecker

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