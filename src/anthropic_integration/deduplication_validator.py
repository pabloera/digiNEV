"""
Validador de Deduplicação via API Anthropic
Garante exclusão de mensagens de mídia da deduplicação e valida contagem de duplicatas.
"""

import pandas as pd
import json
import logging
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
from .base import AnthropicBase
from .api_error_handler import APIErrorHandler, APIQualityChecker

logger = logging.getLogger(__name__)


class DeduplicationValidator(AnthropicBase):
    """
    Validador avançado de deduplicação usando API Anthropic
    
    Capacidades:
    - Detecção inteligente de mensagens de mídia
    - Validação de exclusão correta da deduplicação
    - Análise de padrões de duplicação
    - Detecção de coordenação suspeita
    - Validação de contagem de duplicatas
    - Identificação de falsos positivos/negativos
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.error_handler = APIErrorHandler()
        self.quality_checker = APIQualityChecker(config)
        
        # Padrões para detectar mensagens de mídia
        self.media_patterns = {
            "photo_indicators": [
                r"\[foto\]", r"\[imagem\]", r"\[image\]", r"\[photo\]",
                r"foto anexada", r"imagem anexada", r"enviou uma foto",
                r"enviou uma imagem", r"compartilhou uma foto",
                r"\.(jpg|jpeg|png|gif|bmp|webp)", r"📷", r"📸", r"🖼️"
            ],
            "video_indicators": [
                r"\[vídeo\]", r"\[video\]", r"\[filme\]", r"\[movie\]",
                r"vídeo anexado", r"video anexado", r"enviou um vídeo",
                r"enviou um video", r"compartilhou um vídeo",
                r"\.(mp4|avi|mov|wmv|flv|webm|mkv)", r"🎥", r"📹", r"🎬"
            ],
            "audio_indicators": [
                r"\[áudio\]", r"\[audio\]", r"\[som\]", r"\[voice\]",
                r"áudio anexado", r"audio anexado", r"enviou um áudio",
                r"enviou um audio", r"mensagem de voz", r"nota de voz",
                r"\.(mp3|wav|ogg|m4a|aac|flac)", r"🎵", r"🎶", r"🔊", r"🎤"
            ],
            "document_indicators": [
                r"\[documento\]", r"\[doc\]", r"\[arquivo\]", r"\[file\]",
                r"documento anexado", r"arquivo anexado", r"enviou um arquivo",
                r"compartilhou um documento", r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx)",
                r"📄", r"📃", r"📋", r"📊", r"📈", r"📉"
            ],
            "sticker_indicators": [
                r"\[sticker\]", r"\[adesivo\]", r"\[figurinha\]",
                r"enviou um sticker", r"enviou uma figurinha",
                r"enviou um adesivo", r"😀", r"😂", r"❤️"
            ],
            "location_indicators": [
                r"\[localização\]", r"\[location\]", r"\[local\]",
                r"compartilhou localização", r"enviou localização",
                r"📍", r"🗺️", r"🌍", r"🌎", r"🌏"
            ]
        }
        
        # Padrões para detectar coordenação suspeita
        self.coordination_patterns = {
            "timing_suspicious": "mensagens idênticas em intervalo < 5 minutos",
            "exact_duplicates": "texto exatamente igual em múltiplos canais",
            "template_messages": "estrutura similar com variações mínimas",
            "bot_signatures": "padrões de formatação automatizada"
        }
    
    def validate_deduplication_process(
        self,
        original_df: pd.DataFrame,
        deduplicated_df: pd.DataFrame,
        duplicate_count_column: str = "duplicate_count",
        text_column: str = None
    ) -> Dict[str, Any]:
        """
        Valida processo de deduplicação completo
        
        Args:
            original_df: DataFrame original antes da deduplicação
            deduplicated_df: DataFrame após deduplicação
            duplicate_count_column: Nome da coluna com contagem de duplicatas
            text_column: Nome da coluna de texto (detectado automaticamente se None)
            
        Returns:
            Relatório de validação da deduplicação
        """
        logger.info(f"Validando deduplicação: {len(original_df)} -> {len(deduplicated_df)} registros")
        
        # Detectar coluna de texto automaticamente se não fornecida
        if text_column is None:
            text_column = self._detect_text_column(original_df)
        
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "original_count": len(original_df),
            "deduplicated_count": len(deduplicated_df),
            "reduction_ratio": (len(original_df) - len(deduplicated_df)) / len(original_df),
            "text_column_used": text_column,
            "media_analysis": self._analyze_media_exclusion(original_df, deduplicated_df, text_column),
            "duplicate_count_validation": self._validate_duplicate_counts(deduplicated_df, duplicate_count_column),
            "coordination_analysis": self._analyze_coordination_patterns(deduplicated_df),
            "quality_assessment": {},
            "recommendations": []
        }
        
        # Análise detalhada via API
        api_analysis = self._detailed_deduplication_analysis_api(original_df, deduplicated_df)
        validation_report["api_analysis"] = api_analysis
        
        # Calcular score de qualidade
        validation_report["quality_assessment"] = self._calculate_deduplication_quality(validation_report)
        
        # Gerar recomendações
        validation_report["recommendations"] = self._generate_deduplication_recommendations(validation_report)
        
        logger.info(f"Validação concluída. Qualidade: {validation_report['quality_assessment'].get('overall_score', 'N/A')}")
        
        return validation_report
    
    def detect_media_messages(
        self,
        df: pd.DataFrame,
        text_column: str = "body",
        batch_size: int = 20
    ) -> pd.DataFrame:
        """
        Detecta mensagens de mídia usando padrões e API
        
        Args:
            df: DataFrame para analisar
            text_column: Coluna com texto das mensagens
            batch_size: Tamanho do lote para processamento
            
        Returns:
            DataFrame com flags de mídia detectadas
        """
        logger.info(f"Detectando mensagens de mídia em {len(df)} registros")
        
        result_df = df.copy()
        
        # Detecção baseada em padrões
        result_df = self._detect_media_patterns(result_df, text_column)
        
        # Detecção via API para casos ambíguos
        ambiguous_indices = self._find_ambiguous_media_cases(result_df, text_column)
        
        if ambiguous_indices:
            logger.info(f"Analisando {len(ambiguous_indices)} casos ambíguos via API")
            
            # Processar em lotes
            for i in range(0, len(ambiguous_indices), batch_size):
                batch_indices = ambiguous_indices[i:i + batch_size]
                batch_texts = result_df.loc[batch_indices, text_column].tolist()
                
                # Usar error handler para análise com retry
                api_result = self.error_handler.execute_with_retry(
                    self._analyze_media_with_api,
                    stage="02b_deduplication_validation",
                    operation=f"media_detection_batch_{i//batch_size + 1}",
                    texts=batch_texts
                )
                
                if api_result.success and api_result.data:
                    self._apply_api_media_detection(result_df, batch_indices, api_result.data)
        
        return result_df
    
    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """Detecta automaticamente a coluna de texto principal"""
        
        # CORREÇÃO CRÍTICA: Verificar se o CSV foi parseado corretamente
        if len(df.columns) == 1 and ',' in df.columns[0]:
            # Header mal interpretado - CSV não foi parseado corretamente
            logger.error(f"PARSING CSV INCORRETO detectado. Header: {df.columns[0][:100]}...")
            logger.info("Forçando uso da coluna 'body' como fallback seguro")
            return 'body'
        
        # Verificar se colunas esperadas existem
        expected_columns = ['message_id', 'datetime', 'body', 'channel']
        if not any(col in df.columns for col in expected_columns):
            logger.error(f"Colunas esperadas não encontradas. Colunas disponíveis: {list(df.columns)}")
            # Tentar usar primeira coluna que pareça conter texto
            for col in df.columns:
                if df[col].dtype == 'object':
                    logger.warning(f"Usando coluna {col} como fallback")
                    return col
            return 'body'  # último fallback
        
        # Candidatos conhecidos para coluna de texto (PRIORIDADE: body_cleaned > body)
        text_candidates = ['body_cleaned', 'body', 'texto_cleaned', 'texto', 'text', 'content', 'message', 'mensagem']
        
        # Verificar candidatos conhecidos primeiro
        for candidate in text_candidates:
            if candidate in df.columns:
                # Verificar se a coluna tem conteúdo útil
                non_empty_count = df[candidate].dropna().astype(str).str.len().gt(0).sum()
                if non_empty_count > len(df) * 0.1:  # Pelo menos 10% com conteúdo
                    logger.info(f"Coluna de texto detectada: {candidate} ({non_empty_count}/{len(df)} com conteúdo)")
                    return candidate
                else:
                    logger.warning(f"Coluna {candidate} existe mas tem pouco conteúdo ({non_empty_count}/{len(df)})")
        
        logger.warning("Nenhuma coluna de texto ideal encontrada, usando fallback 'body'")
        
        # Se não encontrou, usar primeira coluna de texto longo
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    sample_length = df[col].dropna().apply(str).str.len().mean()
                    if sample_length > 50:  # Texto com média > 50 caracteres
                        logger.info(f"Coluna de texto detectada por tamanho: {col}")
                        return col
                except:
                    continue
        
        # Fallback para primeira coluna string disponível
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.info(f"Usando primeira coluna de texto disponível: {col}")
                return col
        
        # Se tudo falhar, usar primeira coluna
        fallback_col = df.columns[0] if len(df.columns) > 0 else 'body'
        logger.info(f"Usando fallback para coluna de texto: {fallback_col}")
        return fallback_col
    
    def _analyze_media_exclusion(
        self,
        original_df: pd.DataFrame,
        deduplicated_df: pd.DataFrame,
        text_column: str = None
    ) -> Dict[str, Any]:
        """Analisa se mensagens de mídia foram corretamente excluídas"""
        
        analysis = {
            "media_detection_original": {},
            "media_detection_deduplicated": {},
            "exclusion_validation": {}
        }
        
        # Detectar coluna de texto automaticamente se não fornecida
        if text_column is None:
            text_column = self._detect_text_column(original_df)
        
        # Detectar mídia no dataset original
        original_with_media = self.detect_media_messages(
            original_df.sample(n=min(1000, len(original_df))),
            text_column=text_column
        )
        
        # Contar tipos de mídia
        for media_type in ["photo", "video", "audio", "document", "sticker", "location"]:
            flag_column = f"has_{media_type}"
            if flag_column in original_with_media.columns:
                count = original_with_media[flag_column].sum()
                total = len(original_with_media)
                analysis["media_detection_original"][media_type] = {
                    "count": count,
                    "percentage": round((count / total) * 100, 2) if total > 0 else 0
                }
        
        # Verificar se mensagens de mídia estão no dataset deduplicado
        media_in_deduplicated = self._check_media_in_deduplicated(original_with_media, deduplicated_df)
        analysis["exclusion_validation"] = media_in_deduplicated
        
        return analysis
    
    def _detect_media_patterns(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Detecta mensagens de mídia usando padrões regex"""
        
        result_df = df.copy()
        text_series = result_df[text_column].fillna("").astype(str)
        
        for media_type, patterns in self.media_patterns.items():
            flag_column = f"has_{media_type.replace('_indicators', '')}"
            
            # Combinar todos os padrões para o tipo de mídia
            combined_pattern = "|".join(patterns)
            
            # Detectar padrões (case insensitive)
            result_df[flag_column] = text_series.str.contains(
                combined_pattern, case=False, regex=True, na=False
            )
        
        # Flag geral de mídia
        media_columns = [col for col in result_df.columns if col.startswith("has_") and "indicators" not in col]
        if media_columns:
            result_df["has_any_media"] = result_df[media_columns].any(axis=1)
        
        return result_df
    
    def _find_ambiguous_media_cases(self, df: pd.DataFrame, text_column: str) -> List[int]:
        """Encontra casos ambíguos que precisam de análise via API"""
        
        ambiguous_indices = []
        text_series = df[text_column].fillna("").astype(str)
        
        for idx, text in text_series.items():
            # Critérios para casos ambíguos
            if len(text.strip()) > 0:
                # Textos muito curtos que podem ser legendas de mídia
                if len(text.strip()) < 10 and not any(keyword in text.lower() 
                    for keyword in ["foto", "video", "audio", "sticker", "documento"]):
                    ambiguous_indices.append(idx)
                
                # Textos com emojis que podem indicar mídia
                emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
                if emoji_count > 3 and len(text.split()) < 5:
                    ambiguous_indices.append(idx)
                
                # Textos com menções a "enviou", "compartilhou" sem indicadores claros
                if re.search(r'\b(enviou|compartilhou|mandou)\b', text.lower()) and len(text.split()) < 8:
                    ambiguous_indices.append(idx)
        
        return list(set(ambiguous_indices))[:50]  # Reduzir limite para evitar truncamento JSON
    
    def _analyze_media_with_api(self, texts: List[str]) -> Dict[str, Any]:
        """Usa API para analisar casos ambíguos de mídia"""
        
        # Limitar tamanho do texto para evitar truncamento
        truncated_texts = []
        for i, text in enumerate(texts[:20]):  # Máximo 20 textos por vez
            # Truncar texto individual se muito longo
            truncated_text = text[:100] + "..." if len(text) > 100 else text
            truncated_texts.append(f"{i+1}. {truncated_text}")
        
        texts_sample = "\n".join(truncated_texts)
        
        prompt = f"""
Analise as seguintes mensagens do Telegram brasileiro para determinar se são mensagens de mídia (foto, vídeo, áudio, documento, sticker, localização).

MENSAGENS:
{texts_sample}

Para cada mensagem, determine:
1. Se é uma mensagem de mídia
2. Que tipo de mídia (se aplicável)
3. Se pode ter texto adicional além da mídia

Responda em formato JSON:
{{
  "media_analysis": [
    {{
      "text_id": 1,
      "is_media_message": true/false,
      "media_types": ["photo", "video", "audio", "document", "sticker", "location"],
      "has_additional_text": true/false,
      "confidence": "alto|medio|baixo",
      "reasoning": "explicação breve"
    }}
  ]
}}

CRITÉRIOS:
- Mensagens apenas com emoji/sticker SÃO mídia
- Legendas curtas com mídia SÃO mídia
- Textos descritivos longos NÃO são mídia (mesmo com mídia anexada)
- Considere contexto brasileiro do Telegram
"""
        
        try:
            response = self.create_message(
                prompt,
                stage="02b_deduplication_validation",
                operation="media_analysis"
            )
            
            return self.parse_json_response(response)
            
        except Exception as e:
            logger.error(f"Erro na análise de mídia via API: {e}")
            return {}
    
    def _apply_api_media_detection(
        self,
        df: pd.DataFrame,
        indices: List[int],
        api_results: Dict[str, Any]
    ):
        """Aplica resultados da detecção de mídia via API"""
        
        if "media_analysis" in api_results:
            for analysis in api_results["media_analysis"]:
                text_id = analysis.get("text_id", 1) - 1
                
                if text_id < len(indices):
                    actual_index = indices[text_id]
                    is_media = analysis.get("is_media_message", False)
                    media_types = analysis.get("media_types", [])
                    confidence = analysis.get("confidence", "baixo")
                    
                    # Aplicar apenas resultados de alta confiança
                    if confidence in ["alto", "medio"]:
                        # Atualizar flag geral de mídia
                        df.loc[actual_index, "has_any_media"] = is_media
                        
                        # Atualizar flags específicos
                        for media_type in media_types:
                            flag_column = f"has_{media_type}"
                            if flag_column in df.columns:
                                df.loc[actual_index, flag_column] = True
    
    def _check_media_in_deduplicated(
        self,
        original_with_media: pd.DataFrame,
        deduplicated_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Verifica se mensagens de mídia estão incorretamente no dataset deduplicado"""
        
        validation = {
            "media_incorrectly_included": 0,
            "total_media_messages": 0,
            "exclusion_rate": 0.0,
            "issues_found": []
        }
        
        if "has_any_media" in original_with_media.columns:
            media_messages = original_with_media[original_with_media["has_any_media"] == True]
            validation["total_media_messages"] = len(media_messages)
            
            # Verificar se alguma mensagem de mídia está no dataset deduplicado
            # (usando hash do texto ou identificador único)
            if "body" in media_messages.columns and "body" in deduplicated_df.columns:
                media_texts = set(media_messages["body"].fillna("").astype(str))
                deduplicated_texts = set(deduplicated_df["body"].fillna("").astype(str))
                
                incorrectly_included = media_texts.intersection(deduplicated_texts)
                validation["media_incorrectly_included"] = len(incorrectly_included)
                
                if incorrectly_included:
                    validation["issues_found"].extend([
                        f"Mensagem de mídia incorretamente incluída: {text[:50]}..."
                        for text in list(incorrectly_included)[:5]
                    ])
            
            # Calcular taxa de exclusão
            if validation["total_media_messages"] > 0:
                excluded = validation["total_media_messages"] - validation["media_incorrectly_included"]
                validation["exclusion_rate"] = round((excluded / validation["total_media_messages"]) * 100, 2)
        
        return validation
    
    def intelligent_deduplication(self, df: pd.DataFrame, text_column: str = "body") -> pd.DataFrame:
        """
        Executa deduplicação completa: remove duplicatas e adiciona coluna de frequência
        
        Args:
            df: DataFrame com dados para deduplicar
            text_column: Nome da coluna de texto principal ("body" ou "body_cleaned")
            
        Returns:
            DataFrame deduplicado (SEM duplicatas) + coluna 'duplicate_frequency'
        """
        
        logger.info(f"🔄 INICIANDO DEDUPLICAÇÃO COMPLETA de {len(df)} registros")
        logger.info(f"📝 Usando coluna de texto: '{text_column}'")
        
        try:
            # 1. PRIORIDADE: Usar body_cleaned se disponível, senão body
            dedup_column = None
            if 'body_cleaned' in df.columns:
                dedup_column = 'body_cleaned'
                logger.info("✅ Usando 'body_cleaned' para deduplicação (texto já processado)")
            elif 'body' in df.columns:
                dedup_column = 'body'  
                logger.info("✅ Usando 'body' para deduplicação (texto original)")
            else:
                logger.error(f"❌ Colunas 'body' ou 'body_cleaned' não encontradas. Disponíveis: {list(df.columns)}")
                return df
            
            # 2. Estatísticas iniciais
            total_records = len(df)
            non_empty_content = df[dedup_column].dropna().astype(str).str.strip().str.len().gt(0).sum()
            logger.info(f"📊 Registros com conteúdo em '{dedup_column}': {non_empty_content}/{total_records} ({100*non_empty_content/total_records:.1f}%)")
            
            if non_empty_content == 0:
                logger.warning("⚠️  Nenhum conteúdo encontrado para deduplicação")
                df['duplicate_frequency'] = 1
                return df
            
            # 3. DEDUPLICAÇÃO SIMPLES E EFETIVA por conteúdo de texto
            logger.info(f"🔍 Preparando texto para deduplicação usando '{dedup_column}'...")
            
            # Preparar coluna normalizada para deduplicação
            df_work = df.copy()
            df_work['_normalized_text'] = (
                df_work[dedup_column]
                .fillna('')
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r'\s+', ' ', regex=True)  # Normalizar espaços
            )
            
            # 4. CONTAR FREQUÊNCIAS das mensagens
            logger.info("📊 Calculando frequências de duplicatas...")
            text_counts = df_work['_normalized_text'].value_counts()
            df_work['duplicate_frequency'] = df_work['_normalized_text'].map(text_counts)
            
            # 5. ESTATÍSTICAS de duplicação
            total_unique = len(text_counts)
            total_duplicates = len(df_work) - total_unique
            reduction_rate = (total_duplicates / len(df_work)) * 100
            
            logger.info(f"📈 ESTATÍSTICAS DE DUPLICAÇÃO:")
            logger.info(f"   Total de registros: {len(df_work)}")
            logger.info(f"   Textos únicos: {total_unique}")
            logger.info(f"   Duplicatas encontradas: {total_duplicates}")
            logger.info(f"   Taxa de duplicação: {reduction_rate:.1f}%")
            
            # 6. REMOVER DUPLICATAS mantendo apenas a primeira ocorrência
            logger.info("🗑️  Removendo duplicatas (mantendo primeira ocorrência)...")
            
            # Manter apenas primeira ocorrência de cada texto único
            deduplicated_df = df_work.drop_duplicates(subset=['_normalized_text'], keep='first').copy()
            
            # Remover coluna auxiliar
            deduplicated_df = deduplicated_df.drop('_normalized_text', axis=1)
            
            # 7. VALIDAÇÃO FINAL
            if reduction_rate < 1.0:
                logger.warning(f"⚠️  Taxa de duplicação baixa ({reduction_rate:.1f}%). Dados podem estar já limpos.")
            elif reduction_rate > 70.0:
                logger.warning(f"⚠️  Taxa de duplicação muito alta ({reduction_rate:.1f}%). Verificar dados.")
            else:
                logger.info(f"✅ Taxa de duplicação normal: {reduction_rate:.1f}%")
            
            # 8. ESTATÍSTICAS FINAIS
            final_records = len(deduplicated_df)
            duplicates_removed = len(df) - final_records
            max_frequency = deduplicated_df['duplicate_frequency'].max()
            mean_frequency = deduplicated_df['duplicate_frequency'].mean()
            
            logger.info(f"🎯 DEDUPLICAÇÃO CONCLUÍDA:")
            logger.info(f"   ➤ Dataset original: {len(df)} registros")
            logger.info(f"   ➤ Dataset limpo: {final_records} registros")
            logger.info(f"   ➤ Duplicatas removidas: {duplicates_removed} ({reduction_rate:.1f}%)")
            logger.info(f"   ➤ Maior frequência: {max_frequency}x")
            logger.info(f"   ➤ Frequência média: {mean_frequency:.2f}x")
            logger.info(f"   ➤ Coluna adicionada: 'duplicate_frequency'")
            
            return deduplicated_df
            
        except Exception as e:
            logger.error(f"Erro na deduplicação inteligente: {e}")
            # Fallback para não quebrar o pipeline
            return df
    
    def _ai_validate_duplicates(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Valida duplicatas usando API Anthropic para casos ambíguos
        
        Args:
            df: DataFrame com duplicatas marcadas
            text_column: Coluna de texto
            
        Returns:
            DataFrame deduplicado com validação AI
        """
        
        # Para esta versão, usar deduplicação básica
        # TODO: Implementar validação AI detalhada em versão futura
        deduplicated_df = df.drop_duplicates(subset=['content_hash'], keep='first')
        deduplicated_df = deduplicated_df.drop('content_hash', axis=1)
        
        return deduplicated_df
    
    def _validate_duplicate_counts(self, df: pd.DataFrame, duplicate_count_column: str) -> Dict[str, Any]:
        """Valida contagem de duplicatas"""
        
        validation = {
            "column_exists": duplicate_count_column in df.columns,
            "statistics": {},
            "validation_issues": []
        }
        
        if validation["column_exists"]:
            count_series = df[duplicate_count_column].fillna(0)
            
            validation["statistics"] = {
                "min_count": int(count_series.min()),
                "max_count": int(count_series.max()),
                "mean_count": round(count_series.mean(), 2),
                "total_duplicates_counted": int(count_series.sum()),
                "records_with_duplicates": int((count_series > 1).sum()),
                "percentage_with_duplicates": round(((count_series > 1).sum() / len(df)) * 100, 2)
            }
            
            # Validações
            if validation["statistics"]["min_count"] < 1:
                validation["validation_issues"].append("Encontrados valores menores que 1 na contagem")
            
            if validation["statistics"]["max_count"] > 1000:
                validation["validation_issues"].append(f"Contagem muito alta encontrada: {validation['statistics']['max_count']}")
            
            # Verificar se soma das contagens faz sentido com dataset original
            expected_original_size = validation["statistics"]["total_duplicates_counted"]
            actual_deduplicated_size = len(df)
            
            validation["size_consistency"] = {
                "expected_original_size": expected_original_size,
                "actual_deduplicated_size": actual_deduplicated_size,
                "ratio": round(expected_original_size / actual_deduplicated_size, 2) if actual_deduplicated_size > 0 else 0
            }
        
        return validation
    
    def _analyze_coordination_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões de coordenação suspeita"""
        
        analysis = {
            "potential_coordination": [],
            "statistics": {},
            "suspicious_patterns": []
        }
        
        if "duplicate_count" in df.columns and "body" in df.columns:
            # Mensagens com muitas duplicatas (possível coordenação)
            high_duplicate_threshold = df["duplicate_count"].quantile(0.95)
            high_duplicates = df[df["duplicate_count"] > high_duplicate_threshold]
            
            analysis["statistics"]["high_duplicate_messages"] = len(high_duplicates)
            analysis["statistics"]["high_duplicate_threshold"] = int(high_duplicate_threshold)
            
            # Analisar padrões suspeitos
            if len(high_duplicates) > 0:
                # Mensagens muito similares
                similar_patterns = self._find_similar_message_patterns(high_duplicates)
                analysis["suspicious_patterns"].extend(similar_patterns)
                
                # Adicionar amostras para análise
                for _, row in high_duplicates.head(5).iterrows():
                    analysis["potential_coordination"].append({
                        "text": str(row["body"])[:100] + "...",
                        "duplicate_count": int(row["duplicate_count"]),
                        "canal": row.get("canal", "N/A")
                    })
        
        return analysis
    
    def _find_similar_message_patterns(self, df: pd.DataFrame) -> List[str]:
        """Encontra padrões de mensagens similares que podem indicar coordenação"""
        
        patterns = []
        
        if len(df) > 1:
            texts = df["body"].fillna("").astype(str).tolist()
            
            # Procurar por estruturas similares
            for i, text1 in enumerate(texts[:10]):  # Limitar para performance
                for j, text2 in enumerate(texts[i+1:11], i+1):
                    similarity = self._calculate_text_similarity(text1, text2)
                    if similarity > 0.8:  # 80% de similaridade
                        patterns.append(f"Textos muito similares: índices {i} e {j} (similaridade: {similarity:.2f})")
        
        return patterns
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade simples entre dois textos"""
        
        if not text1 or not text2:
            return 0.0
        
        # Tokenização simples
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _detailed_deduplication_analysis_api(
        self,
        original_df: pd.DataFrame,
        deduplicated_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Análise detalhada via API"""
        
        # Preparar amostra para análise
        sample_size = min(50, len(deduplicated_df))
        sample_df = deduplicated_df.sample(n=sample_size, random_state=42)
        
        # Usar error handler para análise com retry
        result = self.error_handler.execute_with_retry(
            self._analyze_deduplication_quality_api,
            stage="02b_deduplication_validation",
            operation="quality_analysis",
            sample_df=sample_df,
            reduction_ratio=(len(original_df) - len(deduplicated_df)) / len(original_df)
        )
        
        if result.success:
            return result.data
        else:
            logger.warning(f"Falha na análise detalhada via API: {result.error.error_message}")
            return {"error": result.error.error_message}
    
    def _analyze_deduplication_quality_api(
        self,
        sample_df: pd.DataFrame,
        reduction_ratio: float
    ) -> Dict[str, Any]:
        """Usa API para análise de qualidade da deduplicação"""
        
        # Preparar dados para análise
        sample_data = []
        for idx, row in sample_df.head(20).iterrows():
            sample_data.append({
                "body": str(row.get("body", ""))[:200],
                "duplicate_count": int(row.get("duplicate_count", 1)),
                "canal": str(row.get("canal", "N/A"))
            })
        
        prompt = f"""
Analise a qualidade do processo de deduplicação deste dataset brasileiro do Telegram:

ESTATÍSTICAS:
- Taxa de redução: {reduction_ratio:.1%}
- Amostra de registros deduplicados:

{json.dumps(sample_data, indent=2, ensure_ascii=False)}

Avalie:
1. Se a taxa de redução é apropriada
2. Se as contagens de duplicatas parecem corretas
3. Se há padrões suspeitos de coordenação
4. Qualidade geral do processo de deduplicação

Responda em formato JSON:
{{
  "quality_assessment": {{
    "reduction_rate_appropriate": true/false,
    "duplicate_counts_seem_accurate": true/false,
    "coordination_patterns_detected": true/false,
    "overall_quality": "excelente|bom|satisfatorio|ruim",
    "confidence": "alto|medio|baixo"
  }},
  "issues_identified": [
    "problema1", "problema2"
  ],
  "recommendations": [
    "recomendacao1", "recomendacao2"
  ],
  "coordination_analysis": {{
    "suspicious_patterns": ["padrao1", "padrao2"],
    "likely_bot_activity": true/false,
    "organic_vs_coordinated_ratio": "estimativa"
  }}
}}
"""
        
        try:
            response = self.create_message(
                prompt,
                stage="02b_deduplication_validation",
                operation="quality_analysis"
            )
            
            return self.parse_json_response(response)
            
        except Exception as e:
            logger.error(f"Erro na análise de qualidade via API: {e}")
            return {}
    
    def _calculate_deduplication_quality(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula score de qualidade geral da deduplicação"""
        
        quality_score = 1.0
        quality_factors = []
        
        # Fator 1: Exclusão adequada de mídia
        media_analysis = validation_report.get("media_analysis", {})
        exclusion_validation = media_analysis.get("exclusion_validation", {})
        exclusion_rate = exclusion_validation.get("exclusion_rate", 0)
        
        if exclusion_rate < 90:
            quality_score -= 0.3
            quality_factors.append(f"Taxa de exclusão de mídia baixa: {exclusion_rate}%")
        
        # Fator 2: Consistência das contagens
        duplicate_validation = validation_report.get("duplicate_count_validation", {})
        if duplicate_validation.get("validation_issues"):
            quality_score -= 0.2
            quality_factors.append("Problemas na contagem de duplicatas")
        
        # Fator 3: Taxa de redução apropriada (5-60% é normal)
        reduction_ratio = validation_report.get("reduction_ratio", 0)
        if reduction_ratio < 0.05 or reduction_ratio > 0.8:
            quality_score -= 0.2
            quality_factors.append(f"Taxa de redução suspeita: {reduction_ratio:.1%}")
        
        # Fator 4: Análise da API
        api_analysis = validation_report.get("api_analysis", {})
        if "quality_assessment" in api_analysis:
            api_quality = api_analysis["quality_assessment"].get("overall_quality", "satisfatorio")
            if api_quality in ["ruim"]:
                quality_score -= 0.3
                quality_factors.append("Qualidade avaliada como ruim pela API")
        
        return {
            "overall_score": max(0.0, quality_score),
            "quality_level": "excelente" if quality_score >= 0.9 else
                           "bom" if quality_score >= 0.7 else
                           "satisfatorio" if quality_score >= 0.5 else "ruim",
            "quality_factors": quality_factors
        }
    
    def _generate_deduplication_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na validação"""
        
        recommendations = []
        
        # Baseado na qualidade geral
        quality_assessment = validation_report.get("quality_assessment", {})
        quality_level = quality_assessment.get("quality_level", "satisfatorio")
        
        if quality_level == "ruim":
            recommendations.append("Reprocessar deduplicação com parâmetros ajustados")
        
        # Baseado na exclusão de mídia
        media_analysis = validation_report.get("media_analysis", {})
        exclusion_validation = media_analysis.get("exclusion_validation", {})
        
        if exclusion_validation.get("media_incorrectly_included", 0) > 0:
            recommendations.append("Melhorar detecção de mensagens de mídia antes da deduplicação")
        
        # Baseado na análise de coordenação
        coordination_analysis = validation_report.get("coordination_analysis", {})
        if coordination_analysis.get("suspicious_patterns"):
            recommendations.append("Investigar padrões de coordenação identificados")
        
        # Baseado na análise da API
        api_analysis = validation_report.get("api_analysis", {})
        api_recommendations = api_analysis.get("recommendations", [])
        recommendations.extend(api_recommendations)
        
        return list(set(recommendations))  # Remover duplicatas