"""
Analisador Político Inteligente via API Anthropic
Etapa 01c: Análise política profunda e contextualizada do discurso brasileiro
"""

import pandas as pd
import json
import logging
import yaml
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from .base import AnthropicBase
from .api_error_handler import APIErrorHandler, APIQualityChecker

logger = logging.getLogger(__name__)


class PoliticalAnalyzer(AnthropicBase):
    """
    Analisador político especializado em discurso brasileiro
    
    Capacidades:
    - Análise de alinhamento político contextualizada
    - Detecção de narrativas conspiratórias
    - Identificação de negacionismo
    - Análise de tom emocional político
    - Detecção de sinais de coordenação
    - Avaliação de risco de desinformação
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.error_handler = APIErrorHandler()
        self.quality_checker = APIQualityChecker(config)
        
        # Carregar léxico político brasileiro
        self.political_lexicon = self._load_political_lexicon()
        
        # Cache para análises repetidas
        self.analysis_cache = {}
        
        # Configurações de análise
        self.batch_size = 10  # Análise política é mais complexa, lotes menores
        self.confidence_threshold = 0.7
        
    def analyze_political_discourse(
        self,
        df: pd.DataFrame,
        text_column: str = "body_cleaned",
        batch_size: int = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analisa discurso político do dataset
        
        Args:
            df: DataFrame com dados
            text_column: Coluna de texto a analisar
            batch_size: Tamanho do lote (usa configuração padrão se None)
            
        Returns:
            Tuple com DataFrame enriquecido e relatório de análise
        """
        logger.info(f"Iniciando análise política para {len(df)} registros")
        
        if batch_size is None:
            batch_size = self.batch_size
            
        # Validar coluna de texto
        if text_column not in df.columns:
            # Tentar colunas alternativas
            for alt_col in ['body', 'texto', 'text']:
                if alt_col in df.columns:
                    text_column = alt_col
                    logger.info(f"Usando coluna alternativa: {text_column}")
                    break
            else:
                raise ValueError(f"Coluna de texto não encontrada: {text_column}")
        
        # Fazer backup
        backup_file = f"data/interim/political_analysis_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False, sep=';', encoding='utf-8')
        logger.info(f"Backup criado: {backup_file}")
        
        # Preparar DataFrame enriquecido
        enriched_df = df.copy()
        
        # Inicializar colunas de análise política
        political_columns = [
            'political_alignment', 'alignment_confidence',
            'conspiracy_indicators', 'conspiracy_score',
            'negacionism_indicators', 'negacionism_score', 
            'emotional_tone', 'emotional_intensity',
            'discourse_type', 'urgency_level',
            'coordination_signals', 'coordination_score',
            'misinformation_risk', 'brazilian_context_score',
            'political_entities', 'narrative_themes'
        ]
        
        for col in political_columns:
            enriched_df[col] = None
            
        # Processar em lotes
        analysis_report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "text_column": text_column,
            "batch_size": batch_size,
            "batches_processed": 0,
            "api_calls_made": 0,
            "cache_hits": 0,
            "analysis_statistics": {},
            "lexicon_matches": {},
            "quality_scores": []
        }
        
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_df = enriched_df.iloc[i:i + batch_size].copy()
            batch_num = i // batch_size + 1
            
            logger.info(f"Processando lote {batch_num}/{total_batches}")
            
            # Processar lote
            batch_results = self.error_handler.execute_with_retry(
                self._process_political_batch,
                stage="01c_political_analysis",
                operation=f"batch_{batch_num}",
                batch_df=batch_df,
                text_column=text_column
            )
            
            if batch_results.success:
                # Aplicar resultados ao DataFrame principal
                for idx, result in enumerate(batch_results.data):
                    df_idx = i + idx
                    if df_idx < len(enriched_df) and result is not None:
                        # Verificar se result é um dicionário válido
                        if isinstance(result, dict):
                            for col, value in result.items():
                                if col in political_columns:
                                    enriched_df.at[df_idx, col] = value
                        else:
                            logger.warning(f"Resultado inválido para índice {idx}: {type(result)}")
                            # Aplicar análise política vazia
                            empty_analysis = self._get_empty_political_analysis()
                            for col, value in empty_analysis.items():
                                if col in political_columns:
                                    enriched_df.at[df_idx, col] = value
                
                analysis_report["batches_processed"] += 1
                analysis_report["api_calls_made"] += 1
            else:
                logger.error(f"Falha no lote {batch_num}: {batch_results.error.error_message}")
        
        # Análise léxica complementar
        lexicon_results = self._analyze_with_lexicon(enriched_df, text_column)
        analysis_report["lexicon_matches"] = lexicon_results
        
        # Estatísticas finais
        analysis_report["analysis_statistics"] = self._generate_analysis_statistics(enriched_df)
        
        logger.info("Análise política concluída")
        return enriched_df, analysis_report
    
    def _process_political_batch(
        self,
        batch_df: pd.DataFrame,
        text_column: str
    ) -> List[Dict[str, Any]]:
        """Processa um lote para análise política"""
        
        try:
            texts = []
            text_hashes = []
            
            for idx, row in batch_df.iterrows():
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                texts.append(text)
                
                # Gerar hash para cache
                text_hash = hashlib.md5(text.encode()).hexdigest()
                text_hashes.append(text_hash)
            
            # Verificar cache primeiro
            results = []
            texts_to_analyze = []
            indices_to_analyze = []
            
            for i, (text, text_hash) in enumerate(zip(texts, text_hashes)):
                if text_hash in self.analysis_cache:
                    cached_result = self.analysis_cache[text_hash]
                    # Verificar se resultado do cache é válido
                    if isinstance(cached_result, dict):
                        results.append(cached_result)
                    else:
                        results.append(self._get_empty_political_analysis())
                else:
                    results.append(None)  # Placeholder
                    texts_to_analyze.append(text)
                    indices_to_analyze.append(i)
            
            # Analisar textos não encontrados no cache
            if texts_to_analyze:
                try:
                    api_results = self._analyze_political_content_api(texts_to_analyze)
                    
                    # Verificar se api_results é uma lista válida
                    if not isinstance(api_results, list):
                        logger.error(f"API retornou resultado inválido: {type(api_results)}")
                        api_results = [self._get_empty_political_analysis() for _ in texts_to_analyze]
                    
                    # Aplicar resultados da API
                    for i, api_result in enumerate(api_results):
                        if i < len(indices_to_analyze):
                            result_idx = indices_to_analyze[i]
                            text_hash = text_hashes[result_idx]
                            
                            # Verificar se api_result é um dicionário válido
                            if isinstance(api_result, dict):
                                # Salvar no cache
                                self.analysis_cache[text_hash] = api_result
                                results[result_idx] = api_result
                            else:
                                logger.warning(f"API resultado inválido para texto {i}: {type(api_result)}")
                                empty_result = self._get_empty_political_analysis()
                                self.analysis_cache[text_hash] = empty_result
                                results[result_idx] = empty_result
                                
                except Exception as e:
                    logger.error(f"Erro na análise via API: {e}")
                    # Preencher com análises vazias
                    for i in indices_to_analyze:
                        if results[i] is None:
                            results[i] = self._get_empty_political_analysis()
            
            # Garantir que todos os resultados sejam dicionários válidos
            for i in range(len(results)):
                if results[i] is None or not isinstance(results[i], dict):
                    results[i] = self._get_empty_political_analysis()
            
            return results
            
        except Exception as e:
            logger.error(f"Erro crítico no processamento de lote político: {e}")
            # Retornar análises vazias para todos os textos
            return [self._get_empty_political_analysis() for _ in range(len(batch_df))]
    
    def _analyze_political_content_api(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Análise política via API Anthropic"""
        
        try:
            # Validar entrada
            if not texts or not isinstance(texts, list):
                logger.warning("Lista de textos inválida para análise política")
                return [self._get_empty_political_analysis() for _ in (texts or [])]
            
            # Preparar contexto temporal
            current_date = datetime.now()
            temporal_context = self._get_temporal_political_context(current_date)
            
            # Limitar número de textos para evitar prompts muito longos
            texts_to_process = texts[:10]  # Máximo 10 por vez
            
            # Preparar amostra de textos para o prompt
            texts_sample = "\n".join([
                f"{i+1}. {text[:300]}..." if len(text) > 300 else f"{i+1}. {text}"
                for i, text in enumerate(texts_to_process)
            ])
            
            if not texts_sample.strip():
                logger.warning("Nenhum texto válido para análise")
                return [self._get_empty_political_analysis() for _ in texts]
            
            # Construir prompt especializado
            prompt = f"""
Analise as seguintes mensagens do Telegram brasileiro no contexto político 2019-2023.

CONTEXTO TEMPORAL: {temporal_context}

MENSAGENS:
{texts_sample}

Para cada mensagem, forneça análise política detalhada em formato JSON:

{{
  "results": [
    {{
      "text_id": 1,
      "political_alignment": "bolsonarista|antibolsonarista|neutro|indefinido",
      "alignment_confidence": 0.85,
      "conspiracy_indicators": ["tipo1", "tipo2"],
      "conspiracy_score": 0.75,
      "negacionism_indicators": ["covid", "vacinas", "urnas"],
      "negacionism_score": 0.60,
      "emotional_tone": "raiva|medo|esperança|tristeza|alegria|neutro",
      "emotional_intensity": 0.80,
      "discourse_type": "informativo|opinativo|mobilizador|atacante|defensivo",
      "urgency_level": "baixo|medio|alto",
      "coordination_signals": ["hashtag_coordenada", "linguagem_padronizada"],
      "coordination_score": 0.40,
      "misinformation_risk": "baixo|medio|alto",
      "brazilian_context_score": 0.90,
      "political_entities": ["bolsonaro", "lula", "stf"],
      "narrative_themes": ["urna_fraudada", "tratamento_precoce"]
    }}
  ]
}}

CRITÉRIOS ESPECÍFICOS:
1. ALINHAMENTO POLÍTICO: Baseado em apoio/oposição ao governo Bolsonaro
2. CONSPIRAÇÃO: Teorias sem base factual (urnas, globalismo, etc.)
3. NEGACIONISMO: Negação de evidências científicas/institucionais
4. TOM EMOCIONAL: Analisar intensidade da linguagem
5. COORDENAÇÃO: Padrões que sugerem ação organizada
6. CONTEXTO BRASILEIRO: Referências específicas à realidade nacional

IMPORTANTE: 
- Use scores de 0.0 a 1.0 para confiança
- Seja preciso na identificação de narrativas específicas
- Considere nuances do português brasileiro
- Foque no período 2019-2023 para contexto político
- SEMPRE retorne exatamente {len(texts_to_process)} resultados na lista "results"
- Retorne APENAS o JSON, sem texto introdutório ou explicações
"""
            
            try:
                response = self.create_message(
                    prompt,
                    stage="01c_political_analysis",
                    operation="content_analysis"
                )
                
                if not response:
                    logger.warning("Resposta vazia da API Anthropic")
                    return [self._get_empty_political_analysis() for _ in texts]
                
                # Validar qualidade da resposta
                validation = self.quality_checker.validate_output_quality(
                    response,
                    expected_format="json",
                    context={"texts_count": len(texts_to_process)},
                    stage="01c_political_analysis"
                )
                
                if not validation["valid"]:
                    logger.warning(f"Qualidade da resposta baixa: {validation['issues']}")
                
                # Parse da resposta com método ultra-robusto - CORRIGIDO
                try:
                    parsed_response = self.parse_json_response_robust(response, "results")
                    
                    # Verificar se resposta tem a estrutura esperada
                    if isinstance(parsed_response, dict) and "results" in parsed_response:
                        api_results = parsed_response["results"]
                        
                        # Validar se results é uma lista
                        if isinstance(api_results, list) and len(api_results) > 0:
                            # Garantir que temos o número correto de resultados
                            final_results = []
                            for i in range(len(texts)):
                                if i < len(api_results) and isinstance(api_results[i], dict):
                                    final_results.append(api_results[i])
                                else:
                                    final_results.append(self._get_empty_political_analysis())
                            
                            logger.info(f"✅ Análise política bem-sucedida: {len(final_results)} resultados")
                            return final_results
                        else:
                            logger.warning(f"Campo 'results' vazio ou inválido: {api_results}")
                    else:
                        logger.warning(f"Resposta sem campo 'results': {list(parsed_response.keys()) if isinstance(parsed_response, dict) else type(parsed_response)}")
                        
                    # Se chegou aqui, parsing falhou - tentar fallback
                    logger.warning("Usando análise política tradicional como fallback")
                    return [self._get_empty_political_analysis() for _ in texts]
                    
                except Exception as parse_error:
                    logger.error(f"Erro no parsing da resposta política: {parse_error}")
                    logger.error(f"Resposta recebida (primeiros 500 chars): {response[:500]}")
                    return [self._get_empty_political_analysis() for _ in texts]
                    
            except Exception as api_error:
                logger.error(f"Erro na comunicação com API Anthropic: {api_error}")
                return [self._get_empty_political_analysis() for _ in texts]
                
        except Exception as e:
            logger.error(f"Erro crítico na análise política via API: {e}")
            return [self._get_empty_political_analysis() for _ in texts]
    
    def _get_empty_political_analysis(self) -> Dict[str, Any]:
        """Retorna análise política vazia para casos de erro"""
        return {
            "political_alignment": "indefinido",
            "alignment_confidence": 0.0,
            "conspiracy_indicators": [],
            "conspiracy_score": 0.0,
            "negacionism_indicators": [],
            "negacionism_score": 0.0,
            "emotional_tone": "neutro",
            "emotional_intensity": 0.0,
            "discourse_type": "informativo",
            "urgency_level": "baixo",
            "coordination_signals": [],
            "coordination_score": 0.0,
            "misinformation_risk": "baixo",
            "brazilian_context_score": 0.0,
            "political_entities": [],
            "narrative_themes": []
        }
    
    def _load_political_lexicon(self) -> Dict[str, Any]:
        """Carrega léxico político brasileiro"""
        try:
            lexicon_path = Path("config/brazilian_political_lexicon.yaml")
            if lexicon_path.exists():
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("Léxico político não encontrado, usando versão padrão")
                return self._get_default_lexicon()
        except Exception as e:
            logger.error(f"Erro ao carregar léxico político: {e}")
            return self._get_default_lexicon()
    
    def _get_default_lexicon(self) -> Dict[str, Any]:
        """Retorna léxico político padrão"""
        return {
            "brazilian_political_lexicon": {
                "governo_bolsonaro": ["bolsonaro", "presidente", "capitão", "mito"],
                "oposição": ["lula", "pt", "petista", "esquerda"],
                "militarismo": ["forças armadas", "militares", "intervenção militar", "quartel"],
                "teorias_conspiração": ["urna fraudada", "globalismo", "deep state"],
                "saúde_negacionismo": ["tratamento precoce", "ivermectina", "cloroquina"]
            }
        }
    
    def _analyze_with_lexicon(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Análise complementar usando léxico político"""
        lexicon_results = {}
        
        if "brazilian_political_lexicon" in self.political_lexicon:
            lexicon = self.political_lexicon["brazilian_political_lexicon"]
            
            for category, terms in lexicon.items():
                if isinstance(terms, list):
                    # Contar ocorrências de termos da categoria
                    pattern = "|".join([f"\\b{term}\\b" for term in terms])
                    matches = df[text_column].fillna("").str.contains(
                        pattern, case=False, regex=True
                    )
                    lexicon_results[category] = {
                        "matches": matches.sum(),
                        "percentage": (matches.sum() / len(df)) * 100
                    }
        
        return lexicon_results
    
    def _get_temporal_political_context(self, date: datetime) -> str:
        """Gera contexto temporal para análise política"""
        year = date.year
        
        if year >= 2018 and year <= 2022:
            return f"Período Bolsonaro ({year}) - governo federal, polarização política intensa"
        elif year >= 2023:
            return f"Pós-Bolsonaro ({year}) - transição de governo, análise retrospectiva"
        else:
            return f"Período pré-Bolsonaro ({year}) - contexto político anterior"
    
    def _generate_analysis_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera estatísticas da análise política"""
        stats = {
            "political_alignment_distribution": {},
            "average_conspiracy_score": 0.0,
            "average_negacionism_score": 0.0,
            "emotional_tone_distribution": {},
            "high_risk_messages": 0
        }
        
        # Distribuição de alinhamento político
        if 'political_alignment' in df.columns:
            stats["political_alignment_distribution"] = df['political_alignment'].value_counts().to_dict()
        
        # Scores médios
        for score_col in ['conspiracy_score', 'negacionism_score']:
            if score_col in df.columns:
                scores = pd.to_numeric(df[score_col], errors='coerce').dropna()
                if len(scores) > 0:
                    stats[f"average_{score_col}"] = scores.mean()
        
        # Distribuição de tom emocional
        if 'emotional_tone' in df.columns:
            stats["emotional_tone_distribution"] = df['emotional_tone'].value_counts().to_dict()
        
        # Mensagens de alto risco
        if 'misinformation_risk' in df.columns:
            stats["high_risk_messages"] = (df['misinformation_risk'] == 'alto').sum()
        
        return stats
    
    def _parse_political_response(self, response: str) -> Dict[str, Any]:
        """
        Parser especializado para respostas de análise política
        ATUALIZADO: Usa o parser melhorado da classe base
        """
        if not response or not response.strip():
            logger.warning("Resposta vazia da API")
            return {"results": []}
        
        # Usar o parser ultra-robusto da classe base
        return self.parse_json_response_robust(response, "results")