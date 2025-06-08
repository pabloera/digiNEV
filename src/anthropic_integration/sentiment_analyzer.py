"""
Módulo de análise de sentimentos com API Anthropic

Este módulo utiliza a API Anthropic para análise contextualizada de sentimentos
em textos políticos, detectando ironia, radicalização e múltiplas camadas emocionais.
"""

import json
from typing import List, Dict, Any, Optional
from collections import Counter
import pandas as pd
import numpy as np
from .base import AnthropicBase


class AnthropicSentimentAnalyzer(AnthropicBase):
    """Analisa sentimentos considerando contexto político brasileiro"""
    
    def analyze_political_sentiment(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Análise de sentimento contextualizada para textos políticos
        
        Args:
            texts: Lista de textos para analisar
            batch_size: Tamanho do lote para processamento
            
        Returns:
            Lista de análises detalhadas de sentimento
        """
        def process_batch(batch: List[str]) -> List[Dict[str, Any]]:
            prompt = f"""Analise o sentimento e tom dos seguintes textos políticos brasileiros.

Para cada texto, forneça:
1. Sentimento geral: muito_negativo/negativo/neutro/positivo/muito_positivo
2. Confiança na análise: 0-1
3. Emoções detectadas: raiva, medo, esperança, desprezo, orgulho, tristeza, indignação, alegria
4. Presença de ironia/sarcasmo: sim/não
5. Alvo do sentimento: pessoa, instituição, grupo, política, sistema
6. Intensidade do discurso: baixa/média/alta/extrema
7. Indicadores de radicalização: nenhum/leve/moderado/severo
8. Tom dominante: agressivo/defensivo/mobilizador/informativo/conspiratório

Considere o contexto político brasileiro 2019-2023, incluindo:
- Polarização política extrema
- Discursos pró/anti-Bolsonaro
- Ataques ao sistema democrático
- Negacionismo da pandemia
- Teorias conspiratórias
- Linguagem codificada (dogwhistles)

Textos para análise:
{json.dumps([{"id": i, "text": t[:500]} for i, t in enumerate(batch)], ensure_ascii=False)}

Retorne JSON detalhado:
{{
  "results": [
    {{
      "id": 0,
      "sentiment": "negativo",
      "sentiment_score": -0.8,
      "confidence": 0.9,
      "emotions": ["raiva", "desprezo"],
      "irony": false,
      "target": "instituição",
      "target_name": "STF",
      "intensity": "alta",
      "radicalization": "moderado",
      "dominant_tone": "agressivo",
      "context_clues": ["uso de caps", "múltiplas exclamações"]
    }}
  ]
}}"""

            response = self.create_message(prompt, temperature=0.3)
            result = self.parse_claude_response_safe(response, ["results"])
            return result.get('results', [])
        
        # Processar em lotes
        return self.process_batch(texts, batch_size, process_batch)
    
    def detect_sentiment_layers(self, text: str) -> Dict[str, Any]:
        """
        Detecta múltiplas camadas de sentimento em textos complexos
        
        Args:
            text: Texto para análise profunda
            
        Returns:
            Dicionário com análise de camadas
        """
        prompt = f"""Analise as camadas de sentimento neste texto político:

Texto: {text}

Identifique:
1. Sentimento superficial (literal)
2. Sentimento implícito (nas entrelinhas)
3. Uso de ironia ou sarcasmo
4. Dogwhistles emocionais
5. Apelos emocionais manipulativos
6. Gatilhos psicológicos utilizados
7. Estratégias retóricas de persuasão

Contexto: Telegram brasileiro, período bolsonarista.

Retorne JSON:
{{
  "surface_sentiment": "sentimento literal",
  "implicit_sentiment": "sentimento nas entrelinhas",
  "irony_markers": ["marcador1", "marcador2"],
  "emotional_dogwhistles": ["código1", "código2"],
  "manipulation_tactics": ["tática1", "tática2"],
  "psychological_triggers": ["gatilho1", "gatilho2"],
  "rhetorical_strategies": ["estratégia1"],
  "overall_intent": "intenção geral do texto"
}}"""

        response = self.create_message(prompt, temperature=0.3)
        return self.parse_claude_response_safe(response, ["results"])
    
    def analyze_sentiment_evolution(self, messages_by_date: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analisa evolução do sentimento ao longo do tempo
        
        Args:
            messages_by_date: Dicionário com data -> lista de mensagens
            
        Returns:
            Análise temporal do sentimento
        """
        evolution_data = {}
        
        for date, messages in messages_by_date.items():
            # Analisar amostra representativa
            sample_size = min(100, len(messages))
            sample = messages[:sample_size] if len(messages) > sample_size else messages
            
            sentiments = self.analyze_political_sentiment(sample)
            
            # Agregar métricas
            if sentiments:
                sentiment_scores = [s.get('sentiment_score', 0) for s in sentiments]
                radicalization_scores = {
                    'nenhum': 0, 'leve': 0.33, 'moderado': 0.66, 'severo': 1
                }
                rad_values = [radicalization_scores.get(s.get('radicalization', 'nenhum'), 0) 
                             for s in sentiments]
                
                emotions_list = []
                for s in sentiments:
                    emotions_list.extend(s.get('emotions', []))
                
                evolution_data[date] = {
                    'avg_sentiment': np.mean(sentiment_scores),
                    'sentiment_std': np.std(sentiment_scores),
                    'radicalization_index': np.mean(rad_values),
                    'dominant_emotions': Counter(emotions_list).most_common(3),
                    'irony_rate': sum(1 for s in sentiments if s.get('irony')) / len(sentiments),
                    'intensity_distribution': Counter(s.get('intensity', 'média') for s in sentiments),
                    'sample_size': len(sentiments)
                }
        
        return evolution_data
    
    def identify_emotional_patterns(self, df: pd.DataFrame, group_by: str = 'channel') -> Dict[str, Any]:
        """
        Identifica padrões emocionais por grupo
        
        Args:
            df: DataFrame com análises de sentimento
            group_by: Coluna para agrupar
            
        Returns:
            Padrões emocionais por grupo
        """
        patterns = {}
        
        for group_name, group_df in df.groupby(group_by):
            texts = group_df['message'].fillna('').tolist()[:50]  # Amostra
    
    def analyze_sentiment_comprehensive(self, df: pd.DataFrame, text_column: str = "body_cleaned") -> pd.DataFrame:
        """
        Análise abrangente de sentimentos usando API Anthropic
        
        Args:
            df: DataFrame com os dados
            text_column: Nome da coluna de texto
            
        Returns:
            DataFrame com análises de sentimento adicionadas
        """
        self.logger.info(f"Iniciando análise abrangente de sentimentos para {len(df)} registros")
        
        result_df = df.copy()
        texts = result_df[text_column].fillna('').astype(str).tolist()
        
        # Processar em lotes
        batch_size = 20
        all_sentiments = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            self.logger.info(f"Processando lote {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            sentiments = self.analyze_political_sentiment(batch_texts, batch_size=len(batch_texts))
            all_sentiments.extend(sentiments)
        
        # Adicionar colunas de sentimento
        for i, sentiment in enumerate(all_sentiments):
            if i < len(result_df):
                result_df.loc[i, 'sentiment_category'] = sentiment.get('sentiment', 'neutro')
                result_df.loc[i, 'sentiment_confidence'] = sentiment.get('confidence', 0.5)
                result_df.loc[i, 'emotions_detected'] = json.dumps(sentiment.get('emotions', []))
                result_df.loc[i, 'has_irony'] = sentiment.get('irony', False)
                result_df.loc[i, 'sentiment_target'] = sentiment.get('target', 'unknown')
                result_df.loc[i, 'discourse_intensity'] = sentiment.get('intensity', 'média')
                result_df.loc[i, 'radicalization_level'] = sentiment.get('radicalization', 'nenhum')
                result_df.loc[i, 'dominant_tone'] = sentiment.get('tone', 'informativo')
        
        self.logger.info("Análise abrangente de sentimentos concluída")
        return result_df
    
    def generate_sentiment_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera relatório detalhado de análise de sentimentos
        
        Args:
            df: DataFrame com análises de sentimento
            
        Returns:
            Relatório de sentimentos
        """
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_texts": len(df),
            "sentiment_distribution": {},
            "emotion_analysis": {},
            "radicalization_analysis": {},
            "quality_metrics": {}
        }
        
        # Distribuição de sentimentos
        if 'sentiment_category' in df.columns:
            sentiment_counts = df['sentiment_category'].value_counts()
            report["sentiment_distribution"] = sentiment_counts.to_dict()
        
        # Análise de emoções
        if 'emotions_detected' in df.columns:
            all_emotions = []
            for emotions_str in df['emotions_detected'].fillna('[]'):
                try:
                    emotions = json.loads(emotions_str) if emotions_str else []
                    all_emotions.extend(emotions)
                except:
                    pass
            emotion_counts = Counter(all_emotions)
            report["emotion_analysis"] = dict(emotion_counts.most_common(10))
        
        # Análise de radicalização
        if 'radicalization_level' in df.columns:
            rad_counts = df['radicalization_level'].value_counts()
            report["radicalization_analysis"] = rad_counts.to_dict()
        
        # Métricas de qualidade
        if 'sentiment_confidence' in df.columns:
            confidence_scores = df['sentiment_confidence'].fillna(0)
            report["quality_metrics"] = {
                "avg_confidence": float(confidence_scores.mean()),
                "high_confidence_ratio": float((confidence_scores > 0.7).sum() / len(df)),
                "low_confidence_count": int((confidence_scores < 0.3).sum())
            }
        
        return report
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'message', 
                         sample_size: int = 1000) -> pd.DataFrame:
        """
        Analisa sentimentos em um DataFrame
        
        Args:
            df: DataFrame com textos
            text_column: Nome da coluna de texto
            sample_size: Tamanho da amostra para análise
            
        Returns:
            DataFrame com análises de sentimento
        """
        self.logger.info(f"Analisando sentimentos em {len(df)} mensagens")
        
        # Processar amostra se dataset muito grande
        if len(df) > sample_size:
            self.logger.info(f"Processando amostra de {sample_size} mensagens")
            # Amostragem estratificada se possível
            if 'channel' in df.columns:
                sample_df = df.groupby('channel', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // df['channel'].nunique()), 
                                     random_state=42)
                )
            else:
                sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df
        
        # Extrair textos
        texts = sample_df[text_column].fillna('').tolist()
        
        # Analisar sentimentos
        sentiments = self.analyze_political_sentiment(texts, batch_size=10)
        
        # Adicionar resultados ao DataFrame
        for i, sentiment in enumerate(sentiments):
            if i < len(sample_df):
                idx = sample_df.index[i]
                df.loc[idx, 'sentiment'] = sentiment.get('sentiment', 'neutro')
                df.loc[idx, 'sentiment_score'] = sentiment.get('sentiment_score', 0)
                df.loc[idx, 'sentiment_confidence'] = sentiment.get('confidence', 0)
                df.loc[idx, 'emotions'] = ','.join(sentiment.get('emotions', []))
                df.loc[idx, 'irony_detected'] = sentiment.get('irony', False)
                df.loc[idx, 'sentiment_target'] = sentiment.get('target', '')
                df.loc[idx, 'discourse_intensity'] = sentiment.get('intensity', 'média')
                df.loc[idx, 'radicalization_level'] = sentiment.get('radicalization', 'nenhum')
                df.loc[idx, 'dominant_tone'] = sentiment.get('dominant_tone', '')
        
        # Análise de camadas em subamostra
        layer_sample_size = min(50, len(sample_df))
        layer_indices = sample_df.sample(n=layer_sample_size, random_state=42).index
        
        for idx in layer_indices:
            text = df.loc[idx, text_column]
            if text:
                layers = self.detect_sentiment_layers(text)
                df.loc[idx, 'sentiment_layers'] = json.dumps(layers, ensure_ascii=False)
        
        self.logger.info("Análise de sentimentos concluída")
        return df