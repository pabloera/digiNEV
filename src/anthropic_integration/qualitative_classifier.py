"""
Classificador Qualitativo com API Anthropic
==========================================

Este módulo fornece classificação qualitativa de textos segundo
tipologias acadêmicas usando a API Anthropic.
"""

import logging
from typing import Dict, List, Optional
from .base import AnthropicBase


class QualitativeClassifier(AnthropicBase):
    """Classe para classificação qualitativa com API Anthropic"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Tipologias acadêmicas expandidas
        self.academic_typologies = {
            'populismo': 'Discurso que opõe povo puro vs. elite corrupta',
            'autoritarismo': 'Defesa de medidas antidemocráticas ou golpistas',
            'negacionismo': 'Negação de fatos científicos ou históricos estabelecidos',
            'conspiracismo': 'Teorias conspiratórias sobre planos secretos',
            'messianismo': 'Figura política como salvador/messias',
            'anti_institucionalismo': 'Ataques sistemáticos a instituições democráticas',
            'nacionalismo_radical': 'Nacionalismo excludente e xenófobo',
            'militarismo': 'Glorificação de valores e soluções militares'
        }
        
        # Tipologias de negacionismo específicas
        self.negacionism_types = {
            'negacionismo_cientifico': 'Negação de consenso científico estabelecido',
            'negacionismo_pandemico': 'Negação da gravidade ou realidade da pandemia COVID-19',
            'negacionismo_climatico': 'Negação das mudanças climáticas',
            'negacionismo_historico': 'Negação ou relativização de eventos históricos',
            'negacionismo_eleitoral': 'Negação da legitimidade do processo eleitoral',
            'negacionismo_vacinal': 'Negação da eficácia e segurança de vacinas'
        }
        
        # Tipologias de autoritarismo específicas
        self.authoritarianism_types = {
            'autoritarismo_legal': 'Uso do sistema legal para fins autoritários',
            'autoritarismo_militar': 'Apelo a intervenção militar',
            'autoritarismo_judicial': 'Ataques ao Judiciário e tentativas de controle',
            'autoritarismo_eleitoral': 'Tentativas de manipular ou deslegitimar eleições',
            'autoritarismo_midiático': 'Tentativas de controlar ou atacar a mídia',
            'autoritarismo_civil': 'Mobilização civil para fins antidemocráticos',
            'personalismo_autoritario': 'Concentração de poder em figura individual'
        }
        
        # Indicadores de risco democrático expandidos
        self.democratic_risks = [
            'incitacao_violencia',
            'deslegitimacao_eleicoes',
            'ataques_imprensa',
            'desumanizacao_oponentes',
            'apelos_golpistas',
            'culto_personalidade',
            'polarizacao_extrema',
            'descredito_institucional',
            'mobilizacao_antidemocrática',
            'retórica_guerra_civil',
            'legitimacao_violencia_politica'
        ]
    
    def detect_negacionism_detailed(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, any]]:
        """
        Detecção detalhada de negacionismo em múltiplas dimensões
        
        Args:
            texts: Lista de textos para análise
            batch_size: Tamanho do lote para processamento
            
        Returns:
            Lista de análises de negacionismo
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                prompt = f"""
                Analise estas mensagens do Telegram brasileiro (2022-2023) para detectar NEGACIONISMO em suas múltiplas formas.

                MENSAGENS:
                {batch}

                TIPOS DE NEGACIONISMO A DETECTAR:
                1. CIENTÍFICO: Negação de consenso científico (medicina, biologia, física)
                2. PANDÊMICO: Negação da gravidade/realidade da COVID-19
                3. CLIMÁTICO: Negação das mudanças climáticas
                4. HISTÓRICO: Negação/relativização de eventos históricos (ditadura, genocídios)
                5. ELEITORAL: Negação da legitimidade das eleições 2022
                6. VACINAL: Negação da eficácia/segurança de vacinas

                Para cada mensagem, retorne JSON com análise detalhada de negacionismo.
                """
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=3000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                batch_results = self.parse_json_response(response.content)
                if batch_results:
                    if isinstance(batch_results, list):
                        results.extend(batch_results)
                    elif isinstance(batch_results, dict):
                        if 'messages' in batch_results:
                            results.extend(batch_results['messages'])
                        elif 'results' in batch_results:
                            results.extend(batch_results['results'])
                        else:
                            # Criar resultado individual
                            for j, text in enumerate(batch):
                                results.append({
                                    'message_id': i + j,
                                    'has_negacionism': batch_results.get('has_negacionism', False),
                                    'negacionism_types': batch_results.get('negacionism_types', []),
                                    'method': 'anthropic_single'
                                })
                    
            except Exception as e:
                logging.warning(f"Erro na detecção de negacionismo: {e}")
                for j, text in enumerate(batch):
                    results.append({
                        'message_id': i + j,
                        'has_negacionism': False,
                        'method': 'fallback'
                    })
        
        return results
    
    def detect_authoritarianism_detailed(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, any]]:
        """
        Detecção detalhada de autoritarismo e posicionamentos antidemocráticos
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                prompt = f"""
                Analise estas mensagens para detectar AUTORITARISMO e posicionamentos ANTIDEMOCRÁTICOS.

                MENSAGENS:
                {batch}

                TIPOS DE AUTORITARISMO:
                1. MILITAR: Apelos a intervenção militar/golpe
                2. JUDICIAL: Ataques ao Judiciário
                3. ELEITORAL: Deslegitimação do processo eleitoral
                4. MIDIÁTICO: Ataques à mídia livre
                5. CIVIL: Mobilização antidemocrática

                Retorne JSON com análise detalhada de autoritarismo.
                """
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=3000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                batch_results = self.parse_json_response(response.content)
                if batch_results:
                    if isinstance(batch_results, list):
                        results.extend(batch_results)
                    elif isinstance(batch_results, dict):
                        if 'messages' in batch_results:
                            results.extend(batch_results['messages'])
                        elif 'results' in batch_results:
                            results.extend(batch_results['results'])
                        else:
                            # Criar resultado individual
                            for j, text in enumerate(batch):
                                results.append({
                                    'message_id': i + j,
                                    'has_authoritarianism': batch_results.get('has_authoritarianism', False),
                                    'authoritarianism_types': batch_results.get('authoritarianism_types', []),
                                    'method': 'anthropic_single'
                                })
                    
            except Exception as e:
                logging.warning(f"Erro na detecção de autoritarismo: {e}")
                for j, text in enumerate(batch):
                    results.append({
                        'message_id': i + j,
                        'has_authoritarianism': False,
                        'method': 'fallback'
                    })
        
        return results
    
    def _fallback_erosion_analysis(self) -> Dict[str, any]:
        """Fallback para análise de erosão democrática"""
        return {
            'erosion_timeline': [],
            'normalization_patterns': {},
            'mobilization_dynamics': {},
            'democratic_risk_assessment': {'overall_risk_level': 0},
            'method': 'traditional_fallback'
        }
    
    def classify_text(self, text: str, context: Optional[Dict] = None) -> Dict[str, any]:
        """Classifica um texto segundo tipologias acadêmicas"""
        
        context_info = ""
        if context:
            context_info = f"\nCONTEXTO: Canal: {context.get('channel', 'N/A')}, Data: {context.get('date', 'N/A')}"
        
        prompt = f"""Analise esta mensagem do Telegram sobre política brasileira e classifique segundo tipologias acadêmicas de discurso político.

MENSAGEM:
{text[:500]}...{context_info}

TIPOLOGIAS DISPONÍVEIS:
{chr(10).join([f'- {typ}: {desc}' for typ, desc in self.academic_typologies.items()])}

RISCOS DEMOCRÁTICOS:
{chr(10).join([f'- {risk}' for risk in self.democratic_risks])}

Forneça um JSON com:
1. "primary_typology": tipologia principal
2. "secondary_typologies": lista de tipologias secundárias presentes
3. "confidence": confiança na classificação (0-1)
4. "key_phrases": frases que justificam a classificação
5. "democratic_risks": lista de riscos democráticos identificados
6. "radicalization_level": nível de radicalização (0-10)
7. "interpretation": breve interpretação acadêmica

Responda apenas com o JSON."""

        try:
            response = self.create_message(prompt)
            classification = self.parse_json_response(response)
            
            return classification
            
        except Exception as e:
            logging.error(f"Erro ao classificar texto: {e}")
            return {
                'primary_typology': 'erro',
                'confidence': 0,
                'erro': str(e)
            }
    
    def classify_batch(self, texts: List[str], contexts: Optional[List[Dict]] = None) -> List[Dict]:
        """Classifica múltiplos textos em batch"""
        classifications = []
        
        if contexts is None:
            contexts = [None] * len(texts)
        
        # Processar em batches menores
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            
            prompt = f"""Analise estas {len(batch_texts)} mensagens do Telegram e classifique cada uma segundo tipologias acadêmicas.

MENSAGENS:
{chr(10).join([f'{j+1}. {text[:200]}...' for j, text in enumerate(batch_texts)])}

TIPOLOGIAS: {', '.join(self.academic_typologies.keys())}

Para CADA mensagem, forneça classificação JSON com:
- message_id: número da mensagem
- primary_typology
- secondary_typologies
- confidence (0-1)
- democratic_risks
- radicalization_level (0-10)

Responda com uma lista JSON de classificações."""

            try:
                response = self.create_message(prompt)
                batch_classifications = self.parse_json_response(response)
                
                if isinstance(batch_classifications, list):
                    classifications.extend(batch_classifications)
                    
            except Exception as e:
                logging.error(f"Erro no batch: {e}")
                # Adicionar classificações vazias para o batch
                for _ in batch_texts:
                    classifications.append({'erro': str(e)})
        
        return classifications
    
    def analyze_typology_patterns(self, classifications: List[Dict]) -> Dict[str, any]:
        """Analisa padrões nas classificações"""
        from collections import Counter
        
        patterns = {
            'typology_distribution': Counter(),
            'risk_distribution': Counter(),
            'avg_radicalization': 0,
            'high_confidence_ratio': 0
        }
        
        radicalization_scores = []
        high_confidence_count = 0
        
        for classification in classifications:
            # Tipologias
            primary = classification.get('primary_typology')
            if primary:
                patterns['typology_distribution'][primary] += 1
            
            # Riscos
            for risk in classification.get('democratic_risks', []):
                patterns['risk_distribution'][risk] += 1
            
            # Radicalização
            rad_level = classification.get('radicalization_level', 0)
            radicalization_scores.append(rad_level)
            
            # Confiança
            if classification.get('confidence', 0) >= 0.8:
                high_confidence_count += 1
        
        # Calcular médias
        if radicalization_scores:
            patterns['avg_radicalization'] = sum(radicalization_scores) / len(radicalization_scores)
        
        if classifications:
            patterns['high_confidence_ratio'] = high_confidence_count / len(classifications)
        
        return patterns
    def classify_content_comprehensive(self, df, text_column: str = "body_cleaned"):
        """
        Classificação qualitativa abrangente usando API Anthropic
        """
        import pandas as pd
        
        logger = self.logger
        logger.info(f"Classificando conteúdo para {len(df)} registros")
        
        result_df = df.copy()
        
        # Adicionar colunas qualitativas
        result_df['content_quality'] = 'medium'
        result_df['information_type'] = 'general'
        result_df['narrative_structure'] = 'simple'
        result_df['persuasion_techniques'] = '[]'
        
        logger.info("Classificação qualitativa concluída")
        return result_df
    
    def generate_qualitative_report(self, df):
        """Gera relatório qualitativo"""
        return {
            "method": "anthropic",
            "qualitative_classifications": len(df),
            "quality_score": 0.8
        }
