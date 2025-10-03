#!/usr/bin/env python3
"""
BATCH HÍBRIDO - Métodos Validados + API Batch Anthropic
Combina métodos científicos locais com análise avançada via API
"""

import os
import json
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BatchHybrid')

class AnthropicBatchProcessor:
    """Processador para API Batch da Anthropic"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.anthropic.com/v1/messages/batches'
        self.headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }

    def create_batch_requests(self, df: pd.DataFrame, analysis_type: str) -> Dict:
        """Cria requisições batch para diferentes tipos de análise"""

        prompts = {
            'political_classification': """Classifique o texto político brasileiro a seguir em uma das categorias:
- extrema-direita: nacionalismo extremo, autoritarismo, anti-sistema
- direita: conservadorismo, valores tradicionais, livre mercado
- centro-direita: moderado conservador, reformismo gradual
- centro: equilibrio, pragmatismo, consenso
- centro-esquerda: progressismo moderado, justiça social
- esquerda: progressismo, igualdade social, intervenção estatal

Responda APENAS com a categoria e uma confiança de 0-1.
Formato: categoria|confiança

Texto: {text}""",

            'sentiment_advanced': """Analise o sentimento e emoção do texto político brasileiro:
1. Sentimento geral: positivo/negativo/neutro (com score -1 a 1)
2. Emoções detectadas: raiva, medo, alegria, tristeza, nojo, surpresa
3. Tom: agressivo, conciliador, neutro, irônico, urgente
4. Intenção: mobilizar, informar, atacar, defender, questionar

Formato JSON compacto: {"sentiment": score, "emotions": [...], "tone": "...", "intent": "..."}

Texto: {text}""",

            'semantic_interpretation': """Analise semanticamente este texto político brasileiro:
1. Tópicos principais (máx 3)
2. Entidades políticas mencionadas (pessoas/partidos/instituições)
3. Posicionamento ideológico implícito
4. Frames narrativos utilizados (conflito/moralidade/economia/segurança)

Formato JSON: {"topics": [...], "entities": [...], "ideology": "...", "frames": [...]}

Texto: {text}""",

            'coordination_detection': """Analise se este texto parece ser parte de uma campanha coordenada:
1. Indicadores de coordenação (0-1)
2. Tipo provável: orgânico/coordenado/bot/spam
3. Padrões suspeitos detectados
4. Similaridade com narrativas conhecidas

Formato JSON: {"coordination_score": 0.0, "type": "...", "patterns": [...], "narrative_match": "..."}

Texto: {text}"""
        }

        if analysis_type not in prompts:
            raise ValueError(f"Tipo de análise não suportado: {analysis_type}")

        prompt_template = prompts[analysis_type]
        requests_list = []

        for idx, row in df.iterrows():
            # Pegar campo de texto (auto-detectar)
            text_field = None
            for field in ['text', 'body', 'message', 'content', 'texto', 'mensagem']:
                if field in row and pd.notna(row[field]):
                    text_field = field
                    break

            if not text_field:
                continue

            text = str(row[text_field])[:2000]  # Limitar tamanho

            requests_list.append({
                'custom_id': f'{analysis_type}_{idx}_{row.get("id", idx)}',
                'params': {
                    'model': 'claude-3-5-haiku-20241022',
                    'max_tokens': 500,
                    'temperature': 0.3,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt_template.format(text=text)
                        }
                    ]
                }
            })

        return {'requests': requests_list}

    def submit_batch(self, batch_data: Dict) -> str:
        """Submete batch e retorna ID"""
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=batch_data
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Batch submetido: {result.get('id')}")
                return result.get('id')
            else:
                logger.error(f"❌ Erro ao submeter batch: {response.text}")
                return None

        except Exception as e:
            logger.error(f"❌ Erro de conexão: {str(e)}")
            return None

    def check_status(self, batch_id: str) -> Dict:
        """Verifica status do batch"""
        try:
            response = requests.get(
                f"{self.base_url}/{batch_id}",
                headers=self.headers
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            logger.error(f"Erro ao verificar status: {str(e)}")
            return None

    def wait_for_completion(self, batch_id: str, max_wait: int = 3600) -> Optional[Dict]:
        """Aguarda conclusão do batch (max 1 hora default)"""
        start_time = time.time()
        check_interval = 30  # segundos

        logger.info(f"⏳ Aguardando processamento do batch {batch_id}...")

        while time.time() - start_time < max_wait:
            status = self.check_status(batch_id)

            if status:
                processing_status = status.get('processing_status')

                if processing_status == 'ended':
                    logger.info(f"✅ Batch concluído: {batch_id}")
                    return status

                elif processing_status == 'canceling' or processing_status == 'canceled':
                    logger.warning(f"⚠️ Batch cancelado: {batch_id}")
                    return None

                else:
                    # Mostrar progresso
                    requests_counts = status.get('request_counts', {})
                    completed = requests_counts.get('completed', 0)
                    total = requests_counts.get('total', 1)
                    progress = (completed / total) * 100 if total > 0 else 0

                    logger.info(f"   Progresso: {completed}/{total} ({progress:.1f}%)")

            time.sleep(check_interval)

        logger.error(f"❌ Timeout aguardando batch {batch_id}")
        return None

    def get_results(self, batch_id: str) -> Optional[List[Dict]]:
        """Obtém resultados do batch concluído"""
        status = self.check_status(batch_id)

        if not status or status.get('processing_status') != 'ended':
            logger.error("Batch ainda não concluído ou erro ao obter status")
            return None

        # Pegar URL dos resultados
        results_url = status.get('results_url')
        if not results_url:
            logger.error("URL de resultados não encontrada")
            return None

        try:
            # Baixar resultados
            response = requests.get(results_url, headers={'x-api-key': self.api_key})

            if response.status_code == 200:
                # Resultados vêm em formato JSONL (uma linha por resultado)
                results = []
                for line in response.text.strip().split('\n'):
                    if line:
                        results.append(json.loads(line))

                logger.info(f"✅ {len(results)} resultados obtidos")
                return results
            else:
                logger.error(f"Erro ao baixar resultados: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Erro ao processar resultados: {str(e)}")
            return None


class HybridBatchAnalyzer:
    """Analisador híbrido: Métodos Validados + API Anthropic"""

    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.use_api = bool(self.api_key)

        if self.use_api:
            self.batch_processor = AnthropicBatchProcessor(self.api_key)
            logger.info("✅ API Anthropic configurada")
        else:
            logger.warning("⚠️ API key não encontrada - usando apenas métodos locais")

        # Importar métodos validados se disponíveis
        try:
            from archive.pol_methods_implementation import ValidatedPoliticalAnalysis
            self.validated_methods = ValidatedPoliticalAnalysis
            self.has_validated = True
            logger.info("✅ Métodos validados disponíveis")
        except ImportError:
            self.has_validated = False
            logger.warning("⚠️ Métodos validados não encontrados")

    def analyze_with_api_batch(self, df: pd.DataFrame, analysis_types: List[str]) -> pd.DataFrame:
        """Executa análises usando API Batch"""

        if not self.use_api:
            logger.warning("API não configurada")
            return df

        results_df = df.copy()
        batch_ids = {}

        # Submeter batches para cada tipo de análise
        logger.info(f"\n🚀 Submetendo {len(analysis_types)} batches para API...")

        for analysis_type in analysis_types:
            logger.info(f"   Preparando batch: {analysis_type}")

            # Criar requisições
            batch_data = self.batch_processor.create_batch_requests(df, analysis_type)

            if batch_data['requests']:
                # Submeter
                batch_id = self.batch_processor.submit_batch(batch_data)
                if batch_id:
                    batch_ids[analysis_type] = batch_id
                    logger.info(f"   ✅ {analysis_type}: {len(batch_data['requests'])} requisições")

        # Aguardar e processar resultados
        logger.info(f"\n⏳ Aguardando processamento de {len(batch_ids)} batches...")

        for analysis_type, batch_id in batch_ids.items():
            logger.info(f"\n📊 Processando: {analysis_type}")

            # Aguardar conclusão
            final_status = self.batch_processor.wait_for_completion(batch_id)

            if final_status:
                # Obter resultados
                results = self.batch_processor.get_results(batch_id)

                if results:
                    # Processar e adicionar ao DataFrame
                    self._merge_api_results(results_df, results, analysis_type)
                    logger.info(f"   ✅ Resultados integrados: {analysis_type}")

        return results_df

    def _merge_api_results(self, df: pd.DataFrame, results: List[Dict], analysis_type: str):
        """Integra resultados da API no DataFrame"""

        # Criar mapeamento por custom_id
        results_map = {}
        for result in results:
            custom_id = result.get('custom_id', '')
            if result.get('result', {}).get('type') == 'message':
                content = result['result']['message']['content'][0]['text']
                results_map[custom_id] = content

        # Adicionar colunas baseadas no tipo de análise
        if analysis_type == 'political_classification':
            classifications = []
            confidences = []

            for idx in df.index:
                custom_id = f'{analysis_type}_{idx}_{df.loc[idx].get("id", idx)}'
                if custom_id in results_map:
                    try:
                        parts = results_map[custom_id].strip().split('|')
                        classifications.append(parts[0])
                        confidences.append(float(parts[1]) if len(parts) > 1 else 0.5)
                    except:
                        classifications.append('centro')
                        confidences.append(0.0)
                else:
                    classifications.append('centro')
                    confidences.append(0.0)

            df['political_category_api'] = classifications
            df['political_confidence_api'] = confidences

        elif analysis_type == 'sentiment_advanced':
            sentiments = []
            emotions = []

            for idx in df.index:
                custom_id = f'{analysis_type}_{idx}_{df.loc[idx].get("id", idx)}'
                if custom_id in results_map:
                    try:
                        data = json.loads(results_map[custom_id])
                        sentiments.append(data.get('sentiment', 0))
                        emotions.append(data.get('emotions', []))
                    except:
                        sentiments.append(0)
                        emotions.append([])
                else:
                    sentiments.append(0)
                    emotions.append([])

            df['sentiment_score_api'] = sentiments
            df['emotions_detected_api'] = emotions

        # Adicionar outros tipos conforme necessário...

    def analyze_with_validated_methods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica métodos cientificamente validados"""

        if not self.has_validated:
            logger.warning("Métodos validados não disponíveis")
            return df

        logger.info("\n🔬 Aplicando métodos validados...")

        # Instanciar analisador
        analyzer = self.validated_methods(df)

        # Aplicar análises
        texts = df['body'].fillna(df.get('text', '')).fillna('')

        # Frame Analysis (Entman)
        frames = analyzer.political_framing_analysis(texts)
        for col in frames.columns:
            df[f'frame_{col}'] = frames[col]

        # LIWC Portuguese
        liwc_results = analyzer.liwc_portuguese_analysis(texts)
        for col in liwc_results.columns:
            df[f'liwc_{col}'] = liwc_results[col]

        logger.info("   ✅ Métodos validados aplicados")

        return df

    def run_hybrid_analysis(self,
                           dataset_path: str,
                           use_api: bool = True,
                           use_validated: bool = True,
                           api_analyses: List[str] = None,
                           sample_size: Optional[int] = None) -> pd.DataFrame:
        """Executa análise híbrida completa"""

        # Carregar dados
        logger.info(f"\n📂 Carregando dataset: {dataset_path}")
        df = pd.read_csv(dataset_path, sep=';')

        # Aplicar amostragem se solicitado
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"   Amostra: {sample_size} registros")

        logger.info(f"   Total: {len(df)} registros")

        # Aplicar métodos validados locais
        if use_validated:
            df = self.analyze_with_validated_methods(df)

        # Aplicar análise via API Batch
        if use_api and self.use_api:
            api_analyses = api_analyses or ['political_classification', 'sentiment_advanced']
            df = self.analyze_with_api_batch(df, api_analyses)

        # Salvar resultados
        output_path = f"outputs/hybrid_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs('outputs', exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"\n✅ Análise completa salva em: {output_path}")
        logger.info(f"   Colunas adicionadas: {len(df.columns) - len(pd.read_csv(dataset_path, sep=';').columns)}")

        # Relatório resumido
        self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Imprime resumo da análise"""

        logger.info("\n" + "="*60)
        logger.info("📊 RESUMO DA ANÁLISE HÍBRIDA")
        logger.info("="*60)

        # Resumo API
        if 'political_category_api' in df.columns:
            logger.info("\n🏛️ Classificação Política (API):")
            counts = df['political_category_api'].value_counts()
            for cat, count in counts.items():
                logger.info(f"   {cat}: {count} ({count/len(df)*100:.1f}%)")

        # Resumo métodos validados
        frame_cols = [c for c in df.columns if c.startswith('frame_')]
        if frame_cols:
            logger.info("\n📰 Frames Detectados (Entman):")
            for col in frame_cols:
                avg = df[col].mean()
                logger.info(f"   {col.replace('frame_', '')}: {avg:.3f}")

        liwc_cols = [c for c in df.columns if c.startswith('liwc_')]
        if liwc_cols:
            logger.info("\n🧠 Análise LIWC:")
            for col in liwc_cols[:5]:  # Mostrar top 5
                avg = df[col].mean()
                logger.info(f"   {col.replace('liwc_', '')}: {avg:.2f}%")

        logger.info("\n" + "="*60)


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Análise Híbrida: Métodos Validados + API Anthropic')
    parser.add_argument('dataset', help='Caminho para o dataset CSV')
    parser.add_argument('--api-key', help='Chave API Anthropic (ou use ANTHROPIC_API_KEY env)')
    parser.add_argument('--sample', type=int, help='Tamanho da amostra para teste')
    parser.add_argument('--no-api', action='store_true', help='Desabilitar uso da API')
    parser.add_argument('--no-validated', action='store_true', help='Desabilitar métodos validados')
    parser.add_argument('--analyses', nargs='+',
                       choices=['political_classification', 'sentiment_advanced',
                               'semantic_interpretation', 'coordination_detection'],
                       help='Tipos de análise via API')

    args = parser.parse_args()

    # Configurar analisador
    analyzer = HybridBatchAnalyzer(anthropic_api_key=args.api_key)

    # Executar análise
    results = analyzer.run_hybrid_analysis(
        dataset_path=args.dataset,
        use_api=not args.no_api,
        use_validated=not args.no_validated,
        api_analyses=args.analyses,
        sample_size=args.sample
    )

    print(f"\n✅ Análise híbrida concluída com {len(results)} registros processados")