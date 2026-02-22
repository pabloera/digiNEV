#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_06.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any
import json
import time
from pathlib import Path


def _stage_06_affordances_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 06: Affordances Classification (Híbrido: Heurística + API)

    Estratégia otimizada em 3 fases:
    1. Heurística expandida classifica todas as mensagens com scoring
    2. Mensagens de alta confiança (>=0.6) ficam com resultado heurístico
    3. Mensagens de baixa confiança (<0.6) são enviadas à API em batches de 10

    Categorias:
    - noticia, midia_social, video_audio_gif, opiniao,
    - mobilizacao, ataque, interacao, is_forwarded
    """
    try:
        ctx.logger.info("🎯 STAGE 06: Affordances Classification (Híbrido)")

        import os
        import requests
        import json
        import time
        from typing import List, Dict, Any

        text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'
        initial_count = len(df)

        # === FASE 1: Heurística expandida em todas as mensagens ===
        ctx.logger.info(f"   📋 Fase 1: Heurística expandida em {initial_count} mensagens...")
        df = _stage_06_affordances_heuristic_fallback(df)

        # Contar mensagens por nível de confiança
        high_conf_mask = df['affordance_confidence'] >= 0.6
        low_conf_mask = df['affordance_confidence'] < 0.6
        high_conf_count = high_conf_mask.sum()
        low_conf_count = low_conf_mask.sum()

        ctx.logger.info(f"   🟢 Alta confiança heurística: {high_conf_count} ({high_conf_count/initial_count*100:.1f}%)")
        ctx.logger.info(f"   🟡 Baixa confiança (candidatas API): {low_conf_count} ({low_conf_count/initial_count*100:.1f}%)")

        # === FASE 2: Verificar se API está disponível ===
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            ctx.logger.warning("⚠️ ANTHROPIC_API_KEY não encontrada. Usando apenas heurística.")
            # Limpar coluna temporária
            if '_heuristic_scores' in df.columns:
                df = df.drop(columns=['_heuristic_scores'])
            ctx.stats['stages_completed'] += 1
            ctx.stats['features_extracted'] += 10
            return df

        if low_conf_count == 0:
            ctx.logger.info("   ✅ Todas as mensagens classificadas com alta confiança. API não necessária.")
            if '_heuristic_scores' in df.columns:
                df = df.drop(columns=['_heuristic_scores'])
            ctx.stats['stages_completed'] += 1
            ctx.stats['features_extracted'] += 10
            return df

        # === FASE 3: Classificar mensagens de baixa confiança via API (batches de 10) ===
        ctx.logger.info(f"   🤖 Fase 3: Enviando {low_conf_count} mensagens à API em batches de 10...")

        # Modelo configurável via .env (default: Haiku 3.5)
        configured_model = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-haiku-20241022')
        ctx.logger.info(f"   🔧 Modelo API: {configured_model}")

        api_config = {
            'model': configured_model,
            'max_tokens': 800,
            'temperature': 0.1,
            'system_prompt': """Você é um classificador de conteúdo especializado em discurso político brasileiro em redes sociais.

Classifique CADA mensagem numerada de acordo com as categorias de affordances (múltiplas possíveis):

1. noticia: Conteúdo informativo, reportagem, fatos
2. midia_social: Posts de redes sociais, compartilhamentos
3. video_audio_gif: Referências a conteúdo multimídia
4. opiniao: Opiniões pessoais, comentários subjetivos
5. mobilizacao: Chamadas para ação, mobilização política
6. ataque: Ataques pessoais, insultos, agressões verbais
7. interacao: Respostas, menções, conversações diretas
8. is_forwarded: Conteúdo encaminhado/repassado

Responda APENAS com um JSON array válido. Exemplo para 3 mensagens:
[{"id":1,"categorias":["opiniao","ataque"],"confianca":0.9},{"id":2,"categorias":["noticia"],"confianca":0.85},{"id":3,"categorias":["mobilizacao"],"confianca":0.8}]"""
        }

        def classify_batch_with_anthropic(texts: List[str]) -> List[Dict[str, Any]]:
            """Classificar batch de textos (até 10) em uma única chamada API."""
            # Montar mensagem com textos numerados
            numbered_texts = []
            for i, text in enumerate(texts, 1):
                text_sample = str(text)[:400] if not pd.isna(text) else ''
                if len(text_sample.strip()) < 10:
                    text_sample = '(mensagem vazia ou muito curta)'
                numbered_texts.append(f"[{i}] {text_sample}")

            user_content = "Classifique estas mensagens:\n\n" + "\n\n".join(numbered_texts)

            headers = {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'anthropic-beta': 'prompt-caching-2024-07-31'
            }

            payload = {
                'model': api_config['model'],
                'max_tokens': api_config['max_tokens'],
                'temperature': api_config['temperature'],
                'system': [
                    {
                        'type': 'text',
                        'text': api_config['system_prompt'],
                        'cache_control': {'type': 'ephemeral'}
                    }
                ],
                'messages': [{'role': 'user', 'content': user_content}]
            }

            try:
                response = requests.post(
                    'https://api.anthropic.com/v1/messages',
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result['content'][0]['text'].strip()

                    # Parse JSON array
                    try:
                        classifications = json.loads(content)
                        if isinstance(classifications, list):
                            return classifications
                    except json.JSONDecodeError:
                        # Tentar extrair JSON de resposta
                        if '[' in content and ']' in content:
                            json_start = content.find('[')
                            json_end = content.rfind(']') + 1
                            try:
                                classifications = json.loads(content[json_start:json_end])
                                if isinstance(classifications, list):
                                    return classifications
                            except Exception:
                                pass

                elif response.status_code == 429:
                    ctx.logger.warning("⚠️ Rate limit atingido, aguardando 5s...")
                    time.sleep(5)

                else:
                    ctx.logger.warning(f"⚠️ API error: {response.status_code}")

            except requests.RequestException as e:
                ctx.logger.warning(f"⚠️ Erro de conexão: {e}")

            # Retorno vazio em caso de erro
            return []

        # Processar mensagens de baixa confiança
        low_conf_indices = df.index[low_conf_mask].tolist()

        # Verificar se deve usar Batch API (assíncrona) ou chamadas individuais
        use_batch_api = os.getenv('USE_BATCH_API', 'false').lower() in ('true', '1', 'yes')

        if use_batch_api and low_conf_count > 100:
            # === BATCH API (assíncrona, 50% desconto, até 24h) ===
            ctx.logger.info(f"   📦 Usando Batch API assíncrona para {low_conf_count} mensagens...")
            df = _stage_06_submit_batch_api(
                df, low_conf_indices, text_column, api_key, api_config
            )
        else:
            # === CHAMADAS INDIVIDUAIS (síncrono, batches de 10) ===
            if use_batch_api and low_conf_count <= 100:
                ctx.logger.info(f"   ℹ️ Batch API não eficiente para {low_conf_count} mensagens. Usando chamadas diretas.")

            batch_size = 10
            api_calls_made = 0
            api_successes = 0
            api_failures = 0

            for i in range(0, len(low_conf_indices), batch_size):
                batch_indices = low_conf_indices[i:i+batch_size]
                batch_texts = [df.loc[idx, text_column] for idx in batch_indices]

                classifications = classify_batch_with_anthropic(batch_texts)
                api_calls_made += 1

                if classifications:
                    for j, idx in enumerate(batch_indices):
                        if j < len(classifications):
                            cls = classifications[j]
                            if isinstance(cls, dict) and 'categorias' in cls:
                                df.at[idx, 'affordance_categories'] = cls['categorias']
                                df.at[idx, 'affordance_confidence'] = cls.get('confianca', 0.8)
                                for aff_type in ['noticia', 'midia_social', 'video_audio_gif', 'opiniao',
                                                'mobilizacao', 'ataque', 'interacao', 'is_forwarded']:
                                    df.at[idx, f'aff_{aff_type}'] = 1 if aff_type in cls['categorias'] else 0
                                api_successes += 1
                            else:
                                api_failures += 1
                        else:
                            api_failures += 1
                else:
                    api_failures += len(batch_indices)

                time.sleep(0.2)

                if api_calls_made % 100 == 0:
                    progress = min(100, (i / len(low_conf_indices)) * 100)
                    ctx.logger.info(f"   🔄 API Progresso: {progress:.1f}% ({api_calls_made} calls, {api_successes} sucessos)")

        # === ESTATÍSTICAS FINAIS ===
        avg_confidence = df['affordance_confidence'].mean()
        classified_count = len(df[df['affordance_confidence'] > 0.1])

        affordance_types = ['noticia', 'midia_social', 'video_audio_gif', 'opiniao',
                          'mobilizacao', 'ataque', 'interacao', 'is_forwarded']
        category_counts = {}
        for affordance_type in affordance_types:
            count = df[f'aff_{affordance_type}'].sum()
            category_counts[affordance_type] = count

        # Limpar coluna temporária
        if '_heuristic_scores' in df.columns:
            df = df.drop(columns=['_heuristic_scores'])

        ctx.logger.info(f"✅ Classificação Híbrida de Affordances concluída:")
        ctx.logger.info(f"   📊 Heurística: {high_conf_count} mensagens ({high_conf_count/initial_count*100:.1f}%)")
        api_mode = "Batch API" if (use_batch_api and low_conf_count > 100) else "chamadas diretas"
        ctx.logger.info(f"   🤖 API ({api_mode}): {low_conf_count} mensagens processadas")
        ctx.logger.info(f"   ✅ Total classificadas: {classified_count}/{initial_count}")
        ctx.logger.info(f"   🎯 Confiança média: {avg_confidence:.3f}")

        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ctx.logger.info(f"   🔝 Top categorias: {dict(top_categories)}")

        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += len(affordance_types) + 2

        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 06: {e}")
        ctx.stats['processing_errors'] += 1
        # Fallback heurístico
        return _stage_06_affordances_heuristic_fallback(df)
def _stage_06_submit_batch_api(df: pd.DataFrame, low_conf_indices: list,
                                text_column: str, api_key: str, api_config: dict) -> pd.DataFrame:
    """
    Submeter mensagens de baixa confiança à Anthropic Batch API.
    Processa até 10.000 requests por batch com 50% de desconto.

    Args:
        df: DataFrame com dados
        low_conf_indices: Índices de mensagens de baixa confiança
        text_column: Nome da coluna de texto
        api_key: Chave da API Anthropic
        api_config: Configurações do modelo (model, system_prompt, etc.)

    Returns:
        DataFrame atualizado com classificações da Batch API
    """
    import requests
    import json
    import time
    import tempfile
    import os

    batch_size = 10  # Mensagens por request individual dentro do batch
    max_requests_per_batch = 10000  # Limite da Batch API

    ctx.logger.info(f"   📦 Preparando Batch API: {len(low_conf_indices)} mensagens em batches de {batch_size}")

    # === FASE 1: Gerar requests para a Batch API ===
    batch_requests = []
    request_mapping = {}  # custom_id -> lista de índices do DataFrame

    for i in range(0, len(low_conf_indices), batch_size):
        batch_indices = low_conf_indices[i:i+batch_size]
        batch_texts = []
        for idx in batch_indices:
            text = df.loc[idx, text_column]
            text_sample = str(text)[:400] if not pd.isna(text) else ''
            if len(text_sample.strip()) < 10:
                text_sample = '(mensagem vazia ou muito curta)'
            batch_texts.append(text_sample)

        # Montar mensagem com textos numerados
        numbered_texts = [f"[{j+1}] {t}" for j, t in enumerate(batch_texts)]
        user_content = "Classifique estas mensagens:\n\n" + "\n\n".join(numbered_texts)

        custom_id = f"batch_{i//batch_size:06d}"
        request_mapping[custom_id] = batch_indices

        request = {
            "custom_id": custom_id,
            "params": {
                "model": api_config['model'],
                "max_tokens": api_config['max_tokens'],
                "temperature": api_config['temperature'],
                "system": [
                    {
                        "type": "text",
                        "text": api_config['system_prompt'],
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [{"role": "user", "content": user_content}]
            }
        }
        batch_requests.append(request)

    total_requests = len(batch_requests)
    ctx.logger.info(f"   📝 {total_requests} requests gerados para Batch API")

    # === FASE 2: Submeter batches (até 10.000 por vez) ===
    all_results = {}

    for batch_start in range(0, total_requests, max_requests_per_batch):
        batch_chunk = batch_requests[batch_start:batch_start + max_requests_per_batch]
        chunk_num = batch_start // max_requests_per_batch + 1
        total_chunks = (total_requests + max_requests_per_batch - 1) // max_requests_per_batch

        ctx.logger.info(f"   🚀 Submetendo batch {chunk_num}/{total_chunks} ({len(batch_chunk)} requests)...")

        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'anthropic-beta': 'prompt-caching-2024-07-31'
        }

        payload = {"requests": batch_chunk}

        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages/batches',
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                ctx.logger.error(f"   ❌ Batch API erro: {response.status_code} - {response.text[:200]}")
                continue

            batch_response = response.json()
            batch_id = batch_response['id']
            ctx.logger.info(f"   ✅ Batch {batch_id} criado. Status: {batch_response['processing_status']}")

            # === FASE 3: Polling para resultados ===
            results = _stage_06_poll_batch_results(batch_id, api_key, headers)
            all_results.update(results)

        except requests.RequestException as e:
            ctx.logger.error(f"   ❌ Erro ao submeter batch: {e}")
            continue

    # === FASE 4: Aplicar resultados ao DataFrame ===
    api_successes = 0
    api_failures = 0

    for custom_id, classifications in all_results.items():
        if custom_id not in request_mapping:
            continue

        batch_indices = request_mapping[custom_id]

        if classifications:
            for j, idx in enumerate(batch_indices):
                if j < len(classifications):
                    cls = classifications[j]
                    if isinstance(cls, dict) and 'categorias' in cls:
                        df.at[idx, 'affordance_categories'] = cls['categorias']
                        df.at[idx, 'affordance_confidence'] = cls.get('confianca', 0.8)
                        for aff_type in ['noticia', 'midia_social', 'video_audio_gif', 'opiniao',
                                        'mobilizacao', 'ataque', 'interacao', 'is_forwarded']:
                            df.at[idx, f'aff_{aff_type}'] = 1 if aff_type in cls['categorias'] else 0
                        api_successes += 1
                    else:
                        api_failures += 1
                else:
                    api_failures += 1
        else:
            api_failures += len(batch_indices)

    ctx.logger.info(f"   📊 Batch API concluída: {api_successes} sucessos, {api_failures} falhas")
    return df
def _stage_06_poll_batch_results(batch_id: str, api_key: str, headers: dict,
                                  max_wait_seconds: int = 86400, poll_interval: int = 30) -> dict:
    """
    Polling para resultados da Batch API.

    Args:
        batch_id: ID do batch submetido
        api_key: Chave da API
        headers: Headers HTTP
        max_wait_seconds: Tempo máximo de espera (default: 24h)
        poll_interval: Intervalo entre polls (default: 30s)

    Returns:
        Dict de custom_id -> lista de classificações
    """
    import requests
    import json
    import time

    results = {}
    start_time = time.time()

    ctx.logger.info(f"   ⏳ Aguardando resultados do batch {batch_id}...")

    while time.time() - start_time < max_wait_seconds:
        try:
            # Verificar status do batch
            status_response = requests.get(
                f'https://api.anthropic.com/v1/messages/batches/{batch_id}',
                headers={
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01'
                },
                timeout=30
            )

            if status_response.status_code != 200:
                ctx.logger.warning(f"   ⚠️ Erro ao verificar status: {status_response.status_code}")
                time.sleep(poll_interval)
                continue

            batch_status = status_response.json()
            processing_status = batch_status['processing_status']
            request_counts = batch_status.get('request_counts', {})

            processing = request_counts.get('processing', 0)
            succeeded = request_counts.get('succeeded', 0)
            errored = request_counts.get('errored', 0)
            total = processing + succeeded + errored

            elapsed = int(time.time() - start_time)
            ctx.logger.info(
                f"   ⏳ Batch {batch_id}: {processing_status} "
                f"({succeeded}/{total} ok, {errored} erros, {elapsed}s decorridos)"
            )

            if processing_status == 'ended':
                # Coletar resultados
                results_url = batch_status.get('results_url')
                if results_url:
                    results = _stage_06_fetch_batch_results(results_url, api_key)
                else:
                    # Usar endpoint direto
                    results = _stage_06_fetch_batch_results(
                        f'https://api.anthropic.com/v1/messages/batches/{batch_id}/results',
                        api_key
                    )

                ctx.logger.info(f"   ✅ Batch {batch_id} finalizado: {len(results)} resultados coletados")
                return results

        except requests.RequestException as e:
            ctx.logger.warning(f"   ⚠️ Erro no polling: {e}")

        time.sleep(poll_interval)

    ctx.logger.warning(f"   ⚠️ Timeout: batch {batch_id} não concluiu em {max_wait_seconds}s")
    return results
def _stage_06_fetch_batch_results(results_url: str, api_key: str) -> dict:
    """
    Buscar e parsear resultados da Batch API (JSONL).

    Args:
        results_url: URL dos resultados
        api_key: Chave da API

    Returns:
        Dict de custom_id -> lista de classificações
    """
    import requests
    import json

    results = {}

    try:
        response = requests.get(
            results_url,
            headers={
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            },
            timeout=120,
            stream=True
        )

        if response.status_code != 200:
            ctx.logger.error(f"   ❌ Erro ao buscar resultados: {response.status_code}")
            return results

        # Parsear JSONL
        for line in response.iter_lines():
            if not line:
                continue
            try:
                entry = json.loads(line)
                custom_id = entry.get('custom_id', '')
                result = entry.get('result', {})

                if result.get('type') == 'succeeded':
                    message = result.get('message', {})
                    content_blocks = message.get('content', [])

                    # Extrair texto da resposta
                    text_content = ''
                    for block in content_blocks:
                        if block.get('type') == 'text':
                            text_content = block.get('text', '').strip()
                            break

                    # Parsear JSON de classificações
                    classifications = _stage_06_parse_batch_json(text_content)
                    results[custom_id] = classifications

                elif result.get('type') in ('errored', 'canceled', 'expired'):
                    results[custom_id] = []
                    ctx.logger.debug(f"   ⚠️ Request {custom_id}: {result.get('type')}")

            except json.JSONDecodeError:
                continue

    except requests.RequestException as e:
        ctx.logger.error(f"   ❌ Erro ao buscar resultados: {e}")

    return results
def _stage_06_parse_batch_json(text: str) -> list:
    """
    Parsear JSON de classificações da resposta da API.
    Tenta múltiplas estratégias de parsing.

    Args:
        text: Texto da resposta da API

    Returns:
        Lista de dicts com categorias e confiança
    """
    import json

    if not text:
        return []

    # Tentativa 1: JSON direto
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Tentativa 2: Extrair array JSON do texto
    if '[' in text and ']' in text:
        json_start = text.find('[')
        json_end = text.rfind(']') + 1
        try:
            result = json.loads(text[json_start:json_end])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Tentativa 3: Extrair objetos JSON individuais
    try:
        objects = []
        import re
        for match in re.finditer(r'\{[^{}]+\}', text):
            try:
                obj = json.loads(match.group())
                if 'categorias' in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
        if objects:
            return objects
    except Exception:
        pass

    return []
def _stage_06_affordances_heuristic_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback heurístico para classificação de affordances sem API."""
    ctx.logger.info("🔄 Aplicando classificação heurística de affordances...")

    text_column = 'normalized_text' if 'normalized_text' in df.columns else 'body'

    # Padrões heurísticos expandidos (15-20 keywords por categoria)
    patterns = {
        'noticia': [
            'aconteceu', 'notícia', 'informação', 'fato', 'governo', 'brasil',
            'reportagem', 'jornal', 'imprensa', 'publicou', 'divulgou', 'segundo',
            'fonte', 'comunicado', 'nota oficial', 'decreto', 'lei', 'medida',
            'aprovado', 'anunciou', 'declarou', 'dados', 'pesquisa', 'estudo'
        ],
        'midia_social': [
            'compartilhem', 'rt', 'retweet', 'curtir', 'like', 'seguir',
            'inscreva', 'canal', 'grupo', 'telegram', 'whatsapp', 'twitter',
            'instagram', 'youtube', 'facebook', 'tiktok', 'sigam', 'divulguem',
            'espalhem', 'repassem'
        ],
        'video_audio_gif': [
            'vídeo', 'video', 'audio', 'áudio', 'gif', 'assista', 'ouça',
            'podcast', 'live', 'ao vivo', 'transmissão', 'gravação', 'filmou',
            'imagem', 'foto', 'print', 'screenshot', 'clipe', 'documentário'
        ],
        'opiniao': [
            'acho', 'penso', 'na minha opinião', 'acredito', 'creio',
            'considero', 'entendo', 'parece', 'me parece', 'na verdade',
            'sinceramente', 'francamente', 'obviamente', 'claramente',
            'infelizmente', 'felizmente', 'absurdo', 'ridículo', 'inaceitável'
        ],
        'mobilizacao': [
            'vamos', 'precisamos', 'juntos', 'ação', 'mobilizar',
            'protesto', 'manifestação', 'marcha', 'ato', 'convocação',
            'compareçam', 'participem', 'lutar', 'resistir', 'unir',
            'levantar', 'defender', 'cobrar', 'exigir', 'pressionar'
        ],
        'ataque': [
            'idiota', 'burro', 'canalha', 'corrupto', 'mentiroso',
            'ladrão', 'bandido', 'vagabundo', 'safado', 'lixo',
            'vergonha', 'nojo', 'traidor', 'covarde', 'hipócrita',
            'incompetente', 'criminoso', 'fascista', 'comunista', 'genocida'
        ],
        'interacao': [
            '@', 'resposta', 'pergunta', 'dúvida', 'respondendo',
            'concordo', 'discordo', 'exatamente', 'isso mesmo',
            'verdade', 'falso', 'correto', 'errado', 'complementando'
        ],
        'is_forwarded': [
            'encaminhado', 'forward', 'repasse', 'compartilhe',
            'repassando', 'recebi', 'me mandaram', 'vejam isso',
            'olha isso', 'leiam', 'importante', 'urgente', 'atenção'
        ]
    }

    import re

    def classify_text_heuristic(text):
        """Classifica texto por heurística com scoring de confiança."""
        text_lower = str(text).lower() if not pd.isna(text) else ''
        if len(text_lower) < 5:
            return [], {}, 0.0

        categories = []
        scores = {}
        total_matches = 0

        for affordance_type, keywords in patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            scores[affordance_type] = matches
            if matches >= 1:
                categories.append(affordance_type)
                total_matches += matches

        # FIX: regex em normalized_text nunca matcheia (: e // são removidos na normalização)
        # A detecção de URL agora é feita fora do loop, usando 'urls_extracted' do Stage 01
        if re.search(r'https?://', text_lower):
            if 'noticia' not in categories:
                categories.append('noticia')
                scores['noticia'] = scores.get('noticia', 0) + 1
                total_matches += 1

        if text_lower.count('@') >= 2:
            if 'interacao' not in categories:
                categories.append('interacao')
                scores['interacao'] = scores.get('interacao', 0) + 2
                total_matches += 2

        # Confiança baseada no total de matches
        if total_matches == 0:
            confidence = 0.1
        elif total_matches <= 2:
            confidence = 0.4
        elif total_matches <= 4:
            confidence = 0.6
        elif total_matches <= 7:
            confidence = 0.75
        else:
            confidence = 0.85

        return categories, scores, confidence

    # Aplicar classificação vetorizada
    ctx.logger.info(f"   📊 Classificando {len(df)} mensagens por heurística expandida...")
    results = df[text_column].apply(classify_text_heuristic)

    # Extrair resultados
    df['affordance_categories'] = results.apply(lambda x: x[0])
    df['_heuristic_scores'] = results.apply(lambda x: x[1])
    df['affordance_confidence'] = results.apply(lambda x: x[2])

    # Colunas binárias por categoria
    affordance_types = ['noticia', 'midia_social', 'video_audio_gif', 'opiniao',
                      'mobilizacao', 'ataque', 'interacao', 'is_forwarded']
    for affordance_type in affordance_types:
        df[f'aff_{affordance_type}'] = df['affordance_categories'].apply(
            lambda cats: 1 if affordance_type in cats else 0
        )

    # FIX: URL detection — regex em normalized_text nunca matcheia (://  removido)
    # Usar 'urls_extracted' do Stage 01 (preserva URLs reais do body)
    if 'urls_extracted' in df.columns:
        has_url = df['urls_extracted'].apply(
            lambda x: len(x) > 0 if isinstance(x, list) else bool(x) if x else False
        )
        # Marcar textos com URL como 'noticia' se não classificados
        mask_url_not_noticia = has_url & (df['aff_noticia'] == 0)
        df.loc[mask_url_not_noticia, 'aff_noticia'] = 1
        url_boost_count = mask_url_not_noticia.sum()
        if url_boost_count > 0:
            ctx.logger.info(f"   🔗 URL detection via urls_extracted: +{url_boost_count} classificações 'noticia'")

    # Estatísticas
    classified = len(df[df['affordance_confidence'] > 0.1])
    high_conf = len(df[df['affordance_confidence'] >= 0.6])
    low_conf = len(df[(df['affordance_confidence'] > 0.1) & (df['affordance_confidence'] < 0.6)])

    ctx.logger.info(f"✅ Classificação heurística expandida concluída:")
    ctx.logger.info(f"   📊 Total classificadas: {classified}/{len(df)} ({classified/len(df)*100:.1f}%)")
    ctx.logger.info(f"   🟢 Alta confiança (>=0.6): {high_conf} ({high_conf/len(df)*100:.1f}%)")
    ctx.logger.info(f"   🟡 Baixa confiança (<0.6): {low_conf} ({low_conf/len(df)*100:.1f}%)")

    # Contagem por categoria
    for aff_type in affordance_types:
        count = df[f'aff_{aff_type}'].sum()
        if count > 0:
            ctx.logger.info(f"   📌 {aff_type}: {count} ({count/len(df)*100:.1f}%)")

    return df

# ===============================================
# MÉTODOS HELPER PARA ANÁLISE DE QUALIDADE
# ===============================================

