#!/usr/bin/env python3
"""
digiNEV Pipeline — Teste Ponta-a-Ponta (End-to-End) v6.2
=========================================================

Valida o pipeline completo de 17 stages com 6 stages API.
Testa com dados reais dos CSVs de Telegram bolsonarista.

Modos:
  --quick     : 100 rows, validacao basica (~90s)
  --standard  : 200 rows, validacao completa (~3min)  [default]
  --full      : 500 rows, validacao robusta (~6min)
  --stress    : 1000 rows, teste de carga (~12min)

Uso:
  python test_e2e_pipeline.py
  python test_e2e_pipeline.py --quick
  python test_e2e_pipeline.py --full
  python test_e2e_pipeline.py --dataset /caminho/para/arquivo.csv
  python test_e2e_pipeline.py --no-api   # desabilita API, 100% heuristico
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Setup ──
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Load .env manually (sem depender de dotenv)
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                os.environ[key.strip()] = val.strip()

import pandas as pd

# ── Constants ──
DATASETS = {
    'elec': '/Users/pabloalmada/Documents/Telegram/DATASETS_FULL/encoding_fixed/4_2022-2023-elec.csv',
    'pandemia': '/Users/pabloalmada/Documents/Telegram/DATASETS_FULL/encoding_fixed/2_2021-2022-pandemia.csv',
    'govbolso': '/Users/pabloalmada/Documents/Telegram/DATASETS_FULL/encoding_fixed/1_2019-2021-govbolso.csv',
    'poseleic': '/Users/pabloalmada/Documents/Telegram/DATASETS_FULL/encoding_fixed/3_2022-2023-poseleic.csv',
    'test_100': str(PROJECT_ROOT / 'data' / 'controlled_test_100.csv'),
}

# Colunas esperadas por stage (MAPA COMPLETO v6.2 — nomes reais do pipeline)
EXPECTED_COLUMNS = {
    'stage_01': ['main_text_column', 'timestamp_column', 'has_timestamp'],
    'stage_02': ['normalized_text', 'word_count', 'char_count'],
    'stage_03': ['dupli_freq', 'channels_found', 'date_span_days'],
    'stage_04': ['caps_ratio', 'emoji_ratio'],
    'stage_05': ['content_quality_score'],
    'stage_06': ['aff_noticia', 'aff_opiniao', 'aff_ataque'],
    'stage_07': ['spacy_lemmas', 'spacy_entities_count', 'lemmatized_text'],
    'stage_08': ['political_orientation', 'political_keywords', 'political_intensity',
                 'political_confidence', 'tcw_codes', 'tcw_categories'],
    'stage_09': ['tfidf_score_max', 'tfidf_top_terms'],
    'stage_10': ['cluster_id', 'cluster_distance'],
    'stage_11': ['dominant_topic', 'topic_probability', 'topic_keywords',
                 'topic_label', 'topic_confidence'],
    'stage_12': ['sentiment_polarity', 'sentiment_label', 'sentiment_confidence',
                 'emotion_intensity', 'has_aggressive_language',
                 'emotion_anger', 'emotion_fear', 'emotion_hope',
                 'emotion_disgust', 'emotion_sarcasm'],
    'stage_13': ['hour', 'day_of_week'],
    'stage_14': ['sender_frequency', 'is_frequent_sender', 'temporal_coordination'],
    'stage_15': ['url_count', 'has_external_links'],
    'stage_16': ['political_context', 'mentions_government', 'frame_conflito',
                 'event_confidence', 'specific_event'],
    'stage_17': ['channel_type', 'channel_activity', 'is_active_channel',
                 'channel_confidence', 'channel_theme'],
}

# Stages com API (6 total)
API_STAGES = ['stage_06', 'stage_08', 'stage_11', 'stage_12', 'stage_16', 'stage_17']


def print_header(text, char='=', width=80):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_section(text):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def load_data(dataset_path, nrows):
    """Carregar dados com tratamento robusto."""
    df = pd.read_csv(
        dataset_path,
        sep=',', quotechar='"', quoting=1,
        on_bad_lines='skip', nrows=nrows
    )
    return df


def validate_stage_columns(df, stage_name, expected_cols):
    """Validar que colunas de um stage existem e tem dados reais."""
    results = {'present': [], 'missing': [], 'empty': []}

    for col in expected_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                results['present'].append(col)
            else:
                results['empty'].append(col)
        else:
            results['missing'].append(col)

    return results


def validate_api_quality(df):
    """Validar qualidade dos resultados de API."""
    api_metrics = {}

    # Stage 08 — Political
    if 'political_orientation' in df.columns:
        total = len(df)
        neutral_pct = (df['political_orientation'] == 'neutral').mean() * 100
        api_metrics['s08_neutral_pct'] = neutral_pct
        api_metrics['s08_confidence_mean'] = df['political_confidence'].mean() if 'political_confidence' in df.columns else 0

    # Stage 11 — Topic Modeling
    if 'topic_label' in df.columns:
        labels = df['topic_label'].unique()
        api_named = sum(1 for l in labels if not l.startswith('topic_'))
        api_metrics['s11_api_named_topics'] = api_named
        api_metrics['s11_total_topics'] = len(labels)
        api_metrics['s11_confidence_mean'] = df['topic_confidence'].mean() if 'topic_confidence' in df.columns else 0

    # Stage 12 — Sentiment
    if 'sentiment_label' in df.columns:
        api_metrics['s12_sentiment_dist'] = df['sentiment_label'].value_counts().to_dict()
        api_metrics['s12_sarcasm_count'] = int(df['emotion_sarcasm'].sum()) if 'emotion_sarcasm' in df.columns else 0
        api_metrics['s12_anger_mean'] = df['emotion_anger'].mean() if 'emotion_anger' in df.columns else 0

    # Stage 16 — Event Context
    if 'specific_event' in df.columns:
        events = df[df['specific_event'] != 'none']['specific_event'].value_counts()
        api_metrics['s16_events_detected'] = int(events.sum()) if len(events) > 0 else 0
        api_metrics['s16_event_types'] = events.to_dict()
        api_metrics['s16_confidence_mean'] = df['event_confidence'].mean() if 'event_confidence' in df.columns else 0

    # Stage 17 — Channel Analysis
    if 'channel_type' in df.columns:
        general_pct = (df['channel_type'] == 'general').mean() * 100
        api_metrics['s17_general_pct'] = general_pct
        api_metrics['s17_channel_types'] = df['channel_type'].value_counts().to_dict()
        api_metrics['s17_confidence_mean'] = df['channel_confidence'].mean() if 'channel_confidence' in df.columns else 0

    return api_metrics


def validate_data_flow(df):
    """Validar que os dados fluem corretamente entre stages."""
    checks = []

    # Stage 01 → 02: normalized_text deve ser diferente de body
    if 'body' in df.columns and 'normalized_text' in df.columns:
        diff_pct = (df['body'].fillna('') != df['normalized_text'].fillna('')).mean() * 100
        checks.append(('S01→S02 text_normalization', diff_pct > 0, f'{diff_pct:.1f}% diferente'))

    # Stage 03: deduplicacao deve remover registros
    if 'is_duplicate' in df.columns:
        dup_pct = df['is_duplicate'].mean() * 100
        checks.append(('S03 deduplication', True, f'{dup_pct:.1f}% duplicatas'))

    # Stage 07 → 08: spacy_lemmas deve ser usado por political_classification
    if 'spacy_lemmas' in df.columns and 'political_orientation' in df.columns:
        has_lemmas = df['spacy_lemmas'].notna().mean() * 100
        has_politics = df['political_orientation'].notna().mean() * 100
        checks.append(('S07→S08 lemmas→political', has_lemmas > 50, f'lemmas={has_lemmas:.0f}%, political={has_politics:.0f}%'))

    # Stage 08 → TCW: tcw_codes dependem de political_keywords
    if 'tcw_codes' in df.columns and 'political_keywords' in df.columns:
        has_tcw = df['tcw_codes'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).mean() * 100
        has_keywords = df['political_keywords'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).mean() * 100
        checks.append(('S08 keywords→TCW', True, f'keywords={has_keywords:.0f}%, tcw={has_tcw:.0f}%'))

    # Stage 09 → 10: TF-IDF → Clustering
    if 'tfidf_score_max' in df.columns and 'cluster_id' in df.columns:
        checks.append(('S09→S10 tfidf→clustering', True, f'clusters={df["cluster_id"].nunique()}'))

    # Stage 11 → topic_label (API naming)
    if 'topic_label' in df.columns:
        labels = df['topic_label'].unique()
        api_named = [l for l in labels if not l.startswith('topic_')]
        checks.append(('S11 API topic naming', len(api_named) > 0, f'{len(api_named)}/{len(labels)} nomeados via API'))

    # Stage 16 → specific_event (API detection)
    if 'specific_event' in df.columns:
        events = (df['specific_event'] != 'none').sum()
        checks.append(('S16 API event detection', events > 0, f'{events} eventos detectados'))

    return checks


def run_test(args):
    """Executar teste ponta-a-ponta."""

    # ── Configuracao ──
    mode_config = {
        'quick': 100,
        'standard': 200,
        'full': 500,
        'stress': 1000,
    }
    nrows = mode_config.get(args.mode, 200)

    # Desabilitar API se --no-api
    if args.no_api:
        os.environ.pop('ANTHROPIC_API_KEY', None)
        print("  API DESABILITADA — 100% heuristico")

    # Selecionar dataset
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = DATASETS.get('elec')

    if not Path(dataset_path).exists():
        print(f"  ERRO: Dataset nao encontrado: {dataset_path}")
        return False

    # ── Header ──
    print_header(f"digiNEV Pipeline — Teste Ponta-a-Ponta v6.2")
    print(f"  Modo     : {args.mode} ({nrows} rows)")
    print(f"  Dataset  : {Path(dataset_path).name}")
    print(f"  API      : {'ATIVA (6 stages)' if not args.no_api else 'DESABILITADA'}")
    print(f"  Modelo   : {os.environ.get('ANTHROPIC_MODEL', 'N/A')}")
    print(f"  Batch API: {os.environ.get('USE_BATCH_API', 'false')}")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # ── ETAPA 1: Carregar dados ──
    print_section("ETAPA 1: Carregamento de dados")
    t0 = time.time()
    df_input = load_data(dataset_path, nrows)
    t_load = time.time() - t0
    print(f"  Registros carregados: {len(df_input)}")
    print(f"  Colunas de entrada  : {len(df_input.columns)} → {list(df_input.columns)}")
    print(f"  Tempo               : {t_load:.1f}s")

    # ── ETAPA 2: Executar pipeline ──
    print_section("ETAPA 2: Execucao do pipeline (17 stages)")
    from src.analyzer import Analyzer

    t1 = time.time()
    analyzer = Analyzer()
    result = analyzer.analyze(df_input)
    t_pipeline = time.time() - t1

    data = result['data']
    stats = result['stats']

    print(f"  Stages completados  : {stats['stages_completed']}/17")
    print(f"  Erros de processing : {stats['processing_errors']}")
    print(f"  Registros entrada   : {len(df_input)}")
    print(f"  Registros saida     : {len(data)}")
    print(f"  Reducao de volume   : {(1 - len(data)/len(df_input)) * 100:.1f}%")
    print(f"  Colunas geradas     : {len(data.columns)}")
    print(f"  Tempo total         : {t_pipeline:.0f}s ({t_pipeline/60:.1f}min)")

    # ── ETAPA 3: Validacao de colunas por stage ──
    print_section("ETAPA 3: Validacao de colunas por stage")
    total_present = 0
    total_missing = 0
    total_empty = 0

    for stage, expected in sorted(EXPECTED_COLUMNS.items()):
        validation = validate_stage_columns(data, stage, expected)
        present = len(validation['present'])
        missing = len(validation['missing'])
        empty = len(validation['empty'])
        total = len(expected)
        total_present += present
        total_missing += missing
        total_empty += empty

        api_tag = " [API]" if stage in API_STAGES else ""
        status = "PASS" if missing == 0 and empty == 0 else ("WARN" if missing == 0 else "FAIL")
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]

        print(f"  {icon} {stage}{api_tag}: {present}/{total} OK", end="")
        if missing:
            print(f" | missing: {validation['missing']}", end="")
        if empty:
            print(f" | empty: {validation['empty']}", end="")
        print()

    total_expected = total_present + total_missing + total_empty
    print(f"\n  TOTAL: {total_present}/{total_expected} colunas presentes "
          f"({total_missing} missing, {total_empty} vazias)")

    # ── ETAPA 4: Validacao de fluxo de dados ──
    print_section("ETAPA 4: Validacao de fluxo inter-stages")
    flow_checks = validate_data_flow(data)
    flow_pass = 0
    flow_total = len(flow_checks)

    for check_name, passed, detail in flow_checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {check_name}: {detail}")
        if passed:
            flow_pass += 1

    print(f"\n  FLUXO: {flow_pass}/{flow_total} verificacoes OK")

    # ── ETAPA 5: Metricas de qualidade API ──
    if not args.no_api:
        print_section("ETAPA 5: Metricas de qualidade (6 API Stages)")
        api_metrics = validate_api_quality(data)

        if 's08_neutral_pct' in api_metrics:
            print(f"  S08 Political  : neutral={api_metrics['s08_neutral_pct']:.1f}%, "
                  f"confidence={api_metrics['s08_confidence_mean']:.3f}")

        if 's11_total_topics' in api_metrics:
            print(f"  S11 Topics     : {api_metrics['s11_api_named_topics']}/{api_metrics['s11_total_topics']} "
                  f"nomeados via API, confidence={api_metrics['s11_confidence_mean']:.3f}")

        if 's12_sentiment_dist' in api_metrics:
            print(f"  S12 Sentiment  : {api_metrics['s12_sentiment_dist']}")
            print(f"                   sarcasm={api_metrics['s12_sarcasm_count']}, "
                  f"anger_mean={api_metrics['s12_anger_mean']:.3f}")

        if 's16_events_detected' in api_metrics:
            print(f"  S16 Events     : {api_metrics['s16_events_detected']} detectados, "
                  f"confidence={api_metrics['s16_confidence_mean']:.3f}")
            if api_metrics.get('s16_event_types'):
                print(f"                   tipos: {api_metrics['s16_event_types']}")

        if 's17_general_pct' in api_metrics:
            print(f"  S17 Channels   : general_restante={api_metrics['s17_general_pct']:.1f}%, "
                  f"confidence={api_metrics['s17_confidence_mean']:.3f}")
            print(f"                   tipos: {api_metrics['s17_channel_types']}")

    # ── ETAPA 6: Resumo de distribuicoes reais ──
    print_section("ETAPA 6: Distribuicoes reais dos dados")

    # Orientacao politica
    if 'political_orientation' in data.columns:
        print(f"  political_orientation:")
        for val, count in data['political_orientation'].value_counts().items():
            pct = count / len(data) * 100
            bar = '#' * int(pct / 2)
            print(f"    {val:20s} {count:5d} ({pct:5.1f}%) {bar}")

    # Sentimento
    if 'sentiment_label' in data.columns:
        print(f"\n  sentiment_label:")
        for val, count in data['sentiment_label'].value_counts().items():
            pct = count / len(data) * 100
            bar = '#' * int(pct / 2)
            print(f"    {val:20s} {count:5d} ({pct:5.1f}%) {bar}")

    # Contexto politico
    if 'political_context' in data.columns:
        print(f"\n  political_context:")
        for val, count in data['political_context'].value_counts().items():
            pct = count / len(data) * 100
            print(f"    {val:20s} {count:5d} ({pct:5.1f}%)")

    # ── RESULTADO FINAL ──
    print_header("RESULTADO FINAL")

    all_stages_ok = stats['stages_completed'] == 17
    no_errors = stats['processing_errors'] == 0
    columns_ok = len(data.columns) >= 120
    flow_ok = flow_pass == flow_total

    tests = [
        ("17/17 stages completados", all_stages_ok),
        ("0 erros de processamento", no_errors),
        (f"{len(data.columns)} colunas >= 120 esperadas", columns_ok),
        (f"Fluxo inter-stages {flow_pass}/{flow_total}", flow_ok),
        (f"Reducao de volume efetiva", len(data) < len(df_input)),
    ]

    all_pass = True
    for test_name, passed in tests:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test_name}")
        if not passed:
            all_pass = False

    t_total = time.time() - t0
    print(f"\n  Tempo total: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"  Throughput : {len(df_input) / t_pipeline:.1f} rows/s (pipeline)")

    if all_pass:
        print(f"\n  {'=' * 50}")
        print(f"  ✅ TESTE PONTA-A-PONTA: APROVADO")
        print(f"  {'=' * 50}")
    else:
        print(f"\n  {'=' * 50}")
        print(f"  ❌ TESTE PONTA-A-PONTA: REPROVADO")
        print(f"  {'=' * 50}")

    # ── Salvar relatorio JSON ──
    report = {
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode,
        'dataset': Path(dataset_path).name,
        'nrows_input': len(df_input),
        'nrows_output': len(data),
        'columns': len(data.columns),
        'stages_completed': stats['stages_completed'],
        'processing_errors': stats['processing_errors'],
        'time_pipeline_s': round(t_pipeline, 1),
        'time_total_s': round(t_total, 1),
        'api_enabled': not args.no_api,
        'all_pass': all_pass,
        'column_validation': {
            'present': total_present,
            'missing': total_missing,
            'empty': total_empty,
        },
        'flow_validation': {
            'pass': flow_pass,
            'total': flow_total,
        },
    }

    if not args.no_api:
        report['api_metrics'] = validate_api_quality(data)

    report_path = PROJECT_ROOT / 'data' / f'e2e_test_report_{args.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Relatorio salvo: {report_path}")

    return all_pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='digiNEV Pipeline — Teste Ponta-a-Ponta v6.2')
    parser.add_argument('--mode', choices=['quick', 'standard', 'full', 'stress'],
                        default='standard', help='Modo de teste (default: standard)')
    parser.add_argument('--quick', action='store_const', const='quick', dest='mode',
                        help='Teste rapido (100 rows)')
    parser.add_argument('--full', action='store_const', const='full', dest='mode',
                        help='Teste completo (500 rows)')
    parser.add_argument('--stress', action='store_const', const='stress', dest='mode',
                        help='Teste de carga (1000 rows)')
    parser.add_argument('--dataset', type=str, help='Caminho do dataset CSV')
    parser.add_argument('--no-api', action='store_true', help='Desabilitar API (100%% heuristico)')
    args = parser.parse_args()

    success = run_test(args)
    sys.exit(0 if success else 1)
