#!/usr/bin/env python3
"""
digiNEV Pipeline — helpers.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any


def _calculate_emoji_ratio(text: str) -> float:
    """Calcular proporção de emojis no texto."""
    if pd.isna(text) or len(text) == 0:
        return 0.0
    
    import re
    # Regex para detectar emojis Unicode
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    
    emojis = emoji_pattern.findall(str(text))
    emoji_count = sum(len(emoji) for emoji in emojis)
    
    return min(1.0, emoji_count / len(str(text)))


def _calculate_caps_ratio(text: str) -> float:
    """Calcular proporção de letras maiúsculas."""
    if pd.isna(text) or len(text) == 0:
        return 0.0
    
    text_str = str(text)
    letters = [c for c in text_str if c.isalpha()]
    
    if len(letters) == 0:
        return 0.0
    
    caps_count = sum(1 for c in letters if c.isupper())
    return caps_count / len(letters)


def _calculate_repetition_ratio(text: str) -> float:
    """Calcular proporção de caracteres repetitivos."""
    if pd.isna(text) or len(text) <= 1:
        return 0.0
    
    text_str = str(text).lower()
    
    # Contar sequências repetitivas (3+ caracteres iguais)
    repetition_count = 0
    current_char = ''
    current_count = 1
    
    for char in text_str:
        if char == current_char:
            current_count += 1
            if current_count >= 3:
                repetition_count += 1
        else:
            current_char = char
            current_count = 1
    
    return min(1.0, repetition_count / len(text_str))


def _detect_portuguese(text: str) -> bool:
    """Detecção básica de idioma português."""
    if pd.isna(text) or len(text) < 10:
        return True  # Assumir português para textos muito curtos
    
    text_lower = str(text).lower()
    
    # Palavras comuns em português
    portuguese_indicators = [
        'que', 'não', 'com', 'uma', 'para', 'são', 'por', 'mais', 'das', 'dos',
        'mas', 'foi', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem',
        'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você',
        'já', 'eu', 'também', 'só', 'pelo', 'nos', 'é', 'o', 'a', 'de', 'do',
        'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se',
        'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele',
        'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há',
        'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até'
    ]
    
    # Contar palavras portuguesas encontradas
    words = text_lower.split()
    portuguese_count = sum(1 for word in words if word in portuguese_indicators)
    
    # Considerar português se >= 20% das palavras são indicadores
    if len(words) > 0:
        portuguese_ratio = portuguese_count / len(words)
        return portuguese_ratio >= 0.2
    
    return True  # Default para português

def _classify_political_orientation(ctx, lemmas_or_text) -> str:
    """
    Classifica orientação política usando léxico unificado.
    REFORMULADO: aceita lista de lemmas (spaCy) ou string (fallback).
    Token matching via set lookup — O(1) por token, zero falsos positivos.
    """
    if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
        return 'neutral'

    # Converter input para set de tokens (lemmas ou palavras)
    if isinstance(lemmas_or_text, list):
        token_set = set(t.lower() for t in lemmas_or_text if t)
    else:
        token_set = set(str(lemmas_or_text).lower().split())

    if not token_set:
        return 'neutral'

    terms_map = ctx._political_terms_map

    # Macrotemas de direita/extrema-direita
    direita_categories = [
        'identidade_patriotica', 'inimigos_ideologicos', 'teorias_conspiracao',
        'negacionismo', 'autoritarismo_violencia', 'mobilizacao_acao',
        'desinformacao_verdade', 'estrategias_discursivas', 'eventos_simbolicos',
        'corrupcao_transparencia', 'politica_externa'
    ]

    # Contar matches por macrotema via set intersection
    scores = {}
    for cat in direita_categories:
        terms = terms_map.get(cat, [])
        # Para termos compostos (multi-word), verificar no texto concatenado
        single_word_terms = set(t for t in terms if ' ' not in t)
        multi_word_terms = [t for t in terms if ' ' in t]

        count = len(token_set & single_word_terms)
        if multi_word_terms:
            text_joined = ' '.join(sorted(token_set))
            count += sum(1 for t in multi_word_terms if t in text_joined)
        scores[cat] = count

    total_matches = sum(scores.values())
    if total_matches == 0:
        return 'neutral'

    radical_score = scores.get('autoritarismo_violencia', 0) + scores.get('mobilizacao_acao', 0)
    conspiracao_score = scores.get('teorias_conspiracao', 0) + scores.get('negacionismo', 0)
    identidade_score = scores.get('identidade_patriotica', 0) + scores.get('eventos_simbolicos', 0)
    adversario_score = scores.get('inimigos_ideologicos', 0)

    if radical_score >= 2 or (conspiracao_score >= 2 and adversario_score >= 1):
        return 'extrema-direita'
    elif adversario_score >= 2 or conspiracao_score >= 2:
        return 'direita'
    elif identidade_score >= 2:
        return 'centro-direita'
    elif total_matches >= 1:
        return 'direita'
    return 'neutral'


def _extract_political_keywords(ctx, lemmas_or_text) -> list:
    """
    Extrai palavras-chave políticas usando léxico unificado.
    REFORMULADO: token matching via set intersection.
    """
    if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
        return []

    if isinstance(lemmas_or_text, list):
        token_set = set(t.lower() for t in lemmas_or_text if t)
    else:
        token_set = set(str(lemmas_or_text).lower().split())

    if not token_set:
        return []

    terms_map = ctx._political_terms_map
    found = []
    for cat, terms in terms_map.items():
        single_word_terms = set(t for t in terms if ' ' not in t)
        matches = token_set & single_word_terms
        for m in matches:
            if m not in found:
                found.append(m)
                if len(found) >= 10:
                    return found
        # Multi-word terms: fallback substring
        multi_word_terms = [t for t in terms if ' ' in t]
        if multi_word_terms:
            text_joined = ' '.join(sorted(token_set))
            for t in multi_word_terms:
                if t in text_joined and t not in found:
                    found.append(t)
                    if len(found) >= 10:
                        return found
    return found


def _calculate_political_intensity(ctx, lemmas_or_text) -> float:
    """
    Calcula intensidade usando termos de mobilização e autoritarismo.
    REFORMULADO: token matching via set intersection.
    """
    if lemmas_or_text is None or (isinstance(lemmas_or_text, float) and pd.isna(lemmas_or_text)):
        return 0.0

    if isinstance(lemmas_or_text, list):
        token_set = set(t.lower() for t in lemmas_or_text if t)
    else:
        token_set = set(str(lemmas_or_text).lower().split())

    if not token_set:
        return 0.0

    terms_map = ctx._political_terms_map

    # Termos de alta intensidade
    intensity_terms = (
        terms_map.get('mobilizacao_acao', []) +
        terms_map.get('autoritarismo_violencia', []) +
        terms_map.get('desinformacao_verdade', [])
    )

    if not intensity_terms:
        return 0.0

    single_word = set(t for t in intensity_terms if ' ' not in t)
    match_count = len(token_set & single_word)

    # Multi-word fallback
    multi_word = [t for t in intensity_terms if ' ' in t]
    if multi_word:
        text_joined = ' '.join(sorted(token_set))
        match_count += sum(1 for t in multi_word if t in text_joined)

    return min(match_count * 0.15, 1.0)


def _classify_domain_type(domain: str) -> str:
    """Classifica tipo de domínio com categorias expandidas."""
    if not domain or pd.isna(domain):
        return 'unknown'

    domain_lower = str(domain).lower()

    # Categorias expandidas (baseado em domain_authority_analysis do archive)
    trusted_news = ['folha.uol.com.br', 'g1.globo.com', 'estadao.com.br',
                   'oglobo.globo.com', 'uol.com.br', 'bbc.com', 'reuters.com',
                   'globo.com', 'folha.com', 'r7.com', 'terra.com.br']
    government = ['.gov.br', '.leg.br', '.jus.br', '.mil.br']
    video = ['youtube.com', 'youtu.be', 'vimeo.com', 'rumble.com', 'odysee.com']
    social = ['twitter.com', 'x.com', 'facebook.com', 'instagram.com',
             't.me', 'telegram.me', 'whatsapp.com', 'tiktok.com']
    blog = ['blog', 'wordpress', 'medium.com', 'substack.com', 'blogspot']

    if any(trusted in domain_lower for trusted in trusted_news):
        return 'mainstream_news'
    elif any(gov in domain_lower for gov in government):
        return 'government'
    elif any(v in domain_lower for v in video):
        return 'video'
    elif any(s in domain_lower for s in social):
        return 'social'
    elif any(b in domain_lower for b in blog):
        return 'blog'
    else:
        return 'alternative'


def _calculate_domain_trust_score(domain: str) -> float:
    """Calcula score de confiança do domínio (Page et al. 1999, adaptado)."""
    if not domain or pd.isna(domain):
        return 0.0
    dtype = _classify_domain_type(domain)
    trust_map = {
        'government': 0.9, 'mainstream_news': 0.8, 'video': 0.5,
        'social': 0.4, 'blog': 0.3, 'alternative': 0.2, 'unknown': 0.0
    }
    return trust_map.get(dtype, 0.0)


def _calculate_sentiment_polarity(text: str) -> float:
    """Calcula polaridade com dicionário LIWC expandido (Balage Filho et al. 2013)."""
    if not text or pd.isna(text):
        return 0.0

    # Dicionário LIWC-PT expandido (baseado em sci_validated_methods_implementation.py)
    positive_words = [
        'bom', 'boa', 'bons', 'boas', 'ótimo', 'ótima', 'excelente',
        'maravilhoso', 'maravilhosa', 'perfeito', 'perfeita', 'amor',
        'feliz', 'felicidade', 'alegria', 'alegre', 'vitória', 'sucesso',
        'conquista', 'esperança', 'orgulho', 'admiração', 'respeito',
        'liberdade', 'paz', 'progresso', 'avanço', 'melhoria',
        'lindo', 'linda', 'beleza', 'incrível', 'fantástico', 'fantástica',
        'parabéns', 'obrigado', 'obrigada', 'gratidão', 'abençoado',
        'honra', 'glória', 'benção', 'fé', 'força', 'coragem'
    ]
    negative_words = [
        'ruim', 'péssimo', 'péssima', 'terrível', 'horrível', 'ódio',
        'raiva', 'triste', 'tristeza', 'infeliz', 'medo', 'fracasso',
        'derrota', 'vergonha', 'nojo', 'desgraça', 'desastre',
        'culpa', 'miséria', 'sofrimento', 'dor', 'angústia',
        'decepção', 'frustração', 'absurdo', 'ridículo', 'ridícula',
        'lamentável', 'deplorável', 'covarde', 'covardia', 'mentira',
        'mentiroso', 'mentirosa', 'traição', 'traidor', 'traidora',
        'destruição', 'morte', 'desespero', 'pânico', 'terror',
        'criminoso', 'criminosa', 'bandido', 'bandida', 'corrupto', 'corrupção'
    ]

    text_lower = str(text).lower()
    words = text_lower.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0

    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    return (positive_count - negative_count) / total_words


def _calculate_emotion_intensity(text: str, raw_text: str = None) -> float:
    """Calcula intensidade emocional usando texto original (com pontuação)."""
    # Usar raw_text (body original) se disponível, pois normalized_text remove pontuação
    source = raw_text if raw_text else text
    if not source or pd.isna(source):
        return 0.0

    source_str = str(source)
    emotion_markers = source_str.count('!') + source_str.count('?') + source_str.count('...')
    caps_words = sum(1 for word in source_str.split() if word.isupper() and len(word) > 2)

    return min((emotion_markers + caps_words) / 10.0, 1.0)


def _detect_aggressive_language(ctx, text: str) -> bool:
    """Detecta linguagem agressiva usando léxico (autoritarismo_violencia + inimigos)."""
    if not text or pd.isna(text):
        return False

    text_lower = str(text).lower()
    terms_map = ctx._political_terms_map

    # Combinar termos de violência e agressão do léxico
    aggressive_terms = terms_map.get('autoritarismo_violencia', [])
    # Adicionar termos clássicos de agressão pessoal
    extra_aggressive = [
        'ódio', 'matar', 'destruir', 'eliminar', 'acabar',
        'burro', 'idiota', 'imbecil', 'estúpido', 'canalha',
        'vagabundo', 'lixo', 'verme', 'parasita', 'bandido',
        'safado', 'nojento', 'covarde', 'traidor', 'criminoso'
    ]

    all_aggressive = set(aggressive_terms + extra_aggressive)
    return any(term in text_lower for term in all_aggressive)


def _detect_political_context(text: str) -> str:
    """Detecta contexto político."""
    if not text or pd.isna(text):
        return 'none'

    text_lower = str(text).lower()

    if any(word in text_lower for word in ['eleição', 'voto', 'urna', 'candidato', 'campanha', 'debate']):
        return 'electoral'
    elif any(word in text_lower for word in ['governo', 'ministro', 'presidente', 'planalto', 'congresso']):
        return 'government'
    elif any(word in text_lower for word in ['manifestação', 'protesto', 'greve', 'ato', 'marcha']):
        return 'protest'
    elif any(word in text_lower for word in ['economia', 'inflação', 'desemprego', 'pib', 'dólar']):
        return 'economic'
    elif any(word in text_lower for word in ['pandemia', 'covid', 'vacina', 'lockdown', 'quarentena']):
        return 'pandemic'
    else:
        return 'general'


def _mentions_government(text: str) -> bool:
    """Verifica se menciona governo."""
    if not text or pd.isna(text):
        return False

    government_terms = [
        'governo', 'presidente', 'ministro', 'secretário', 'federal',
        'planalto', 'congresso', 'senado', 'câmara', 'deputado',
        'senador', 'governador', 'prefeito', 'bolsonaro', 'lula'
    ]
    text_lower = str(text).lower()
    return any(term in text_lower for term in government_terms)


def _mentions_opposition(ctx, text: str) -> bool:
    """Verifica se menciona oposição."""
    if not text or pd.isna(text):
        return False

    terms_map = ctx._political_terms_map
    opposition_terms = terms_map.get('inimigos_ideologicos', [])
    extra = ['oposição', 'contra', 'resistência', 'impeachment', 'fora']
    all_terms = set(opposition_terms + extra)

    text_lower = str(text).lower()
    return any(term in text_lower for term in all_terms)


def _detect_election_context(text: str) -> bool:
    """Detecta contexto eleitoral."""
    if not text or pd.isna(text):
        return False

    election_terms = [
        'eleição', 'eleições', 'voto', 'votos', 'urna', 'urnas',
        'candidato', 'candidata', 'campanha', 'debate', 'apuração',
        'segundo turno', 'primeiro turno', 'tse', 'propaganda eleitoral'
    ]
    text_lower = str(text).lower()
    return any(term in text_lower for term in election_terms)


def _detect_protest_context(ctx, text: str) -> bool:
    """Detecta contexto de protesto."""
    if not text or pd.isna(text):
        return False

    terms_map = ctx._political_terms_map
    mobilizacao = terms_map.get('mobilizacao_acao', [])
    extra = ['manifestação', 'protesto', 'greve', 'ocupação', 'ato', 'marcha']
    all_terms = set(mobilizacao + extra)

    text_lower = str(text).lower()
    return any(term in text_lower for term in all_terms)


def _classify_channel_type(channel: str) -> str:
    """Classifica tipo de canal."""
    if not channel or pd.isna(channel):
        return 'unknown'

    channel_lower = str(channel).lower()

    if any(word in channel_lower for word in ['news', 'notícia', 'jornal', 'imprensa']):
        return 'news'
    elif any(word in channel_lower for word in ['brasil', 'patriota', 'conservador', 'direita', 'bolso']):
        return 'political'
    elif any(word in channel_lower for word in ['humor', 'meme', 'engraçado', 'zueira']):
        return 'entertainment'
    elif any(word in channel_lower for word in ['gospel', 'igreja', 'cristo', 'deus']):
        return 'religious'
    else:
        return 'general'

# ==========================================
# FRAME ANALYSIS - Entman (1993)
# ==========================================
def _analyze_political_frames(text: str) -> dict:
    """Identifica frames políticos (Entman 1993, J Communication 43(4): 51-58)."""
    if not text or pd.isna(text):
        return {'conflito': 0.0, 'responsabilizacao': 0.0, 'moralista': 0.0, 'economico': 0.0}

    frames = {
        'conflito': ['contra', 'ataque', 'briga', 'guerra', 'batalha', 'confronto',
                    'disputa', 'embate', 'oposição', 'adversário', 'inimigo'],
        'responsabilizacao': ['culpa', 'responsável', 'causou', 'provocou', 'deve',
                             'culpado', 'responsabilidade', 'negligência', 'omissão'],
        'moralista': ['certo', 'errado', 'justo', 'moral', 'ética', 'valores',
                     'pecado', 'virtude', 'honra', 'vergonha', 'dever', 'dignidade'],
        'economico': ['economia', 'dinheiro', 'custo', 'gasto', 'investimento', 'pib',
                     'inflação', 'desemprego', 'salário', 'imposto', 'dívida', 'mercado']
    }

    text_lower = str(text).lower()
    result = {}
    for frame, keywords in frames.items():
        score = sum(1 for word in keywords if word in text_lower)
        result[frame] = score / len(keywords)
    return result

# ==========================================
# MANN-KENDALL TREND TEST - Mann (1945); Kendall (1975)
# ==========================================
def _mann_kendall_trend_test(time_series) -> dict:
    """Teste não-paramétrico para tendência (Mann 1945, Kendall 1975)."""
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return {'statistic': 0, 'p_value': 1.0, 'trend': 'unavailable'}

    n = len(time_series)
    if n < 4:
        return {'statistic': 0, 'p_value': 1.0, 'trend': 'insufficient_data'}

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(time_series[j] - time_series[i])

    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    if p_value < 0.05:
        trend = 'increasing' if z > 0 else 'decreasing'
    else:
        trend = 'no_trend'

    return {'statistic': float(s), 'p_value': float(p_value), 'trend': trend}

# ==========================================
# INFORMATION CASCADE DETECTION - Leskovec et al. (2007)
# ==========================================
def _detect_information_cascades(df) -> pd.DataFrame:
    """Detecta cascatas de informação (Leskovec et al. 2007, ACM Trans Web)."""
    cascades = []
    if 'is_fwrd' not in df.columns:
        return pd.DataFrame(cascades)

    forwarded = df[df['is_fwrd'] == True].copy()
    if len(forwarded) < 3:
        return pd.DataFrame(cascades)

    forwarded['cascade_id'] = forwarded.groupby('body').ngroup()
    for cascade_id in forwarded['cascade_id'].unique():
        cascade = forwarded[forwarded['cascade_id'] == cascade_id]
        if len(cascade) > 2:
            cascades.append({
                'cascade_id': cascade_id,
                'size': len(cascade),
                'channels': cascade['channel'].nunique() if 'channel' in cascade.columns else 0,
                'content_preview': str(cascade['body'].iloc[0])[:100]
            })

    return pd.DataFrame(cascades)


def main():
    """Teste do módulo helpers."""
    logging.basicConfig(level=logging.INFO)
    print("helpers.py module loaded successfully")
    print(f"Functions available: {[name for name in dir() if name.startswith('_') and not name.startswith('__')]}")


if __name__ == "__main__":
    main()
