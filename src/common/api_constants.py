#!/usr/bin/env python3
"""
API Constants - Constantes Centralizadas da API
===============================================

Centraliza todas as constantes de API (preços, modelos, configurações)
que estavam duplicadas em múltiplos arquivos.

Eliminação de duplicação identificada na auditoria v5.0.0 - TASK-020
Arquivos consolidados: base.py, cost_monitor.py, political_analyzer.py
"""

from typing import Dict, Any

# =============================================================================
# MODELOS ANTHROPIC DISPONÍVEIS
# =============================================================================

ANTHROPIC_MODELS = {
    # Modelos Sonnet 4 (Mais recentes)
    "claude-sonnet-4-20250514": {
        "name": "Claude Sonnet 4",
        "tier": "premium",
        "context_window": 200000,
        "capabilities": ["analysis", "reasoning", "writing", "code"],
        "recommended_for": ["complex_analysis", "topic_interpretation", "research"]
    },
    
    # Modelos Sonnet 3.5 (Balanceados)
    "claude-3-5-sonnet-20241022": {
        "name": "Claude 3.5 Sonnet",
        "tier": "standard",
        "context_window": 200000,
        "capabilities": ["analysis", "reasoning", "writing", "code"],
        "recommended_for": ["political_analysis", "sentiment_analysis", "general_processing"]
    },
    
    # Modelos Haiku (Rápidos e econômicos)
    "claude-3-5-haiku-20241022": {
        "name": "Claude 3.5 Haiku",
        "tier": "fast",
        "context_window": 200000,
        "capabilities": ["analysis", "classification", "extraction"],
        "recommended_for": ["batch_processing", "classification", "quick_analysis"]
    }
}

# =============================================================================
# PREÇOS DE TOKENS (USD por token)
# =============================================================================

TOKEN_PRICES = {
    # Sonnet 4 - Premium pricing
    "claude-sonnet-4-20250514": {
        "input": 0.000015,   # $15 per million input tokens
        "output": 0.000075   # $75 per million output tokens
    },
    
    # Sonnet 3.5 - Standard pricing
    "claude-3-5-sonnet-20241022": {
        "input": 0.000003,   # $3 per million input tokens
        "output": 0.000015   # $15 per million output tokens
    },
    
    # Haiku 3.5 - Fast and economical
    "claude-3-5-haiku-20241022": {
        "input": 0.00000025, # $0.25 per million input tokens
        "output": 0.00000125 # $1.25 per million output tokens
    }
}

# =============================================================================
# VOYAGE.AI MODELS E PREÇOS
# =============================================================================

VOYAGE_MODELS = {
    "voyage-3.5-lite": {
        "name": "Voyage 3.5 Lite",
        "dimensions": 1024,
        "max_tokens": 32000,
        "price_per_1m_tokens": 0.12,  # $0.12 per million tokens
        "recommended_for": ["general_embeddings", "semantic_search", "clustering"]
    },
    
    "voyage-large-2": {
        "name": "Voyage Large 2",
        "dimensions": 1536,
        "max_tokens": 16000,
        "price_per_1m_tokens": 0.12,  # $0.12 per million tokens
        "recommended_for": ["high_quality_embeddings", "research"]
    }
}

# =============================================================================
# CONFIGURAÇÕES DE MODELO POR OPERAÇÃO
# =============================================================================

STAGE_MODEL_MAPPING = {
    # Estágios que requerem alta qualidade
    "political_analysis": {
        "default": "claude-3-5-sonnet-20241022",
        "enhanced": "claude-sonnet-4-20250514",
        "fast": "claude-3-5-haiku-20241022"
    },
    
    "topic_interpretation": {
        "default": "claude-3-5-sonnet-20241022", 
        "enhanced": "claude-sonnet-4-20250514",
        "fast": "claude-3-5-haiku-20241022"
    },
    
    "qualitative_analysis": {
        "default": "claude-3-5-sonnet-20241022",
        "enhanced": "claude-sonnet-4-20250514", 
        "fast": "claude-3-5-haiku-20241022"
    },
    
    # Estágios que podem usar modelos mais rápidos
    "sentiment_analysis": {
        "default": "claude-3-5-haiku-20241022",
        "enhanced": "claude-3-5-sonnet-20241022",
        "fast": "claude-3-5-haiku-20241022"
    },
    
    "network_analysis": {
        "default": "claude-3-5-haiku-20241022",
        "enhanced": "claude-3-5-sonnet-20241022", 
        "fast": "claude-3-5-haiku-20241022"
    }
}

# =============================================================================
# LIMITES E CONFIGURAÇÕES DE SEGURANÇA
# =============================================================================

COST_LIMITS = {
    "daily_limit_usd": 50.0,
    "hourly_limit_usd": 10.0,
    "single_request_limit_usd": 5.0,
    "warning_threshold_percentage": 80.0,
    "auto_downgrade_threshold_percentage": 90.0
}

RATE_LIMITS = {
    "requests_per_minute": {
        "claude-sonnet-4-20250514": 40,
        "claude-3-5-sonnet-20241022": 50,
        "claude-3-5-haiku-20241022": 100
    },
    "tokens_per_minute": {
        "claude-sonnet-4-20250514": 40000,
        "claude-3-5-sonnet-20241022": 50000,
        "claude-3-5-haiku-20241022": 100000
    }
}

# =============================================================================
# CONFIGURAÇÕES DE FALLBACK
# =============================================================================

FALLBACK_STRATEGIES = {
    "model_unavailable": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ],
    
    "cost_exceeded": [
        "claude-3-5-haiku-20241022"
    ],
    
    "rate_limited": {
        "wait_time_seconds": 60,
        "max_retries": 3,
        "exponential_backoff": True
    }
}

# =============================================================================
# CONFIGURAÇÕES DE QUALIDADE POR TIPO DE ANÁLISE
# =============================================================================

QUALITY_PROFILES = {
    "research": {
        "preferred_model": "claude-sonnet-4-20250514",
        "min_confidence": 0.8,
        "enable_validation": True,
        "max_cost_per_request": 1.0
    },
    
    "production": {
        "preferred_model": "claude-3-5-sonnet-20241022", 
        "min_confidence": 0.7,
        "enable_validation": False,
        "max_cost_per_request": 0.5
    },
    
    "development": {
        "preferred_model": "claude-3-5-haiku-20241022",
        "min_confidence": 0.6,
        "enable_validation": False,
        "max_cost_per_request": 0.1
    }
}

# =============================================================================
# METADADOS E VERSIONING
# =============================================================================

API_METADATA = {
    "constants_version": "5.0.0",
    "last_updated": "2025-06-14",
    "pricing_last_verified": "2025-06-14",
    "consolidation_source": [
        "src/anthropic_integration/base.py",
        "src/anthropic_integration/cost_monitor.py", 
        "src/anthropic_integration/political_analyzer.py"
    ],
    "audit_task": "TASK-020"
}

# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def get_model_price(model_name: str, token_type: str = "input") -> float:
    """
    Retorna o preço por token para um modelo específico
    
    Args:
        model_name: Nome do modelo
        token_type: 'input' ou 'output'
        
    Returns:
        float: Preço por token em USD
    """
    if model_name not in TOKEN_PRICES:
        raise ValueError(f"Modelo não encontrado: {model_name}")
        
    if token_type not in ["input", "output"]:
        raise ValueError(f"Tipo de token inválido: {token_type}")
        
    return TOKEN_PRICES[model_name][token_type]


def calculate_request_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calcula o custo total de uma requisição
    
    Args:
        model_name: Nome do modelo
        input_tokens: Número de tokens de input
        output_tokens: Número de tokens de output
        
    Returns:
        float: Custo total em USD
    """
    input_cost = input_tokens * get_model_price(model_name, "input")
    output_cost = output_tokens * get_model_price(model_name, "output")
    return input_cost + output_cost


def get_recommended_model(operation: str, quality_profile: str = "production") -> str:
    """
    Retorna o modelo recomendado para uma operação e perfil de qualidade
    
    Args:
        operation: Nome da operação (ex: 'political_analysis')
        quality_profile: Perfil de qualidade ('research', 'production', 'development')
        
    Returns:
        str: Nome do modelo recomendado
    """
    if operation in STAGE_MODEL_MAPPING:
        if quality_profile == "research":
            return STAGE_MODEL_MAPPING[operation].get("enhanced", STAGE_MODEL_MAPPING[operation]["default"])
        elif quality_profile == "development":
            return STAGE_MODEL_MAPPING[operation].get("fast", STAGE_MODEL_MAPPING[operation]["default"])
        else:  # production
            return STAGE_MODEL_MAPPING[operation]["default"]
    
    # Fallback para operações não mapeadas
    return QUALITY_PROFILES[quality_profile]["preferred_model"]


def is_within_cost_limits(proposed_cost: float, limit_type: str = "single_request") -> bool:
    """
    Verifica se um custo proposto está dentro dos limites
    
    Args:
        proposed_cost: Custo proposto em USD
        limit_type: Tipo de limite ('single_request', 'hourly', 'daily')
        
    Returns:
        bool: True se dentro do limite
    """
    limit_key = f"{limit_type}_limit_usd"
    if limit_key not in COST_LIMITS:
        return True  # Se não há limite definido, assume que está OK
        
    return proposed_cost <= COST_LIMITS[limit_key]


if __name__ == "__main__":
    # Teste básico das funcionalidades
    print("🧪 Testando API Constants...")
    
    # Teste preços
    sonnet_price = get_model_price("claude-3-5-sonnet-20241022", "input")
    print(f"Preço Sonnet 3.5 input: ${sonnet_price} por token")
    
    # Teste cálculo de custo
    cost = calculate_request_cost("claude-3-5-sonnet-20241022", 1000, 500)
    print(f"Custo de requisição (1000 in, 500 out): ${cost:.6f}")
    
    # Teste recomendação de modelo
    model = get_recommended_model("political_analysis", "research")
    print(f"Modelo recomendado para análise política (research): {model}")
    
    # Teste limites
    within_limits = is_within_cost_limits(1.0, "single_request")
    print(f"$1.00 está dentro dos limites: {within_limits}")
    
    print("✅ API Constants funcionando corretamente!")
    print(f"📊 {len(TOKEN_PRICES)} modelos configurados")
    print(f"🎯 {len(STAGE_MODEL_MAPPING)} operações mapeadas")