#!/usr/bin/env python3
"""
digiNEV Pipeline — stage_10.py
Auto-extracted from analyzer.py (TAREFA 11 modularização)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Any
# sklearn imported lazily inside function


def _stage_10_clustering_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 10: Análise de clustering baseado em features linguísticas.

    Agrupa documentos similares usando características extraídas.
    """
    try:
        ctx.logger.info("🔄 Stage 10: Análise de clustering")
        
        # Features numéricas para clustering
        numeric_features = []
        # FIX: 'text_length' não existe — Stage 04 gera 'char_count'
        for col in ['word_count', 'char_count', 'tfidf_score_mean', 'political_intensity']:
            if col in df.columns:
                numeric_features.append(col)
        
        if len(numeric_features) < 2:
            ctx.logger.warning("⚠️ Features insuficientes para clustering")
            df['cluster_id'] = 0
            df['cluster_distance'] = 0.0
            df['cluster_size'] = len(df)
        else:
            from sklearn.preprocessing import StandardScaler

            # Preparar dados
            feature_data = df[numeric_features].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)

            # Tentar HDBSCAN (instalado no pyproject.toml, auto-detecção de k)
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(5, len(df) // 50),
                    min_samples=3,
                    metric='euclidean'
                )
                clusters = clusterer.fit_predict(scaled_data)
                # HDBSCAN retorna -1 para noise
                df['cluster_id'] = clusters
                df['cluster_distance'] = 1.0 - clusterer.probabilities_  # prob → distância
                n_found = len(set(clusters) - {-1})
                n_noise = (clusters == -1).sum()
                ctx.logger.info(f"📊 HDBSCAN: {n_found} clusters, {n_noise} noise points")

            except (ImportError, Exception) as e:
                # Fallback para K-Means
                ctx.logger.warning(f"⚠️ HDBSCAN indisponível ({e}), usando KMeans fallback")
                from sklearn.cluster import KMeans
                n_clusters = min(5, len(df) // 10 + 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_data)
                df['cluster_id'] = clusters
                df['cluster_distance'] = [
                    min(((scaled_data[i] - center) ** 2).sum() for center in kmeans.cluster_centers_)
                    for i in range(len(scaled_data))
                ]

            # Tamanho dos clusters
            cluster_sizes = pd.Series(df['cluster_id']).value_counts().to_dict()
            df['cluster_size'] = df['cluster_id'].map(cluster_sizes)
        
        ctx.stats['stages_completed'] += 1
        ctx.stats['features_extracted'] += 3
        
        ctx.logger.info(f"✅ Stage 10 concluído: {len(df)} registros processados")
        return df

    except Exception as e:
        ctx.logger.error(f"❌ Erro Stage 10: {e}")
        ctx.stats['processing_errors'] += 1
        return df

