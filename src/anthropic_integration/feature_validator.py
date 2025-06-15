"""
Validador e Enriquecedor de Features Básicas
Este módulo valida e enriquece features já existentes nos datasets sem duplicação.
Focado em processamento local e eficiente.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

# Import emoji with fallback
try:
    import emoji
    EMOJI_AVAILABLE = True
    logger.info(f"Módulo emoji carregado com sucesso - v{emoji.__version__}")
except ImportError:
    EMOJI_AVAILABLE = False
    logger.warning("⚠️  Módulo emoji não disponível. Contagem de emojis usará regex como fallback.")

class FeatureValidator:
    """
    Validador de features existentes e enriquecedor mínimo

    Responsabilidades:
    - Detectar features já existentes para evitar reprocessamento
    - Validar features já extraídas (hashtags, urls, domains)
    - Revisar e consolidar media_type
    - Adicionar APENAS 4 features essenciais: text_length, word_count, is_very_short, is_very_long
    - NÃO duplicar extrações já existentes
    - NÃO adicionar padrões estruturais (datasets já têm tudo necessário)
    
    TDD Phase 3 Enhancement: Added standard validation interface
    """

    def __init__(self, config: dict = None):
        # Store configuration for TDD interface
        self.config = config or {}
        
        # Default required columns for TDD interface
        self.required_columns = self.config.get('required_columns', [
            'id', 'body', 'date', 'channel'
        ])
        
        # Expected data types for TDD interface
        self.expected_types = self.config.get('expected_types', {
            'id': ['int64', 'float64', 'object'],
            'body': ['object', 'string'],
            'date': ['datetime64[ns]', 'object'],
            'channel': ['object', 'string']
        })
        
        # Padrões para detecção de mídia
        self.media_patterns = {
            'photo': r'\b(foto|imagem|jpeg|jpg|png|gif|picture|pic|img)\b',
            'video': r'\b(vídeo|video|mp4|avi|mov|filme|filmagem|gravação)\b',
            'audio': r'\b(áudio|audio|mp3|wav|voz|podcast|gravação de voz)\b',
            'document': r'\b(documento|pdf|doc|docx|arquivo|file)\b',
            'sticker': r'\b(sticker|figurinha|adesivo)\b',
            'poll': r'\b(enquete|votação|poll|sondagem)\b'
        }

        # Padrões para detecção de forwarding
        self.forward_patterns = [
            r'encaminhada de',
            r'forwarded from',
            r'compartilhado de',
            r'repassando',
            r'vejam só',
            r'olhem isso',
            r'recebi agora'
        ]

    def validate_and_enrich_features(
        self,
        df: pd.DataFrame,
        text_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Valida features existentes e adiciona apenas enriquecimentos necessários

        Args:
            df: DataFrame com dados
            text_columns: Colunas de texto a processar (detecta automaticamente se None)

        Returns:
            Tuple com DataFrame enriquecido e relatório de validação
        """
        logger.info(f"Iniciando validação e enriquecimento de features para {len(df)} registros")

        # Detectar colunas de texto se não especificadas
        if text_columns is None:
            text_columns = self._detect_text_columns(df)
            logger.info(f"Colunas de texto detectadas: {text_columns}")

        # Fazer cópia para não modificar original
        enriched_df = df.copy()

        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "text_columns": text_columns,
            "existing_features_detected": [],
            "validations_performed": [],
            "enrichments_added": [],
            "skipped_features": [],
            "issues_found": []
        }

        # Detectar features já existentes
        existing_features = self._detect_existing_features(enriched_df)
        validation_report["existing_features_detected"] = existing_features
        logger.info(f"Features já existentes detectadas: {existing_features}")

        # 1. Validar media_type (já existe, apenas consolidar)
        if 'media_type' in existing_features:
            enriched_df, media_report = self._validate_existing_media_type(enriched_df)
            validation_report["media_validation"] = media_report
            validation_report["skipped_features"].append("media_type (já existe)")
        else:
            enriched_df, media_report = self._validate_media_type(enriched_df, text_columns)
            validation_report["media_validation"] = media_report

        # 2. Validar hashtags (já existem, apenas limpar)
        if 'hashtag' in existing_features or 'hashtags' in existing_features:
            enriched_df, hashtag_report = self._validate_hashtags(enriched_df)
            validation_report["hashtag_validation"] = hashtag_report
            validation_report["skipped_features"].append("hashtag (já existe)")

        # 3. Validar URLs e domínios (já existem, apenas verificar)
        if 'url' in existing_features and 'domain' in existing_features:
            enriched_df, url_report = self._validate_existing_urls_domains(enriched_df)
            validation_report["url_validation"] = url_report
            validation_report["skipped_features"].append("url/domain (já existem)")
        elif 'url' in existing_features:
            enriched_df, url_report = self._validate_urls_domains(enriched_df)
            validation_report["url_validation"] = url_report

        # 4. Adicionar métricas básicas de texto (apenas se não existirem)
        missing_metrics = self._check_missing_text_metrics(enriched_df)
        if missing_metrics:
            enriched_df, metrics_report = self._add_text_metrics(enriched_df, text_columns, missing_metrics)
            validation_report["text_metrics"] = metrics_report
        else:
            logger.info("Métricas de texto já existem, pulando extração")
            validation_report["skipped_features"].append("text_metrics (já existem)")
            validation_report["text_metrics"] = {"primary_text_column": text_columns[0] if text_columns else None, "metrics_added": []}

        # 5. Detectar padrões estruturais (apenas se não existirem)
        missing_patterns = self._check_missing_structural_patterns(enriched_df)
        if missing_patterns:
            enriched_df, patterns_report = self._detect_structural_patterns(enriched_df, text_columns, missing_patterns)
            validation_report["structural_patterns"] = patterns_report
        else:
            logger.info("Padrões estruturais já existem, pulando detecção")
            validation_report["skipped_features"].append("structural_patterns (já existem)")
            validation_report["structural_patterns"] = {"patterns_detected": [], "forwarded_messages": 0, "messages_with_mentions": 0}

        # 6. Adicionar flags de qualidade (apenas se não existirem)
        missing_quality = self._check_missing_quality_flags(enriched_df)
        if missing_quality:
            enriched_df, quality_report = self._add_quality_flags(enriched_df, text_columns, missing_quality)
            validation_report["quality_flags"] = quality_report
        else:
            logger.info("Flags de qualidade já existem, pulando criação")
            validation_report["skipped_features"].append("quality_flags (já existem)")
            validation_report["quality_flags"] = {"quality_flags_added": [], "low_quality_messages": 0}

        logger.info(f"Validação concluída. Features puladas: {len(validation_report['skipped_features'])}")
        return enriched_df, validation_report

    def _detect_existing_features(self, df: pd.DataFrame) -> List[str]:
        """
        Detecta features já existentes no dataset para evitar reprocessamento
        """
        existing_features = []

        # Mapear features conhecidas
        feature_mapping = {
            'media_type': ['media_type', 'tipo_midia'],
            'hashtag': ['hashtag', 'hashtags'],
            'url': ['url', 'urls'],
            'domain': ['domain', 'domains', 'dominio'],
            'mentions': ['mentions', 'mencoes'],
            'is_fwrd': ['is_fwrd', 'is_forwarded', 'encaminhada'],
            'body_cleaned': ['body_cleaned', 'texto_limpo'],
            'text_length': ['text_length', 'comprimento_texto'],
            'word_count': ['word_count', 'contagem_palavras'],
            'emoji_count': ['emoji_count', 'contagem_emojis'],
            'caps_ratio': ['caps_ratio', 'proporcao_maiuscula'],
            'excessive_punctuation': ['excessive_punctuation', 'pontuacao_excessiva']
        }

        for feature, possible_columns in feature_mapping.items():
            for col in possible_columns:
                if col in df.columns:
                    existing_features.append(feature)
                    break

        return existing_features

    def _detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Detecta colunas de texto no DataFrame"""
        text_columns = []

        # Priorizar colunas conhecidas
        priority_columns = ['body_cleaned', 'body', 'texto', 'text', 'message']

        for col in priority_columns:
            if col in df.columns:
                text_columns.append(col)

        # Adicionar outras colunas de texto se existirem
        for col in df.columns:
            if col not in text_columns and df[col].dtype == 'object':
                # Verificar se tem conteúdo de texto significativo
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    avg_length = sample.astype(str).str.len().mean()
                    if avg_length > 20:  # Provável texto
                        text_columns.append(col)

        return text_columns

    def _validate_existing_media_type(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Valida media_type existente sem reprocessar
        """
        report = {
            "original_media_types": df['media_type'].value_counts().to_dict(),
            "validation_method": "existing_column",
            "issues_found": []
        }

        # Verificar valores inválidos
        valid_types = ['text', 'photo', 'video', 'audio', 'document', 'sticker', 'poll', 'url']
        invalid_mask = ~df['media_type'].isin(valid_types)
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            report["issues_found"].append(f"{invalid_count} registros com media_type inválido")
            # Corrigir valores inválidos para 'text'
            df.loc[invalid_mask, 'media_type'] = 'text'
            report["corrections_made"] = invalid_count

        report["final_media_types"] = df['media_type'].value_counts().to_dict()
        return df, report

    def _validate_media_type(self, df: pd.DataFrame, text_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Valida e consolida media_type baseado no conteúdo"""
        report = {
            "original_media_types": {},
            "updated_count": 0,
            "detection_method": "content_analysis"
        }

        # Contar tipos originais se existir a coluna
        if 'media_type' in df.columns:
            report["original_media_types"] = df['media_type'].value_counts().to_dict()
        else:
            df['media_type'] = 'text'  # Default
            report["created_column"] = True

        # Analisar conteúdo para detectar tipo de mídia (vetorizado)
        # Combinar todas as colunas de texto de forma vetorizada
        combined_text_series = pd.Series('', index=df.index)
        for col in text_columns:
            if col in df.columns:
                combined_text_series += ' ' + df[col].astype(str).str.lower().fillna('')

        # Detectar tipo de mídia para todas as linhas usando operações vetorizadas
        detected_types = pd.Series('text', index=df.index)  # Default
        
        for media_type, pattern in self.media_patterns.items():
            # Usar str.contains para operação vetorizada
            matches = combined_text_series.str.contains(pattern, case=False, na=False, regex=True)
            detected_types.loc[matches] = media_type

        # Atualizar coluna de forma vetorizada
        needs_update = (
            df['media_type'].isna() | 
            (df['media_type'].notna() & (df['media_type'] != detected_types))
        )
        
        df.loc[needs_update, 'media_type'] = detected_types.loc[needs_update]
        report["updated_count"] = needs_update.sum()

        report["final_media_types"] = df['media_type'].value_counts().to_dict()
        return df, report

    def _validate_hashtags(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Valida hashtags existentes"""
        report = {
            "hashtag_column": None,
            "total_hashtags": 0,
            "empty_hashtags": 0,
            "malformed_hashtags": 0
        }

        # Identificar coluna de hashtags
        hashtag_col = None
        if 'hashtag' in df.columns:
            hashtag_col = 'hashtag'
        elif 'hashtags' in df.columns:
            hashtag_col = 'hashtags'

        if hashtag_col:
            report["hashtag_column"] = hashtag_col

            # Contar e validar
            non_empty = df[hashtag_col].notna()
            report["total_hashtags"] = non_empty.sum()
            report["empty_hashtags"] = (~non_empty).sum()

            # Verificar hashtags malformadas (sem #) - vetorizado
            if report["total_hashtags"] > 0:
                hashtag_sample = df[non_empty][hashtag_col].astype(str).head(1000)
                # Operação vetorizada para detectar hashtags malformadas
                is_non_empty = hashtag_sample.str.len() > 0
                starts_with_hash = hashtag_sample.str.strip().str.startswith('#')
                malformed_mask = is_non_empty & ~starts_with_hash
                malformed = malformed_mask.sum()
                report["malformed_hashtags"] = malformed

                # Corrigir hashtags malformadas de forma vetorizada
                if malformed > 0:
                    hashtag_series = df[hashtag_col].astype(str)
                    needs_correction = (
                        hashtag_series.notna() & 
                        (hashtag_series.str.strip() != '') & 
                        ~hashtag_series.str.strip().str.startswith('#')
                    )
                    df.loc[needs_correction, hashtag_col] = '#' + hashtag_series.loc[needs_correction].str.strip()

        return df, report

    def _validate_existing_urls_domains(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Valida URLs e domínios já existentes
        """
        report = {
            "url_column": "url",
            "domain_column": "domain",
            "urls_validated": df['url'].notna().sum(),
            "domains_validated": df['domain'].notna().sum(),
            "validation_method": "existing_columns"
        }

        return df, report

    def _validate_urls_domains(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Valida URLs e domínios existentes"""
        report = {
            "url_column": None,
            "domain_column": None,
            "urls_validated": 0,
            "domains_extracted": 0
        }

        # Identificar colunas
        url_col = 'url' if 'url' in df.columns else ('urls' if 'urls' in df.columns else None)
        domain_col = 'domain' if 'domain' in df.columns else ('domains' if 'domains' in df.columns else None)

        if url_col:
            report["url_column"] = url_col
            report["urls_validated"] = df[url_col].notna().sum()

            # Se não existir coluna de domínios, criar
            if not domain_col:
                df['domains'] = df[url_col].apply(self._extract_domain_from_url)
                report["domains_extracted"] = df['domains'].notna().sum()
                report["domain_column"] = 'domains'

        return df, report

    def _extract_domain_from_url(self, url_string: str) -> str:
        """Extrai domínio de uma URL"""
        if pd.isna(url_string) or not url_string:
            return None

        try:
            # Se for lista de URLs, pegar primeiro
            if ',' in str(url_string):
                url_string = str(url_string).split(',')[0].strip()

            # Adicionar protocolo se não tiver
            if not url_string.startswith(('http://', 'https://')):
                url_string = 'https://' + url_string

            parsed = urlparse(url_string)
            domain = parsed.netloc

            # Remover www. se presente
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain if domain else None

        except Exception:
            return None

    def _check_missing_text_metrics(self, df: pd.DataFrame) -> List[str]:
        """
        Verifica quais métricas de texto estão faltando - REDUZIDO para apenas 2 métricas essenciais
        """
        expected_metrics = ['text_length', 'word_count']
        missing_metrics = []

        for metric in expected_metrics:
            if metric not in df.columns:
                missing_metrics.append(metric)

        return missing_metrics

    def _add_text_metrics(self, df: pd.DataFrame, text_columns: List[str], missing_metrics: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Adiciona apenas métricas de texto essenciais - REDUZIDO para 2 métricas"""
        if missing_metrics is None:
            missing_metrics = ['text_length', 'word_count']

        report = {
            "metrics_added": [],
            "primary_text_column": None
        }

        # Usar primeira coluna de texto disponível (preferência: body_cleaned > body)
        primary_col = None
        for col in ['body_cleaned', 'body'] + text_columns:
            if col in df.columns:
                primary_col = col
                break

        if primary_col:
            report["primary_text_column"] = primary_col

            # Adicionar apenas métricas essenciais
            if 'text_length' in missing_metrics:
                df['text_length'] = df[primary_col].fillna('').astype(str).str.len()
                report["metrics_added"].append('text_length')

            if 'word_count' in missing_metrics:
                df['word_count'] = df[primary_col].fillna('').astype(str).str.split().str.len()
                report["metrics_added"].append('word_count')

        return df, report

    def _check_missing_structural_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        Verifica quais padrões estruturais estão faltando - DESABILITADO (não adiciona nenhum)
        """
        # DESABILITADO: Não adicionar padrões estruturais, datasets já têm tudo necessário
        return []

    def _detect_structural_patterns(self, df: pd.DataFrame, text_columns: List[str], missing_patterns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Detecta apenas padrões estruturais faltantes"""
        if missing_patterns is None:
            missing_patterns = ['is_forwarded', 'mention_count', 'has_mentions', 'has_telegram_link']

        report = {
            "patterns_detected": [],
            "forwarded_messages": 0,
            "messages_with_mentions": 0
        }

        # Usar primeira coluna de texto disponível
        primary_col = None
        for col in ['body', 'body_cleaned'] + text_columns:
            if col in df.columns:
                primary_col = col
                break

        if primary_col:
            # Detectar mensagens encaminhadas (apenas se não existir is_fwrd)
            if 'is_forwarded' in missing_patterns and 'is_fwrd' not in df.columns:
                forward_pattern = '|'.join(self.forward_patterns)
                df['is_forwarded'] = df[primary_col].fillna('').astype(str).str.contains(
                    forward_pattern, case=False, regex=True
                )
                report["forwarded_messages"] = df['is_forwarded'].sum()
                report["patterns_detected"].append('is_forwarded')
            elif 'is_fwrd' in df.columns:
                report["forwarded_messages"] = df['is_fwrd'].sum()

            # Detectar menções (apenas se não existir coluna mentions)
            if 'mention_count' in missing_patterns and 'mentions' not in df.columns:
                df['mention_count'] = df[primary_col].fillna('').astype(str).apply(
                    lambda x: len(re.findall(r'@\w+', x))
                )
                report["patterns_detected"].append('mention_count')

            if 'has_mentions' in missing_patterns:
                if 'mentions' in df.columns:
                    df['has_mentions'] = df['mentions'].fillna('').astype(str) != ''
                elif 'mention_count' in df.columns:
                    df['has_mentions'] = df['mention_count'] > 0
                else:
                    df['has_mentions'] = df[primary_col].fillna('').astype(str).str.contains(r'@\w+', regex=True)
                report["messages_with_mentions"] = df['has_mentions'].sum()
                report["patterns_detected"].append('has_mentions')

            # Detectar links Telegram
            if 'has_telegram_link' in missing_patterns:
                df['has_telegram_link'] = df[primary_col].fillna('').astype(str).str.contains(
                    r't\.me/\w+|telegram\.me/\w+', case=False, regex=True
                )
                report["patterns_detected"].append('has_telegram_link')

        return df, report

    def _check_missing_quality_flags(self, df: pd.DataFrame) -> List[str]:
        """
        Verifica quais flags de qualidade estão faltando - REDUZIDO para apenas 2 flags essenciais
        """
        expected_flags = ['is_very_short', 'is_very_long']
        missing_flags = []

        for flag in expected_flags:
            if flag not in df.columns:
                missing_flags.append(flag)

        return missing_flags

    def _add_quality_flags(self, df: pd.DataFrame, text_columns: List[str], missing_flags: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Adiciona apenas flags de qualidade essenciais - REDUZIDO para 2 flags"""
        if missing_flags is None:
            missing_flags = ['is_very_short', 'is_very_long']

        report = {
            "quality_flags_added": [],
            "low_quality_messages": 0
        }

        # Adicionar apenas flags essenciais baseadas em word_count
        if 'is_very_short' in missing_flags:
            # Usar word_count se existir, senão calcular dinamicamente
            if 'word_count' in df.columns:
                df['is_very_short'] = df['word_count'] < 3
            else:
                primary_col = text_columns[0] if text_columns else 'body_cleaned'
                if primary_col not in df.columns:
                    primary_col = 'body' if 'body' in df.columns else text_columns[0]
                df['is_very_short'] = df[primary_col].fillna('').astype(str).str.split().str.len() < 3
            report["quality_flags_added"].append('is_very_short')

        if 'is_very_long' in missing_flags:
            # Usar word_count se existir, senão calcular dinamicamente
            if 'word_count' in df.columns:
                df['is_very_long'] = df['word_count'] > 500
            else:
                primary_col = text_columns[0] if text_columns else 'body_cleaned'
                if primary_col not in df.columns:
                    primary_col = 'body' if 'body' in df.columns else text_columns[0]
                df['is_very_long'] = df[primary_col].fillna('').astype(str).str.split().str.len() > 500
            report["quality_flags_added"].append('is_very_long')

        # Calcular estatística final
        if 'is_very_short' in df.columns:
            report["low_quality_messages"] = df['is_very_short'].sum()

        return df, report
    
    # TDD Phase 3 Methods - Standard validation interface
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        TDD interface: Validate complete dataset structure and quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        result = {
            'is_valid': True,
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'issues': []
        }
        
        try:
            # Check required columns
            missing_columns = self._check_required_columns_tdd(df)
            if missing_columns:
                result['is_valid'] = False
                result['missing_columns'] = missing_columns
                result['issues'].append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            type_issues = self._check_data_types_tdd(df)
            if type_issues:
                result['data_type_issues'] = type_issues
                result['issues'].append("Data type validation issues found")
            
            # Check data quality
            quality_issues = self._check_data_quality_tdd(df)
            if quality_issues:
                result['quality_issues'] = quality_issues
                result['issues'].append("Data quality issues found")
            
            # Overall validation status
            if result['issues']:
                result['is_valid'] = False
            
            logger.info(f"📊 TDD Dataset validation complete: {'✅ Valid' if result['is_valid'] else '❌ Issues found'}")
            
        except Exception as e:
            logger.error(f"TDD Validation error: {e}")
            result['is_valid'] = False
            result['error'] = str(e)
            result['issues'].append(f"Validation error: {e}")
        
        return result
    
    def validate_schema_consistency(self, datasets: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        TDD interface: Validate schema consistency across multiple datasets.
        
        Args:
            datasets: List of DataFrames to compare
            
        Returns:
            Dict with consistency validation results
        """
        if len(datasets) < 2:
            return {
                'schema_consistent': True,
                'message': 'Only one dataset provided, nothing to compare'
            }
        
        result = {
            'schema_consistent': True,
            'schemas': [],
            'differences': []
        }
        
        try:
            # Extract schema information from each dataset
            schemas = []
            for i, df in enumerate(datasets):
                schema = {
                    'dataset_index': i,
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'shape': df.shape
                }
                schemas.append(schema)
            
            result['schemas'] = schemas
            
            # Compare schemas
            base_schema = schemas[0]
            for i, schema in enumerate(schemas[1:], 1):
                # Check column differences
                base_cols = set(base_schema['columns'])
                current_cols = set(schema['columns'])
                
                if base_cols != current_cols:
                    result['schema_consistent'] = False
                    missing_in_current = base_cols - current_cols
                    extra_in_current = current_cols - base_cols
                    
                    difference = {
                        'datasets_compared': f"0 vs {i}",
                        'missing_columns': list(missing_in_current),
                        'extra_columns': list(extra_in_current)
                    }
                    result['differences'].append(difference)
                
                # Check data type differences for common columns
                common_cols = base_cols & current_cols
                for col in common_cols:
                    if base_schema['dtypes'][col] != schema['dtypes'][col]:
                        result['schema_consistent'] = False
                        difference = {
                            'datasets_compared': f"0 vs {i}",
                            'column': col,
                            'base_type': base_schema['dtypes'][col],
                            'current_type': schema['dtypes'][col]
                        }
                        result['differences'].append(difference)
            
            logger.info(f"🔍 TDD Schema consistency check: {'✅ Consistent' if result['schema_consistent'] else '❌ Inconsistent'}")
            
        except Exception as e:
            logger.error(f"TDD Schema consistency check error: {e}")
            result['schema_consistent'] = False
            result['error'] = str(e)
        
        return result
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """TDD interface alias for validate_dataset."""
        return self.validate_dataset(df)
    
    def _check_required_columns_tdd(self, df: pd.DataFrame) -> List[str]:
        """TDD interface: Check if all required columns are present."""
        missing = []
        for col in self.required_columns:
            if col not in df.columns:
                missing.append(col)
        return missing
    
    def _check_data_types_tdd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """TDD interface: Check data types against expected types."""
        issues = {}
        
        for col in df.columns:
            if col in self.expected_types:
                current_type = str(df[col].dtype)
                expected_types = self.expected_types[col]
                
                # Special handling for date columns - check if they can be parsed as dates
                if col == 'date' and current_type == 'object':
                    # Try to parse a sample to see if it's actually date-like
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        try:
                            # Try to convert to datetime - if it fails, it's not a valid date
                            pd.to_datetime(sample.iloc[0])
                        except (ValueError, TypeError):
                            issues[col] = {
                                'current_type': current_type,
                                'expected_types': expected_types,
                                'validation_error': 'Cannot parse as datetime'
                            }
                elif current_type not in expected_types:
                    issues[col] = {
                        'current_type': current_type,
                        'expected_types': expected_types
                    }
        
        return issues
    
    def _check_data_quality_tdd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """TDD interface: Check data quality issues."""
        issues = {}
        
        # Check for missing values in critical columns
        if 'id' in df.columns:
            missing_ids = df['id'].isna().sum()
            if missing_ids > 0:
                issues['missing_ids'] = missing_ids
        
        # Check for empty body text
        if 'body' in df.columns:
            empty_bodies = (df['body'].isna() | (df['body'].str.strip() == '')).sum()
            if empty_bodies > 0:
                issues['empty_bodies'] = empty_bodies
        
        # Check for null dates
        if 'date' in df.columns:
            null_dates = df['date'].isna().sum()
            if null_dates > 0:
                issues['null_dates'] = null_dates
        
        # Check for missing channels
        if 'channel' in df.columns:
            missing_channels = df['channel'].isna().sum()
            if missing_channels > 0:
                issues['missing_channels'] = missing_channels
        
        # Calculate overall missing value statistics
        total_missing = df.isna().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        if missing_percentage > 5:  # More than 5% missing
            issues['overall_missing_percentage'] = missing_percentage
        
        return issues
