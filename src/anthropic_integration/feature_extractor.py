"""
Advanced Feature Extractor via Anthropic API
Implements complete feature extraction with pattern identification and error correction.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .api_error_handler import APIErrorHandler, APIQualityChecker
from .base import AnthropicBase

logger = logging.getLogger(__name__)

class FeatureExtractor(AnthropicBase):
    """
    Advanced feature extractor using Anthropic API

    Capabilities:
    - Intelligent extraction of hashtags, URLs and domains
    - Behavior pattern detection
    - Automatic content classification
    - Identification of Brazilian context-specific characteristics
    - Error correction detected by the API itself
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.error_handler = APIErrorHandler()
        self.quality_checker = APIQualityChecker(config)

        # Specific patterns for Brazilian/Bolsonarist context
        self.brazilian_patterns = {
            "political_keywords": [
                "bolsonaro", "lula", "pt", "psl", "pl", "tse", "stf", "governo",
                "presidente", "deputado", "senador", "ministro", "eleições",
                "urna", "voto", "democracia", "ditadura", "comunismo", "socialismo"
            ],
            "conspiracy_keywords": [
                "fake news", "mídia", "globo", "manipulação", "censura", "verdade",
                "acordem", "despertem", "sistema", "elite", "illuminati", "maçonaria"
            ],
            "health_keywords": [
                "covid", "corona", "vírus", "vacina", "cloroquina", "ivermectina",
                "lockdown", "quarentena", "pandemia", "sus", "anvisa", "oms"
            ]
        }

    def extract_comprehensive_features(
        self,
        df: pd.DataFrame,
        text_column: str = "body",
        batch_size: int = 50
    ) -> pd.DataFrame:
        """
        Extract comprehensive features from dataset using API

        Args:
            df: DataFrame with data
            text_column: Name of text column
            batch_size: Batch size for processing

        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Starting feature extraction for {len(df)} records")

        # Create backup before starting
        backup_file = f"data/interim/feature_extraction_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False, sep=';', encoding='utf-8')
        logger.info(f"Backup created: {backup_file}")

        # Process in batches
        result_dfs = []
        total_batches = (len(df) + batch_size - 1) // batch_size

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            batch_num = i // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches}")

            # Use error handler for processing with retry
            result = self.error_handler.execute_with_retry(
                self._process_batch_features,
                stage="01b_feature_extraction",
                operation=f"batch_{batch_num}",
                batch_df=batch_df,
                text_column=text_column
            )

            if result.success:
                result_dfs.append(result.data)
            else:
                logger.error(f"Batch {batch_num} failed: {result.error.error_message}")
                # Add batch without extra features (preserve original data)
                result_dfs.append(batch_df)

        # Combine results
        final_df = pd.concat(result_dfs, ignore_index=True)

        # Final validation
        validation_result = self._validate_extracted_features(final_df, df)

        return final_df

    def _process_batch_features(
        self,
        batch_df: pd.DataFrame,
        text_column: str
    ) -> pd.DataFrame:
        """Process a batch of data for feature extraction"""

        # Prepare texts for analysis
        texts = batch_df[text_column].fillna("").astype(str).tolist()

        # Extract basic features first
        batch_df = self._extract_basic_features(batch_df, text_column)

        # Use API for advanced analysis
        advanced_features = self._extract_advanced_features_api(texts)

        # Integrate advanced features
        for i, features in enumerate(advanced_features):
            if i < len(batch_df):
                for key, value in features.items():
                    # Check if column exists, otherwise create
                    if key not in batch_df.columns:
                        batch_df[key] = None
                    # Use .at for safer assignment
                    batch_df.at[batch_df.index[i], key] = value

        return batch_df

    def _extract_basic_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Extract basic features ONLY if they don't exist
        Avoid duplication of features already present in dataset
        """

        # Check and extract hashtags only if they don't exist
        if 'hashtag' not in df.columns and 'hashtags' not in df.columns:
            df['hashtags_extracted'] = df[text_column].apply(
                lambda x: self._extract_hashtags(str(x))
            )
            logger.info("Hashtags extracted - column didn't exist")
        else:
            logger.info("Hashtags already exist in dataset - skipping extraction")

        # Check and extract URLs only if they don't exist
        if 'url' not in df.columns and 'urls' not in df.columns:
            df['urls_extracted'] = df[text_column].apply(
                lambda x: self._extract_urls(str(x))
            )
            logger.info("URLs extracted - column didn't exist")
        else:
            logger.info("URLs already exist in dataset - skipping extraction")

        # Check and extract domains only if they don't exist
        if 'domain' not in df.columns and 'domains' not in df.columns:
            # Use existing or newly extracted URLs
            url_column = None
            if 'urls_extracted' in df.columns:
                url_column = 'urls_extracted'
            elif 'urls' in df.columns:
                url_column = 'urls'
            elif 'url' in df.columns:
                url_column = 'url'

            if url_column:
                df['domains_extracted'] = df[url_column].apply(
                    lambda x: self._extract_domains_from_urls(x)
                )
                logger.info("Domains extracted - column didn't exist")
        else:
            logger.info("Domains already exist in dataset - skipping extraction")

        # Check existing media_type before creating individual flags
        if 'media_type' in df.columns:
            logger.info("Media_type already exists - using validation instead of individual flags")
            # If media_type exists, don't create has_photo, has_video, has_audio
        else:
            # Create media flags only if media_type doesn't exist
            df['has_photo'] = df[text_column].str.contains(
                r'foto|imagem|jpeg|jpg|png|gif', case=False, na=False
            )
            df['has_video'] = df[text_column].str.contains(
                r'vídeo|video|mp4|avi|mov', case=False, na=False
            )
            df['has_audio'] = df[text_column].str.contains(
                r'áudio|audio|mp3|wav|voz', case=False, na=False
            )
            logger.info("Media flags created - media_type didn't exist")

        # Basic metrics are always useful, but check if they already exist
        if 'text_length' not in df.columns:
            df['text_length'] = df[text_column].str.len()
        if 'word_count' not in df.columns:
            df['word_count'] = df[text_column].str.split().str.len()
        if 'has_emoji' not in df.columns and 'emoji_count' not in df.columns:
            df['has_emoji'] = df[text_column].str.contains(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]',
                na=False
            )

        return df

    def _extract_advanced_features_api(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Use API to extract advanced features"""

        prompt = self._build_feature_extraction_prompt(texts[:10])  # Maximum 10 at a time

        try:
            response = self.create_message(
                prompt,
                stage="01b_feature_extraction",
                operation="advanced_analysis"
            )

            # Validate response quality
            validation = self.quality_checker.validate_output_quality(
                response,
                expected_format="json",
                context={"texts_count": len(texts)},
                stage="01b_feature_extraction"
            )

            if not validation["valid"]:
                logger.warning(f"Low response quality: {validation['issues']}")

            # Parse response with robust method
            parsed_response = self.parse_claude_response_safe(response, ["results"])
            return parsed_response.get("results", [{}] * len(texts))

        except Exception as e:
            logger.error(f"Error in advanced extraction via API: {e}")
            return [{}] * len(texts)

    def _build_feature_extraction_prompt(self, texts: List[str]) -> str:
        """Build prompt for feature extraction"""

        texts_sample = "\n".join([f"{i+1}. {text[:200]}..." for i, text in enumerate(texts)])

        return f"""
Analyze the following Brazilian Telegram messages (political context 2019-2023) and extract detailed features.

TEXTS:
{texts_sample}

For each text, provide analysis in JSON format:

{{
  "results": [
    {{
      "text_id": 1,
      "sentiment_category": "positivo|negativo|neutro",
      "political_alignment": "bolsonarista|antibolsonarista|neutro|indefinido",
      "conspiracy_indicators": ["indicator1", "indicator2"],
      "negacionism_indicators": ["type1", "type2"],
      "discourse_type": "informativo|opinião|mobilizador|atacante|defensivo",
      "urgency_level": "low|medium|high",
      "emotional_tone": "raiva|medo|esperança|tristeza|alegria|neutro",
      "target_entities": ["person", "institution", "group"],
      "call_to_action": true/false,
      "misinformation_risk": "low|medium|high",
      "coordination_signals": ["signal1", "signal2"],
      "brazilian_context_markers": ["marker1", "marker2"],
      "quality_issues": ["error1", "error2"]
    }}
  ]
}}

SPECIFIC INSTRUCTIONS:
1. Identify patterns specific to Brazilian political context
2. Detect coordination signals (similar messages, timing)
3. Evaluate misinformation risk based on context
4. Identify quality issues in the text itself
5. Classify political alignment based on language and themes
6. Detect indicators of Brazilian conspiracy theories
7. Identify negationism (scientific, democratic, etc.)

RESPOND ONLY WITH JSON, NO ADDITIONAL EXPLANATIONS.
"""

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, text, re.IGNORECASE)
        return [tag.lower() for tag in hashtags]

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls

    def _extract_domains_from_urls(self, urls: List[str]) -> List[str]:
        """Extract domains from list of URLs"""
        domains = []
        for url in urls:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                if domain:
                    domains.append(domain.lower())
            except:
                continue
        return list(set(domains))

    def _validate_extracted_features(self, final_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate extracted features"""

        validation_report = {
            "original_rows": len(original_df),
            "final_rows": len(final_df),
            "rows_preserved": len(final_df) == len(original_df),
            "new_columns": [col for col in final_df.columns if col not in original_df.columns],
            "missing_values_analysis": {},
            "data_quality_issues": []
        }

        # Analysis of missing values in new columns
        for col in validation_report["new_columns"]:
            if col in final_df.columns:
                missing_count = final_df[col].isna().sum()
                missing_pct = (missing_count / len(final_df)) * 100
                validation_report["missing_values_analysis"][col] = {
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_pct, 2)
                }

        # Identify quality issues
        if not validation_report["rows_preserved"]:
            validation_report["data_quality_issues"].append("Number of rows not preserved")

        if len(validation_report["new_columns"]) < 5:
            validation_report["data_quality_issues"].append("Few new features extracted")

        # Log report
        logger.info(f"Feature validation completed: {validation_report}")

        return validation_report

    def correct_extraction_errors(
        self,
        df: pd.DataFrame,
        error_patterns: List[str] = None
    ) -> pd.DataFrame:
        """
        Correct errors detected in feature extraction

        Args:
            df: DataFrame with extracted features
            error_patterns: Error patterns to correct

        Returns:
            DataFrame with corrections applied
        """
        logger.info("Starting correction of extraction errors")

        corrected_df = df.copy()
        corrections_applied = []

        # Basic corrections
        corrections_applied.extend(self._fix_basic_extraction_errors(corrected_df))

        # Use API for advanced corrections if necessary
        if error_patterns:
            api_corrections = self._fix_errors_with_api(corrected_df, error_patterns)
            corrections_applied.extend(api_corrections)

        logger.info(f"Corrections applied: {len(corrections_applied)}")

        return corrected_df

    def _fix_basic_extraction_errors(self, df: pd.DataFrame) -> List[str]:
        """Apply basic corrections for common errors"""
        corrections = []

        # Fix malformed hashtags
        if 'hashtags_extracted' in df.columns:
            original_count = df['hashtags_extracted'].apply(len).sum()
            df['hashtags_extracted'] = df['hashtags_extracted'].apply(
                lambda x: [tag for tag in x if len(tag) > 1 and tag.startswith('#')]
            )
            new_count = df['hashtags_extracted'].apply(len).sum()
            if new_count != original_count:
                corrections.append(f"Hashtags corrected: {original_count} -> {new_count}")

        # Fix invalid URLs
        if 'urls_extracted' in df.columns:
            original_count = df['urls_extracted'].apply(len).sum()
            df['urls_extracted'] = df['urls_extracted'].apply(
                lambda x: [url for url in x if self._is_valid_url(url)]
            )
            new_count = df['urls_extracted'].apply(len).sum()
            if new_count != original_count:
                corrections.append(f"URLs corrected: {original_count} -> {new_count}")

        # Fix boolean flags
        boolean_columns = ['has_photo', 'has_video', 'has_audio', 'has_emoji', 'call_to_action']
        for col in boolean_columns:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = df[col].astype(bool)
                if original_type != bool:
                    corrections.append(f"Column {col} converted to boolean")

        return corrections

    def _fix_errors_with_api(self, df: pd.DataFrame, error_patterns: List[str]) -> List[str]:
        """Use API to correct specific errors"""
        corrections = []

        # Implement specific corrections via API as needed
        # For example, re-analysis of texts with detected problems

        return corrections

    def _is_valid_url(self, url: str) -> bool:
        """Validate if URL is valid"""
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def generate_feature_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate report of extracted features adapted to existing structure"""

        # Identify original columns vs. new features
        original_columns = {
            'datetime', 'body', 'url', 'hashtag', 'channel', 'is_fwrd',
            'mentions', 'sender', 'media_type', 'domain', 'body_cleaned'
        }
        new_feature_columns = [col for col in df.columns if col not in original_columns]

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "original_columns": len(original_columns.intersection(set(df.columns))),
            "new_features_added": len(new_feature_columns),
            "features_extracted": {},
            "data_quality": {},
            "feature_coverage": {},
            "recommendations": []
        }

        # Analysis of feature coverage
        for feature in new_feature_columns:
            if feature in df.columns:
                non_null_count = df[feature].notna().sum()
                coverage = (non_null_count / len(df)) * 100
                report["feature_coverage"][feature] = {
                    "non_null_count": int(non_null_count),
                    "coverage_percentage": round(coverage, 2)
                }

        # Specific analysis of original columns utilized
        if 'hashtag' in df.columns:
            hashtag_usage = df['hashtag'].notna().sum()
            report["features_extracted"]["hashtag_analysis"] = {
                "messages_with_hashtags": int(hashtag_usage),
                "hashtag_usage_rate": round((hashtag_usage / len(df)) * 100, 2)
            }

        if 'mentions' in df.columns:
            mentions_usage = df['mentions'].notna().sum()
            report["features_extracted"]["mention_analysis"] = {
                "messages_with_mentions": int(mentions_usage),
                "mention_usage_rate": round((mentions_usage / len(df)) * 100, 2)
            }

        if 'url' in df.columns:
            url_usage = df['url'].notna().sum()
            report["features_extracted"]["url_analysis"] = {
                "messages_with_urls": int(url_usage),
                "url_usage_rate": round((url_usage / len(df)) * 100, 2)
            }

        # Quality analysis
        report["data_quality"] = {
            "text_metrics_available": 'text_length' in df.columns,
            "temporal_features_available": 'hour_of_day' in df.columns,
            "political_analysis_available": 'political_alignment' in df.columns,
            "sentiment_analysis_available": 'sentiment_category' in df.columns
        }

        # Recommendations based on structure
        if len(new_feature_columns) < 10:
            report["recommendations"].append("Few new features extracted - check API")

        low_coverage_features = [
            feature for feature, data in report["feature_coverage"].items()
            if data["coverage_percentage"] < 50
        ]
        if low_coverage_features:
            report["recommendations"].append(
                f"Features with low coverage: {', '.join(low_coverage_features)}"
            )

        if not report["data_quality"]["political_analysis_available"]:
            report["recommendations"].append("Political analysis was not extracted successfully")

        return report

    # TDD Phase 3 Methods - Standard feature extraction interface
    def extract_features(self, df: pd.DataFrame, text_column: str = None) -> pd.DataFrame:
        """
        TDD interface: Extract comprehensive features from DataFrame.
        
        Args:
            df: DataFrame to process
            text_column: Text column name (auto-detected if None)
            
        Returns:
            DataFrame with extracted features
        """
        try:
            logger.info(f"🔧 TDD feature extraction started for {len(df)} records")
            
            # Auto-detect text column if not provided
            if text_column is None:
                text_column = self._detect_text_column(df)
            
            if text_column not in df.columns:
                logger.error(f"Text column '{text_column}' not found in DataFrame")
                return df.copy()
            
            result_df = df.copy()
            
            # Extract basic text features
            result_df = self._extract_basic_text_features(result_df, text_column)
            
            # Extract URL features
            result_df = self._extract_url_features(result_df, text_column)
            
            # Extract hashtag features
            result_df = self._extract_hashtag_features(result_df, text_column)
            
            # Extract mention features
            result_df = self._extract_mention_features(result_df, text_column)
            
            # Extract sentiment features
            result_df = self._extract_sentiment_features(result_df, text_column)
            
            # Extract political features
            result_df = self._extract_political_features(result_df, text_column)
            
            features_added = len(result_df.columns) - len(df.columns)
            logger.info(f"✅ TDD feature extraction completed: {features_added} features added")
            
            return result_df
            
        except Exception as e:
            logger.error(f"TDD feature extraction error: {e}")
            return df.copy()
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """TDD interface alias for extract_features."""
        return self.extract_features(df)
    
    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """Detect text column automatically."""
        text_candidates = ['body', 'text', 'content', 'message', 'mensagem']
        
        for candidate in text_candidates:
            if candidate in df.columns:
                # Check if column has text content
                non_empty = df[candidate].dropna().astype(str).str.len().gt(0).sum()
                if non_empty > len(df) * 0.1:  # At least 10% with content
                    return candidate
        
        # Fallback to first object column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        return 'body'  # Final fallback
    
    def _extract_basic_text_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract basic text features."""
        try:
            # Text length
            df['text_length'] = df[text_column].fillna('').astype(str).str.len()
            
            # Word count
            df['word_count'] = df[text_column].fillna('').astype(str).str.split().str.len()
            
            # Character count (excluding spaces)
            df['char_count'] = df[text_column].fillna('').astype(str).str.replace(' ', '').str.len()
            
            # Sentence count (approximate)
            df['sentence_count'] = df[text_column].fillna('').astype(str).str.count(r'[.!?]+') + 1
            
            # Uppercase ratio
            df['uppercase_ratio'] = df[text_column].fillna('').astype(str).apply(
                lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting basic text features: {e}")
            return df
    
    def _extract_url_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract URL-related features."""
        try:
            import re
            url_pattern = r'https?://[^\s]+'
            
            # Extract URLs
            df['urls'] = df[text_column].fillna('').astype(str).apply(
                lambda x: re.findall(url_pattern, x)
            )
            
            # URL count
            df['url_count'] = df['urls'].apply(len)
            
            # Has URLs boolean
            df['has_urls'] = df['url_count'] > 0
            
            # Extract domains
            df['domains'] = df['urls'].apply(
                lambda urls: [self._extract_domain(url) for url in urls] if urls else []
            )
            
            # Domain count
            df['domain_count'] = df['domains'].apply(len)
            
            # Unique domain count
            df['unique_domain_count'] = df['domains'].apply(lambda x: len(set(x)) if x else 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting URL features: {e}")
            return df
    
    def _extract_hashtag_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract hashtag-related features."""
        try:
            import re
            hashtag_pattern = r'#\w+'
            
            # Extract hashtags
            df['hashtags'] = df[text_column].fillna('').astype(str).apply(
                lambda x: re.findall(hashtag_pattern, x.lower())
            )
            
            # Hashtag count
            df['hashtag_count'] = df['hashtags'].apply(len)
            
            # Has hashtags boolean
            df['has_hashtags'] = df['hashtag_count'] > 0
            
            # Unique hashtag count
            df['unique_hashtag_count'] = df['hashtags'].apply(lambda x: len(set(x)) if x else 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting hashtag features: {e}")
            return df
    
    def _extract_mention_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract mention-related features."""
        try:
            import re
            mention_pattern = r'@\w+'
            
            # Extract mentions
            mentions_list = df[text_column].fillna('').astype(str).apply(
                lambda x: re.findall(mention_pattern, x.lower())
            )
            df['mentions'] = mentions_list.apply(lambda x: x if x else None)
            
            # Mention count
            df['mention_count'] = mentions_list.apply(len)
            
            # Has mentions boolean
            df['has_mentions'] = df['mention_count'] > 0
            
            # Unique mention count
            df['unique_mention_count'] = df['mentions'].apply(lambda x: len(set(x)) if x else 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting mention features: {e}")
            return df
    
    def _extract_sentiment_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract sentiment-related features."""
        try:
            # Exclamation count
            df['exclamation_count'] = df[text_column].fillna('').astype(str).str.count('!')
            
            # Question count
            df['question_count'] = df[text_column].fillna('').astype(str).str.count(r'\?')
            
            # Caps ratio (percentage of text in caps)
            df['caps_ratio'] = df[text_column].fillna('').astype(str).apply(
                lambda x: sum(c.isupper() for c in x if c.isalpha()) / len([c for c in x if c.isalpha()]) if any(c.isalpha() for c in x) else 0
            )
            
            # Repeated chars (!!!!, ????, etc.)
            df['repeated_chars'] = df[text_column].fillna('').astype(str).str.count(r'(.)\1{2,}')
            
            # Positive/negative words (simple heuristic)
            positive_words = ['bom', 'ótimo', 'excelente', 'maravilhoso', 'feliz', 'alegre', 'perfeito']
            negative_words = ['ruim', 'péssimo', 'terrível', 'horrível', 'triste', 'revoltante', 'inaceitável']
            
            df['positive_word_count'] = df[text_column].fillna('').astype(str).str.lower().apply(
                lambda x: sum(word in x for word in positive_words)
            )
            
            df['negative_word_count'] = df[text_column].fillna('').astype(str).str.lower().apply(
                lambda x: sum(word in x for word in negative_words)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting sentiment features: {e}")
            return df
    
    def _extract_political_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract political discourse features."""
        try:
            # Political entities (simplified)
            political_entities = ['bolsonaro', 'lula', 'pt', 'psl', 'stf', 'tse', 'congresso', 'governo']
            
            df['political_entities'] = df[text_column].fillna('').astype(str).str.lower().apply(
                lambda x: [entity for entity in political_entities if entity in x]
            )
            
            df['political_entity_count'] = df['political_entities'].apply(len)
            
            # Discourse markers
            conspiracy_markers = ['fake news', 'mídia', 'manipulação', 'verdade', 'acordem']
            authoritarian_markers = ['ordem', 'disciplina', 'autoridade', 'controle']
            
            df['conspiracy_markers'] = df[text_column].fillna('').astype(str).str.lower().apply(
                lambda x: sum(marker in x for marker in conspiracy_markers)
            )
            
            df['authoritarian_markers'] = df[text_column].fillna('').astype(str).str.lower().apply(
                lambda x: sum(marker in x for marker in authoritarian_markers)
            )
            
            # Discourse markers list
            df['discourse_markers'] = df.apply(
                lambda row: (['conspiracy'] if row['conspiracy_markers'] > 0 else []) + 
                           (['authoritarian'] if row['authoritarian_markers'] > 0 else []),
                axis=1
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting political features: {e}")
            return df
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            import re
            match = re.search(r'https?://([^/]+)', url)
            if match:
                domain = match.group(1)
                # Remove www. prefix
                return domain.replace('www.', '')
            return ''
        except:
            return ''
