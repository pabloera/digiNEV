"""
Intelligent Query System for Political Discourse Analysis
Provides natural language querying interface with advanced AI capabilities
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import argparse

from .base import AnthropicBase
from .semantic_search_engine import SemanticSearchEngine
from .voyage_embeddings import VoyageEmbeddingAnalyzer

logger = logging.getLogger(__name__)


class IntelligentQuerySystem(AnthropicBase):
    """
    Intelligent Query System for Political Discourse Analysis
    
    Capabilities:
    - Natural language query processing and understanding
    - Intent detection and query expansion
    - Multi-modal search (semantic + keyword + temporal)
    - Query suggestion and autocomplete
    - Context-aware result ranking
    - Interactive query refinement
    - Research assistant functionality
    - Export and reporting capabilities
    """
    
    def __init__(self, config: Dict[str, Any], search_engine: SemanticSearchEngine = None):
        super().__init__(config)
        
        # Initialize search engine
        if search_engine:
            self.search_engine = search_engine
        else:
            embedding_analyzer = VoyageEmbeddingAnalyzer(config)
            self.search_engine = SemanticSearchEngine(config, embedding_analyzer)
        
        # Query processing configuration
        query_config = config.get('intelligent_query', {})
        self.max_suggestions = query_config.get('max_suggestions', 5)
        self.enable_query_expansion = query_config.get('enable_expansion', True)
        self.enable_context_tracking = query_config.get('enable_context', True)
        self.default_result_limit = query_config.get('default_results', 20)
        
        # Query context and history
        self.query_history = []
        self.current_context = {}
        self.session_insights = []
        
        # Predefined query templates for political analysis
        self.query_templates = {
            'discourse_analysis': [
                "Como {político} fala sobre {tópico}?",
                "Qual a evolução do discurso sobre {tópico}?",
                "Quais canais mais discutem {tópico}?",
                "Como {evento} influenciou o discurso?"
            ],
            'comparative_analysis': [
                "Compare discurso sobre {tópico1} vs {tópico2}",
                "Diferenças entre canais sobre {tópico}",
                "Evolução temporal de {tópico}",
                "Polarização em torno de {tópico}"
            ],
            'influence_tracking': [
                "Quais canais influenciam discussões sobre {tópico}?",
                "Redes de influência em {tópico}",
                "Propagação de narrativas sobre {tópico}",
                "Coordenação de mensagens sobre {tópico}"
            ],
            'content_discovery': [
                "Temas emergentes em {período}",
                "Padrões suspeitos de coordenação",
                "Teorias conspiratórias sobre {tópico}",
                "Desinformação relacionada a {tópico}"
            ]
        }
        
        # Brazilian political context
        self.political_entities = {
            'politicians': ['bolsonaro', 'lula', 'moro', 'doria', 'ciro', 'marina', 'haddad'],
            'institutions': ['stf', 'congresso', 'senado', 'câmara', 'tse', 'pf', 'pcc'],
            'events': ['eleições 2022', 'pandemia', 'impeachment', 'operação lava jato', 'mensalão'],
            'topics': ['vacinas', 'economia', 'educação', 'segurança', 'corrupção', 'democracia']
        }
        
        logger.info("IntelligentQuerySystem initialized successfully")
    
    def process_natural_language_query(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process natural language query with full intelligence
        
        Args:
            query: Natural language query in Portuguese
            context: Optional context from previous queries
            
        Returns:
            Comprehensive query results with analysis
        """
        logger.info(f"Processing natural language query: '{query}'")
        
        start_time = datetime.now()
        
        # Step 1: Query preprocessing and understanding
        query_analysis = self._analyze_query_intent(query)
        
        # Step 2: Query expansion and enrichment
        if self.enable_query_expansion:
            expanded_queries = self._expand_query(query, query_analysis)
        else:
            expanded_queries = [query]
        
        # Step 3: Execute searches
        search_results = []
        for expanded_query in expanded_queries:
            result = self.search_engine.semantic_search(
                query=expanded_query,
                top_k=self.default_result_limit,
                include_metadata=True
            )
            if result.get('results'):
                search_results.append({
                    'query': expanded_query,
                    'results': result['results'],
                    'total_found': result['total_results']
                })
        
        # Step 4: Aggregate and rank results
        aggregated_results = self._aggregate_search_results(search_results)
        
        # Step 5: Generate insights and analysis
        query_insights = self._generate_query_insights(
            query, 
            query_analysis, 
            aggregated_results
        )
        
        # Step 6: Suggest follow-up queries
        follow_up_suggestions = self._generate_follow_up_queries(
            query, 
            query_analysis, 
            aggregated_results
        )
        
        # Step 7: Update context and history
        self._update_query_context(query, query_analysis, aggregated_results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'original_query': query,
            'query_analysis': query_analysis,
            'expanded_queries': expanded_queries,
            'total_results_found': len(aggregated_results),
            'results': aggregated_results,
            'insights': query_insights,
            'follow_up_suggestions': follow_up_suggestions,
            'processing_time_seconds': processing_time,
            'timestamp': start_time.isoformat()
        }
    
    def interactive_research_session(self) -> Dict[str, Any]:
        """
        Start an interactive research session with guided queries
        
        Returns:
            Session results and insights
        """
        logger.info("Starting interactive research session")
        
        session_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'queries_processed': [],
            'discoveries': [],
            'insights_generated': []
        }
        
        print("\n" + "="*60)
        print("🔍 SISTEMA DE CONSULTA INTELIGENTE")
        print("Análise de Discurso Político Brasileiro (2019-2023)")
        print("="*60)
        
        print("\nTipos de consulta disponíveis:")
        print("1. Análise de discurso: 'Como Bolsonaro fala sobre vacinas?'")
        print("2. Evolução temporal: 'Evolução do discurso sobre democracia'")
        print("3. Comparação: 'Compare discurso sobre STF vs TSE'")
        print("4. Descoberta: 'Teorias conspiratórias sobre eleições'")
        print("5. Influência: 'Quais canais mais influentes sobre COVID?'")
        print("\nComandos especiais:")
        print("- 'exemplos' - Mostrar exemplos de consultas")
        print("- 'contexto' - Ver contexto atual da sessão")
        print("- 'insights' - Gerar insights automáticos")
        print("- 'sair' - Finalizar sessão")
        
        while True:
            try:
                user_input = input("\n💬 Sua consulta: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    break
                elif user_input.lower() == 'exemplos':
                    self._show_query_examples()
                    continue
                elif user_input.lower() == 'contexto':
                    self._show_session_context()
                    continue
                elif user_input.lower() == 'insights':
                    insights = self._generate_session_insights()
                    print(f"\n📊 Insights da Sessão:")
                    for insight in insights:
                        print(f"• {insight}")
                    continue
                
                # Process the query
                print(f"\n🔍 Processando: '{user_input}'...")
                
                result = self.process_natural_language_query(user_input)
                session_data['queries_processed'].append(result)
                
                # Display results
                self._display_query_results(result)
                
                # Show follow-up suggestions
                if result.get('follow_up_suggestions'):
                    print(f"\n💡 Sugestões de consultas relacionadas:")
                    for i, suggestion in enumerate(result['follow_up_suggestions'][:3], 1):
                        print(f"{i}. {suggestion}")
                
            except KeyboardInterrupt:
                print("\n\nSessão interrompida pelo usuário.")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                logger.error(f"Error in interactive session: {e}")
        
        # Finalize session
        session_data['end_time'] = datetime.now().isoformat()
        session_data['total_queries'] = len(session_data['queries_processed'])
        
        # Generate final session report
        final_insights = self._generate_final_session_report(session_data)
        session_data['final_insights'] = final_insights
        
        print(f"\n📋 Sessão finalizada!")
        print(f"Total de consultas: {session_data['total_queries']}")
        print(f"Insights gerados: {len(final_insights)}")
        
        return session_data
    
    def batch_query_processing(
        self, 
        queries: List[str], 
        output_format: str = 'json'
    ) -> Dict[str, Any]:
        """
        Process multiple queries in batch mode
        
        Args:
            queries: List of queries to process
            output_format: Output format ('json', 'csv', 'report')
            
        Returns:
            Batch processing results
        """
        logger.info(f"Processing {len(queries)} queries in batch mode")
        
        batch_results = {
            'batch_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_queries': len(queries),
            'processed_queries': [],
            'failed_queries': [],
            'aggregate_insights': {},
            'processing_summary': {}
        }
        
        for i, query in enumerate(queries, 1):
            try:
                logger.info(f"Processing query {i}/{len(queries)}: {query}")
                
                result = self.process_natural_language_query(query)
                batch_results['processed_queries'].append(result)
                
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                batch_results['failed_queries'].append({
                    'query': query,
                    'error': str(e)
                })
        
        # Generate aggregate insights
        batch_results['aggregate_insights'] = self._generate_batch_insights(
            batch_results['processed_queries']
        )
        
        # Generate processing summary
        batch_results['processing_summary'] = {
            'successful_queries': len(batch_results['processed_queries']),
            'failed_queries': len(batch_results['failed_queries']),
            'success_rate': len(batch_results['processed_queries']) / len(queries),
            'total_results_found': sum(
                len(q['results']) for q in batch_results['processed_queries']
            ),
            'processing_time': sum(
                q['processing_time_seconds'] for q in batch_results['processed_queries']
            )
        }
        
        # Export results if requested
        if output_format != 'json':
            self._export_batch_results(batch_results, output_format)
        
        return batch_results
    
    def generate_query_suggestions(
        self, 
        partial_query: str = "", 
        context_type: str = "general"
    ) -> List[str]:
        """
        Generate intelligent query suggestions
        
        Args:
            partial_query: Partial query text for completion
            context_type: Type of context for suggestions
            
        Returns:
            List of suggested queries
        """
        suggestions = []
        
        # Template-based suggestions
        if context_type in self.query_templates:
            templates = self.query_templates[context_type]
            for template in templates[:self.max_suggestions]:
                # Simple template filling for demonstration
                filled_template = template.replace('{tópico}', 'COVID')
                filled_template = filled_template.replace('{político}', 'Bolsonaro')
                suggestions.append(filled_template)
        
        # Context-aware suggestions based on query history
        if self.query_history:
            recent_topics = self._extract_topics_from_history()
            for topic in recent_topics[:2]:
                suggestions.append(f"Evolução do discurso sobre {topic}")
                suggestions.append(f"Principais canais que discutem {topic}")
        
        # AI-powered suggestions if available
        if self.api_available and partial_query:
            ai_suggestions = self._generate_ai_query_suggestions(partial_query)
            suggestions.extend(ai_suggestions)
        
        return suggestions[:self.max_suggestions]
    
    def export_query_results(
        self, 
        results: Dict[str, Any], 
        format_type: str = 'json',
        output_path: str = None
    ) -> str:
        """
        Export query results to various formats
        
        Args:
            results: Query results to export
            format_type: Export format ('json', 'csv', 'markdown', 'pdf')
            output_path: Optional output path
            
        Returns:
            Path to exported file
        """
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"query_results_{timestamp}.{format_type}"
        
        if format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        elif format_type == 'csv':
            # Export results as CSV
            if results.get('results'):
                df_results = pd.DataFrame([
                    {
                        'similarity_score': r['similarity_score'],
                        'text': r['text'][:200] + "...",
                        'channel': r.get('metadata', {}).get('channel', ''),
                        'datetime': r.get('metadata', {}).get('datetime', '')
                    }
                    for r in results['results']
                ])
                df_results.to_csv(output_path, index=False, encoding='utf-8')
        
        elif format_type == 'markdown':
            self._export_markdown_report(results, output_path)
        
        logger.info(f"Results exported to: {output_path}")
        return output_path
    
    # Helper Methods
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and extract key information"""
        query_lower = query.lower()
        
        # Detect query type
        query_type = 'general'
        if any(word in query_lower for word in ['como', 'qual', 'quando']):
            query_type = 'question'
        elif any(word in query_lower for word in ['compare', 'diferença', 'versus']):
            query_type = 'comparison'
        elif any(word in query_lower for word in ['evolução', 'mudança', 'temporal']):
            query_type = 'temporal'
        elif any(word in query_lower for word in ['influência', 'rede', 'propagação']):
            query_type = 'network'
        
        # Extract entities
        entities = {
            'politicians': [p for p in self.political_entities['politicians'] if p in query_lower],
            'institutions': [i for i in self.political_entities['institutions'] if i in query_lower],
            'topics': [t for t in self.political_entities['topics'] if t in query_lower],
            'events': [e for e in self.political_entities['events'] if e in query_lower]
        }
        
        # Detect temporal indicators
        temporal_indicators = []
        if any(word in query_lower for word in ['2019', '2020', '2021', '2022', '2023']):
            temporal_indicators.extend([word for word in ['2019', '2020', '2021', '2022', '2023'] if word in query_lower])
        
        return {
            'query_type': query_type,
            'entities_detected': entities,
            'temporal_indicators': temporal_indicators,
            'complexity_level': 'high' if len([e for e in entities.values() if e]) > 2 else 'medium',
            'requires_ai_analysis': query_type in ['comparison', 'network']
        }
    
    def _expand_query(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded = [query]
        
        # Add entity-based expansions
        entities = analysis.get('entities_detected', {})
        
        # Add synonyms for key political terms
        synonyms = {
            'governo': ['administração', 'gestão', 'poder executivo'],
            'democracia': ['democratização', 'regime democrático', 'sistema democrático'],
            'eleição': ['pleito', 'votação', 'processo eleitoral'],
            'corrupção': ['má gestão', 'desvio', 'irregularidade']
        }
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity in synonyms:
                    for synonym in synonyms[entity]:
                        expanded.append(query.replace(entity, synonym))
        
        return expanded[:3]  # Limit expansions
    
    def _aggregate_search_results(self, search_results: List[Dict]) -> List[Dict]:
        """Aggregate and deduplicate search results"""
        all_results = []
        seen_texts = set()
        
        for search_result in search_results:
            for result in search_result['results']:
                text_key = result['text'][:100]  # Use first 100 chars as key
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_results.append(result)
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return all_results[:self.default_result_limit]
    
    def _generate_query_insights(
        self, 
        query: str, 
        analysis: Dict[str, Any], 
        results: List[Dict]
    ) -> Dict[str, Any]:
        """Generate insights about the query results"""
        if not results:
            return {'status': 'no_results'}
        
        # Basic insights
        insights = {
            'total_results': len(results),
            'avg_similarity': np.mean([r['similarity_score'] for r in results]),
            'top_similarity': max(r['similarity_score'] for r in results),
            'result_quality': 'high' if np.mean([r['similarity_score'] for r in results]) > 0.8 else 'medium'
        }
        
        # Channel analysis
        channels = [r.get('metadata', {}).get('channel') for r in results]
        channels = [c for c in channels if c]
        if channels:
            from collections import Counter
            channel_counts = Counter(channels)
            insights['top_channels'] = dict(channel_counts.most_common(3))
            insights['channel_diversity'] = len(set(channels))
        
        # Temporal analysis
        dates = [r.get('metadata', {}).get('datetime') for r in results]
        dates = [d for d in dates if d]
        if dates:
            insights['temporal_spread'] = {
                'earliest': min(dates),
                'latest': max(dates),
                'date_count': len(set(dates))
            }
        
        # AI-powered insights if available
        if self.api_available:
            ai_insights = self._generate_ai_insights(query, results[:5])
            insights.update(ai_insights)
        
        return insights
    
    def _generate_follow_up_queries(
        self, 
        query: str, 
        analysis: Dict[str, Any], 
        results: List[Dict]
    ) -> List[str]:
        """Generate intelligent follow-up query suggestions"""
        suggestions = []
        
        # Based on query type
        query_type = analysis.get('query_type', 'general')
        
        if query_type == 'question':
            suggestions.extend([
                f"Evolução temporal de: {query.replace('Como ', '').replace('?', '')}",
                f"Comparar com outros políticos: {query}",
                f"Canais que mais discutem: {query.replace('Como ', '').replace('?', '')}"
            ])
        elif query_type == 'temporal':
            suggestions.extend([
                f"Principais influenciadores sobre: {query}",
                f"Impacto das mudanças em: {query}",
                f"Reações do público a: {query}"
            ])
        
        # Based on entities found
        entities = analysis.get('entities_detected', {})
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity = entity_list[0]
                suggestions.append(f"Redes de influência relacionadas a {entity}")
                suggestions.append(f"Coordenação de mensagens sobre {entity}")
        
        return suggestions[:self.max_suggestions]
    
    def _update_query_context(
        self, 
        query: str, 
        analysis: Dict[str, Any], 
        results: List[Dict]
    ):
        """Update query context and history"""
        self.query_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'result_count': len(results)
        })
        
        # Update current context
        entities = analysis.get('entities_detected', {})
        for entity_type, entity_list in entities.items():
            if entity_list:
                self.current_context[entity_type] = entity_list
        
        # Limit history size
        if len(self.query_history) > 20:
            self.query_history = self.query_history[-20:]
    
    def _display_query_results(self, result: Dict[str, Any]):
        """Display query results in a user-friendly format"""
        print(f"\n📊 Resultados para: '{result['original_query']}'")
        print(f"⏱️  Processamento: {result['processing_time_seconds']:.2f}s")
        print(f"📈 Total encontrado: {result['total_results_found']} documentos")
        
        # Show insights
        insights = result.get('insights', {})
        if insights.get('avg_similarity'):
            print(f"🎯 Relevância média: {insights['avg_similarity']:.3f}")
        
        if insights.get('top_channels'):
            print(f"📺 Principais canais: {', '.join(list(insights['top_channels'].keys())[:3])}")
        
        # Show top results
        print(f"\n🔝 Top {min(5, len(result['results']))} resultados:")
        for i, res in enumerate(result['results'][:5], 1):
            similarity = res['similarity_score']
            text = res['text'][:150] + "..." if len(res['text']) > 150 else res['text']
            channel = res.get('metadata', {}).get('channel', 'Desconhecido')
            
            print(f"\n{i}. 📊 {similarity:.3f} | 📺 {channel}")
            print(f"   💬 {text}")
    
    def _show_query_examples(self):
        """Show example queries to help users"""
        print("\n📝 Exemplos de Consultas:")
        
        examples = [
            "Como Bolsonaro fala sobre vacinas?",
            "Evolução do discurso sobre democracia de 2019 a 2023",
            "Teorias conspiratórias sobre eleições 2022",
            "Compare discurso sobre STF vs TSE",
            "Quais canais mais influentes sobre COVID?",
            "Coordenação de mensagens sobre urnas eletrônicas",
            "Desinformação relacionada a hidroxicloroquina",
            "Polarização em torno do isolamento social"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
    
    def _show_session_context(self):
        """Show current session context"""
        print(f"\n📋 Contexto da Sessão:")
        print(f"Consultas realizadas: {len(self.query_history)}")
        
        if self.current_context:
            print("Entidades identificadas:")
            for entity_type, entities in self.current_context.items():
                if entities:
                    print(f"  • {entity_type}: {', '.join(entities)}")
        
        if self.query_history:
            print("\nÚltimas consultas:")
            for query_info in self.query_history[-3:]:
                print(f"  • {query_info['query']}")
    
    def _generate_session_insights(self) -> List[str]:
        """Generate insights about the current session"""
        insights = []
        
        if len(self.query_history) >= 3:
            insights.append(f"Você realizou {len(self.query_history)} consultas nesta sessão")
            
            # Find common themes
            all_entities = []
            for query_info in self.query_history:
                entities = query_info.get('analysis', {}).get('entities_detected', {})
                for entity_list in entities.values():
                    all_entities.extend(entity_list)
            
            if all_entities:
                from collections import Counter
                common_entities = Counter(all_entities).most_common(3)
                insights.append(f"Temas mais consultados: {', '.join([e[0] for e in common_entities])}")
        
        return insights
    
    def _generate_final_session_report(self, session_data: Dict[str, Any]) -> List[str]:
        """Generate final session report"""
        insights = []
        
        total_queries = session_data['total_queries']
        insights.append(f"Sessão processou {total_queries} consultas")
        
        if session_data['queries_processed']:
            total_results = sum(len(q['results']) for q in session_data['queries_processed'])
            insights.append(f"Total de {total_results} documentos analisados")
            
            avg_processing_time = np.mean([q['processing_time_seconds'] for q in session_data['queries_processed']])
            insights.append(f"Tempo médio de processamento: {avg_processing_time:.2f}s")
        
        return insights
    
    def _generate_batch_insights(self, processed_queries: List[Dict]) -> Dict[str, Any]:
        """Generate insights from batch processing"""
        if not processed_queries:
            return {}
        
        total_results = sum(len(q['results']) for q in processed_queries)
        avg_processing_time = np.mean([q['processing_time_seconds'] for q in processed_queries])
        
        # Find most common query types
        query_types = []
        for query in processed_queries:
            query_type = query.get('query_analysis', {}).get('query_type', 'general')
            query_types.append(query_type)
        
        from collections import Counter
        type_distribution = Counter(query_types)
        
        return {
            'total_documents_analyzed': total_results,
            'average_processing_time': avg_processing_time,
            'query_type_distribution': dict(type_distribution),
            'most_common_query_type': type_distribution.most_common(1)[0][0] if type_distribution else 'unknown'
        }
    
    def _export_batch_results(self, batch_results: Dict[str, Any], format_type: str):
        """Export batch results to specified format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == 'csv':
            # Create summary CSV
            summary_data = []
            for query_result in batch_results['processed_queries']:
                summary_data.append({
                    'query': query_result['original_query'],
                    'results_found': len(query_result['results']),
                    'avg_similarity': np.mean([r['similarity_score'] for r in query_result['results']]) if query_result['results'] else 0,
                    'processing_time': query_result['processing_time_seconds']
                })
            
            df_summary = pd.DataFrame(summary_data)
            output_path = f"batch_results_summary_{timestamp}.csv"
            df_summary.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Batch summary exported to: {output_path}")
    
    def _export_markdown_report(self, results: Dict[str, Any], output_path: str):
        """Export results as markdown report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Relatório de Consulta Semântica\n\n")
            f.write(f"**Consulta:** {results['original_query']}\n\n")
            f.write(f"**Data:** {results['timestamp']}\n\n")
            f.write(f"**Resultados encontrados:** {results['total_results_found']}\n\n")
            
            # Write insights
            insights = results.get('insights', {})
            if insights:
                f.write("## Insights\n\n")
                for key, value in insights.items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")
            
            # Write top results
            f.write("## Principais Resultados\n\n")
            for i, result in enumerate(results['results'][:10], 1):
                f.write(f"### {i}. Similaridade: {result['similarity_score']:.3f}\n\n")
                f.write(f"{result['text']}\n\n")
                
                metadata = result.get('metadata', {})
                if metadata.get('channel'):
                    f.write(f"**Canal:** {metadata['channel']}\n\n")
                if metadata.get('datetime'):
                    f.write(f"**Data:** {metadata['datetime']}\n\n")
                f.write("---\n\n")
    
    def _extract_topics_from_history(self) -> List[str]:
        """Extract topics from query history"""
        topics = []
        for query_info in self.query_history[-5:]:  # Last 5 queries
            entities = query_info.get('analysis', {}).get('entities_detected', {})
            for entity_list in entities.values():
                topics.extend(entity_list)
        
        # Return unique topics
        return list(set(topics))
    
    def _generate_ai_query_suggestions(self, partial_query: str) -> List[str]:
        """Generate query suggestions using AI"""
        if not self.api_available:
            return []
        
        try:
            prompt = f"""
Dado o início de consulta "{partial_query}", sugira 3 consultas completas e específicas para análise de discurso político brasileiro no Telegram (2019-2023).

Foque em:
- Análise de políticos (Bolsonaro, Lula, etc.)
- Instituições (STF, TSE, Congresso)
- Temas políticos (democracia, eleições, pandemia)
- Padrões de influência e coordenação

Responda em JSON:
{{
  "suggestions": ["consulta1", "consulta2", "consulta3"]
}}
"""
            
            response = self.create_message(
                prompt,
                stage="intelligent_query",
                operation="suggestion_generation",
                temperature=0.7
            )
            
            parsed = self.parse_json_response(response)
            return parsed.get('suggestions', [])
            
        except Exception as e:
            logger.warning(f"AI suggestion generation failed: {e}")
            return []
    
    def _generate_ai_insights(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """Generate AI-powered insights about query results"""
        if not self.api_available or not results:
            return {}
        
        try:
            sample_texts = [r['text'][:200] for r in results[:3]]
            texts_sample = '\n'.join([f"{i+1}. {text}..." for i, text in enumerate(sample_texts)])
            
            prompt = f"""
Analise os resultados desta consulta semântica:

CONSULTA: "{query}"

RESULTADOS ENCONTRADOS:
{texts_sample}

Gere insights em JSON:
{{
  "content_themes": ["tema1", "tema2"],
  "discourse_pattern": "padrão identificado",
  "political_alignment": "tendência política detectada",
  "credibility_indicators": ["indicador1", "indicador2"],
  "research_value": "alto|medio|baixo",
  "recommended_analysis": "tipo de análise recomendada"
}}
"""
            
            response = self.create_message(
                prompt,
                stage="intelligent_query",
                operation="result_analysis",
                temperature=0.3
            )
            
            return self.parse_json_response(response)
            
        except Exception as e:
            logger.warning(f"AI insight generation failed: {e}")
            return {}


def create_intelligent_query_system(
    config: Dict[str, Any], 
    search_engine: SemanticSearchEngine = None
) -> IntelligentQuerySystem:
    """
    Factory function to create IntelligentQuerySystem instance
    
    Args:
        config: Configuration dictionary
        search_engine: Optional pre-initialized search engine
        
    Returns:
        IntelligentQuerySystem instance
    """
    return IntelligentQuerySystem(config, search_engine)


# CLI Interface
def main():
    """Command-line interface for the intelligent query system"""
    parser = argparse.ArgumentParser(description='Sistema de Consulta Inteligente - Análise de Discurso Político')
    parser.add_argument('--config', default='config/settings.yaml', help='Configuration file path')
    parser.add_argument('--data', required=True, help='Path to processed dataset')
    parser.add_argument('--interactive', action='store_true', help='Start interactive session')
    parser.add_argument('--query', help='Single query to process')
    parser.add_argument('--batch', help='File with queries to process in batch')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize system
    query_system = create_intelligent_query_system(config)
    
    # Load and index data
    df = pd.read_csv(args.data, sep=';', encoding='utf-8')
    logger.info(f"Loaded {len(df)} documents")
    
    # Build search index
    index_result = query_system.search_engine.build_search_index(df)
    if not index_result.get('success'):
        logger.error("Failed to build search index")
        return
    
    logger.info("Search index built successfully")
    
    # Process based on mode
    if args.interactive:
        query_system.interactive_research_session()
    elif args.query:
        result = query_system.process_natural_language_query(args.query)
        if args.output:
            query_system.export_query_results(result, 'json', args.output)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif args.batch:
        with open(args.batch, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        batch_result = query_system.batch_query_processing(queries)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, indent=2, ensure_ascii=False, default=str)
        else:
            print(json.dumps(batch_result, indent=2, ensure_ascii=False, default=str))
    else:
        print("Specify --interactive, --query, or --batch mode")


if __name__ == "__main__":
    main()