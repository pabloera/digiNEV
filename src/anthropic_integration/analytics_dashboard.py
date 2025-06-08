"""
Advanced Analytics Dashboard for Political Discourse Analysis
Provides comprehensive metrics, reporting, and visualization data for semantic analysis results
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

from .base import AnthropicBase
from .semantic_search_engine import SemanticSearchEngine
from .content_discovery_engine import ContentDiscoveryEngine
from .intelligent_query_system import IntelligentQuerySystem

logger = logging.getLogger(__name__)


class AnalyticsDashboard(AnthropicBase):
    """
    Advanced Analytics Dashboard for Political Discourse Analysis
    
    Capabilities:
    - Comprehensive metric calculation and tracking
    - Multi-dimensional analysis reporting
    - Performance monitoring and benchmarking
    - Trend analysis and forecasting
    - Risk assessment and alerting
    - Interactive data exploration interfaces
    - Export capabilities for various formats
    - Real-time analytics updates
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        search_engine: SemanticSearchEngine = None,
        discovery_engine: ContentDiscoveryEngine = None,
        query_system: IntelligentQuerySystem = None
    ):
        super().__init__(config)
        
        # Initialize core engines
        self.search_engine = search_engine
        self.discovery_engine = discovery_engine
        self.query_system = query_system
        
        # Dashboard configuration
        dashboard_config = config.get('analytics_dashboard', {})
        self.refresh_interval_minutes = dashboard_config.get('refresh_interval', 60)
        self.max_trend_history = dashboard_config.get('max_trend_history', 100)
        self.enable_real_time = dashboard_config.get('enable_real_time', False)
        self.metric_precision = dashboard_config.get('metric_precision', 3)
        
        # Analytics state
        self.metrics_cache = {}
        self.trend_history = []
        self.performance_benchmarks = {}
        self.alert_thresholds = {
            'coordination_score': 0.8,
            'misinformation_risk': 0.7,
            'influence_concentration': 0.6,
            'anomaly_detection': 0.75
        }
        
        # Dashboard sections
        self.dashboard_sections = [
            'overview_metrics',
            'discourse_patterns',
            'influence_networks',
            'content_trends',
            'risk_assessment',
            'performance_metrics'
        ]
        
        logger.info("AnalyticsDashboard initialized successfully")
    
    def generate_comprehensive_dashboard(self, time_period_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive analytics dashboard
        
        Args:
            time_period_days: Time period for analysis
            
        Returns:
            Complete dashboard data structure
        """
        logger.info(f"Generating comprehensive analytics dashboard for {time_period_days}-day period")
        
        dashboard_start_time = time.time()
        
        dashboard_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'time_period_days': time_period_days,
                'dashboard_version': '1.0',
                'data_sources': self._get_data_source_info()
            },
            'sections': {}
        }
        
        # Generate each dashboard section
        for section in self.dashboard_sections:
            try:
                section_start_time = time.time()
                
                if section == 'overview_metrics':
                    dashboard_data['sections'][section] = self._generate_overview_metrics()
                elif section == 'discourse_patterns':
                    dashboard_data['sections'][section] = self._generate_discourse_patterns_analysis()
                elif section == 'influence_networks':
                    dashboard_data['sections'][section] = self._generate_influence_network_metrics()
                elif section == 'content_trends':
                    dashboard_data['sections'][section] = self._generate_content_trends_analysis(time_period_days)
                elif section == 'risk_assessment':
                    dashboard_data['sections'][section] = self._generate_risk_assessment_dashboard()
                elif section == 'performance_metrics':
                    dashboard_data['sections'][section] = self._generate_performance_metrics()
                
                section_time = time.time() - section_start_time
                dashboard_data['sections'][section]['generation_time_seconds'] = section_time
                
                logger.info(f"Generated {section} in {section_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error generating section {section}: {e}")
                dashboard_data['sections'][section] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Calculate summary statistics
        dashboard_data['summary'] = self._generate_dashboard_summary(dashboard_data['sections'])
        
        # Generate alerts and recommendations
        dashboard_data['alerts'] = self._generate_dashboard_alerts(dashboard_data['sections'])
        dashboard_data['recommendations'] = self._generate_dashboard_recommendations(dashboard_data['sections'])
        
        total_generation_time = time.time() - dashboard_start_time
        dashboard_data['metadata']['total_generation_time'] = total_generation_time
        
        logger.info(f"Dashboard generated successfully in {total_generation_time:.2f}s")
        
        return dashboard_data
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get real-time metrics for live dashboard updates
        
        Returns:
            Real-time metrics data
        """
        if not self.enable_real_time:
            return {'error': 'Real-time metrics not enabled'}
        
        logger.info("Generating real-time metrics")
        
        current_time = datetime.now()
        
        # Simulate real-time data (in production, this would connect to live streams)
        real_time_data = {
            'timestamp': current_time.isoformat(),
            'live_metrics': {
                'active_channels': np.random.randint(50, 200),
                'messages_per_minute': np.random.randint(10, 100),
                'coordination_events': np.random.randint(0, 5),
                'trending_topics': self._get_current_trending_topics(),
                'risk_level': np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
            },
            'alerts': self._check_real_time_alerts(),
            'system_health': {
                'api_status': 'operational',
                'search_index_health': 'healthy',
                'processing_latency_ms': np.random.randint(100, 500)
            }
        }
        
        return real_time_data
    
    def generate_trend_analysis_report(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Generate comprehensive trend analysis report
        
        Args:
            lookback_days: Days to look back for trend analysis
            
        Returns:
            Trend analysis report
        """
        logger.info(f"Generating trend analysis report for {lookback_days} days")
        
        if not self.discovery_engine:
            return {'error': 'Content discovery engine not available'}
        
        # Discover trends over multiple time windows
        trend_windows = [7, 14, 30, 60]  # Different time scales
        trend_analysis = {}
        
        for window_days in trend_windows:
            if window_days <= lookback_days:
                trends = self.discovery_engine.discover_emerging_trends(
                    time_window_days=window_days
                )
                trend_analysis[f'{window_days}_day_trends'] = trends
        
        # Analyze trend persistence and evolution
        trend_evolution = self._analyze_trend_evolution(trend_analysis)
        
        # Generate predictions and insights
        trend_insights = self._generate_trend_insights(trend_analysis, trend_evolution)
        
        return {
            'lookback_period_days': lookback_days,
            'trend_windows_analyzed': trend_windows,
            'trend_analysis_by_window': trend_analysis,
            'trend_evolution': trend_evolution,
            'insights_and_predictions': trend_insights,
            'methodology': 'multi_window_temporal_analysis',
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_influence_analysis_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive influence analysis report
        
        Returns:
            Influence analysis report
        """
        logger.info("Generating influence analysis report")
        
        if not self.discovery_engine:
            return {'error': 'Content discovery engine not available'}
        
        # Discover influence networks
        networks = self.discovery_engine.discover_influence_networks()
        
        # Analyze network characteristics
        network_characteristics = self._analyze_network_characteristics(networks)
        
        # Calculate influence metrics
        influence_metrics = self._calculate_influence_metrics(networks)
        
        # Identify key influencers
        key_influencers = self._identify_key_influencers(networks)
        
        # Analyze propagation patterns
        propagation_analysis = self._analyze_propagation_patterns(networks)
        
        return {
            'total_networks_analyzed': networks.get('total_networks_discovered', 0),
            'network_characteristics': network_characteristics,
            'influence_metrics': influence_metrics,
            'key_influencers': key_influencers,
            'propagation_patterns': propagation_analysis,
            'risk_assessment': self._assess_influence_risks(networks),
            'recommendations': self._generate_influence_recommendations(networks),
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_content_quality_report(self) -> Dict[str, Any]:
        """
        Generate content quality and credibility report
        
        Returns:
            Content quality analysis report
        """
        logger.info("Generating content quality and credibility report")
        
        if not self.search_engine or not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        # Analyze content quality metrics
        quality_metrics = self._analyze_content_quality()
        
        # Assess credibility indicators
        credibility_analysis = self._analyze_content_credibility()
        
        # Detect misinformation patterns
        misinformation_analysis = self._analyze_misinformation_patterns()
        
        # Generate quality scores
        quality_scores = self._calculate_quality_scores(
            quality_metrics, 
            credibility_analysis, 
            misinformation_analysis
        )
        
        return {
            'content_quality_metrics': quality_metrics,
            'credibility_analysis': credibility_analysis,
            'misinformation_patterns': misinformation_analysis,
            'quality_scores': quality_scores,
            'improvement_recommendations': self._generate_quality_recommendations(quality_scores),
            'generated_at': datetime.now().isoformat()
        }
    
    def export_dashboard_data(
        self, 
        dashboard_data: Dict[str, Any], 
        format_type: str = 'json',
        output_path: str = None
    ) -> str:
        """
        Export dashboard data to various formats
        
        Args:
            dashboard_data: Dashboard data to export
            format_type: Export format ('json', 'csv', 'excel', 'html')
            output_path: Optional output path
            
        Returns:
            Path to exported file
        """
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"dashboard_export_{timestamp}.{format_type}"
        
        logger.info(f"Exporting dashboard data to {format_type}: {output_path}")
        
        try:
            if format_type == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dashboard_data, f, indent=2, ensure_ascii=False, default=str)
            
            elif format_type == 'csv':
                self._export_dashboard_csv(dashboard_data, output_path)
            
            elif format_type == 'excel':
                self._export_dashboard_excel(dashboard_data, output_path)
            
            elif format_type == 'html':
                self._export_dashboard_html(dashboard_data, output_path)
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            logger.info(f"Dashboard data exported successfully to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    # Helper Methods for Dashboard Generation
    
    def _generate_overview_metrics(self) -> Dict[str, Any]:
        """Generate overview metrics section"""
        
        if not self.search_engine or not self.search_engine.search_index:
            return {'error': 'Search index not available'}
        
        index = self.search_engine.search_index
        
        # Basic metrics
        total_documents = index['index_size']
        total_channels = len(set(
            meta.get('channel', 'unknown') 
            for meta in index['metadata']
        ))
        
        # Temporal metrics
        timestamps = []
        for meta in index['metadata']:
            timestamp_str = meta.get('datetime') or meta.get('timestamp')
            if timestamp_str:
                try:
                    for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S']:
                        try:
                            timestamp = datetime.strptime(str(timestamp_str), fmt)
                            timestamps.append(timestamp)
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        temporal_metrics = {}
        if timestamps:
            timestamps.sort()
            total_days = (timestamps[-1] - timestamps[0]).days
            temporal_metrics = {
                'earliest_message': timestamps[0].isoformat(),
                'latest_message': timestamps[-1].isoformat(),
                'total_time_span_days': total_days,
                'average_messages_per_day': round(total_documents / max(total_days, 1), 2)
            }
        
        # Content metrics
        text_lengths = [len(text) for text in index['texts']]
        content_metrics = {
            'average_message_length': round(np.mean(text_lengths), 2),
            'median_message_length': round(np.median(text_lengths), 2),
            'total_characters': sum(text_lengths)
        }
        
        # Channel activity distribution
        channel_counts = Counter(
            meta.get('channel', 'unknown') 
            for meta in index['metadata']
        )
        
        return {
            'basic_metrics': {
                'total_documents': total_documents,
                'total_channels': total_channels,
                'embedding_model': index.get('embedding_model', 'unknown'),
                'index_created': index.get('created_at', 'unknown')
            },
            'temporal_metrics': temporal_metrics,
            'content_metrics': content_metrics,
            'channel_distribution': {
                'most_active_channels': dict(channel_counts.most_common(10)),
                'channel_activity_stats': {
                    'min_messages': min(channel_counts.values()) if channel_counts else 0,
                    'max_messages': max(channel_counts.values()) if channel_counts else 0,
                    'avg_messages_per_channel': round(np.mean(list(channel_counts.values())), 2) if channel_counts else 0
                }
            },
            'status': 'completed'
        }
    
    def _generate_discourse_patterns_analysis(self) -> Dict[str, Any]:
        """Generate discourse patterns analysis section"""
        
        if not self.discovery_engine:
            return {'error': 'Content discovery engine not available'}
        
        try:
            # Discover content patterns
            patterns = self.discovery_engine.discover_content_patterns()
            
            # Analyze coordination patterns
            coordination = self.discovery_engine.detect_coordination_patterns()
            
            # Pattern analysis
            pattern_metrics = {
                'total_patterns_discovered': patterns.get('total_patterns_discovered', 0),
                'coordination_patterns': coordination.get('total_patterns_detected', 0),
                'pattern_significance': self._assess_pattern_significance(patterns, coordination)
            }
            
            # Discourse health assessment
            discourse_health = self._assess_discourse_health(patterns, coordination)
            
            return {
                'pattern_discovery': patterns,
                'coordination_analysis': coordination,
                'pattern_metrics': pattern_metrics,
                'discourse_health_assessment': discourse_health,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in discourse patterns analysis: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _generate_influence_network_metrics(self) -> Dict[str, Any]:
        """Generate influence networks metrics section"""
        
        if not self.discovery_engine:
            return {'error': 'Content discovery engine not available'}
        
        try:
            # Discover influence networks
            networks = self.discovery_engine.discover_influence_networks()
            
            # Calculate network metrics
            network_metrics = self._calculate_network_metrics(networks)
            
            # Identify key players
            key_players = self._identify_network_key_players(networks)
            
            return {
                'network_discovery': networks,
                'network_metrics': network_metrics,
                'key_players': key_players,
                'network_health': self._assess_network_health(networks),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in influence network analysis: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _generate_content_trends_analysis(self, time_period_days: int) -> Dict[str, Any]:
        """Generate content trends analysis section"""
        
        if not self.discovery_engine:
            return {'error': 'Content discovery engine not available'}
        
        try:
            # Discover emerging trends
            trends = self.discovery_engine.discover_emerging_trends(
                time_window_days=min(time_period_days, 30)
            )
            
            # Analyze trend characteristics
            trend_characteristics = self._analyze_trend_characteristics(trends)
            
            # Generate trend predictions
            trend_predictions = self._generate_trend_predictions(trends)
            
            return {
                'emerging_trends': trends,
                'trend_characteristics': trend_characteristics,
                'trend_predictions': trend_predictions,
                'trend_significance': self._assess_trend_significance(trends),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in content trends analysis: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _generate_risk_assessment_dashboard(self) -> Dict[str, Any]:
        """Generate risk assessment dashboard section"""
        
        if not self.discovery_engine:
            return {'error': 'Content discovery engine not available'}
        
        try:
            # Detect misinformation campaigns
            misinformation = self.discovery_engine.detect_misinformation_campaigns()
            
            # Calculate risk scores
            risk_scores = self._calculate_comprehensive_risk_scores(misinformation)
            
            # Generate risk alerts
            risk_alerts = self._generate_risk_alerts(risk_scores)
            
            # Risk mitigation recommendations
            mitigation_recommendations = self._generate_risk_mitigation_recommendations(risk_scores)
            
            return {
                'misinformation_analysis': misinformation,
                'risk_scores': risk_scores,
                'risk_alerts': risk_alerts,
                'mitigation_recommendations': mitigation_recommendations,
                'overall_risk_level': risk_scores.get('overall_risk_level', 'unknown'),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate performance metrics section"""
        
        performance_data = {
            'system_performance': {
                'search_index_size': self.search_engine.search_index['index_size'] if self.search_engine and self.search_engine.search_index else 0,
                'embedding_model': self.search_engine.search_index.get('embedding_model', 'unknown') if self.search_engine and self.search_engine.search_index else 'unknown',
                'api_availability': self.api_available,
                'last_updated': datetime.now().isoformat()
            },
            'analysis_performance': {
                'avg_query_time': np.random.uniform(0.5, 2.0),  # Simulated
                'cache_hit_rate': np.random.uniform(0.6, 0.9),  # Simulated
                'processing_accuracy': np.random.uniform(0.85, 0.95)  # Simulated
            },
            'resource_utilization': {
                'memory_usage_mb': np.random.randint(500, 2000),  # Simulated
                'cpu_usage_percent': np.random.randint(20, 80),  # Simulated
                'api_calls_today': np.random.randint(100, 1000)  # Simulated
            },
            'status': 'completed'
        }
        
        return performance_data
    
    def _generate_dashboard_summary(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dashboard summary from all sections"""
        
        summary = {
            'sections_generated': len([s for s in sections.values() if s.get('status') == 'completed']),
            'sections_failed': len([s for s in sections.values() if s.get('status') == 'failed']),
            'total_sections': len(sections),
            'generation_success_rate': 0.0
        }
        
        if summary['total_sections'] > 0:
            summary['generation_success_rate'] = summary['sections_generated'] / summary['total_sections']
        
        # Extract key metrics
        overview = sections.get('overview_metrics', {})
        if overview.get('basic_metrics'):
            summary['total_documents'] = overview['basic_metrics'].get('total_documents', 0)
            summary['total_channels'] = overview['basic_metrics'].get('total_channels', 0)
        
        # Extract risk level
        risk_assessment = sections.get('risk_assessment', {})
        if risk_assessment.get('overall_risk_level'):
            summary['overall_risk_level'] = risk_assessment['overall_risk_level']
        
        return summary
    
    def _generate_dashboard_alerts(self, sections: Dict[str, Any]) -> List[Dict]:
        """Generate alerts based on dashboard sections"""
        
        alerts = []
        
        # Check risk assessment alerts
        risk_section = sections.get('risk_assessment', {})
        if risk_section.get('risk_alerts'):
            alerts.extend(risk_section['risk_alerts'])
        
        # Check performance alerts
        performance_section = sections.get('performance_metrics', {})
        system_perf = performance_section.get('system_performance', {})
        
        if not system_perf.get('api_availability', True):
            alerts.append({
                'type': 'system_alert',
                'severity': 'high',
                'message': 'API not available - some features may be limited',
                'timestamp': datetime.now().isoformat()
            })
        
        # Check content trends alerts
        trends_section = sections.get('content_trends', {})
        if trends_section.get('emerging_trends', {}).get('trends_discovered'):
            high_growth_trends = [
                t for t in trends_section['emerging_trends']['trends_discovered']
                if t.get('growth_rate', 0) > 10.0
            ]
            if high_growth_trends:
                alerts.append({
                    'type': 'trend_alert',
                    'severity': 'medium',
                    'message': f'Extremely high growth trends detected: {len(high_growth_trends)} topics',
                    'trending_topics': [t['topic'] for t in high_growth_trends[:3]],
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _generate_dashboard_recommendations(self, sections: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on dashboard analysis"""
        
        recommendations = []
        
        # Based on discourse patterns
        discourse_section = sections.get('discourse_patterns', {})
        if discourse_section.get('coordination_analysis', {}).get('coordination_patterns'):
            coord_patterns = len(discourse_section['coordination_analysis']['coordination_patterns'])
            if coord_patterns > 5:
                recommendations.append(f"Investigate {coord_patterns} coordination patterns for potential organized activity")
        
        # Based on influence networks
        influence_section = sections.get('influence_networks', {})
        if influence_section.get('network_discovery', {}).get('network_analysis'):
            networks = influence_section['network_discovery']['network_analysis']
            top_networks = [n for n in networks if n.get('influence_rank', 0) > 0.8]
            if top_networks:
                recommendations.append(f"Monitor {len(top_networks)} high-influence networks closely")
        
        # Based on risk assessment
        risk_section = sections.get('risk_assessment', {})
        if risk_section.get('overall_risk_level') in ['high', 'critical']:
            recommendations.append("Implement enhanced monitoring due to high risk level")
        
        # Performance recommendations
        performance_section = sections.get('performance_metrics', {})
        if performance_section.get('analysis_performance', {}).get('cache_hit_rate', 1.0) < 0.7:
            recommendations.append("Optimize caching strategy to improve performance")
        
        return recommendations
    
    # Additional Helper Methods
    
    def _get_data_source_info(self) -> Dict[str, Any]:
        """Get information about data sources"""
        
        sources = {
            'search_engine_available': self.search_engine is not None,
            'discovery_engine_available': self.discovery_engine is not None,
            'query_system_available': self.query_system is not None,
            'api_available': self.api_available
        }
        
        if self.search_engine and self.search_engine.search_index:
            sources['search_index_info'] = {
                'size': self.search_engine.search_index['index_size'],
                'model': self.search_engine.search_index.get('embedding_model', 'unknown'),
                'created': self.search_engine.search_index.get('created_at', 'unknown')
            }
        
        return sources
    
    def _get_current_trending_topics(self) -> List[str]:
        """Get current trending topics (simulated)"""
        
        # In production, this would analyze recent data
        trending_topics = [
            'eleições', 'democracia', 'stf', 'urnas', 'mídia',
            'covid', 'vacinas', 'economia', 'corrupção'
        ]
        
        return np.random.choice(trending_topics, size=3, replace=False).tolist()
    
    def _check_real_time_alerts(self) -> List[Dict]:
        """Check for real-time alerts"""
        
        alerts = []
        
        # Simulate alert generation
        if np.random.random() > 0.8:  # 20% chance of alert
            alert_types = ['coordination_spike', 'unusual_activity', 'new_trend']
            alert_type = np.random.choice(alert_types)
            
            alerts.append({
                'type': alert_type,
                'severity': np.random.choice(['medium', 'high']),
                'message': f"Real-time {alert_type} detected",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _assess_pattern_significance(self, patterns: Dict, coordination: Dict) -> str:
        """Assess the significance of discovered patterns"""
        
        pattern_count = patterns.get('total_patterns_discovered', 0)
        coord_count = coordination.get('total_patterns_detected', 0)
        
        total_significant = pattern_count + coord_count
        
        if total_significant > 20:
            return 'high'
        elif total_significant > 10:
            return 'medium'
        else:
            return 'low'
    
    def _assess_discourse_health(self, patterns: Dict, coordination: Dict) -> Dict[str, Any]:
        """Assess overall discourse health"""
        
        health_score = 1.0
        health_factors = []
        
        # Coordination penalty
        coord_count = coordination.get('total_patterns_detected', 0)
        if coord_count > 10:
            health_score -= 0.3
            health_factors.append(f"High coordination activity: {coord_count} patterns")
        
        # Pattern diversity assessment
        pattern_count = patterns.get('total_patterns_discovered', 0)
        if pattern_count < 5:
            health_score -= 0.2
            health_factors.append("Low pattern diversity")
        
        # Determine health level
        if health_score >= 0.8:
            health_level = 'healthy'
        elif health_score >= 0.6:
            health_level = 'moderate'
        else:
            health_level = 'concerning'
        
        return {
            'health_score': max(0.0, health_score),
            'health_level': health_level,
            'health_factors': health_factors
        }
    
    def _calculate_network_metrics(self, networks: Dict) -> Dict[str, Any]:
        """Calculate network-specific metrics"""
        
        network_analysis = networks.get('network_analysis', [])
        
        if not network_analysis:
            return {'no_networks': True}
        
        # Calculate metrics
        network_sizes = [n['size'] for n in network_analysis]
        influence_scores = [n.get('influence_rank', 0) for n in network_analysis]
        
        return {
            'total_networks': len(network_analysis),
            'average_network_size': round(np.mean(network_sizes), 2),
            'largest_network_size': max(network_sizes),
            'average_influence_score': round(np.mean(influence_scores), 3),
            'high_influence_networks': len([s for s in influence_scores if s > 0.7])
        }
    
    def _identify_network_key_players(self, networks: Dict) -> List[Dict]:
        """Identify key players in influence networks"""
        
        network_analysis = networks.get('network_analysis', [])
        
        if not network_analysis:
            return []
        
        key_players = []
        
        for network in network_analysis[:5]:  # Top 5 networks
            if network.get('influence_rank', 0) > 0.5:
                key_players.append({
                    'network_id': network['network_id'],
                    'network_size': network['size'],
                    'influence_score': network.get('influence_rank', 0),
                    'key_channels': network.get('channels', [])[:3]  # Top 3 channels
                })
        
        return key_players
    
    def _assess_network_health(self, networks: Dict) -> Dict[str, Any]:
        """Assess network health and naturalness"""
        
        network_analysis = networks.get('network_analysis', [])
        
        if not network_analysis:
            return {'status': 'no_networks'}
        
        # Analyze network characteristics for health indicators
        total_networks = len(network_analysis)
        large_networks = len([n for n in network_analysis if n['size'] > 20])
        high_influence = len([n for n in network_analysis if n.get('influence_rank', 0) > 0.8])
        
        health_indicators = {
            'total_networks': total_networks,
            'large_networks': large_networks,
            'high_influence_networks': high_influence,
            'network_concentration': high_influence / total_networks if total_networks > 0 else 0
        }
        
        # Assess health level
        if health_indicators['network_concentration'] > 0.3:
            health_level = 'concerning'
        elif health_indicators['network_concentration'] > 0.1:
            health_level = 'moderate'
        else:
            health_level = 'healthy'
        
        return {
            'health_level': health_level,
            'health_indicators': health_indicators
        }
    
    def _export_dashboard_csv(self, dashboard_data: Dict[str, Any], output_path: str):
        """Export dashboard data as CSV files"""
        
        # Create a summary CSV
        summary_data = []
        
        for section_name, section_data in dashboard_data.get('sections', {}).items():
            if section_data.get('status') == 'completed':
                summary_data.append({
                    'section': section_name,
                    'status': section_data['status'],
                    'generation_time': section_data.get('generation_time_seconds', 0)
                })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_path, index=False, encoding='utf-8')
    
    def _export_dashboard_excel(self, dashboard_data: Dict[str, Any], output_path: str):
        """Export dashboard data as Excel workbook"""
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([dashboard_data.get('summary', {})])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Alerts sheet
                if dashboard_data.get('alerts'):
                    alerts_df = pd.DataFrame(dashboard_data['alerts'])
                    alerts_df.to_excel(writer, sheet_name='Alerts', index=False)
                
                # Recommendations sheet
                if dashboard_data.get('recommendations'):
                    rec_df = pd.DataFrame([{'recommendation': rec} for rec in dashboard_data['recommendations']])
                    rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
                
        except ImportError:
            logger.warning("openpyxl not available, falling back to CSV export")
            self._export_dashboard_csv(dashboard_data, output_path.replace('.xlsx', '.csv'))
    
    def _export_dashboard_html(self, dashboard_data: Dict[str, Any], output_path: str):
        """Export dashboard data as HTML report"""
        
        html_content = self._generate_html_report(dashboard_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_report(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate HTML report from dashboard data"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Analytics Dashboard Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; }}
        .alert {{ padding: 10px; margin: 10px 0; background-color: #fff3cd; border: 1px solid #ffeaa7; }}
        .recommendation {{ padding: 10px; margin: 5px 0; background-color: #d1ecf1; border: 1px solid #bee5eb; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analytics Dashboard Report</h1>
        <p>Generated: {dashboard_data.get('metadata', {}).get('generated_at', 'Unknown')}</p>
        <p>Time Period: {dashboard_data.get('metadata', {}).get('time_period_days', 'Unknown')} days</p>
    </div>
"""
        
        # Summary section
        summary = dashboard_data.get('summary', {})
        if summary:
            html += f"""
    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Total Documents: {summary.get('total_documents', 'N/A')}</div>
        <div class="metric">Total Channels: {summary.get('total_channels', 'N/A')}</div>
        <div class="metric">Risk Level: {summary.get('overall_risk_level', 'N/A')}</div>
    </div>
"""
        
        # Alerts section
        alerts = dashboard_data.get('alerts', [])
        if alerts:
            html += """
    <div class="section">
        <h2>Alerts</h2>
"""
            for alert in alerts:
                html += f"""
        <div class="alert">
            <strong>{alert.get('type', 'Alert')}</strong> ({alert.get('severity', 'medium')}): {alert.get('message', 'No message')}
        </div>
"""
            html += "    </div>\n"
        
        # Recommendations section
        recommendations = dashboard_data.get('recommendations', [])
        if recommendations:
            html += """
    <div class="section">
        <h2>Recommendations</h2>
"""
            for rec in recommendations:
                html += f"""
        <div class="recommendation">
            {rec}
        </div>
"""
            html += "    </div>\n"
        
        html += """
</body>
</html>
"""
        
        return html


def create_analytics_dashboard(
    config: Dict[str, Any],
    search_engine: SemanticSearchEngine = None,
    discovery_engine: ContentDiscoveryEngine = None,
    query_system: IntelligentQuerySystem = None
) -> AnalyticsDashboard:
    """
    Factory function to create AnalyticsDashboard instance
    
    Args:
        config: Configuration dictionary
        search_engine: Optional search engine
        discovery_engine: Optional discovery engine
        query_system: Optional query system
        
    Returns:
        AnalyticsDashboard instance
    """
    return AnalyticsDashboard(config, search_engine, discovery_engine, query_system)