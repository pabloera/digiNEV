#!/usr/bin/env python3
"""
MAIN PIPELINE CONTROLLER - BOLSONARISMO v4.6
============================================

Controlador principal do pipeline com checkpoints e recuperação automática.
Integra com o sistema unificado de anthropic_integration.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineController:
    """Controlador principal do pipeline com sistema de checkpoints"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or Path.cwd())
        self.checkpoints_dir = self.base_path / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Pipeline stages configuration
        self.stages = [
            {'id': '01_chunk_processing', 'name': 'Chunk Processing', 'critical': True},
            {'id': '02a_encoding_validation', 'name': 'Encoding Validation', 'critical': True},
            {'id': '02b_deduplication', 'name': 'Deduplication', 'critical': True},
            {'id': '01b_features_validation', 'name': 'Features Validation', 'critical': True},
            {'id': '01c_political_analysis', 'name': 'Political Analysis', 'critical': False},
            {'id': '03_text_cleaning', 'name': 'Text Cleaning', 'critical': True},
            {'id': '04_sentiment_analysis', 'name': 'Sentiment Analysis', 'critical': False},
            {'id': '05_topic_modeling', 'name': 'Topic Modeling', 'critical': False},
            {'id': '06_tfidf_extraction', 'name': 'TF-IDF Extraction', 'critical': False},
            {'id': '07_clustering', 'name': 'Clustering', 'critical': False},
            {'id': '08_hashtag_normalization', 'name': 'Hashtag Normalization', 'critical': False},
            {'id': '09_domain_analysis', 'name': 'Domain Analysis', 'critical': False},
            {'id': '10_temporal_analysis', 'name': 'Temporal Analysis', 'critical': False},
            {'id': '11_network_analysis', 'name': 'Network Analysis', 'critical': False},
            {'id': '12_qualitative_analysis', 'name': 'Qualitative Analysis', 'critical': False},
            {'id': '13_smart_pipeline_review', 'name': 'Smart Pipeline Review', 'critical': False},
            {'id': '14_topic_interpretation', 'name': 'Topic Interpretation', 'critical': False},
            {'id': '15_semantic_search', 'name': 'Semantic Search', 'critical': False},
            {'id': '16_pipeline_validation', 'name': 'Pipeline Validation', 'critical': True}
        ]
        
    def save_checkpoint(self, stage_id: str, data: Dict[str, Any]) -> bool:
        """Salva checkpoint para uma etapa específica"""
        try:
            checkpoint_file = self.checkpoints_dir / f"{stage_id}_checkpoint.json"
            checkpoint_data = {
                'stage_id': stage_id,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'status': 'completed'
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved: {stage_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {stage_id}: {e}")
            return False
    
    def load_checkpoint(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """Carrega checkpoint de uma etapa específica"""
        try:
            checkpoint_file = self.checkpoints_dir / f"{stage_id}_checkpoint.json"
            
            if not checkpoint_file.exists():
                return None
                
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            if checkpoint_data.get('status') == 'completed':
                logger.info(f"Checkpoint loaded: {stage_id}")
                return checkpoint_data
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint {stage_id}: {e}")
            
        return None
    
    def clear_checkpoints(self) -> bool:
        """Limpa todos os checkpoints existentes"""
        try:
            for checkpoint_file in self.checkpoints_dir.glob("*_checkpoint.json"):
                checkpoint_file.unlink()
            logger.info("All checkpoints cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear checkpoints: {e}")
            return False
    
    def get_resume_point(self) -> str:
        """Determina a partir de qual etapa retomar a execução"""
        completed_stages = []
        
        for stage in self.stages:
            if self.load_checkpoint(stage['id']):
                completed_stages.append(stage['id'])
            else:
                break
                
        if completed_stages:
            last_completed = completed_stages[-1]
            next_stage_idx = next((i for i, s in enumerate(self.stages) if s['id'] == last_completed), -1) + 1
            
            if next_stage_idx < len(self.stages):
                resume_stage = self.stages[next_stage_idx]['id']
                logger.info(f"Resuming from stage: {resume_stage} (after {last_completed})")
                return resume_stage
        
        logger.info("Starting from beginning")
        return self.stages[0]['id']
    
    def run_stage(self, stage: Dict[str, Any], datasets: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Executa uma etapa específica do pipeline"""
        stage_id = stage['id']
        stage_name = stage['name']
        
        logger.info(f"Executing stage: {stage_name} ({stage_id})")
        
        try:
            # Import unified pipeline
            from anthropic_integration.unified_pipeline import UnifiedAnthropicPipeline
            
            # Create pipeline instance
            pipeline = UnifiedAnthropicPipeline(config, str(self.base_path))
            
            # Execute specific stage
            stage_method = getattr(pipeline, f"run_{stage_id}", None)
            if not stage_method:
                logger.error(f"Stage method not found: run_{stage_id}")
                return {'success': False, 'error': f"Method not found: run_{stage_id}"}
            
            # Execute stage
            start_time = time.time()
            result = stage_method(datasets)
            execution_time = time.time() - start_time
            
            # Enhance result with metadata
            if isinstance(result, dict):
                result.update({
                    'stage_id': stage_id,
                    'stage_name': stage_name,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                result = {
                    'success': True,
                    'stage_id': stage_id,
                    'stage_name': stage_name,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat(),
                    'data': result
                }
            
            # Save checkpoint if successful
            if result.get('success', True):
                self.save_checkpoint(stage_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            return {
                'success': False,
                'stage_id': stage_id,
                'stage_name': stage_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_pipeline(self, datasets: List[str] = None, config: Dict[str, Any] = None, 
                    resume: bool = True, clear_cache: bool = False) -> Dict[str, Any]:
        """Executa o pipeline completo com controle de checkpoints"""
        
        logger.info("Starting pipeline execution with checkpoint control")
        
        if clear_cache:
            self.clear_checkpoints()
        
        # Default configuration
        if not config:
            config = {
                'anthropic': {'enable_api_integration': True},
                'processing': {'chunk_size': 10000},
                'data': {
                    'path': 'data/uploads',
                    'interim_path': 'data/interim'
                }
            }
        
        # Auto-discover datasets if not provided
        if not datasets:
            from pathlib import Path
            import glob
            
            data_paths = ['data/uploads', 'data/DATASETS_FULL']
            datasets = []
            for path in data_paths:
                if os.path.exists(path):
                    datasets.extend(glob.glob(os.path.join(path, '*.csv')))
            
            if not datasets:
                return {'success': False, 'error': 'No datasets found'}
        
        # Determine starting point
        start_stage_id = self.get_resume_point() if resume else self.stages[0]['id']
        start_idx = next((i for i, s in enumerate(self.stages) if s['id'] == start_stage_id), 0)
        
        # Pipeline execution results
        execution_results = {
            'start_time': datetime.now().isoformat(),
            'datasets': [Path(d).name for d in datasets],
            'stages_executed': [],
            'stages_failed': [],
            'overall_success': False,
            'total_execution_time': 0
        }
        
        overall_start_time = time.time()
        
        # Execute stages sequentially
        for i in range(start_idx, len(self.stages)):
            stage = self.stages[i]
            
            try:
                result = self.run_stage(stage, datasets, config)
                
                if result.get('success', True):
                    execution_results['stages_executed'].append({
                        'stage_id': stage['id'],
                        'stage_name': stage['name'],
                        'execution_time': result.get('execution_time', 0),
                        'success': True
                    })
                    logger.info(f"✅ Stage completed: {stage['name']}")
                else:
                    execution_results['stages_failed'].append({
                        'stage_id': stage['id'],
                        'stage_name': stage['name'],
                        'error': result.get('error', 'Unknown error'),
                        'critical': stage.get('critical', False)
                    })
                    
                    # Stop execution if critical stage fails
                    if stage.get('critical', False):
                        logger.error(f"❌ Critical stage failed: {stage['name']} - stopping execution")
                        break
                    else:
                        logger.warning(f"⚠️ Non-critical stage failed: {stage['name']} - continuing")
                        
            except Exception as e:
                logger.error(f"❌ Unexpected error in stage {stage['name']}: {e}")
                execution_results['stages_failed'].append({
                    'stage_id': stage['id'],
                    'stage_name': stage['name'],
                    'error': str(e),
                    'critical': stage.get('critical', False)
                })
                
                if stage.get('critical', False):
                    break
        
        # Calculate final results
        execution_results['total_execution_time'] = time.time() - overall_start_time
        execution_results['end_time'] = datetime.now().isoformat()
        execution_results['overall_success'] = (
            len(execution_results['stages_executed']) > 0 and 
            len([f for f in execution_results['stages_failed'] if f.get('critical', False)]) == 0
        )
        
        # Save final execution summary
        summary_file = self.checkpoints_dir / f"execution_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(execution_results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save execution summary: {e}")
        
        logger.info(f"Pipeline execution completed. Success: {execution_results['overall_success']}")
        return execution_results

def main():
    """Entry point principal"""
    print("🎯 PIPELINE CONTROLLER - BOLSONARISMO v4.6")
    print("=" * 50)
    
    try:
        controller = PipelineController()
        
        # Execute pipeline with checkpoint control
        results = controller.run_pipeline(resume=True)
        
        # Display results
        print(f"\n{'✅' if results['overall_success'] else '❌'} Pipeline {'COMPLETED' if results['overall_success'] else 'FAILED'}")
        print(f"⏱️ Total time: {results['total_execution_time']:.1f}s")
        print(f"📊 Datasets: {len(results['datasets'])}")
        print(f"✅ Stages completed: {len(results['stages_executed'])}")
        print(f"❌ Stages failed: {len(results['stages_failed'])}")
        
        if results['stages_executed']:
            print("\n📋 Completed stages:")
            for stage in results['stages_executed']:
                print(f"   ✅ {stage['stage_name']} ({stage['execution_time']:.1f}s)")
        
        if results['stages_failed']:
            print("\n⚠️ Failed stages:")
            for stage in results['stages_failed']:
                print(f"   ❌ {stage['stage_name']}: {stage['error']}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()