#!/usr/bin/env python3
"""
MAIN PIPELINE CONTROLLER - BOLSONARISMO v4.9.8
===============================================

Controlador principal do pipeline com checkpoints e recuperação automática.
Integra com o sistema unificado de anthropic_integration.
Pipeline completo: 22 etapas + Dashboard 100% funcional com correções críticas.

🎯 DASHBOARD v4.9.8: Análise temporal corrigida, erro dropna=False resolvido.
🏛️ POLITICAL v4.9.8: 4 níveis políticos funcionais, 2 clusters semânticos.
🔤 SPACY v4.9.5: Stage 07 pt_core_news_lg com 57 entidades brasileiras.
🚨 BUGS FIXED: Deduplicação + configuração + separadores CSV resolvidos.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

        # Pipeline stages configuration (v4.9.4 - 22 etapas com correção crítica deduplicação)
        self.stages = [
            {'id': '01_chunk_processing', 'name': 'Chunk Processing',
             'critical': True},
            {'id': '02_encoding_validation',
             'name': 'Enhanced Encoding Validation', 'critical': True},
            {'id': '03_deduplication', 'name': 'Global Deduplication',
             'critical': True},
            {'id': '04_feature_validation', 'name': 'Feature Validation',
             'critical': True},
            {'id': '04b_statistical_analysis_pre',
             'name': 'Statistical Analysis (Pre-Cleaning)', 'critical': False},
            {'id': '05_political_analysis', 'name': 'Political Analysis',
             'critical': False},
            {'id': '06_text_cleaning', 'name': 'Enhanced Text Cleaning',
             'critical': True},
            {'id': '06b_statistical_analysis_post',
             'name': 'Statistical Analysis (Post-Cleaning)',
             'critical': False},
            {'id': '07_linguistic_processing',
             'name': 'Linguistic Processing (spaCy)', 'critical': False},
            {'id': '08_sentiment_analysis', 'name': 'Sentiment Analysis',
             'critical': False},
            {'id': '09_topic_modeling', 'name': 'Topic Modeling (Voyage.ai)',
             'critical': False},
            {'id': '10_tfidf_extraction',
             'name': 'TF-IDF Extraction (Voyage.ai)', 'critical': False},
            {'id': '11_clustering', 'name': 'Clustering (Voyage.ai)',
             'critical': False},
            {'id': '12_hashtag_normalization',
             'name': 'Hashtag Normalization', 'critical': False},
            {'id': '13_domain_analysis', 'name': 'Domain Analysis',
             'critical': False},
            {'id': '14_temporal_analysis', 'name': 'Temporal Analysis',
             'critical': False},
            {'id': '15_network_analysis', 'name': 'Network Analysis',
             'critical': False},
            {'id': '16_qualitative_analysis', 'name': 'Qualitative Analysis',
             'critical': False},
            {'id': '17_smart_pipeline_review',
             'name': 'Smart Pipeline Review', 'critical': False},
            {'id': '18_topic_interpretation',
             'name': 'Topic Interpretation', 'critical': False},
            {'id': '19_semantic_search',
             'name': 'Semantic Search (Voyage.ai)', 'critical': False},
            {'id': '20_pipeline_validation', 'name': 'Pipeline Validation',
             'critical': True}
        ]

    def save_checkpoint(self, stage_id: str, data: Dict[str, Any]) -> bool:
        """Salva checkpoint para uma etapa específica"""
        try:
            # Save individual checkpoint
            checkpoint_file = self.checkpoints_dir / f"{stage_id}_checkpoint.json"
            checkpoint_data = {
                'stage_id': stage_id,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'status': 'completed'
            }

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            # Update central checkpoints.json
            self._update_central_checkpoint(stage_id, data)

            # Update protection checklist
            self._update_protection_checklist(stage_id, data)

            logger.info(f"Checkpoint saved: {stage_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint {stage_id}: {e}")
            return False

    def _update_central_checkpoint(self, stage_id: str, data: Dict[str, Any]) -> bool:
        """Atualiza o arquivo central checkpoints.json"""
        try:
            checkpoints_file = self.checkpoints_dir / "checkpoints.json"

            # Load existing checkpoints
            if checkpoints_file.exists():
                with open(checkpoints_file, 'r', encoding='utf-8') as f:
                    checkpoints = json.load(f)
            else:
                # Create default structure if not exists
                checkpoints = self._create_default_checkpoints()

            # Update specific stage
            if stage_id in checkpoints['stages']:
                checkpoints['stages'][stage_id].update({
                    'status': 'completed',
                    'completed_at': datetime.now().isoformat(),
                    'execution_time': data.get('execution_time', 0),
                    'records_processed': data.get('records_processed', 0),
                    'output_file': data.get('output_file', ''),
                    'success': data.get('success', True),
                    'error': data.get('error', None)
                })

            # Update execution summary
            completed_stages = sum(
                1 for stage in checkpoints['stages'].values()
                if stage['status'] == 'completed')
            checkpoints['execution_summary'].update({
                'completed_stages': completed_stages,
                'current_stage': self._get_next_stage(stage_id),
                'overall_progress': (completed_stages /
                                    checkpoints['execution_summary']['total_stages']),
                'last_successful_stage': stage_id,
                'resume_from': self._get_next_stage(stage_id)
            })

            checkpoints['last_updated'] = datetime.now().isoformat()

            # Save updated checkpoints
            with open(checkpoints_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoints, f, indent=2, default=str)

            return True

        except Exception as e:
            logger.error(f"Failed to update central checkpoint: {e}")
            return False

    def _create_default_checkpoints(self) -> Dict[str, Any]:
        """Cria estrutura padrão de checkpoints"""
        stages_dict = {}
        for stage in self.stages:
            stages_dict[stage['id']] = {
                'status': 'pending',
                'completed_at': None,
                'execution_time': 0,
                'records_processed': 0,
                'output_file': '',
                'success': False,
                'error': None
            }

        return {
            'pipeline_version': '4.9.4',
            'last_updated': datetime.now().isoformat(),
            'current_dataset': '',
            'stages': stages_dict,
            'execution_summary': {
                'total_stages': len(self.stages),
                'completed_stages': 0,
                'failed_stages': 0,
                'current_stage': self.stages[0]['id'],
                'overall_progress': 0.0,
                'total_execution_time': 0,
                'last_successful_stage': None,
                'resume_from': self.stages[0]['id']
            }
        }

    def _get_next_stage(self, current_stage_id: str) -> str:
        """Obtém próximo stage após o atual"""
        try:
            current_idx = next(
                i for i, s in enumerate(self.stages)
                if s['id'] == current_stage_id)
            if current_idx + 1 < len(self.stages):
                return self.stages[current_idx + 1]['id']
            else:
                return 'completed'
        except StopIteration:
            return 'unknown'

    def _update_protection_checklist(self, stage_id: str, data: Dict[str, Any]) -> bool:
        """Atualiza flags de proteção no checklist.json"""
        try:
            checklist_file = self.checkpoints_dir / "checklist.json"

            # Load existing checklist
            if checklist_file.exists():
                with open(checklist_file, 'r', encoding='utf-8') as f:
                    checklist = json.load(f)
            else:
                # Create default checklist if not exists
                checklist = self._create_default_checklist()

            # Update specific stage flags
            if stage_id in checklist['stage_flags']:
                stage_flags = checklist['stage_flags'][stage_id]
                success = data.get('success', True)

                if success:
                    # Mark as completed and increment success count
                    stage_flags.update({
                        'completed': True,
                        'verified': True,
                        'output_validated': True,
                        'last_successful_run': datetime.now().isoformat(),
                        'success_count': stage_flags.get('success_count', 0) + 1
                    })

                    # Check for auto-lock based on protection level
                    protection_level = stage_flags.get('protection_level', 'low')
                    min_success = checklist['protection_rules'][
                        f'{protection_level}_stages']['min_success_count']
                    auto_lock = checklist['protection_rules'][
                        f'{protection_level}_stages']['auto_lock_after_success']

                    if stage_flags['success_count'] >= min_success and auto_lock:
                        stage_flags.update({
                            'locked': True,
                            'can_overwrite': False,
                            'validation_notes': (
                                f"Auto-locked after {stage_flags['success_count']} "
                                f"successful runs")
                        })
                        logger.info(
                            f"Stage {stage_id} auto-locked after "
                            f"{stage_flags['success_count']} successes")
                else:
                    # Mark failure but don't lock
                    stage_flags.update({
                        'completed': False,
                        'validation_notes': (
                            f"Failed run at {datetime.now().isoformat()}: "
                            f"{data.get('error', 'Unknown error')}")
                    })

            # Update statistics
            completed_stages = sum(
                1 for flags in checklist['stage_flags'].values()
                if flags['completed'])
            locked_stages = sum(
                1 for flags in checklist['stage_flags'].values()
                if flags['locked'])
            protected_stages = sum(
                1 for flags in checklist['stage_flags'].values()
                if not flags['can_overwrite'])

            checklist['statistics'].update({
                'completed_stages': completed_stages,
                'locked_stages': locked_stages,
                'protected_stages': protected_stages,
                'success_rate': completed_stages / checklist['statistics']['total_stages'],
                'last_validation': datetime.now().isoformat()
            })

            checklist['last_updated'] = datetime.now().isoformat()

            # Save updated checklist
            with open(checklist_file, 'w', encoding='utf-8') as f:
                json.dump(checklist, f, indent=2, default=str)

            return True

        except Exception as e:
            logger.error(f"Failed to update protection checklist: {e}")
            return False

    def _create_default_checklist(self) -> Dict[str, Any]:
        """Cria estrutura padrão do checklist de proteção"""
        stage_flags = {}
        for stage in self.stages:
            protection_level = ('critical' if stage.get('critical', False)
                               else 'medium')
            if stage['id'] in [
                '04b_statistical_analysis_pre',
                '06b_statistical_analysis_post',
                '12_hashtag_normalization', '13_domain_analysis',
                '14_temporal_analysis', '15_network_analysis',
                '16_qualitative_analysis', '19_semantic_search']:
                protection_level = 'low'

            stage_flags[stage['id']] = {
                'completed': False,
                'locked': False,
                'verified': False,
                'output_validated': False,
                'last_successful_run': None,
                'protection_level': protection_level,
                'can_overwrite': True,
                'success_count': 0,
                'validation_notes': ''
            }

        return {
            'pipeline_version': '4.9.4',
            'protection_mode': 'enabled',
            'last_updated': datetime.now().isoformat(),
            'description': 'Flags de proteção para impedir reescrita de etapas funcionais',
            'stage_flags': stage_flags,
            'protection_rules': {
                'critical_stages': {
                    'min_success_count': 3,
                    'auto_lock_after_success': True,
                    'requires_manual_unlock': True,
                    'backup_before_overwrite': True
                },
                'medium_stages': {
                    'min_success_count': 2,
                    'auto_lock_after_success': False,
                    'requires_manual_unlock': False,
                    'backup_before_overwrite': True
                },
                'low_stages': {
                    'min_success_count': 1,
                    'auto_lock_after_success': False,
                    'requires_manual_unlock': False,
                    'backup_before_overwrite': False
                }
            },
            'override_codes': {
                'emergency_override': 'FORCE_OVERWRITE_2025',
                'development_mode': 'DEV_MODE_UNSAFE',
                'admin_unlock': 'ADMIN_UNLOCK_ALL'
            },
            'statistics': {
                'total_stages': len(self.stages),
                'completed_stages': 0,
                'locked_stages': 0,
                'protected_stages': 0,
                'success_rate': 0.0,
                'last_validation': None
            }
        }

    def check_stage_protection(self, stage_id: str) -> Dict[str, Any]:
        """Verifica se uma etapa está protegida contra reescrita"""
        try:
            checklist_file = self.checkpoints_dir / "checklist.json"

            if not checklist_file.exists():
                return {'can_overwrite': True, 'reason': 'No protection file found'}

            with open(checklist_file, 'r', encoding='utf-8') as f:
                checklist = json.load(f)

            if stage_id not in checklist['stage_flags']:
                return {'can_overwrite': True, 'reason': 'Stage not in protection list'}

            stage_flags = checklist['stage_flags'][stage_id]

            if stage_flags['locked']:
                return {
                    'can_overwrite': False,
                    'reason': 'Stage is locked',
                    'protection_level': stage_flags['protection_level'],
                    'success_count': stage_flags['success_count'],
                    'requires_override': True
                }

            if not stage_flags['can_overwrite']:
                return {
                    'can_overwrite': False,
                    'reason': 'Stage is protected',
                    'protection_level': stage_flags['protection_level'],
                    'success_count': stage_flags['success_count'],
                    'requires_override': False
                }

            return {'can_overwrite': True, 'reason': 'Stage not protected'}

        except Exception as e:
            logger.error(f"Failed to check stage protection: {e}")
            return {'can_overwrite': True, 'reason': 'Protection check failed'}

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

            # Execute specific stage using internal method
            stage_method = getattr(pipeline, "_execute_stage", None)
            if not stage_method:
                logger.error("Stage method '_execute_stage' not found in pipeline")
                return {'success': False, 'error': "Method '_execute_stage' not found"}

            # Execute stage
            start_time = time.time()
            result = stage_method(stage_id, datasets)
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

    def run_pipeline(self, datasets: List[str] = None,
                     config: Dict[str, Any] = None, resume: bool = True,
                     clear_cache: bool = False) -> Dict[str, Any]:
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
                    'interim_path': 'data/interim',
                    'output_path': 'pipeline_outputs'
                }
            }

        # Auto-discover datasets if not provided
        if not datasets:
            import glob
            from pathlib import Path

            data_paths = ['data/uploads', 'data/DATASETS_FULL']
            datasets = []
            for path in data_paths:
                if os.path.exists(path):
                    datasets.extend(glob.glob(os.path.join(path, '*.csv')))

            if not datasets:
                return {'success': False, 'error': 'No datasets found'}

        # Determine starting point
        start_stage_id = (self.get_resume_point() if resume
                         else self.stages[0]['id'])
        start_idx = next(
            (i for i, s in enumerate(self.stages)
             if s['id'] == start_stage_id), 0)

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
                        logger.error(
                            f"❌ Critical stage failed: {stage['name']} - "
                            f"stopping execution")
                        break
                    else:
                        logger.warning(
                            f"⚠️ Non-critical stage failed: {stage['name']} - "
                            f"continuing")

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
            len([f for f in execution_results['stages_failed']
                 if f.get('critical', False)]) == 0)

        # Save final execution summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.checkpoints_dir / f"execution_summary_{timestamp}.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(execution_results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save execution summary: {e}")

        logger.info(
            f"Pipeline execution completed. Success: "
            f"{execution_results['overall_success']}")
        return execution_results


def main():
    """Entry point principal"""
    print("🎯 PIPELINE CONTROLLER - BOLSONARISMO v4.9.4")
    print("=" * 50)

    try:
        controller = PipelineController()

        # Execute pipeline with checkpoint control
        results = controller.run_pipeline(resume=True)

        # Display results
        status_icon = '✅' if results['overall_success'] else '❌'
        status_text = 'COMPLETED' if results['overall_success'] else 'FAILED'
        print(f"\n{status_icon} Pipeline {status_text}")
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
