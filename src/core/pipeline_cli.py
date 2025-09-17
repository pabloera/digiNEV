#!/usr/bin/env python3
"""
Pipeline CLI Wrapper - Unified Entry Point
=========================================

Simple CLI wrapper around PipelineExecutor for consistent execution
across all interfaces (run_pipeline.py, dashboard, tests)

Usage:
    python -m src.core.pipeline_cli run [datasets...]
    python -m src.core.pipeline_cli status
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pipeline_executor import PipelineExecutor


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def emit_progress(stage_id: str, status: str, progress: float = 0.0, message: str = ""):
    """Emit structured progress in JSONL format"""
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'stage_id': stage_id,
        'status': status,  # pending, running, completed, error
        'progress': progress,
        'message': message
    }
    print(json.dumps(progress_data), flush=True)


def run_pipeline(datasets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run pipeline with progress emission"""
    
    emit_progress("pipeline", "running", 0.0, "Iniciando pipeline...")
    
    try:
        # Initialize executor
        executor = PipelineExecutor()
        
        # Setup dashboard integration
        dashboard_ready = executor.setup_dashboard_integration()
        emit_progress("setup", "completed", 10.0, "Dashboard integration configurado")
        
        # Discover datasets if not provided
        if datasets is None:
            datasets = executor.discover_datasets()
            
        if not datasets:
            emit_progress("discovery", "error", 0.0, "Nenhum dataset encontrado")
            return {'overall_success': False, 'error': 'No datasets found'}
            
        emit_progress("discovery", "completed", 20.0, f"{len(datasets)} datasets encontrados")
        
        # Execute pipeline
        emit_progress("execution", "running", 30.0, "Executando pipeline...")
        results = executor.execute_pipeline(datasets)
        
        # Write results for dashboard
        if dashboard_ready and results.get('overall_success'):
            dashboard_results_dir = Path("src/dashboard/data/dashboard_results")
            dashboard_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Write latest.json
            latest_file = dashboard_results_dir / "latest.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    'overall_success': results['overall_success'],
                    'stages_completed': len(results.get('stages_completed', {})),
                    'datasets_processed': results.get('datasets_processed', []),
                    'execution_time': results.get('execution_time', 0)
                }, f, indent=2, ensure_ascii=False)
            
            emit_progress("integration", "completed", 90.0, "Resultados integrados ao dashboard")
        
        if results.get('overall_success'):
            emit_progress("pipeline", "completed", 100.0, "Pipeline concluído com sucesso")
        else:
            emit_progress("pipeline", "error", 0.0, f"Pipeline falhou: {results.get('error', 'Erro desconhecido')}")
            
        return results
        
    except Exception as e:
        emit_progress("pipeline", "error", 0.0, f"Erro crítico: {str(e)}")
        return {'overall_success': False, 'error': str(e)}


def show_status() -> Dict[str, Any]:
    """Show current pipeline status"""
    try:
        # Check latest results
        latest_file = Path("src/dashboard/data/dashboard_results/latest.json")
        if latest_file.exists():
            with open(latest_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return status
        else:
            return {'status': 'idle', 'message': 'Nenhuma execução encontrada'}
    except Exception as e:
        return {'status': 'error', 'message': f'Erro ao verificar status: {e}'}


def main():
    """Main CLI entry point"""
    setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.core.pipeline_cli {run|status} [datasets...]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "run":
        datasets = sys.argv[2:] if len(sys.argv) > 2 else None
        results = run_pipeline(datasets)
        
        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)
        
    elif command == "status":
        status = show_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
