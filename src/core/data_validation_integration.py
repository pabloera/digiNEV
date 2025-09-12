"""
Integração do Data Validator no Pipeline (Fase 2)

Este módulo integra o Data Validator como o primeiro estágio obrigatório
do pipeline, conforme especificado no plano de aprimoramento.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from ..data.data_validator import DataValidator, ValidationResult
from ..common.config_loader import get_config_loader

logger = logging.getLogger(__name__)

class DataValidationStage:
    """
    Estágio de validação de dados como primeiro passo do pipeline.
    
    Este estágio implementa o padrão "Gatekeeper" para garantir que
    apenas dados válidos sejam processados pelo pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o estágio de validação.
        
        Args:
            config: Configuração do sistema
        """
        self.config = config
        self.validator = DataValidator(config)
        
    def execute(self, datasets: List[str]) -> Dict[str, Any]:
        """
        Executa a validação de dados como primeiro estágio do pipeline.
        
        Args:
            datasets: Lista de caminhos para datasets
            
        Returns:
            Resultado da validação com datasets válidos
            
        Raises:
            ValueError: Se nenhum dataset válido for encontrado
        """
        logger.info("🔍 STAGE 0: Data Validation (Gatekeeper)")
        logger.info("=" * 50)
        
        valid_datasets = []
        all_results = {}
        
        for dataset_path in datasets:
            dataset_path_obj = Path(dataset_path)
            
            if dataset_path_obj.is_file():
                # Validar arquivo individual
                result = self.validator.validate_file_structure(dataset_path_obj)
                all_results[str(dataset_path_obj)] = result
                
                if result.is_valid:
                    valid_datasets.append(str(dataset_path_obj))
                    logger.info(f"✅ {dataset_path_obj.name}: VÁLIDO")
                else:
                    logger.warning(f"❌ {dataset_path_obj.name}: INVÁLIDO - movido para quarentena")
                    try:
                        self.validator.quarantine_file(dataset_path_obj, result)
                    except Exception as e:
                        logger.error(f"Erro ao mover {dataset_path_obj.name} para quarentena: {e}")
                        
            elif dataset_path_obj.is_dir():
                # Validar diretório
                dir_results = self.validator.validate_dataset(dataset_path_obj)
                all_results.update(dir_results)
                
                # Obter arquivos válidos do diretório
                dir_valid_files = self.validator.get_valid_files(dataset_path_obj)
                valid_datasets.extend([str(f) for f in dir_valid_files])
                
            else:
                logger.error(f"Caminho inválido: {dataset_path}")
        
        # Verificar se temos dados válidos para continuar
        if not valid_datasets:
            error_msg = (
                "❌ PIPELINE INTERROMPIDO: Nenhum dataset válido encontrado!\n"
                "Todos os arquivos falharam na validação e foram movidos para quarentena.\n"
                "Verifique o diretório de quarentena para mais detalhes."
            )
            logger.error(error_msg)
            raise ValueError("Nenhum dataset válido encontrado após validação")
        
        # Estatísticas finais
        total_files = len(all_results)
        valid_files = len(valid_datasets)
        invalid_files = total_files - valid_files
        
        logger.info("📊 RESULTADOS DA VALIDAÇÃO:")
        logger.info(f"   Total de arquivos: {total_files}")
        logger.info(f"   Arquivos válidos: {valid_files}")
        logger.info(f"   Arquivos em quarentena: {invalid_files}")
        logger.info(f"   Taxa de sucesso: {(valid_files/total_files)*100:.1f}%")
        
        return {
            'stage_id': '00_data_validation',
            'stage_name': 'Data Validation (Gatekeeper)', 
            'success': True,
            'valid_datasets': valid_datasets,
            'validation_results': all_results,
            'statistics': {
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': invalid_files,
                'success_rate': (valid_files/total_files) if total_files > 0 else 0.0
            },
            'records_processed': sum(
                result.row_count for result in all_results.values() 
                if result.is_valid
            ),
            'execution_time': 0,  # Será preenchido pelo controlador
            'output_file': str(self.validator.quarantine_dir / "validation_log.txt"),
            'message': f"Validação concluída: {valid_files}/{total_files} arquivos válidos"
        }

def integrate_data_validation_in_pipeline(original_execute_method):
    """
    Decorator para integrar validação de dados no pipeline existente.
    
    Args:
        original_execute_method: Método execute original do pipeline
        
    Returns:
        Método execute modificado com validação
    """
    def execute_with_validation(self, datasets: List[str], config: Dict[str, Any] = None):
        """
        Executa pipeline com validação de dados como primeiro estágio.
        
        Args:
            datasets: Lista de datasets
            config: Configuração do pipeline
            
        Returns:
            Resultado da execução completa
        """
        logger.info("🚀 INICIANDO PIPELINE COM VALIDAÇÃO DE DADOS")
        
        # STAGE 0: Data Validation (Gatekeeper)
        try:
            validation_stage = DataValidationStage(config or {})
            validation_result = validation_stage.execute(datasets)
            
            # Usar apenas datasets válidos para o restante do pipeline
            valid_datasets = validation_result['valid_datasets']
            
            logger.info(f"✅ Validação concluída: {len(valid_datasets)} datasets válidos")
            logger.info("🔄 Prosseguindo com pipeline principal...")
            
            # Executar pipeline original com datasets validados
            pipeline_result = original_execute_method(self, valid_datasets, config)
            
            # Adicionar resultados da validação ao resultado final
            if isinstance(pipeline_result, dict):
                pipeline_result['validation_stage'] = validation_result
                
                # Atualizar estatísticas globais
                if 'stage_results' not in pipeline_result:
                    pipeline_result['stage_results'] = {}
                pipeline_result['stage_results']['00_data_validation'] = validation_result
                
            return pipeline_result
            
        except ValueError as e:
            # Falha na validação - não executar pipeline
            logger.error(f"Pipeline interrompido devido a falha na validação: {e}")
            return {
                'overall_success': False,
                'error': str(e),
                'validation_stage': {
                    'success': False,
                    'error': str(e)
                },
                'message': 'Pipeline interrompido: nenhum dataset válido encontrado'
            }
        except Exception as e:
            logger.error(f"Erro inesperado na validação: {e}")
            # Tentar continuar com datasets originais se validação falhar
            logger.warning("⚠️ Continuando sem validação devido a erro inesperado")
            return original_execute_method(self, datasets, config)
    
    return execute_with_validation