"""
Configurações de Performance do Sistema
=======================================

Configura otimizações de performance para todos os componentes do pipeline.
"""

import logging
import os

logger = logging.getLogger(__name__)


def configure_numexpr_threads():
    """
    Configura NumExpr para usar todos os cores disponíveis de forma otimizada
    """
    try:
        import numexpr
        
        # Detectar número de cores
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        
        # Configurar NumExpr para usar todos os cores (com limite de segurança)
        optimal_threads = min(num_cores, 16)  # Máximo 16 threads para evitar overhead
        
        # Definir variável de ambiente se não estiver definida
        if 'NUMEXPR_MAX_THREADS' not in os.environ:
            os.environ['NUMEXPR_MAX_THREADS'] = str(optimal_threads)
            logger.info(f"✅ NumExpr configurado para {optimal_threads} threads ({num_cores} cores detectados)")
        else:
            current_setting = os.environ['NUMEXPR_MAX_THREADS']
            logger.info(f"✅ NumExpr já configurado: {current_setting} threads")
            
        # Verificar configuração ativa
        actual_threads = numexpr.nthreads
        logger.info(f"✅ NumExpr usando {actual_threads} threads ativos")
        
        return True
        
    except ImportError:
        logger.info("NumExpr não disponível - configuração ignorada")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Erro ao configurar NumExpr: {e}")
        return False


def configure_numpy_threads():
    """
    Configura NumPy/OpenBLAS para performance otimizada
    """
    try:
        import numpy as np
        import multiprocessing
        
        num_cores = multiprocessing.cpu_count()
        
        # Configurar threads do NumPy se não estiver definido
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = str(num_cores)
            logger.info(f"✅ OpenMP configurado para {num_cores} threads")
            
        if 'OPENBLAS_NUM_THREADS' not in os.environ:
            os.environ['OPENBLAS_NUM_THREADS'] = str(num_cores)
            logger.info(f"✅ OpenBLAS configurado para {num_cores} threads")
            
        # Verificar configuração do NumPy
        try:
            import numpy.show_config
            logger.info("✅ NumPy configurado com BLAS otimizado")
        except:
            logger.info("✅ NumPy loaded (BLAS status not verifiable)")
            
        return True
        
    except Exception as e:
        logger.warning(f"⚠️  Erro ao configurar NumPy: {e}")
        return False


def configure_pandas_performance():
    """
    Configura Pandas para performance otimizada
    """
    try:
        import pandas as pd
        
        # Configurar opções de performance do Pandas
        pd.set_option('compute.use_numexpr', True)
        pd.set_option('compute.use_bottleneck', True)
        
        # Configurar limite de memória para operações
        pd.set_option('mode.copy_on_write', True)  # Reduz uso de memória
        
        logger.info("✅ Pandas configurado para performance otimizada")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️  Erro ao configurar Pandas: {e}")
        return False


def configure_all_performance():
    """
    Aplica todas as configurações de performance
    """
    logger.info("🚀 Configurando otimizações de performance...")
    
    results = {
        'numexpr': configure_numexpr_threads(),
        'numpy': configure_numpy_threads(), 
        'pandas': configure_pandas_performance()
    }
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"✅ Performance configurada: {success_count}/{total_count} otimizações aplicadas")
    
    if success_count == total_count:
        logger.info("🎯 Sistema totalmente otimizado para máxima performance")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.warning(f"⚠️  Algumas otimizações falharam: {failed}")
    
    return results


# Aplicar configurações automaticamente na importação
if __name__ != "__main__":
    configure_all_performance()