"""
Patch para compatibilidade Gensim-SciPy
=========================================

Este módulo implementa uma solução para problemas de compatibilidade
entre Gensim e versões mais recentes do SciPy.
"""

import logging
import warnings

logger = logging.getLogger(__name__)


def patch_gensim_scipy_compatibility():
    """
    Aplica patch para resolver problemas de compatibilidade Gensim-SciPy
    
    Resolve:
    - ImportError: cannot import name 'triu' from 'scipy.linalg'
    - Problemas de compatibilidade com SciPy 1.13+
    """
    try:
        # Tentar importar scipy.linalg.triu
        from scipy.linalg import triu
        logger.info("✅ SciPy triu disponível - compatibilidade OK")
        return True
    except ImportError:
        logger.warning("⚠️  scipy.linalg.triu não encontrado - aplicando patch")
        
        # Aplicar patch: criar triu fallback usando numpy
        try:
            import numpy as np
            import scipy.linalg
            
            # Criar função triu usando numpy se não existir
            if not hasattr(scipy.linalg, 'triu'):
                def triu_fallback(m, k=0):
                    """Fallback para triu usando numpy.triu"""
                    return np.triu(m, k=k)
                
                # Monkey patch
                scipy.linalg.triu = triu_fallback
                logger.info("✅ Patch aplicado: scipy.linalg.triu usando numpy.triu")
                return True
        except Exception as e:
            logger.error(f"❌ Falha ao aplicar patch Gensim-SciPy: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro inesperado ao verificar compatibilidade SciPy: {e}")
        return False


def safe_import_gensim():
    """
    Importa Gensim com patch de compatibilidade aplicado
    
    Returns:
        tuple: (gensim_module, success_flag)
    """
    # Aplicar patch primeiro
    patch_success = patch_gensim_scipy_compatibility()
    
    if not patch_success:
        logger.warning("Patch SciPy falhou, tentando importar Gensim sem patch")
    
    try:
        import gensim
        logger.info(f"✅ Gensim {gensim.__version__} importado com sucesso")
        
        # Testar funcionalidade básica
        from gensim.models import LdaModel
        logger.info("✅ Gensim.models.LdaModel disponível")
        
        return gensim, True
        
    except ImportError as e:
        if "triu" in str(e):
            logger.error(f"❌ Falha na importação Gensim (problema triu não resolvido): {e}")
        else:
            logger.error(f"❌ Falha na importação Gensim: {e}")
        
        return None, False
        
    except Exception as e:
        logger.error(f"❌ Erro inesperado ao importar Gensim: {e}")
        return None, False


def get_lda_model_safe():
    """
    Obtém LdaModel do Gensim com fallback para scikit-learn
    
    Returns:
        class: Classe LDA disponível (Gensim ou scikit-learn)
    """
    gensim, success = safe_import_gensim()
    
    if success and gensim:
        try:
            from gensim.models import LdaModel
            logger.info("✅ Usando Gensim LdaModel (preferido)")
            return LdaModel, "gensim"
        except Exception as e:
            logger.warning(f"⚠️  Falha ao importar Gensim LdaModel: {e}")
    
    # Fallback para scikit-learn
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        logger.info("✅ Usando scikit-learn LDA como fallback")
        return LatentDirichletAllocation, "sklearn"
    except ImportError as e:
        logger.error(f"❌ Nem Gensim nem scikit-learn LDA disponíveis: {e}")
        return None, "none"


# Aplicar patch automaticamente na importação
if __name__ != "__main__":
    patch_gensim_scipy_compatibility()