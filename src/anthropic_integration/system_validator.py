"""
System Validator - Validação de dependências e configurações

Este módulo valida que todas as dependências necessárias estão instaladas
e que as configurações estão corretas antes de executar o pipeline.
"""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)


class SystemValidator:
    """Valida sistema e dependências antes da execução do pipeline"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.validation_results = {
            "dependencies": {"passed": [], "failed": []},
            "config_files": {"passed": [], "failed": []},
            "directories": {"passed": [], "failed": []},
            "environment": {"passed": [], "failed": []},
            "overall_status": "unknown"
        }

    def validate_dependencies(self) -> bool:
        """Valida todas as dependências necessárias"""
        logger.info("Validando dependências do sistema...")

        # Dependências essenciais
        essential_deps = [
            'pandas', 'numpy', 'yaml', 'anthropic', 'dotenv',
            'pathlib', 'logging', 'json', 'datetime'
        ]

        # Dependências opcionais
        optional_deps = [
            'gensim', 'sklearn', 'networkx', 'chardet',
            'ftfy', 'tqdm', 'pickle'
        ]

        all_passed = True

        # Testar dependências essenciais
        for dep in essential_deps:
            try:
                if dep == 'dotenv':
                    import dotenv
                elif dep == 'anthropic':
                    import anthropic
                else:
                    importlib.import_module(dep)
                self.validation_results["dependencies"]["passed"].append(dep)
                logger.debug(f"✅ Dependência {dep} encontrada")
            except ImportError as e:
                self.validation_results["dependencies"]["failed"].append({
                    "dependency": dep,
                    "error": str(e),
                    "critical": True
                })
                logger.error(f"❌ Dependência crítica {dep} não encontrada: {e}")
                all_passed = False

        # Testar dependências opcionais
        for dep in optional_deps:
            try:
                importlib.import_module(dep)
                self.validation_results["dependencies"]["passed"].append(dep)
                logger.debug(f"✅ Dependência opcional {dep} encontrada")
            except ImportError as e:
                self.validation_results["dependencies"]["failed"].append({
                    "dependency": dep,
                    "error": str(e),
                    "critical": False
                })
                logger.warning(f"⚠️ Dependência opcional {dep} não encontrada: {e}")

        return all_passed

    def validate_config_files(self) -> bool:
        """Valida arquivos de configuração"""
        logger.info("Validando arquivos de configuração...")

        config_files = [
            "config/settings.yaml",
            "config/logging.yaml",
            "config/processing.yaml"
        ]

        all_passed = True

        for config_file in config_files:
            config_path = self.project_root / config_file
            try:
                if config_path.exists():
                    # Tentar carregar o YAML para validar sintaxe
                    with open(config_path, 'r') as f:
                        yaml.safe_load(f)
                    self.validation_results["config_files"]["passed"].append(config_file)
                    logger.debug(f"✅ Arquivo de configuração {config_file} válido")
                else:
                    raise FileNotFoundError(f"Arquivo não encontrado: {config_path}")

            except Exception as e:
                self.validation_results["config_files"]["failed"].append({
                    "file": config_file,
                    "error": str(e),
                    "critical": config_file == "config/settings.yaml"
                })

                if config_file == "config/settings.yaml":
                    logger.error(f"❌ Arquivo crítico {config_file} inválido: {e}")
                    all_passed = False
                else:
                    logger.warning(f"⚠️ Arquivo opcional {config_file} inválido: {e}")

        return all_passed

    def validate_directories(self) -> bool:
        """Valida estrutura de diretórios"""
        logger.info("Validando estrutura de diretórios...")

        required_dirs = [
            "src",
            "src/anthropic_integration",
            "src/data",
            "config",
            "data",
            "logs"
        ]

        optional_dirs = [
            "data/DATASETS_FULL",
            "data/interim",
            "checkpoints",
            "temp"
        ]

        all_passed = True

        # Validar diretórios obrigatórios
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.validation_results["directories"]["passed"].append(dir_path)
                logger.debug(f"✅ Diretório {dir_path} encontrado")
            else:
                self.validation_results["directories"]["failed"].append({
                    "directory": dir_path,
                    "error": "Diretório não encontrado",
                    "critical": True
                })
                logger.error(f"❌ Diretório obrigatório {dir_path} não encontrado")
                all_passed = False

        # Validar diretórios opcionais (criar se não existir)
        for dir_path in optional_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.validation_results["directories"]["passed"].append(dir_path)
                logger.debug(f"✅ Diretório {dir_path} encontrado")
            else:
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    self.validation_results["directories"]["passed"].append(dir_path)
                    logger.info(f"📁 Diretório {dir_path} criado")
                except Exception as e:
                    self.validation_results["directories"]["failed"].append({
                        "directory": dir_path,
                        "error": str(e),
                        "critical": False
                    })
                    logger.warning(f"⚠️ Não foi possível criar diretório {dir_path}: {e}")

        return all_passed

    def validate_environment(self) -> bool:
        """Valida variáveis de ambiente"""
        logger.info("Validando variáveis de ambiente...")

        # Verificar arquivo .env
        env_file = self.project_root / '.env'
        if env_file.exists():
            self.validation_results["environment"]["passed"].append(".env file exists")
            logger.debug("✅ Arquivo .env encontrado")
        else:
            self.validation_results["environment"]["failed"].append({
                "item": ".env file",
                "error": "Arquivo .env não encontrado",
                "critical": False
            })
            logger.warning("⚠️ Arquivo .env não encontrado")

        # Verificar ANTHROPIC_API_KEY
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            # Validar formato da chave (não expor o valor)
            if api_key.startswith('sk-ant-'):
                self.validation_results["environment"]["passed"].append("ANTHROPIC_API_KEY format valid")
                logger.debug("✅ ANTHROPIC_API_KEY com formato válido")
            else:
                self.validation_results["environment"]["failed"].append({
                    "item": "ANTHROPIC_API_KEY",
                    "error": "Formato de chave inválido",
                    "critical": False
                })
                logger.warning("⚠️ ANTHROPIC_API_KEY com formato suspeito")
        else:
            self.validation_results["environment"]["failed"].append({
                "item": "ANTHROPIC_API_KEY",
                "error": "Variável de ambiente não definida",
                "critical": False
            })
            logger.warning("⚠️ ANTHROPIC_API_KEY não definida")

        # Verificar versão do Python
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.validation_results["environment"]["passed"].append(f"Python {python_version.major}.{python_version.minor}")
            logger.debug(f"✅ Python {python_version.major}.{python_version.minor} compatível")
        else:
            self.validation_results["environment"]["failed"].append({
                "item": "Python version",
                "error": f"Python {python_version.major}.{python_version.minor} é muito antigo",
                "critical": True
            })
            logger.error(f"❌ Python {python_version.major}.{python_version.minor} é muito antigo (requer 3.8+)")
            return False

        return True

    def run_full_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Executa validação completa do sistema"""
        logger.info("🔍 Iniciando validação completa do sistema...")

        # Executar todas as validações
        deps_ok = self.validate_dependencies()
        config_ok = self.validate_config_files()
        dirs_ok = self.validate_directories()
        env_ok = self.validate_environment()

        # Determinar status geral
        if deps_ok and config_ok and dirs_ok and env_ok:
            self.validation_results["overall_status"] = "healthy"
            logger.info("✅ Validação do sistema: SUCESSO")
        elif deps_ok and config_ok and dirs_ok:  # Ambiente pode ter problemas não críticos
            self.validation_results["overall_status"] = "warning"
            logger.warning("⚠️ Validação do sistema: AVISOS (sistema funcional)")
        else:
            self.validation_results["overall_status"] = "error"
            logger.error("❌ Validação do sistema: ERRO (problemas críticos)")

        return self.validation_results["overall_status"] in ["healthy", "warning"], self.validation_results

    def generate_report(self) -> str:
        """Gera relatório de validação"""
        report = []
        report.append("📋 RELATÓRIO DE VALIDAÇÃO DO SISTEMA")
        report.append("=" * 50)

        # Status geral
        status_icon = {
            "healthy": "✅",
            "warning": "⚠️",
            "error": "❌",
            "unknown": "❓"
        }

        report.append(f"\n{status_icon.get(self.validation_results['overall_status'], '❓')} Status Geral: {self.validation_results['overall_status'].upper()}")

        # Dependências
        report.append(f"\n📦 Dependências:")
        report.append(f"   ✅ Aprovadas: {len(self.validation_results['dependencies']['passed'])}")
        report.append(f"   ❌ Falhas: {len(self.validation_results['dependencies']['failed'])}")

        for failure in self.validation_results['dependencies']['failed']:
            critical = "CRÍTICO" if failure.get('critical', False) else "OPCIONAL"
            report.append(f"      - {failure['dependency']} ({critical}): {failure['error']}")

        # Arquivos de configuração
        report.append(f"\n⚙️ Configurações:")
        report.append(f"   ✅ Válidas: {len(self.validation_results['config_files']['passed'])}")
        report.append(f"   ❌ Inválidas: {len(self.validation_results['config_files']['failed'])}")

        for failure in self.validation_results['config_files']['failed']:
            critical = "CRÍTICO" if failure.get('critical', False) else "OPCIONAL"
            report.append(f"      - {failure['file']} ({critical}): {failure['error']}")

        # Diretórios
        report.append(f"\n📁 Diretórios:")
        report.append(f"   ✅ Encontrados: {len(self.validation_results['directories']['passed'])}")
        report.append(f"   ❌ Problemas: {len(self.validation_results['directories']['failed'])}")

        # Ambiente
        report.append(f"\n🌍 Ambiente:")
        report.append(f"   ✅ Configurações OK: {len(self.validation_results['environment']['passed'])}")
        report.append(f"   ❌ Problemas: {len(self.validation_results['environment']['failed'])}")

        return "\n".join(report)


def validate_system(project_root: str = None) -> Tuple[bool, str]:
    """
    Função de conveniência para validar sistema

    Args:
        project_root: Diretório raiz do projeto

    Returns:
        Tuple com (sucesso, relatório)
    """
    validator = SystemValidator(project_root)
    success, results = validator.run_full_validation()
    report = validator.generate_report()

    return success, report


if __name__ == "__main__":
    # Teste da validação
    success, report = validate_system()
    print(report)

    if success:
        print("\n🎯 Sistema pronto para execução!")
        sys.exit(0)
    else:
        print("\n💥 Sistema tem problemas críticos!")
        sys.exit(1)
