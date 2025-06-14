#!/usr/bin/env python3
"""
Script Consolidado de Ferramentas de Manutenção v4.9.8
======================================================

Consolida todas as ferramentas de manutenção do sistema enhanced em um único script.
Inclui validação, diagnósticos, e ferramentas de verificação do sistema.

🔧 CONSOLIDAÇÃO: Unifica ferramentas de manutenção
✅ VALIDAÇÃO: Sistema de validação completo do enhanced config
🎯 DIAGNÓSTICO: Ferramentas de análise e verificação
🛠️ UTILITÁRIOS: Funções auxiliares para manutenção
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Encontrar raiz do projeto
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

class SystemValidator:
    """Validador completo do sistema enhanced consolidado"""
    
    def __init__(self):
        self.project_root = project_root
        
    def test_enhanced_loader(self) -> bool:
        """Testa o carregamento do EnhancedConfigLoader consolidado"""
        logger.info("🧪 Testando EnhancedConfigLoader consolidado...")
        
        try:
            from anthropic_integration.base import get_enhanced_config_loader, load_operation_config
            
            # Testar singleton
            loader1 = get_enhanced_config_loader()
            loader2 = get_enhanced_config_loader()
            
            if loader1 is not loader2:
                logger.error("❌ Singleton pattern falhou")
                return False
                
            logger.info("✅ Singleton pattern funcionando")
            
            # Testar carregamento de stages
            test_operations = [
                'political_analysis',
                'sentiment_analysis', 
                'network_analysis',
                'qualitative_analysis',
                'pipeline_review',
                'topic_interpretation',
                'validation'
            ]
            
            for operation in test_operations:
                try:
                    config = load_operation_config(operation)
                    if 'model' not in config:
                        logger.error(f"❌ Configuração inválida para {operation}: falta 'model'")
                        return False
                    logger.info(f"✅ {operation}: {config.get('model', 'N/A')}")
                except Exception as e:
                    logger.error(f"❌ Erro ao carregar config para {operation}: {e}")
                    return False
            
            logger.info("✅ EnhancedConfigLoader consolidado validado com sucesso")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Erro de importação: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
            return False

    def test_anthropic_base(self) -> bool:
        """Testa inicialização do AnthropicBase com enhanced config"""
        logger.info("🧪 Testando AnthropicBase com enhanced config...")
        
        try:
            from anthropic_integration.base import AnthropicBase
            
            # Testar inicialização sem stage_operation específico (usa configuração padrão)
            test_config = {'anthropic': {'model': 'claude-3-5-sonnet-20241022'}}
            base1 = AnthropicBase(config=test_config)
            logger.info(f"✅ AnthropicBase sem stage: {getattr(base1, 'model', 'N/A')}")
            
            # Testar inicialização com stage_operation
            base2 = AnthropicBase(stage_operation="political_analysis")
            logger.info(f"✅ AnthropicBase com political_analysis: {getattr(base2, 'model', 'N/A')}")
            
            # Verificar se enhanced config foi carregada
            if hasattr(base2, 'enhanced_config') and base2.enhanced_config:
                logger.info("✅ Enhanced config carregada com sucesso")
            else:
                logger.warning("⚠️ Enhanced config não carregada (usando fallback)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao testar AnthropicBase: {e}")
            return False

    def test_cost_monitor(self) -> bool:
        """Testa o sistema de monitoramento de custos consolidado"""
        logger.info("🧪 Testando sistema de monitoramento de custos...")
        
        try:
            from anthropic_integration.cost_monitor import get_cost_monitor
            
            # Inicializar cost monitor
            monitor = get_cost_monitor(self.project_root)
            
            # Testar registro de uso
            cost = monitor.record_usage(
                model="claude-3-5-sonnet-20241022",
                input_tokens=100,
                output_tokens=50,
                stage="test_stage",
                operation="validation"
            )
            
            if cost > 0:
                logger.info(f"✅ Custo calculado: ${cost:.6f}")
            else:
                logger.warning("⚠️ Custo calculado como 0")
            
            # Testar relatório
            report = monitor.get_daily_report()
            logger.info(f"✅ Relatório diário gerado: {report.get('total_cost', 0):.6f} USD")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao testar cost monitor: {e}")
            return False

    def test_component_initialization(self) -> bool:
        """Testa inicialização dos componentes principais"""
        logger.info("🧪 Testando inicialização de componentes...")
        
        test_config = {
            'anthropic': {
                'api_key': '${ANTHROPIC_API_KEY}',
                'model': 'claude-3-5-sonnet-20241022'
            }
        }
        
        components_to_test = [
            ('PoliticalAnalyzer', 'political_analyzer'),
            ('AnthropicSentimentAnalyzer', 'sentiment_analyzer'),
            ('SmartPipelineReviewer', 'smart_pipeline_reviewer'),
            ('TopicInterpreter', 'topic_interpreter'),
            ('CompletePipelineValidator', 'pipeline_validator'),
            ('QualitativeClassifier', 'qualitative_classifier'),
            ('IntelligentNetworkAnalyzer', 'intelligent_network_analyzer')
        ]
        
        success_count = 0
        
        for class_name, module_name in components_to_test:
            try:
                module = __import__(f'anthropic_integration.{module_name}', fromlist=[class_name])
                component_class = getattr(module, class_name)
                
                # Inicializar componente
                if class_name == 'CompletePipelineValidator':
                    component = component_class(test_config, str(self.project_root))
                else:
                    component = component_class(test_config)
                
                # Verificar se enhanced config foi aplicada
                model = getattr(component, 'model', 'N/A')
                logger.info(f"✅ {class_name}: {model}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar {class_name}: {e}")
        
        logger.info(f"✅ {success_count}/{len(components_to_test)} componentes inicializados com sucesso")
        return success_count == len(components_to_test)

    def run_validation_suite(self) -> bool:
        """Executa suite completa de validação"""
        logger.info("🚀 Iniciando validação completa do sistema enhanced...")
        
        tests = [
            ("Enhanced Loader", self.test_enhanced_loader),
            ("Anthropic Base", self.test_anthropic_base),
            ("Component Initialization", self.test_component_initialization),
            ("Cost Monitor", self.test_cost_monitor)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 TESTE: {test_name}")
            logger.info('='*50)
            
            try:
                if test_func():
                    logger.info(f"✅ {test_name}: PASSOU")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name}: FALHOU")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERRO - {e}")
        
        # Relatório final
        logger.info(f"\n{'='*50}")
        logger.info("📊 RELATÓRIO FINAL DE VALIDAÇÃO")
        logger.info('='*50)
        logger.info(f"✅ Testes passaram: {passed}/{total}")
        logger.info(f"❌ Testes falharam: {total - passed}/{total}")
        logger.info(f"📈 Taxa de sucesso: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 TODAS AS VALIDAÇÕES PASSARAM!")
            logger.info("✅ Sistema enhanced está funcionando corretamente")
            return True
        else:
            logger.error("⚠️ ALGUMAS VALIDAÇÕES FALHARAM!")
            logger.error("❌ Sistema enhanced precisa de correções")
            return False


class SystemDiagnostics:
    """Ferramentas de diagnóstico do sistema"""
    
    def __init__(self):
        self.project_root = project_root
    
    def check_file_structure(self) -> bool:
        """Verifica estrutura de arquivos do projeto"""
        logger.info("🔍 Verificando estrutura de arquivos...")
        
        required_files = [
            "config/settings.yaml",
            "src/anthropic_integration/base.py",
            "src/anthropic_integration/cost_monitor.py",
            "run_pipeline.py",
            "src/main.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Arquivos ausentes: {missing_files}")
            return False
        
        logger.info("✅ Estrutura de arquivos correta")
        return True
    
    def check_configuration_integrity(self) -> bool:
        """Verifica integridade das configurações"""
        logger.info("🔍 Verificando integridade das configurações...")
        
        settings_path = self.project_root / "config" / "settings.yaml"
        
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            
            # Verificar seções essenciais
            required_sections = ['anthropic', 'enhanced_models']
            for section in required_sections:
                if section not in settings:
                    logger.error(f"❌ Seção ausente em settings.yaml: {section}")
                    return False
            
            # Verificar configurações específicas de stage
            anthropic_config = settings.get('anthropic', {})
            if 'stage_specific_configs' not in anthropic_config:
                logger.error("❌ stage_specific_configs ausente na configuração anthropic")
                return False
            
            logger.info("✅ Configurações íntegras")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar configurações: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """Verifica dependências críticas"""
        logger.info("🔍 Verificando dependências críticas...")
        
        critical_imports = [
            'anthropic',
            'yaml', 
            'pandas',
            'numpy',
            'voyageai'
        ]
        
        missing_deps = []
        for dep in critical_imports:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"❌ Dependências ausentes: {missing_deps}")
            return False
        
        logger.info("✅ Dependências críticas disponíveis")
        return True
    
    def check_voyage_configuration(self) -> bool:
        """Verifica configuração Voyage.ai padronizada"""
        logger.info("🔍 Verificando configuração Voyage.ai...")
        
        try:
            settings_path = self.project_root / "config" / "settings.yaml"
            voyage_path = self.project_root / "config" / "voyage_embeddings.yaml"
            
            # Verificar settings.yaml
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            
            embeddings_model = settings.get('embeddings', {}).get('model')
            if embeddings_model != 'voyage-3.5-lite':
                logger.error(f"❌ settings.yaml: modelo incorreto '{embeddings_model}' (esperado: voyage-3.5-lite)")
                return False
            
            # Verificar voyage_embeddings.yaml
            if voyage_path.exists():
                with open(voyage_path, 'r', encoding='utf-8') as f:
                    voyage_config = yaml.safe_load(f)
                
                voyage_model = voyage_config.get('embeddings', {}).get('model')
                if voyage_model != 'voyage-3.5-lite':
                    logger.error(f"❌ voyage_embeddings.yaml: modelo incorreto '{voyage_model}' (esperado: voyage-3.5-lite)")
                    return False
            
            # Verificar código fonte
            voyage_code_path = self.project_root / "src" / "anthropic_integration" / "voyage_embeddings.py"
            if voyage_code_path.exists():
                with open(voyage_code_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'voyage-3.5-lite' not in content:
                    logger.error("❌ voyage_embeddings.py: modelo não padronizado no código")
                    return False
            
            logger.info("✅ Configuração Voyage.ai padronizada (voyage-3.5-lite)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar configuração Voyage.ai: {e}")
            return False
    
    def generate_system_report(self) -> Dict[str, Any]:
        """Gera relatório completo do sistema"""
        logger.info("📊 Gerando relatório do sistema...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_structure": self.check_file_structure(),
            "configuration_integrity": self.check_configuration_integrity(),
            "dependencies": self.check_dependencies(),
            "voyage_configuration": self.check_voyage_configuration(),
            "system_info": {
                "project_root": str(self.project_root),
                "python_version": sys.version,
                "script_location": str(script_dir)
            }
        }
        
        # Salvar relatório
        report_path = self.project_root / "logs" / f"system_report_{int(time.time())}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Relatório salvo: {report_path}")
        return report


class MaintenanceUtilities:
    """Utilitários gerais de manutenção"""
    
    def __init__(self):
        self.project_root = project_root
    
    def cleanup_old_files(self) -> bool:
        """Remove arquivos antigos e desnecessários"""
        logger.info("🧹 Limpando arquivos antigos...")
        
        # Padrões de arquivos para limpeza
        cleanup_patterns = [
            "**/*.pyc",
            "**/__pycache__",
            "**/.*_cache",
            "logs/*.log.old",
            "temp/*"
        ]
        
        cleaned_count = 0
        for pattern in cleanup_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    cleaned_count += 1
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                    cleaned_count += 1
        
        logger.info(f"✅ {cleaned_count} arquivos/diretórios limpos")
        return True
    
    def check_log_sizes(self) -> Dict[str, Any]:
        """Verifica tamanhos dos logs"""
        logger.info("📋 Verificando tamanhos dos logs...")
        
        logs_dir = self.project_root / "logs"
        log_info = {}
        
        if logs_dir.exists():
            for log_file in logs_dir.glob("**/*.log"):
                size_mb = log_file.stat().st_size / (1024 * 1024)
                log_info[str(log_file.relative_to(logs_dir))] = f"{size_mb:.2f} MB"
        
        logger.info(f"📊 Informações de logs: {log_info}")
        return log_info


def main():
    """Função principal com interface de linha de comando"""
    parser = argparse.ArgumentParser(description="Ferramentas de Manutenção do Sistema Enhanced")
    parser.add_argument('action', choices=[
        'validate', 'diagnose', 'report', 'cleanup', 'all'
    ], help='Ação a executar')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verboso')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 Iniciando ferramentas de manutenção...")
    
    validator = SystemValidator()
    diagnostics = SystemDiagnostics()
    utilities = MaintenanceUtilities()
    
    success = True
    
    try:
        if args.action in ['validate', 'all']:
            logger.info("\n" + "="*60)
            logger.info("🧪 EXECUTANDO VALIDAÇÃO COMPLETA")
            logger.info("="*60)
            if not validator.run_validation_suite():
                success = False
        
        if args.action in ['diagnose', 'all']:
            logger.info("\n" + "="*60)
            logger.info("🔍 EXECUTANDO DIAGNÓSTICOS")
            logger.info("="*60)
            diagnostics.check_file_structure()
            diagnostics.check_configuration_integrity()
            diagnostics.check_dependencies()
            diagnostics.check_voyage_configuration()
        
        if args.action in ['report', 'all']:
            logger.info("\n" + "="*60)
            logger.info("📊 GERANDO RELATÓRIO DO SISTEMA")
            logger.info("="*60)
            report = diagnostics.generate_system_report()
            logger.info(f"✅ Relatório gerado com sucesso")
        
        if args.action in ['cleanup', 'all']:
            logger.info("\n" + "="*60)
            logger.info("🧹 EXECUTANDO LIMPEZA")
            logger.info("="*60)
            utilities.cleanup_old_files()
            utilities.check_log_sizes()
        
        if success:
            logger.info("\n🎉 MANUTENÇÃO CONCLUÍDA COM SUCESSO!")
        else:
            logger.error("\n⚠️ MANUTENÇÃO CONCLUÍDA COM PROBLEMAS")
            
        return success
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Manutenção interrompida pelo usuário")
        return False
    except Exception as e:
        logger.error(f"\n❌ Erro durante manutenção: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)