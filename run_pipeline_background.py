#!/usr/bin/env python3
"""
Executa o pipeline em background e monitora progresso
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path
import json
import glob
from datetime import datetime

class PipelineRunner:
    def __init__(self):
        self.process = None
        self.project_root = Path(__file__).parent
        self.last_checkpoint_count = 0
        self.last_file_count = 0
        
    def start_pipeline(self):
        """Inicia o pipeline em background"""
        print("🚀 Iniciando pipeline em background...")
        
        # Comando para executar o pipeline
        cmd = [sys.executable, "run_pipeline.py"]
        
        # Iniciar processo em background
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.project_root,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"✅ Pipeline iniciado com PID: {self.process.pid}")
        return self.process.pid
    
    def monitor_progress(self, check_interval=30):
        """Monitora o progresso do pipeline"""
        print(f"📊 Monitorando progresso (verificação a cada {check_interval}s)")
        print("Pressione Ctrl+C para parar o monitoramento\n")
        
        try:
            while True:
                # Verificar se processo ainda está rodando
                if self.process and self.process.poll() is not None:
                    print("⚠️ Pipeline terminou")
                    break
                
                # Mostrar status atual
                self.show_current_status()
                
                # Esperar antes da próxima verificação
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoramento interrompido pelo usuário")
            self.stop_pipeline()
    
    def show_current_status(self):
        """Mostra status atual de forma compacta"""
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Status do Pipeline:")
        print("-" * 50)
        
        # Contar arquivos processados por etapa
        interim_dir = self.project_root / "data" / "interim"
        
        stages = {
            "02_encoding_fixed": "*_02_encoding_fixed.csv",
            "03_deduplicated": "*_03_deduplicated.csv", 
            "04_features_extracted": "*_04_features_extracted.csv",
            "06_text_cleaned": "*_06_text_cleaned.csv",
            "07_sentiment_analyzed": "*_07_sentiment_analyzed.csv",
            "08_topic_modeled": "*_08_topic_modeled.csv"
        }
        
        total_files = 0
        for stage_name, pattern in stages.items():
            files = glob.glob(str(interim_dir / pattern))
            count = len(files)
            total_files += count
            
            if count > 0:
                print(f"✅ {stage_name}: {count}/5 datasets")
            else:
                print(f"⏳ {stage_name}: Aguardando...")
                break  # Parar na primeira etapa não completa
        
        # Contar checkpoints
        checkpoints_dir = self.project_root / "checkpoints"
        checkpoint_count = len(glob.glob(str(checkpoints_dir / "*.json")))
        
        if checkpoint_count > self.last_checkpoint_count:
            print(f"🔖 Novos checkpoints: {checkpoint_count} (+{checkpoint_count - self.last_checkpoint_count})")
            self.last_checkpoint_count = checkpoint_count
        
        # Verificar último arquivo modificado
        csv_files = glob.glob(str(interim_dir / "*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=os.path.getmtime)
            mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
            time_diff = datetime.now() - mod_time
            
            if time_diff.total_seconds() < 60:
                print(f"📄 Última atividade: {Path(latest_file).name} ({time_diff.seconds}s atrás)")
            else:
                print(f"📄 Última atividade: {mod_time.strftime('%H:%M:%S')}")
        
        # Mostrar total processado
        print(f"📊 Total processado: {total_files} arquivos")
        
        # Status do processo
        if self.process:
            if self.process.poll() is None:
                print("🔥 Status: Executando")
            else:
                print("⚠️ Status: Finalizado")
        
        print()
    
    def stop_pipeline(self):
        """Para o pipeline"""
        if self.process and self.process.poll() is None:
            print("🛑 Parando pipeline...")
            self.process.terminate()
            
            # Aguardar término gracioso
            try:
                self.process.wait(timeout=10)
                print("✅ Pipeline parado com sucesso")
            except subprocess.TimeoutExpired:
                print("⚠️ Forçando parada do pipeline...")
                self.process.kill()
                self.process.wait()
                print("✅ Pipeline forcibly stopped")
    
    def get_final_status(self):
        """Mostra status final"""
        print("\n📋 STATUS FINAL DO PIPELINE")
        print("=" * 60)
        
        # Verificar arquivo de resultados mais recente
        logs_dir = self.project_root / "logs" / "pipeline"
        latest_results = logs_dir / "latest_pipeline_results.json"
        
        if latest_results.exists():
            with open(latest_results, 'r') as f:
                results = json.load(f)
                
            print(f"Status: {'✅ SUCESSO' if results.get('overall_success') else '❌ FALHOU'}")
            
            if 'execution_summary' in results:
                summary = results['execution_summary']
                print(f"Duração total: {summary.get('total_duration_seconds', 0):.1f}s")
                print(f"Etapas concluídas: {summary.get('completed_stages', 0)}/15")
                
            if 'errors' in results and results['errors']:
                print("\nErros encontrados:")
                for error in results['errors'][:3]:  # Mostrar até 3 erros
                    print(f"  - {error.get('stage', 'N/A')}: {error.get('error', 'Unknown')[:100]}")
        
        # Contar arquivos finais
        interim_dir = self.project_root / "data" / "interim"
        total_files = len(glob.glob(str(interim_dir / "*.csv")))
        total_size = sum(os.path.getsize(f) for f in glob.glob(str(interim_dir / "*.csv")))
        
        print(f"\nArquivos gerados: {total_files}")
        print(f"Tamanho total: {total_size / (1024*1024*1024):.2f} GB")

def main():
    runner = PipelineRunner()
    
    try:
        # Verificar se já existe um processo rodando
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'] and 'run_pipeline.py' in ' '.join(proc.info['cmdline']):
                    print(f"⚠️ Pipeline já está rodando com PID: {proc.info['pid']}")
                    response = input("Deseja monitorar o processo existente? (y/n): ")
                    if response.lower() == 'y':
                        runner.process = proc
                        runner.monitor_progress()
                        return
                    else:
                        print("Encerrando...")
                        return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Iniciar novo pipeline
        pid = runner.start_pipeline()
        
        # Aguardar um pouco para o pipeline inicializar
        time.sleep(5)
        
        # Monitorar progresso
        runner.monitor_progress()
        
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário")
    
    finally:
        # Mostrar status final
        runner.get_final_status()
        
        # Limpar processo
        if runner.process:
            runner.stop_pipeline()

if __name__ == "__main__":
    main()