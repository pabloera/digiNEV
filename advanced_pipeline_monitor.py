#!/usr/bin/env python3
"""
Monitor Avançado do Pipeline - Execução Contínua com Monitoramento em Tempo Real
"""

import time
import subprocess
import signal
import sys
import os
import glob
import json
from datetime import datetime
from pathlib import Path

class AdvancedPipelineMonitor:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.process = None
        self.monitoring = True
        
    def get_detailed_status(self):
        """Coleta status detalhado do pipeline"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "files_by_stage": {},
            "latest_files": [],
            "checkpoints": [],
            "api_costs": 0,
            "total_size_gb": 0
        }
        
        # Verificar arquivos por etapa
        interim_dir = self.project_root / "data" / "interim"
        
        stage_patterns = {
            "02_encoding_fixed": "*_02_encoding_fixed.csv",
            "02b_deduplicated": "*_02b_deduplicated.csv", 
            "01b_features_extracted": "*_01b_features_extracted.csv",
            "03_text_cleaned": "*_03_text_cleaned.csv",
            "04_sentiment_analyzed": "*_04_sentiment_analyzed.csv",
            "05_topic_modeled": "*_05_topic_modeled.csv",
            "06_tfidf_extracted": "*_06_tfidf_extracted.csv",
            "07_clustered": "*_07_clustered.csv",
            "08_hashtags_normalized": "*_08_hashtags_normalized.csv",
            "09_domains_analyzed": "*_09_domains_analyzed.csv",
            "10_temporal_analyzed": "*_10_temporal_analyzed.csv",
            "11_network_analyzed": "*_11_network_analyzed.csv",
            "12_qualitative_analyzed": "*_12_qualitative_analyzed.csv",
            "13_final_processed": "*_13_final_processed.csv"
        }
        
        for stage, pattern in stage_patterns.items():
            files = glob.glob(str(interim_dir / pattern))
            status["files_by_stage"][stage] = len(files)
        
        # Últimos arquivos modificados
        all_csvs = glob.glob(str(interim_dir / "*.csv"))
        if all_csvs:
            all_csvs.sort(key=os.path.getmtime, reverse=True)
            for f in all_csvs[:5]:
                mod_time = datetime.fromtimestamp(os.path.getmtime(f))
                status["latest_files"].append({
                    "file": Path(f).name,
                    "modified": mod_time.strftime("%H:%M:%S"),
                    "size_mb": round(os.path.getsize(f) / (1024*1024), 1)
                })
        
        # Checkpoints recentes
        checkpoints_dir = self.project_root / "checkpoints"
        recent_checkpoints = glob.glob(str(checkpoints_dir / "*.json"))
        recent_checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        for cp in recent_checkpoints[:3]:
            cp_time = datetime.fromtimestamp(os.path.getmtime(cp))
            status["checkpoints"].append({
                "file": Path(cp).name,
                "time": cp_time.strftime("%H:%M:%S")
            })
        
        # Custos da API
        try:
            costs_file = self.project_root / "logs" / "anthropic_costs.json"
            if costs_file.exists():
                with open(costs_file, 'r') as f:
                    costs_data = json.load(f)
                    if isinstance(costs_data, list) and costs_data:
                        status["api_costs"] = sum(item.get("cost", 0) for item in costs_data[-10:])
        except:
            pass
        
        # Tamanho total
        if all_csvs:
            total_bytes = sum(os.path.getsize(f) for f in all_csvs)
            status["total_size_gb"] = round(total_bytes / (1024**3), 2)
        
        return status
    
    def print_status(self, status):
        """Imprime status formatado"""
        print(f"\n⏰ {status['timestamp'][11:19]} - STATUS DO PIPELINE")
        print("=" * 60)
        
        # Progresso por etapa
        stages = status["files_by_stage"]
        total_stages_with_files = sum(1 for count in stages.values() if count == 5)
        
        print(f"📊 Progresso: {total_stages_with_files}/14 etapas completas")
        print(f"💾 Dados processados: {status['total_size_gb']} GB")
        print(f"💰 Custos API (últimas 10 operações): ${status['api_costs']:.3f}")
        
        # Etapas ativas
        print(f"\n🔄 Etapas com arquivos processados:")
        for stage, count in stages.items():
            if count > 0:
                emoji = "✅" if count == 5 else f"🔄 {count}/5"
                print(f"   {emoji} {stage}")
        
        # Últimos arquivos
        if status["latest_files"]:
            print(f"\n📄 Últimos arquivos modificados:")
            for file_info in status["latest_files"]:
                print(f"   {file_info['modified']} - {file_info['file']} ({file_info['size_mb']}MB)")
        
        # Checkpoints
        if status["checkpoints"]:
            print(f"\n🔖 Checkpoints recentes:")
            for cp in status["checkpoints"]:
                print(f"   {cp['time']} - {cp['file']}")
    
    def run_pipeline_continuous(self):
        """Executa pipeline de forma contínua com monitoramento"""
        print("🚀 MONITOR AVANÇADO DO PIPELINE")
        print("=" * 60)
        print("Pressione Ctrl+C para parar\n")
        
        # Status inicial
        initial_status = self.get_detailed_status()
        self.print_status(initial_status)
        
        try:
            while self.monitoring:
                # Iniciar processo do pipeline
                print(f"\n🔄 Iniciando execução do pipeline...")
                
                process = subprocess.Popen(
                    [sys.executable, "run_pipeline.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.project_root
                )
                
                # Monitorar por 5 minutos ou até o processo terminar
                start_time = time.time()
                last_status_time = start_time
                
                while process.poll() is None and (time.time() - start_time) < 300:  # 5 min
                    time.sleep(10)  # Check every 10 seconds
                    
                    # Mostrar status a cada 30 segundos
                    if time.time() - last_status_time >= 30:
                        current_status = self.get_detailed_status()
                        self.print_status(current_status)
                        last_status_time = time.time()
                
                # Se processo ainda está rodando, deixar continuar e monitorar menos frequentemente
                if process.poll() is None:
                    print(f"\n⏳ Pipeline ainda executando... aguardando conclusão")
                    
                    # Monitorar de forma menos frequente
                    while process.poll() is None:
                        time.sleep(60)  # Check every minute
                        current_status = self.get_detailed_status()
                        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Pipeline em execução...")
                        print(f"📊 Etapas completas: {sum(1 for count in current_status['files_by_stage'].values() if count == 5)}/14")
                        print(f"💾 Dados: {current_status['total_size_gb']} GB")
                
                # Processo terminou
                return_code = process.returncode
                final_status = self.get_detailed_status()
                
                print(f"\n🏁 Pipeline terminou com código: {return_code}")
                self.print_status(final_status)
                
                # Verificar se deve continuar
                completed_stages = sum(1 for count in final_status["files_by_stage"].values() if count == 5)
                
                if completed_stages >= 14:
                    print(f"\n🎉 PIPELINE COMPLETO! Todas as 14 etapas foram processadas!")
                    break
                elif return_code == 0:
                    print(f"\n✅ Execução bem-sucedida. Verificando necessidade de continuação...")
                    time.sleep(5)
                else:
                    print(f"\n⚠️ Execução com erro. Tentando novamente em 10 segundos...")
                    time.sleep(10)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoramento interrompido pelo usuário")
            if process and process.poll() is None:
                print("🔄 Terminando processo do pipeline...")
                process.terminate()
                process.wait()
        
        print(f"\n📋 Monitor finalizado.")

def main():
    monitor = AdvancedPipelineMonitor()
    monitor.run_pipeline_continuous()

if __name__ == "__main__":
    main()