#!/usr/bin/env python3
"""
Monitorador de Custos da API Anthropic
Rastreia uso e calcula custos estimados

Autor: Pablo Almada
Data: 2025-05-29
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AnthropicCostMonitor:
    """
    Monitor de custos da API Anthropic
    Rastreia tokens de entrada/saída e calcula custos estimados
    """

    # Preços por token (valores de maio 2025 - verificar atualizações)
    TOKEN_PRICES = {
        'claude-sonnet-4-20250514': {
            'input': 0.000003,   # $3 per million input tokens
            'output': 0.000015   # $15 per million output tokens
        },
        'claude-3-5-sonnet-20241022': {
            'input': 0.000003,
            'output': 0.000015
        },
        'claude-3-5-haiku-20241022': {
            'input': 0.00000025,
            'output': 0.00000125
        }
    }

    def __init__(self, project_dir: Path):
        """
        Inicializa o monitor de custos

        Args:
            project_dir: Diretório base do projeto
        """
        self.project_dir = Path(project_dir)
        self.cost_file = self.project_dir / 'logs' / 'anthropic_costs.json'
        self.session_start = datetime.now()
        self.lock = threading.Lock()

        # Criar diretório de logs se não existir
        self.cost_file.parent.mkdir(parents=True, exist_ok=True)

        # Carregar dados existentes
        self.cost_data = self._load_cost_data()

    def _load_cost_data(self) -> Dict[str, Any]:
        """Carrega dados de custo existentes"""
        if self.cost_file.exists():
            try:
                with open(self.cost_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar dados de custo: {e}")

        return {
            'total_cost': 0.0,
            'sessions': [],
            'daily_usage': {},
            'by_model': {},
            'by_stage': {}
        }

    def _save_cost_data(self):
        """Salva dados de custo no arquivo"""
        try:
            with open(self.cost_file, 'w', encoding='utf-8') as f:
                json.dump(self.cost_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Erro ao salvar dados de custo: {e}")

    def record_usage(self,
                    model: str,
                    input_tokens: int,
                    output_tokens: int,
                    stage: str = 'unknown',
                    operation: str = 'general') -> float:
        """
        Registra uso da API e calcula custo

        Args:
            model: Modelo usado
            input_tokens: Tokens de entrada
            output_tokens: Tokens de saída
            stage: Etapa do pipeline
            operation: Operação específica

        Returns:
            Custo estimado da operação
        """
        with self.lock:
            # Calcular custo
            if model in self.TOKEN_PRICES:
                prices = self.TOKEN_PRICES[model]
                input_cost = input_tokens * prices['input']
                output_cost = output_tokens * prices['output']
                total_cost = input_cost + output_cost
            else:
                # Usar preços do modelo padrão se modelo não conhecido
                prices = self.TOKEN_PRICES['claude-sonnet-4-20250514']
                input_cost = input_tokens * prices['input']
                output_cost = output_tokens * prices['output']
                total_cost = input_cost + output_cost
                logger.warning(f"Modelo {model} não encontrado, usando preços padrão")

            # Registrar dados
            now = datetime.now()
            today = now.strftime('%Y-%m-%d')

            # Atualizar total
            self.cost_data['total_cost'] += total_cost

            # Atualizar uso diário
            if today not in self.cost_data['daily_usage']:
                self.cost_data['daily_usage'][today] = {
                    'cost': 0.0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'requests': 0
                }

            daily = self.cost_data['daily_usage'][today]
            daily['cost'] += total_cost
            daily['input_tokens'] += input_tokens
            daily['output_tokens'] += output_tokens
            daily['requests'] += 1

            # Atualizar por modelo
            if model not in self.cost_data['by_model']:
                self.cost_data['by_model'][model] = {
                    'cost': 0.0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'requests': 0
                }

            model_data = self.cost_data['by_model'][model]
            model_data['cost'] += total_cost
            model_data['input_tokens'] += input_tokens
            model_data['output_tokens'] += output_tokens
            model_data['requests'] += 1

            # Atualizar por etapa
            if stage not in self.cost_data['by_stage']:
                self.cost_data['by_stage'][stage] = {
                    'cost': 0.0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'requests': 0
                }

            stage_data = self.cost_data['by_stage'][stage]
            stage_data['cost'] += total_cost
            stage_data['input_tokens'] += input_tokens
            stage_data['output_tokens'] += output_tokens
            stage_data['requests'] += 1

            # Registrar evento individual
            usage_record = {
                'timestamp': now.isoformat(),
                'model': model,
                'stage': stage,
                'operation': operation,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost
            }

            # Adicionar à sessão atual
            current_session = self._get_current_session()
            current_session['operations'].append(usage_record)
            current_session['total_cost'] += total_cost

            # Salvar dados
            self._save_cost_data()

            logger.info(f"API Usage - Stage: {stage}, Tokens: {input_tokens}→{output_tokens}, Cost: ${total_cost:.6f}")

            return total_cost

    def _get_current_session(self) -> Dict[str, Any]:
        """Obtém ou cria sessão atual"""
        session_id = self.session_start.strftime('%Y%m%d_%H%M%S')

        # Procurar sessão existente
        for session in self.cost_data['sessions']:
            if session['session_id'] == session_id:
                return session

        # Criar nova sessão
        new_session = {
            'session_id': session_id,
            'start_time': self.session_start.isoformat(),
            'total_cost': 0.0,
            'operations': []
        }

        self.cost_data['sessions'].append(new_session)
        return new_session

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo de uso da API

        Returns:
            Dicionário com estatísticas de uso
        """
        with self.lock:
            today = datetime.now().strftime('%Y-%m-%d')

            return {
                'total_cost': self.cost_data['total_cost'],
                'today_cost': self.cost_data['daily_usage'].get(today, {}).get('cost', 0.0),
                'sessions_count': len(self.cost_data['sessions']),
                'by_model': dict(self.cost_data['by_model']),
                'by_stage': dict(self.cost_data['by_stage']),
                'last_7_days': self._get_last_n_days_usage(7),
                'current_session': self._get_current_session()
            }

    def _get_last_n_days_usage(self, n_days: int) -> Dict[str, float]:
        """Retorna uso dos últimos N dias"""
        from datetime import datetime, timedelta

        usage = {}
        base_date = datetime.now()

        for i in range(n_days):
            date = (base_date - timedelta(days=i)).strftime('%Y-%m-%d')
            usage[date] = self.cost_data['daily_usage'].get(date, {}).get('cost', 0.0)

        return usage

    def check_cost_limits(self,
                         daily_limit: float = 10.0,
                         session_limit: float = 5.0) -> Dict[str, Any]:
        """
        Verifica limites de custo

        Args:
            daily_limit: Limite diário em USD
            session_limit: Limite por sessão em USD

        Returns:
            Status dos limites
        """
        today = datetime.now().strftime('%Y-%m-%d')
        today_cost = self.cost_data['daily_usage'].get(today, {}).get('cost', 0.0)
        session_cost = self._get_current_session()['total_cost']

        return {
            'daily_limit': daily_limit,
            'daily_usage': today_cost,
            'daily_remaining': max(0, daily_limit - today_cost),
            'daily_exceeded': today_cost > daily_limit,
            'session_limit': session_limit,
            'session_usage': session_cost,
            'session_remaining': max(0, session_limit - session_cost),
            'session_exceeded': session_cost > session_limit
        }

    def generate_cost_report(self) -> str:
        """
        Gera relatório detalhado de custos

        Returns:
            Relatório formatado
        """
        summary = self.get_usage_summary()
        limits = self.check_cost_limits()

        report = f"""
📊 RELATÓRIO DE CUSTOS API ANTHROPIC
{'='*50}

💰 CUSTOS TOTAIS:
   Total Geral: ${summary['total_cost']:.4f}
   Hoje: ${summary['today_cost']:.4f}
   Sessão Atual: ${summary['current_session']['total_cost']:.4f}

🚦 LIMITES:
   Diário: ${limits['daily_usage']:.4f} / ${limits['daily_limit']:.2f} ({'❌ EXCEDIDO' if limits['daily_exceeded'] else '✅ OK'})
   Sessão: ${limits['session_usage']:.4f} / ${limits['session_limit']:.2f} ({'❌ EXCEDIDO' if limits['session_exceeded'] else '✅ OK'})

📈 USO POR MODELO:
"""

        for model, data in summary['by_model'].items():
            report += f"   {model}: ${data['cost']:.4f} ({data['requests']} req)\n"

        report += "\n🔧 USO POR ETAPA:\n"
        for stage, data in summary['by_stage'].items():
            report += f"   {stage}: ${data['cost']:.4f} ({data['requests']} req)\n"

        report += f"\n📅 ÚLTIMOS 7 DIAS:\n"
        for date, cost in summary['last_7_days'].items():
            report += f"   {date}: ${cost:.4f}\n"

        return report


# Instância global para facilitar uso
_cost_monitor = None


def get_cost_monitor(project_dir: Optional[Path] = None) -> AnthropicCostMonitor:
    """
    Retorna instância global do monitor de custos

    Args:
        project_dir: Diretório do projeto (usado apenas na primeira chamada)

    Returns:
        Instância do monitor
    """
    global _cost_monitor

    if _cost_monitor is None:
        if project_dir is None:
            # Usar diretório atual como fallback
            project_dir = Path.cwd()
        _cost_monitor = AnthropicCostMonitor(project_dir)

    return _cost_monitor
