#!/usr/bin/env python3
"""
Sistema Consolidado de Monitoramento de Custos Anthropic v4.9.8
===============================================================

Combina funcionalidades dos sistemas original e enhanced em uma única implementação:
- Monitoramento básico de custos (original)
- Enhanced monitoring com alertas e auto-downgrade
- Sistema unificado sem divisão de código

🔧 CONSOLIDAÇÃO: Unifica cost_monitor.py + cost_monitor_enhanced.py
💰 FEATURES: Monitoramento completo, alertas automáticos, controle de orçamento
📊 RELATÓRIOS: Análise detalhada de custos por modelo/stage/operação
"""

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConsolidatedCostMonitor:
    """
    Monitor consolidado de custos da API Anthropic
    
    Funcionalidades:
    - Rastreamento básico de uso e custos
    - Enhanced monitoring com alertas automáticos
    - Auto-downgrade quando orçamento excede threshold
    - Relatórios detalhados por modelo, stage e operação
    - Singleton pattern para uso global
    """

    # Preços por token (valores atualizados junho 2025)
    TOKEN_PRICES = {
        'claude-sonnet-4-20250514': {
            'input': 0.000015,   # $15 per million input tokens
            'output': 0.000075   # $75 per million output tokens
        },
        'claude-3-5-sonnet-20241022': {
            'input': 0.000003,   # $3 per million input tokens
            'output': 0.000015   # $15 per million output tokens
        },
        'claude-3-5-haiku-20241022': {
            'input': 0.00000025, # $0.25 per million input tokens
            'output': 0.00000125 # $1.25 per million output tokens
        }
    }

    def __init__(self, project_dir: Path, config: Dict[str, Any] = None):
        """
        Inicializa o monitor consolidado de custos

        Args:
            project_dir: Diretório base do projeto
            config: Configuração de custos (opcional)
        """
        self.project_dir = Path(project_dir)
        self.config = config or {}
        
        # Configuração de diretórios
        self.logs_dir = self.project_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        # Arquivos de dados
        self.cost_file = self.logs_dir / 'consolidated_cost_monitor.json'
        self.daily_reports_dir = self.logs_dir / 'cost_reports'
        self.daily_reports_dir.mkdir(exist_ok=True)
        
        # Configuração de orçamento e alertas
        self.monthly_budget = self.config.get('monthly_budget_limit', 200.0)
        self.alert_threshold = self.config.get('budget_threshold', 0.8)
        self.auto_downgrade_enabled = self.config.get('auto_downgrade', {}).get('enable', True)
        self.fallback_model = self.config.get('auto_downgrade', {}).get('fallback_model', 'claude-3-5-haiku-20241022')
        
        # Thread safety
        self.session_start = datetime.now()
        self.lock = threading.Lock()
        
        # Carregar dados existentes
        self.cost_data = self._load_cost_data()

    def _load_cost_data(self) -> Dict[str, Any]:
        """Carrega dados de custo existentes"""
        try:
            if self.cost_file.exists():
                with open(self.cost_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"✅ Dados de custo carregados: {self.cost_file}")
                return data
            else:
                logger.info("📊 Inicializando novo sistema de monitoramento de custos")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados de custo: {e}")

        # Estrutura unificada (original + enhanced)
        return {
            # Estrutura original
            'total_cost': 0.0,
            'sessions': [],
            'daily_usage': {},
            'by_model': {},
            'by_stage': {},
            # Estrutura enhanced
            'daily_totals': {},
            'monthly_totals': {},
            'alerts': [],
            'model_usage': {}
        }

    def _save_cost_data(self):
        """Salva dados de custo no arquivo"""
        try:
            with open(self.cost_file, 'w', encoding='utf-8') as f:
                json.dump(self.cost_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar dados de custo: {e}")

    def record_usage(self,
                    model: str,
                    input_tokens: int,
                    output_tokens: int,
                    stage: str = 'unknown',
                    operation: str = 'general',
                    session_id: str = None) -> float:
        """
        Registra uso da API e calcula custo (método consolidado)

        Args:
            model: Modelo usado
            input_tokens: Tokens de entrada
            output_tokens: Tokens de saída
            stage: Etapa do pipeline
            operation: Operação específica
            session_id: ID da sessão (opcional)

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
                prices = self.TOKEN_PRICES['claude-3-5-sonnet-20241022']
                input_cost = input_tokens * prices['input']
                output_cost = output_tokens * prices['output']
                total_cost = input_cost + output_cost
                logger.warning(f"Modelo {model} não encontrado, usando preços padrão")

            # Timestamps
            now = datetime.now()
            timestamp = now.isoformat()
            today = now.strftime('%Y-%m-%d')
            month = now.strftime('%Y-%m')

            # Criar registro de uso
            usage_record = {
                'timestamp': timestamp,
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost,
                'stage': stage,
                'operation': operation,
                'session_id': session_id or 'default'
            }

            # ===== ESTRUTURA ORIGINAL =====
            # Atualizar total geral
            self.cost_data['total_cost'] += total_cost

            # Atualizar uso diário (formato original)
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

            # Atualizar por modelo (formato original)
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

            # Atualizar por stage (formato original)
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

            # ===== ESTRUTURA ENHANCED =====
            # Atualizar totais diários (formato enhanced)
            if today not in self.cost_data['daily_totals']:
                self.cost_data['daily_totals'][today] = {
                    'total_cost': 0.0,
                    'total_tokens': 0,
                    'requests': 0,
                    'models': {}
                }

            daily_enhanced = self.cost_data['daily_totals'][today]
            daily_enhanced['total_cost'] += total_cost
            daily_enhanced['total_tokens'] += input_tokens + output_tokens
            daily_enhanced['requests'] += 1

            if model not in daily_enhanced['models']:
                daily_enhanced['models'][model] = {'cost': 0.0, 'tokens': 0, 'requests': 0}
            daily_enhanced['models'][model]['cost'] += total_cost
            daily_enhanced['models'][model]['tokens'] += input_tokens + output_tokens
            daily_enhanced['models'][model]['requests'] += 1

            # Atualizar totais mensais (formato enhanced)
            if month not in self.cost_data['monthly_totals']:
                self.cost_data['monthly_totals'][month] = {
                    'total_cost': 0.0,
                    'total_tokens': 0,
                    'requests': 0,
                    'models': {},
                    'stages': {}
                }

            monthly = self.cost_data['monthly_totals'][month]
            monthly['total_cost'] += total_cost
            monthly['total_tokens'] += input_tokens + output_tokens
            monthly['requests'] += 1

            if model not in monthly['models']:
                monthly['models'][model] = {'cost': 0.0, 'tokens': 0, 'requests': 0}
            monthly['models'][model]['cost'] += total_cost
            monthly['models'][model]['tokens'] += input_tokens + output_tokens
            monthly['models'][model]['requests'] += 1

            if stage not in monthly['stages']:
                monthly['stages'][stage] = {'cost': 0.0, 'tokens': 0, 'requests': 0}
            monthly['stages'][stage]['cost'] += total_cost
            monthly['stages'][stage]['tokens'] += input_tokens + output_tokens
            monthly['stages'][stage]['requests'] += 1

            # Adicionar à sessão se fornecida
            if session_id:
                if session_id not in self.cost_data['sessions']:
                    self.cost_data['sessions'][session_id] = []
                self.cost_data['sessions'][session_id].append(usage_record)

            # Verificar alertas de orçamento
            self._check_budget_alerts(monthly['total_cost'])

            # Salvar dados
            self._save_cost_data()

            logger.info(f"💰 Custo registrado: ${total_cost:.4f} ({model}, {stage}:{operation})")
            return total_cost

    def _check_budget_alerts(self, current_monthly_cost: float):
        """Verifica e emite alertas de orçamento"""
        budget_usage = current_monthly_cost / self.monthly_budget

        # Alert de 80%
        if budget_usage >= self.alert_threshold and budget_usage < 1.0:
            alert_msg = f"⚠️ ALERTA: Orçamento mensal em {budget_usage:.1%} (${current_monthly_cost:.2f}/${self.monthly_budget})"
            self._add_alert("budget_warning", alert_msg)
            logger.warning(alert_msg)

        # Alert de 100%
        elif budget_usage >= 1.0:
            alert_msg = f"🚨 CRÍTICO: Orçamento mensal excedido em {budget_usage:.1%} (${current_monthly_cost:.2f}/${self.monthly_budget})"
            self._add_alert("budget_exceeded", alert_msg)
            logger.error(alert_msg)

    def _add_alert(self, alert_type: str, message: str):
        """Adiciona alerta ao log"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message
        }
        self.cost_data['alerts'].append(alert)

        # Manter apenas últimos 100 alertas
        if len(self.cost_data['alerts']) > 100:
            self.cost_data['alerts'] = self.cost_data['alerts'][-100:]

    def should_auto_downgrade(self) -> bool:
        """Verifica se deve fazer auto-downgrade"""
        if not self.auto_downgrade_enabled:
            return False

        month = datetime.now().strftime('%Y-%m')
        monthly_cost = self.cost_data['monthly_totals'].get(month, {}).get('total_cost', 0.0)
        budget_usage = monthly_cost / self.monthly_budget

        if budget_usage >= self.alert_threshold:
            logger.warning(f"🔽 Auto-downgrade ativado: {budget_usage:.1%} >= {self.alert_threshold:.1%}")
            return True

        return False

    def get_recommended_model(self, preferred_model: str) -> str:
        """
        Retorna modelo recomendado (pode fazer downgrade automático)

        Args:
            preferred_model: Modelo preferido

        Returns:
            Modelo recomendado (pode ser downgrade)
        """
        if self.should_auto_downgrade():
            if preferred_model != self.fallback_model:
                logger.warning(f"🔽 Downgrade automático: {preferred_model} → {self.fallback_model}")
                return self.fallback_model

        return preferred_model

    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo consolidado de uso (compatível com formato original)
        """
        now = datetime.now()
        
        # Calcular últimos 7 dias
        last_7_days = {}
        for i in range(7):
            date = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            cost = self.cost_data['daily_usage'].get(date, {}).get('cost', 0.0)
            last_7_days[date] = cost

        return {
            'total_cost': self.cost_data['total_cost'],
            'session_duration': str(now - self.session_start),
            'daily_usage': self.cost_data['daily_usage'],
            'by_model': self.cost_data['by_model'],
            'by_stage': self.cost_data['by_stage'],
            'last_7_days': last_7_days,
            'current_session': str(self.session_start),
            # Enhanced data
            'monthly_budget': self.monthly_budget,
            'budget_usage_percent': self._get_current_budget_usage() * 100,
            'auto_downgrade_enabled': self.auto_downgrade_enabled,
            'alerts_count': len(self.cost_data['alerts'])
        }

    def _get_current_budget_usage(self) -> float:
        """Calcula uso atual do orçamento mensal"""
        month = datetime.now().strftime('%Y-%m')
        monthly_cost = self.cost_data['monthly_totals'].get(month, {}).get('total_cost', 0.0)
        return monthly_cost / self.monthly_budget

    def get_daily_report(self, date: str = None) -> Dict[str, Any]:
        """Gera relatório diário (enhanced format)"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Usar dados enhanced se disponíveis, senão usar formato original
        if date in self.cost_data['daily_totals']:
            daily_data = self.cost_data['daily_totals'][date]
        else:
            daily_data = self.cost_data['daily_usage'].get(date, {})

        return {
            'date': date,
            'total_cost': daily_data.get('total_cost', daily_data.get('cost', 0.0)),
            'total_tokens': daily_data.get('total_tokens', 
                daily_data.get('input_tokens', 0) + daily_data.get('output_tokens', 0)),
            'total_requests': daily_data.get('requests', 0),
            'models': daily_data.get('models', {}),
            'cost_per_request': daily_data.get('total_cost', daily_data.get('cost', 0)) / 
                max(daily_data.get('requests', 1), 1),
            'tokens_per_request': daily_data.get('total_tokens', 
                daily_data.get('input_tokens', 0) + daily_data.get('output_tokens', 0)) / 
                max(daily_data.get('requests', 1), 1)
        }

    def get_monthly_report(self, month: str = None) -> Dict[str, Any]:
        """Gera relatório mensal (enhanced format)"""
        if month is None:
            month = datetime.now().strftime('%Y-%m')

        monthly_data = self.cost_data['monthly_totals'].get(month, {})
        total_cost = monthly_data.get('total_cost', 0.0)
        budget_usage = total_cost / self.monthly_budget

        return {
            'month': month,
            'total_cost': total_cost,
            'budget_limit': self.monthly_budget,
            'budget_usage_percent': budget_usage * 100,
            'remaining_budget': self.monthly_budget - total_cost,
            'total_tokens': monthly_data.get('total_tokens', 0),
            'total_requests': monthly_data.get('requests', 0),
            'models': monthly_data.get('models', {}),
            'stages': monthly_data.get('stages', {}),
            'avg_cost_per_request': total_cost / max(monthly_data.get('requests', 1), 1),
            'projected_monthly_cost': total_cost * (30 / datetime.now().day) if datetime.now().day > 0 else total_cost
        }

    def generate_report(self) -> str:
        """
        Gera relatório formatado de uso (formato original mantido)
        """
        summary = self.get_summary()
        
        report = f"""
💰 RELATÓRIO DE CUSTOS ANTHROPIC - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

💵 CUSTO TOTAL: ${summary['total_cost']:.4f}
⏱️ DURAÇÃO DA SESSÃO: {summary['session_duration']}
📊 ORÇAMENTO MENSAL: ${summary['monthly_budget']:.2f} ({summary['budget_usage_percent']:.1f}% usado)
🔽 AUTO-DOWNGRADE: {'Ativado' if summary['auto_downgrade_enabled'] else 'Desativado'}
⚠️ ALERTAS: {summary['alerts_count']} alertas registrados

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


# Instância singleton para compatibilidade
_consolidated_monitor = None


def get_cost_monitor(project_dir: Optional[Path] = None, config: Dict[str, Any] = None) -> ConsolidatedCostMonitor:
    """
    Retorna instância singleton do monitor consolidado (compatível com ambos os sistemas)

    Args:
        project_dir: Diretório do projeto
        config: Configuração de custos (para enhanced features)

    Returns:
        Instância do monitor consolidado
    """
    global _consolidated_monitor

    if _consolidated_monitor is None:
        if project_dir is None:
            project_dir = Path.cwd()
        _consolidated_monitor = ConsolidatedCostMonitor(project_dir, config)
        logger.info("💰 ConsolidatedCostMonitor inicializado")

    return _consolidated_monitor


# Alias para compatibilidade com enhanced system
def get_enhanced_cost_monitor(project_root: Path, config: Dict[str, Any] = None) -> ConsolidatedCostMonitor:
    """
    Alias para compatibilidade com sistema enhanced
    """
    return get_cost_monitor(project_root, config)