"""
Enhanced Cost Monitor v4.9.8
============================

Sistema avançado de monitoramento de custos para modelos Claude
com alertas automáticos, controle de orçamento e downgrade automático.

🔧 UPGRADE: Monitoramento por stage específico
💰 CONTROLE: Auto-downgrade quando orçamento excede threshold
📊 RELATÓRIOS: Análise detalhada de custos por modelo
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnhancedCostMonitor:
    """Monitor avançado de custos com controle automático"""

    def __init__(self, project_root: Path, config: Dict[str, Any] = None):
        """
        Inicializa o monitor

        Args:
            project_root: Diretório raiz do projeto
            config: Configuração de custos
        """
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        self.cost_file = self.logs_dir / "enhanced_cost_monitor.json"
        self.daily_reports_dir = self.logs_dir / "cost_reports"
        self.daily_reports_dir.mkdir(exist_ok=True)

        # Configuração de custos por modelo (USD por 1K tokens)
        self.cost_per_1k_tokens = {
            "claude-3-5-haiku-20241022": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-sonnet-4-20250514": {"input": 0.015, "output": 0.075},
            "claude-3-5-haiku-latest": {"input": 0.00025, "output": 0.00125},  # Legacy
        }

        # Configuração de orçamento
        self.config = config or {}
        self.monthly_budget = self.config.get('monthly_budget_limit', 200.0)
        self.alert_threshold = self.config.get('budget_threshold', 0.8)
        self.auto_downgrade_enabled = self.config.get('auto_downgrade', {}).get('enable', True)
        self.fallback_model = self.config.get('auto_downgrade', {}).get('fallback_model', 'claude-3-5-haiku-20241022')

        # Carregar dados existentes
        self.usage_data = self._load_usage_data()

    def _load_usage_data(self) -> Dict[str, Any]:
        """Carrega dados de uso existentes"""
        try:
            if self.cost_file.exists():
                with open(self.cost_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"✅ Dados de custo carregados: {self.cost_file}")
                return data
            else:
                return {
                    "sessions": {},
                    "daily_totals": {},
                    "monthly_totals": {},
                    "alerts": [],
                    "model_usage": {}
                }
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados de custo: {e}")
            return {
                "sessions": {},
                "daily_totals": {},
                "monthly_totals": {},
                "alerts": [],
                "model_usage": {}
            }

    def _save_usage_data(self):
        """Salva dados de uso"""
        try:
            with open(self.cost_file, 'w', encoding='utf-8') as f:
                json.dump(self.usage_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar dados de custo: {e}")

    def record_usage(self, model: str, input_tokens: int, output_tokens: int, 
                    stage: str = 'unknown', operation: str = 'general', 
                    session_id: str = None) -> float:
        """
        Registra uso de API e calcula custo

        Args:
            model: Modelo usado
            input_tokens: Tokens de entrada
            output_tokens: Tokens de saída
            stage: Stage do pipeline
            operation: Operação específica
            session_id: ID da sessão

        Returns:
            Custo calculado em USD
        """
        # Calcular custo
        cost_config = self.cost_per_1k_tokens.get(model, {"input": 0.003, "output": 0.015})
        input_cost = (input_tokens / 1000) * cost_config["input"]
        output_cost = (output_tokens / 1000) * cost_config["output"]
        total_cost = input_cost + output_cost

        # Registrar uso
        timestamp = datetime.now().isoformat()
        today = datetime.now().strftime('%Y-%m-%d')
        month = datetime.now().strftime('%Y-%m')

        usage_record = {
            "timestamp": timestamp,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "stage": stage,
            "operation": operation,
            "session_id": session_id or "default"
        }

        # Atualizar totais diários
        if today not in self.usage_data["daily_totals"]:
            self.usage_data["daily_totals"][today] = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "requests": 0,
                "models": {}
            }

        daily = self.usage_data["daily_totals"][today]
        daily["total_cost"] += total_cost
        daily["total_tokens"] += input_tokens + output_tokens
        daily["requests"] += 1

        if model not in daily["models"]:
            daily["models"][model] = {"cost": 0.0, "tokens": 0, "requests": 0}
        daily["models"][model]["cost"] += total_cost
        daily["models"][model]["tokens"] += input_tokens + output_tokens
        daily["models"][model]["requests"] += 1

        # Atualizar totais mensais
        if month not in self.usage_data["monthly_totals"]:
            self.usage_data["monthly_totals"][month] = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "requests": 0,
                "models": {},
                "stages": {}
            }

        monthly = self.usage_data["monthly_totals"][month]
        monthly["total_cost"] += total_cost
        monthly["total_tokens"] += input_tokens + output_tokens
        monthly["requests"] += 1

        if model not in monthly["models"]:
            monthly["models"][model] = {"cost": 0.0, "tokens": 0, "requests": 0}
        monthly["models"][model]["cost"] += total_cost
        monthly["models"][model]["tokens"] += input_tokens + output_tokens
        monthly["models"][model]["requests"] += 1

        if stage not in monthly["stages"]:
            monthly["stages"][stage] = {"cost": 0.0, "tokens": 0, "requests": 0}
        monthly["stages"][stage]["cost"] += total_cost
        monthly["stages"][stage]["tokens"] += input_tokens + output_tokens
        monthly["stages"][stage]["requests"] += 1

        # Adicionar à sessão
        if session_id:
            if session_id not in self.usage_data["sessions"]:
                self.usage_data["sessions"][session_id] = []
            self.usage_data["sessions"][session_id].append(usage_record)

        # Verificar alertas
        self._check_budget_alerts(monthly["total_cost"])

        # Salvar dados
        self._save_usage_data()

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
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message
        }
        self.usage_data["alerts"].append(alert)

        # Manter apenas últimos 100 alertas
        if len(self.usage_data["alerts"]) > 100:
            self.usage_data["alerts"] = self.usage_data["alerts"][-100:]

    def should_auto_downgrade(self) -> bool:
        """Verifica se deve fazer auto-downgrade"""
        if not self.auto_downgrade_enabled:
            return False

        month = datetime.now().strftime('%Y-%m')
        monthly_cost = self.usage_data["monthly_totals"].get(month, {}).get("total_cost", 0.0)
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

    def get_daily_report(self, date: str = None) -> Dict[str, Any]:
        """
        Gera relatório diário

        Args:
            date: Data no formato YYYY-MM-DD (default: hoje)

        Returns:
            Relatório diário
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        daily_data = self.usage_data["daily_totals"].get(date, {})
        return {
            "date": date,
            "total_cost": daily_data.get("total_cost", 0.0),
            "total_tokens": daily_data.get("total_tokens", 0),
            "total_requests": daily_data.get("requests", 0),
            "models": daily_data.get("models", {}),
            "cost_per_request": daily_data.get("total_cost", 0) / max(daily_data.get("requests", 1), 1),
            "tokens_per_request": daily_data.get("total_tokens", 0) / max(daily_data.get("requests", 1), 1)
        }

    def get_monthly_report(self, month: str = None) -> Dict[str, Any]:
        """
        Gera relatório mensal

        Args:
            month: Mês no formato YYYY-MM (default: mês atual)

        Returns:
            Relatório mensal
        """
        if month is None:
            month = datetime.now().strftime('%Y-%m')

        monthly_data = self.usage_data["monthly_totals"].get(month, {})
        total_cost = monthly_data.get("total_cost", 0.0)
        budget_usage = total_cost / self.monthly_budget

        return {
            "month": month,
            "total_cost": total_cost,
            "budget_limit": self.monthly_budget,
            "budget_usage_percent": budget_usage * 100,
            "remaining_budget": self.monthly_budget - total_cost,
            "total_tokens": monthly_data.get("total_tokens", 0),
            "total_requests": monthly_data.get("requests", 0),
            "models": monthly_data.get("models", {}),
            "stages": monthly_data.get("stages", {}),
            "avg_cost_per_request": total_cost / max(monthly_data.get("requests", 1), 1),
            "projected_monthly_cost": total_cost * (30 / datetime.now().day) if datetime.now().day > 0 else total_cost
        }

    def save_daily_report(self, date: str = None):
        """Salva relatório diário em arquivo"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        report = self.get_daily_report(date)
        report_file = self.daily_reports_dir / f"cost_report_{date}.json"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"📊 Relatório diário salvo: {report_file}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar relatório diário: {e}")


# Instância singleton
_enhanced_cost_monitor_instance = None


def get_enhanced_cost_monitor(project_root: Path, config: Dict[str, Any] = None) -> EnhancedCostMonitor:
    """
    Obtém instância singleton do EnhancedCostMonitor

    Args:
        project_root: Diretório raiz do projeto
        config: Configuração de custos

    Returns:
        Instância do EnhancedCostMonitor
    """
    global _enhanced_cost_monitor_instance

    if _enhanced_cost_monitor_instance is None:
        _enhanced_cost_monitor_instance = EnhancedCostMonitor(project_root, config)
        logger.info("💰 EnhancedCostMonitor inicializado")

    return _enhanced_cost_monitor_instance