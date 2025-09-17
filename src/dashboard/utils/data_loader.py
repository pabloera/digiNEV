"""
Data Loader para Dashboard digiNEV
Carrega e gerencia dados de saída do pipeline para visualização
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoader:
    """Carregador de dados para o dashboard digiNEV"""
    
    def __init__(self, project_root: Path):
        """
        Inicializa o carregador de dados
        
        Args:
            project_root: Caminho raiz do projeto
        """
        self.project_root = project_root
        self.output_dir = project_root / "pipeline_outputs"
        self.data_dir = project_root / "data"
        
        # Cache de dados
        self._cache = {}
        self._cache_timestamps = {}
        
        # Mapeamento de arquivos esperados
        self.expected_files = {
            'dataset_stats': '01_dataset_stats.csv',
            'political_analysis': '05_political_analysis.csv',
            'cleaned_text': '06_cleaned_text.csv',
            'sentiment_analysis': '08_sentiment_analysis.csv',
            'topic_modeling': '09_topic_modeling.csv',
            'clustering_results': '11_clustering_results.csv',
            'domain_analysis': '13_domain_analysis.csv',
            'temporal_analysis': '14_temporal_analysis.csv',
            'network_metrics': '15_network_metrics.csv',
            'qualitative_coding': '16_qualitative_coding.csv',
            'semantic_search_index': '19_semantic_search_index.csv',
            'validation_report': '20_validation_report.csv'
        }
    
    def get_data_status(self) -> Dict[str, Any]:
        """
        Retorna status dos dados disponíveis
        
        Returns:
            Dicionário com informações de status
        """
        available_files = 0
        missing_files = []
        file_sizes = {}
        
        for data_type, filename in self.expected_files.items():
            file_path = self.output_dir / filename
            if file_path.exists():
                available_files += 1
                file_sizes[data_type] = file_path.stat().st_size
            else:
                missing_files.append(data_type)
        
        # Verificar última execução
        last_execution = "N/A"
        if self.output_dir.exists():
            try:
                # Pegar timestamp do arquivo mais recente
                files = list(self.output_dir.glob("*.csv")) + list(self.output_dir.glob("*.json"))
                if files:
                    latest_file = max(files, key=lambda f: f.stat().st_mtime)
                    last_execution = datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            except Exception as e:
                logger.warning(f"Erro ao determinar última execução: {e}")
        
        return {
            'available_files': available_files,
            'total_files': len(self.expected_files),
            'missing_files': missing_files,
            'file_sizes': file_sizes,
            'last_execution': last_execution,
            'output_dir_exists': self.output_dir.exists()
        }
    
    def load_data(self, data_type: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Carrega dados de um tipo específico
        
        Args:
            data_type: Tipo de dados a carregar
            use_cache: Se deve usar cache
            
        Returns:
            DataFrame com os dados ou None se não encontrado
        """
        if data_type not in self.expected_files:
            logger.error(f"Tipo de dados desconhecido: {data_type}")
            return None
        
        filename = self.expected_files[data_type]
        file_path = self.output_dir / filename
        
        # Verificar cache
        if use_cache and data_type in self._cache:
            cached_time = self._cache_timestamps.get(data_type, 0)
            if file_path.exists():
                file_time = file_path.stat().st_mtime
                if cached_time >= file_time:
                    logger.debug(f"Usando dados cached para {data_type}")
                    return self._cache[data_type]
        
        # Carregar dados
        if not file_path.exists():
            logger.warning(f"Arquivo não encontrado: {file_path}")
            return None
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
                logger.info(f"Carregados {len(df)} registros de {data_type}")
            elif filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])
                logger.info(f"Carregados dados JSON de {data_type}")
            else:
                logger.error(f"Formato de arquivo não suportado: {filename}")
                return None
            
            # Atualizar cache
            if use_cache:
                self._cache[data_type] = df
                self._cache_timestamps[data_type] = file_path.stat().st_mtime
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar {data_type}: {e}")
            return None
    
    def load_multiple_data(self, data_types: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Carrega múltiplos tipos de dados
        
        Args:
            data_types: Lista de tipos de dados
            
        Returns:
            Dicionário com dados carregados
        """
        results = {}
        for data_type in data_types:
            results[data_type] = self.load_data(data_type)
        
        return results
    
    def get_available_data_types(self) -> List[str]:
        """
        Retorna lista de tipos de dados disponíveis
        
        Returns:
            Lista de tipos de dados disponíveis
        """
        available = []
        for data_type, filename in self.expected_files.items():
            file_path = self.output_dir / filename
            if file_path.exists():
                available.append(data_type)
        
        return available
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo dos dados disponíveis
        
        Returns:
            Dicionário com resumo dos dados
        """
        summary = {}
        available_types = self.get_available_data_types()
        
        for data_type in available_types:
            df = self.load_data(data_type)
            if df is not None:
                summary[data_type] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                    'columns_list': list(df.columns)
                }
        
        return summary
    
    def search_data(self, query: str, data_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Busca dados baseado em query
        
        Args:
            query: Termo de busca
            data_types: Tipos de dados para buscar (None = todos)
            
        Returns:
            Resultados da busca por tipo de dados
        """
        if data_types is None:
            data_types = self.get_available_data_types()
        
        results = {}
        query_lower = query.lower()
        
        for data_type in data_types:
            df = self.load_data(data_type)
            if df is None:
                continue
            
            # Buscar em colunas de texto
            text_columns = df.select_dtypes(include=['object']).columns
            matches = pd.DataFrame()
            
            for col in text_columns:
                try:
                    mask = df[col].astype(str).str.lower().str.contains(query_lower, na=False)
                    if mask.any():
                        matches = pd.concat([matches, df[mask]], ignore_index=True)
                except Exception as e:
                    logger.warning(f"Erro ao buscar em {col}: {e}")
            
            if not matches.empty:
                # Remover duplicatas
                matches = matches.drop_duplicates()
                results[data_type] = matches
        
        return results
    
    def export_data(self, data_type: str, format: str = 'csv') -> Optional[Path]:
        """
        Exporta dados em formato específico
        
        Args:
            data_type: Tipo de dados
            format: Formato de exportação (csv, json, excel)
            
        Returns:
            Caminho do arquivo exportado ou None
        """
        df = self.load_data(data_type)
        if df is None:
            return None
        
        # Criar diretório de exportação
        export_dir = self.project_root / "exports"
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if format == 'csv':
                export_path = export_dir / f"{data_type}_{timestamp}.csv"
                df.to_csv(export_path, index=False)
            elif format == 'json':
                export_path = export_dir / f"{data_type}_{timestamp}.json"
                df.to_json(export_path, orient='records', indent=2)
            elif format == 'excel':
                export_path = export_dir / f"{data_type}_{timestamp}.xlsx"
                df.to_excel(export_path, index=False)
            else:
                logger.error(f"Formato não suportado: {format}")
                return None
            
            logger.info(f"Dados exportados para: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Erro ao exportar dados: {e}")
            return None
    
    def clear_cache(self):
        """Limpa o cache de dados"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache de dados limpo")