"""
Módulo com os modelos Pydantic para validação dos arquivos de configuração.

Este módulo centraliza a estrutura esperada de todos os arquivos .yaml
do projeto, garantindo que qualquer configuração carregada seja
validada em termos de tipo, presença e valores antes de ser usada
pela aplicação.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt, FilePath, DirectoryPath

# Modelos para as seções de configuração

class PathsConfig(BaseModel):
    """Schema para a configuração de caminhos (paths.yaml)."""
    root_dir: DirectoryPath
    data_dir: DirectoryPath
    output_dir: DirectoryPath
    config_dir: DirectoryPath
    log_dir: DirectoryPath
    cache_dir: DirectoryPath

class CoreConfig(BaseModel):
    """Schema para a configuração do núcleo (core.yaml)."""
    project_name: str = "digiNEV"
    pipeline_mode: str = Field(..., pattern=r"^(academic|production)$")
    log_level: str = Field(..., pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

class NetworkDashboardConfig(BaseModel):
    """Schema para configuração do dashboard."""
    main: Dict[str, Any] = {}
    data_analysis: Dict[str, Any] = {}
    theme: Dict[str, Any] = {}

class NetworkAPIConfig(BaseModel):
    """Schema para configuração de APIs."""
    anthropic: Dict[str, Any] = {}
    voyage: Dict[str, Any] = {}

class NetworkConfig(BaseModel):
    """Schema para a configuração de rede (network.yaml)."""
    dashboard: NetworkDashboardConfig = NetworkDashboardConfig()
    apis: NetworkAPIConfig = NetworkAPIConfig()
    development: Dict[str, Any] = {}
    production: Dict[str, Any] = {}
    connection_pool: Dict[str, Any] = {}

class ProcessingConfig(BaseModel):
    """Schema para a configuração de processamento (processing.yaml)."""
    max_workers: Optional[PositiveInt] = None
    chunk_size: PositiveInt = 1000

class ApiLimitsConfig(BaseModel):
    """Schema para os limites de API (api_limits.yaml)."""
    anthropic_requests_per_minute: PositiveInt
    voyage_requests_per_minute: PositiveInt

class AcademicSettingsConfig(BaseModel):
    """Schema para as configurações acadêmicas (academic_settings.yaml)."""
    analysis_depth: str
    include_experimental_features: bool

# Modelo Mestre que agrega todas as configurações

class MasterConfig(BaseModel):
    """
    Modelo mestre que agrega e valida a estrutura completa
    da configuração carregada de master.yaml.
    """
    paths: PathsConfig
    core: CoreConfig
    network: NetworkConfig
    processing: ProcessingConfig
    api_limits: ApiLimitsConfig
    academic_settings: AcademicSettingsConfig
    
    # Adicione outras configurações de alto nível aqui se master.yaml as incluir
    # Ex: anthropic: Dict[str, Any]