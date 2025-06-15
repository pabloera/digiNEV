"""
Test configuration and fixtures for the Digital Discourse Monitor pipeline.
Provides shared test data, mock objects, and test utilities.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Create and return test data directory."""
    test_dir = project_root / "tests" / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def sample_telegram_data():
    """Create sample Telegram data for testing."""
    
    # Create exactly 100 messages
    all_messages = [
        # Political messages (10)
        'O presidente falou sobre economia hoje #economia #brasil',
        'Política brasileira precisa de mudanças urgentes #política',
        'Manifestação na Paulista reuniu milhares #manifestacao @jornal',
        'Corrupção é problema histórico do país #anticorrupcao',
        'Eleições 2024 serão decisivas #eleicoes2024',
        'STF toma decisão importante sobre democracia',
        'Congresso Nacional aprova nova lei',
        'Ministério da Saúde anuncia medidas #saude',
        'Operação da Polícia Federal #pf #corrupcao',
        'Reforma tributária será votada amanhã',
        
        # Sentiment variety (10)
        'Estou muito feliz com os resultados! Excelente! 😊',
        'Que situação terrível... Muito triste 😢',
        'REVOLTANTE!!! Não aceito essa situação!!!',
        'Amo minha família e amigos ❤️',
        'Notícia neutra sobre o clima hoje',
        'Parabéns pelo aniversário! 🎉',
        'Decepcionado com os resultados 😞',
        'Esperançoso com as mudanças 🙏',
        'Preocupado com o futuro do país',
        'Orgulhoso de ser brasileiro 🇧🇷',
        
        # URLs and links (10)
        'Veja esta notícia: https://globo.com/politica/noticia1',
        'Link importante: https://folha.uol.com.br/poder/2023',
        'YouTube: https://youtube.com/watch?v=abc123',
        'Facebook: https://facebook.com/post/456',
        'Twitter: https://twitter.com/usuario/status/789',
        'Instagram: https://instagram.com/p/xyz',
        'Notícia G1: https://g1.globo.com/politica',
        'UOL: https://uol.com.br/noticias/brasil',
        'CNN: https://cnnbrasil.com.br/politica',
        'BBC: https://bbc.com/portuguese/brasil',
        
        # Hashtags analysis (10)
        '#bolsonaro #lula #política #brasil #eleições',
        '#economia #inflação #pib #mercado #dólar',
        '#saúde #covid19 #vacina #sus #medicina',
        '#educação #universidade #enem #professor',
        '#meio_ambiente #amazônia #sustentabilidade',
        '#justiça #stf #direitos #constituição',
        '#segurança #violência #criminalidade',
        '#cultura #arte #música #cinema #literatura',
        '#esporte #futebol #copa #olimpíadas',
        '#tecnologia #inovação #startup #digital',
        
        # Duplicates for deduplication testing (5)
        'O presidente falou sobre economia hoje #economia #brasil',
        'Política brasileira precisa de mudanças urgentes #política',
        'Manifestação na Paulista reuniu milhares #manifestacao @jornal',
        'Corrupção é problema histórico do país #anticorrupcao',
        'Eleições 2024 serão decisivas #eleicoes2024',
    ]
    
    # Fill remaining slots to reach 100
    remaining_count = 100 - len(all_messages)
    for i in range(remaining_count):
        all_messages.append(f'Mensagem genérica número {i+46} para teste')
    
    return pd.DataFrame({
        'id': list(range(1, 101)),
        'body': all_messages,
        'date': pd.date_range('2023-01-01', periods=100, freq='h'),
        'channel': [f'canal_{i % 10}' for i in range(100)],
        'author': [f'autor_{i % 20}' for i in range(100)],
        'message_id': [f'msg_{i:04d}' for i in range(1, 101)],
        'forwards': [i % 50 for i in range(100)],
        'views': [(i * 10) % 1000 for i in range(100)],
        'replies': [i % 20 for i in range(100)]
    })


@pytest.fixture
def sample_political_data():
    """Create sample data with political content for testing."""
    return pd.DataFrame({
        'id': range(1, 21),
        'body': [
            # Negacionismo
            'Vacinas são experimentais e perigosas #antivacina',
            'Mudanças climáticas são farsa globalista #climategate',
            'Urnas eletrônicas são fraudulentas #fraudeeleitoral',
            'Mídia tradicional só mente #fakemedia',
            # Teorias conspiratórias
            'Globalistas querem destruir o Brasil #globalismo',
            'Comunismo infiltrado nas universidades #comunismo',
            'Nova ordem mundial controla tudo #novaordem',
            'Agenda 2030 é plano de dominação #agenda2030',
            # Autoritarismo
            'STF precisa ser fechado #fechaostf',
            'Congresso não representa o povo #fechaocongresso',
            'Imprensa é inimiga do povo #imprensagolpista',
            'Ditadura militar foi necessária #intervencao',
            # Discurso democrático
            'Precisamos fortalecer as instituições #democracia',
            'Eleições livres são fundamentais #votolivre',
            'Imprensa livre é essencial #imprensalivre',
            'Estado de direito deve prevalecer #justica',
            # Conteúdo neutro
            'Chuva forte hoje na cidade',
            'Receita de bolo deliciosa',
            'Feliz aniversário para todos',
            'Bom dia, pessoal!'
        ],
        'date': pd.date_range('2023-01-01', periods=20, freq='D'),
        'channel': ['canal_politico'] * 20
    })


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = json.dumps({
        "sentiment": "positive",
        "confidence": 0.85,
        "themes": ["politics", "economy"],
        "classification": "democratic"
    })
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_voyage_client():
    """Mock Voyage AI client for testing."""
    mock_client = Mock()
    # Mock embeddings response
    mock_client.embed.return_value = Mock(
        embeddings=[[0.1, 0.2, 0.3] for _ in range(10)]
    )
    return mock_client


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        'anthropic': {
            'enable_api_integration': False,  # Disable for testing
            'api_key': 'test_key',
            'model': 'claude-3-haiku-20240307',
            'max_tokens': 1000,
            'batch_size': 10
        },
        'voyage_embeddings': {
            'enable_sampling': True,
            'max_messages': 1000,
            'model': 'voyage-3-lite'
        },
        'processing': {
            'chunk_size': 100,
            'encoding': 'utf-8',
            'memory_limit': '1GB'
        },
        'data': {
            'path': 'tests/test_data',
            'interim_path': 'tests/test_data/interim',
            'output_path': 'tests/test_data/output',
            'dashboard_path': 'src/dashboard/data'
        }
    }


@pytest.fixture
def temp_csv_file(sample_telegram_data):
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_telegram_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_checkpoints():
    """Mock checkpoints data."""
    return {
        'execution_summary': {
            'completed_stages': 5,
            'total_stages': 22,
            'overall_progress': 0.23,
            'resume_from': '06_text_cleaning'
        },
        'stages': {
            '01_chunk_processing': {
                'status': 'completed',
                'success': True,
                'timestamp': '2023-01-01T10:00:00'
            },
            '02_encoding_validation': {
                'status': 'completed', 
                'success': True,
                'timestamp': '2023-01-01T10:05:00'
            }
        }
    }


@pytest.fixture
def mock_protection_checklist():
    """Mock protection checklist."""
    return {
        'stage_flags': {
            '01_chunk_processing': {
                'completed': True,
                'verified': True,
                'locked': False,
                'can_overwrite': False,
                'success_count': 3,
                'protection_level': 'high'
            },
            '02_encoding_validation': {
                'completed': True,
                'verified': True,
                'locked': True,
                'can_overwrite': False,
                'success_count': 5,
                'protection_level': 'critical'
            }
        },
        'statistics': {
            'total_stages': 22,
            'completed_stages': 2,
            'locked_stages': 1,
            'protected_stages': 2,
            'success_rate': 0.95
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(project_root, test_data_dir):
    """Setup test environment before each test."""
    # Create necessary directories
    (test_data_dir / "interim").mkdir(exist_ok=True)
    (test_data_dir / "output").mkdir(exist_ok=True)
    (test_data_dir / "uploads").mkdir(exist_ok=True)
    
    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    yield
    
    # Cleanup after tests (optional)
    pass


@pytest.fixture
def sample_sentiment_results():
    """Sample sentiment analysis results."""
    return [
        {'text': 'Muito feliz hoje!', 'sentiment': 'positive', 'confidence': 0.9},
        {'text': 'Situação terrível', 'sentiment': 'negative', 'confidence': 0.8},
        {'text': 'Dia normal de trabalho', 'sentiment': 'neutral', 'confidence': 0.7}
    ]


@pytest.fixture
def sample_topic_results():
    """Sample topic modeling results."""
    return {
        'topics': {
            0: {
                'words': ['política', 'eleições', 'democracia', 'voto'],
                'label': 'Processo Democrático'
            },
            1: {
                'words': ['economia', 'inflação', 'dólar', 'pib'],
                'label': 'Economia Nacional'
            }
        },
        'document_topics': [0, 1, 0, 1, 0]
    }


@pytest.fixture
def sample_network_data():
    """Sample network analysis data."""
    return {
        'nodes': [
            {'id': 'canal1', 'type': 'channel'},
            {'id': 'canal2', 'type': 'channel'},
            {'id': 'topic1', 'type': 'topic'}
        ],
        'edges': [
            {'source': 'canal1', 'target': 'topic1', 'weight': 5},
            {'source': 'canal2', 'target': 'topic1', 'weight': 3}
        ]
    }


def create_test_data_file(data: pd.DataFrame, filename: str, test_data_dir: Path) -> str:
    """Helper function to create test data files."""
    file_path = test_data_dir / filename
    data.to_csv(file_path, index=False, encoding='utf-8')
    return str(file_path)


def assert_dataframe_columns(df: pd.DataFrame, expected_columns: List[str]):
    """Assert that DataFrame has expected columns."""
    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    
    assert not missing_cols, f"Missing columns: {missing_cols}"
    assert not extra_cols, f"Unexpected columns: {extra_cols}"


def assert_valid_analysis_result(result: Dict[str, Any], required_keys: List[str]):
    """Assert that analysis result has required structure."""
    missing_keys = set(required_keys) - set(result.keys())
    assert not missing_keys, f"Missing result keys: {missing_keys}"
    assert result.get('success', False), "Analysis should be successful"


class MockAnthropicResponse:
    """Mock Anthropic API response for testing."""
    
    def __init__(self, content_text: str):
        self.content = [Mock()]
        self.content[0].text = content_text


def mock_anthropic_sentiment_response():
    """Create mock sentiment analysis response."""
    return json.dumps({
        "results": [
            {
                "text_index": 0,
                "sentiment": "positive",
                "confidence": 0.85,
                "irony_detected": False,
                "political_stance": "center",
                "themes": ["politics", "optimism"]
            }
        ]
    })


def mock_anthropic_political_response():
    """Create mock political analysis response."""
    return json.dumps({
        "0": {
            "classification": {
                "primary": "democratic_discourse",
                "secondary": ["institutional_support"],
                "confidence": 0.9
            },
            "reasoning": "Content supports democratic institutions",
            "risk_level": "low"
        }
    })
