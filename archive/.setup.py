"""
Setup script para Batch Analyzer independente
Instala o módulo como pacote Python
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lê o README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="batch-analyzer-br",
    version="1.0.0",
    author="Academic Research Team",
    author_email="research@example.com",
    description="Sistema independente de análise em lote para discurso político brasileiro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/batch-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: Portuguese (Brazilian)",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "spacy>=3.7.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "anthropic>=0.18.0",
        "voyageai>=0.2.0",
        "plotly>=5.14.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "diskcache>=5.6.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "academic": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "statsmodels>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "batch-analyzer=batch_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "docs/*.md", "requirements.txt"],
    },
)