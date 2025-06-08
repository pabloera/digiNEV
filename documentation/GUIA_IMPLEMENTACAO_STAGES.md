# Guia de Implementação dos 13 Stages Centralizados

## Princípio Fundamental

**REGRA ABSOLUTA**: Nenhum stage deve ser implementado sem Anthropic API, exceto para:
1. **Carregamento de arquivos** (leitura CSV básica)
2. **Funções muito simples** (contagens, validações estruturais)
3. **Operações de I/O** (salvar checkpoints)

## Status de Implementação dos Stages

### ✅ **Stages Totalmente Implementados com Anthropic**

#### **Stage 03: Text Cleaning**
- **Módulo**: `src/anthropic_integration/text_cleaner.py`
- **Classe**: `AnthropicTextCleaner`
- **Método**: `clean_text_intelligent()`
- **Status**: ✅ Completamente implementado
- **Funcionalidade**: Limpeza contextual preservando significado político

#### **Stage 04: Sentiment Analysis**
- **Módulo**: `src/anthropic_integration/sentiment_analyzer.py`
- **Classe**: `AnthropicSentimentAnalyzer`
- **Método**: `analyze_sentiment_comprehensive()`
- **Status**: ✅ Completamente implementado
- **Funcionalidade**: Análise multi-dimensional de sentimentos políticos

#### **Stage 05: Topic Modeling**
- **Módulo**: `src/anthropic_integration/topic_interpreter.py`
- **Classe**: `TopicInterpreter`
- **Método**: `interpret_topics_comprehensive()`
- **Status**: ✅ Completamente implementado
- **Funcionalidade**: Interpretação semântica de tópicos LDA

#### **Stage 07: Clustering**
- **Módulo**: `src/anthropic_integration/cluster_validator.py`
- **Classe**: `ClusterValidator`
- **Método**: `validate_clusters_comprehensive()`
- **Status**: ✅ Completamente implementado
- **Funcionalidade**: Validação e interpretação semântica de clusters

#### **Stage 12: Qualitative Analysis**
- **Módulo**: `src/anthropic_integration/qualitative_classifier.py`
- **Classe**: `QualitativeClassifier`
- **Método**: `classify_content_comprehensive()`
- **Status**: ✅ Completamente implementado
- **Funcionalidade**: Classificação de conspiração e negacionismo

### 🆕 **Stages com Módulos Antropic Recém-Criados**

#### **Stage 02: Encoding Fix**
- **Módulo**: `src/anthropic_integration/smart_encoding_fixer.py`
- **Classe**: `SmartEncodingFixer`
- **Método**: `fix_encoding_intelligent()`
- **Status**: ✅ Módulo criado, integração centralizada
- **Funcionalidade**: Correção contextual de encoding

#### **Stage 02b: Deduplication**
- **Módulo**: `src/anthropic_integration/intelligent_deduplicator.py`
- **Classe**: `IntelligentDeduplicator`
- **Método**: `deduplicate_intelligent()`
- **Status**: ✅ Módulo criado, integração centralizada
- **Funcionalidade**: Deduplicação semântica avançada

#### **Stage 06: TF-IDF Extraction**
- **Módulo**: `src/anthropic_integration/semantic_tfidf_analyzer.py`
- **Classe**: `SemanticTfidfAnalyzer`
- **Método**: `extract_semantic_tfidf()`
- **Status**: ✅ Módulo criado, integração centralizada
- **Funcionalidade**: TF-IDF com interpretação semântica

#### **Stage 09: Domain Extraction**
- **Módulo**: `src/anthropic_integration/intelligent_domain_analyzer.py`
- **Classe**: `IntelligentDomainAnalyzer`
- **Método**: `analyze_domains_intelligent()`
- **Status**: ✅ Módulo criado, integração centralizada
- **Funcionalidade**: Classificação e análise de credibilidade de domínios

#### **Stage 10: Temporal Analysis**
- **Módulo**: `src/anthropic_integration/smart_temporal_analyzer.py`
- **Classe**: `SmartTemporalAnalyzer`
- **Método**: `analyze_temporal_patterns()`
- **Status**: ✅ Módulo criado, integração centralizada
- **Funcionalidade**: Detecção e interpretação de eventos temporais

#### **Stage 11: Network Structure**
- **Módulo**: `src/anthropic_integration/intelligent_network_analyzer.py`
- **Classe**: `IntelligentNetworkAnalyzer`
- **Método**: `analyze_networks_intelligent()`
- **Status**: ✅ Módulo criado, integração centralizada
- **Funcionalidade**: Análise de redes com interpretação de comunidades

#### **Stage 13: Review & Reproducibility**
- **Módulo**: `src/anthropic_integration/smart_pipeline_reviewer.py`
- **Classe**: `SmartPipelineReviewer`
- **Método**: `review_pipeline_comprehensive()`
- **Status**: ✅ Módulo criado, integração centralizada
- **Funcionalidade**: Revisão inteligente de qualidade do pipeline

### ⚙️ **Stages com Implementação Básica (Sem AI por Design)**

#### **Stage 01: Data Validation**
- **Razão**: Performance e eficiência para validação estrutural
- **Implementação**: Validação básica de estrutura CSV
- **Funcionalidade**: 
  - Verificação de colunas obrigatórias
  - Contagem de linhas e colunas
  - Detecção de valores nulos
  - Validação de tipos básicos

```python
def _execute_traditional_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validação tradicional de dados - ÚNICO caso sem AI"""
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'validation_method': 'traditional'
    }
    return df, validation_results
```

#### **Stage 01b: Feature Extraction**
- **Módulo**: `src/utils/auto_column_detector.py` (AutoColumnDetectorAI)
- **Status**: ✅ Já utiliza Anthropic
- **Funcionalidade**: Extração inteligente de características políticas

#### **Stage 08: Hashtag Normalization**
- **Status**: 🔄 Precisa de módulo Anthropic específico
- **Implementação Atual**: Normalização básica por regex
- **Implementação Necessária**: Agrupamento semântico inteligente

```python
# Implementação básica temporária (apenas para I/O)
def _execute_traditional_hashtag_normalization(self, df: pd.DataFrame):
    """Normalização básica - TEMPORÁRIA até módulo AI"""
    # Apenas limpeza básica e lowercase
    # TODO: Implementar módulo Anthropic específico
```

## Padrões de Implementação

### 1. **Template de Módulo Anthropic**

```python
class IntelligentStageModule(AnthropicBase):
    """
    Módulo inteligente para Stage XX
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configurações específicas do stage
        stage_config = config.get('stage_config_section', {})
        self.param1 = stage_config.get('param1', default_value)
    
    def analyze_intelligent(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Método principal de análise inteligente
        
        Args:
            df: DataFrame com dados
            **kwargs: Parâmetros específicos
            
        Returns:
            Tuple[DataFrame processado, Métricas de análise]
        """
        self.logger.info("Iniciando análise inteligente")
        
        # Processamento em chunks se necessário
        if len(df) > 10000:
            return self._process_in_chunks(df, **kwargs)
        
        # Análise direta para datasets menores
        return self._analyze_full_dataset(df, **kwargs)
    
    def _process_in_chunks(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Processamento em chunks para datasets grandes"""
        chunk_size = 5000
        results = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_result = self._analyze_chunk(chunk, **kwargs)
            results.append(chunk_result)
        
        # Consolidar resultados
        return self._consolidate_results(results)
    
    def _analyze_chunk(self, chunk: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de um chunk específico com AI"""
        # Preparar dados para análise
        sample_data = self._prepare_sample_for_ai(chunk)
        
        prompt = f"""
        Analise este chunk de dados do Telegram brasileiro (2019-2023):
        
        CONTEXTO: {self._get_brazilian_context()}
        
        DADOS: {sample_data}
        
        TAREFA ESPECÍFICA: [Descrever tarefa do stage]
        
        Responda em JSON com análise específica para este stage.
        """
        
        try:
            response = self.create_message(
                prompt=prompt,
                stage=f'XX_stage_name',
                operation='analyze_chunk'
            )
            
            analysis = self.parse_json_response(response)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise AI: {e}")
            raise
```

### 2. **Template de Integração no Pipeline Executor**

```python
def execute_stage_XX_description(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Stage XX: Descrição - Funcionalidade específica"""
    self.logger.info("🎯 Executando Stage XX: Descrição")
    
    stage_instance = self.stage_factory.create_stage('XX_stage_name')
    
    if hasattr(stage_instance, 'analyze_intelligent'):  # Usando AI
        return stage_instance.analyze_intelligent(
            df, 
            param1=self.config.get('stage_config', {}).get('param1', default)
        )
    else:  # APENAS para funções muito simples
        return self._execute_simple_operation(df)

def _execute_simple_operation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Operação simples - APENAS para:
    - Carregamento de arquivos
    - Contagens básicas  
    - Validações estruturais
    - I/O operations
    """
    # Implementação extremamente básica
    simple_result = {
        'method': 'simple_operation',
        'rows_processed': len(df),
        'operation': 'basic_structural_check'
    }
    
    return df, simple_result
```

### 3. **Template de Factory Integration**

```python
def _create_stage_XX(self, **kwargs) -> Any:
    """Stage XX: Descrição"""
    use_anthropic = self.config.get('stage_config', {}).get('use_anthropic', True)  # Padrão TRUE
    
    if use_anthropic and self.anthropic_available:
        try:
            from src.anthropic_integration.intelligent_module import IntelligentModule
            self.logger.info("🤖 Stage XX: Usando análise inteligente")
            return IntelligentModule(self.config)
        except Exception as e:
            self.logger.warning(f"Falha na AI Stage XX: {e}")
            if self._is_complex_analysis():
                raise  # FAIL se análise complexa
    
    # Fallback APENAS para operações simples
    if self._is_simple_operation():
        self.logger.info("🔧 Stage XX: Usando operação básica")
        return None  # Delega para implementação básica
    else:
        raise Exception("Stage XX requer Anthropic API para análise complexa")

def _is_complex_analysis(self) -> bool:
    """Verifica se é análise complexa que requer AI"""
    complex_stages = [
        'sentiment_analysis', 'topic_modeling', 'clustering',
        'domain_analysis', 'temporal_analysis', 'network_analysis',
        'qualitative_analysis', 'text_cleaning', 'deduplication'
    ]
    return any(stage in self.current_stage for stage in complex_stages)

def _is_simple_operation(self) -> bool:
    """Verifica se é operação simples permitida sem AI"""
    simple_operations = [
        'data_validation',  # Validação estrutural
        'file_loading',     # Carregamento de arquivo
        'checkpoint_save'   # Salvar checkpoint
    ]
    return any(op in self.current_stage for op in simple_operations)
```

## Diretrizes de Implementação

### ✅ **O Que DEVE Ser Implementado com Anthropic**

1. **Análise Semântica**
   - Interpretação de conteúdo político
   - Classificação de sentimentos
   - Detecção de temas e narrativas

2. **Processamento Contextual**
   - Limpeza preservando significado
   - Deduplicação semântica
   - Normalização inteligente

3. **Análise Complexa**
   - Detecção de eventos temporais
   - Análise de redes sociais
   - Classificação de desinformação

4. **Interpretação de Resultados**
   - Validação de clusters
   - Interpretação de tópicos
   - Revisão de qualidade

### ❌ **O Que PODE Ser Implementado Tradicionalmente**

1. **Operações de I/O**
   ```python
   # Carregamento básico de arquivo
   df = pd.read_csv(file_path, sep=';', encoding='utf-8')
   
   # Salvamento de checkpoint
   df.to_csv(output_path, sep=';', encoding='utf-8', index=False)
   ```

2. **Validações Estruturais**
   ```python
   # Verificação de colunas obrigatórias
   required_columns = ['texto', 'canal', 'timestamp']
   missing_columns = [col for col in required_columns if col not in df.columns]
   
   # Contagem básica
   total_rows = len(df)
   total_columns = len(df.columns)
   ```

3. **Operações Matemáticas Simples**
   ```python
   # Estatísticas descritivas básicas
   null_counts = df.isnull().sum()
   data_types = df.dtypes
   ```

### 🚫 **O Que NÃO DEVE Ser Implementado Tradicionalmente**

1. **Análise de Conteúdo**
   - ❌ Classificação de sentimentos por regras
   - ❌ Detecção de temas por palavras-chave
   - ❌ Limpeza de texto por regex complexos

2. **Interpretação Semântica**
   - ❌ Agrupamento por similaridade lexical
   - ❌ Classificação de domínios por listas
   - ❌ Detecção de eventos por thresholds

3. **Análise Contextual**
   - ❌ Interpretação de redes por métricas básicas
   - ❌ Validação de resultados por estatísticas
   - ❌ Classificação qualitativa por regras

## Checklist de Implementação

### Para Cada Novo Stage:

- [ ] **Módulo Anthropic criado** em `src/anthropic_integration/`
- [ ] **Classe herda de AnthropicBase**
- [ ] **Método principal implementado** (ex: `analyze_intelligent()`)
- [ ] **Processamento de chunks** para datasets grandes
- [ ] **Prompts contextualizados** para política brasileira
- [ ] **Tratamento de erros** com logging apropriado
- [ ] **Factory integration** em `stage_factory.py`
- [ ] **Executor integration** em `pipeline_executor.py`
- [ ] **Configuração** em `settings.yaml`
- [ ] **Fallback mínimo** apenas para operações triviais

### Para Validação:

- [ ] **Stage executa com Anthropic** quando `use_anthropic: true`
- [ ] **Fallback funciona** apenas para operações simples
- [ ] **Erro appropriado** quando AI necessária mas indisponível
- [ ] **Logging claro** sobre qual método está sendo usado
- [ ] **Resultados consistentes** entre execuções
- [ ] **Performance adequada** para datasets grandes

## Conclusão

Esta implementação garante que **todos os 13 stages utilizem Anthropic API** para análises complexas, mantendo fallbacks **apenas para operações triviais** como carregamento de arquivos e validações estruturais básicas. 

A arquitetura centralizada elimina a necessidade de scripts separados e garante que **todas as atualizações sejam feitas nos arquivos principais**, conforme especificado nos requisitos do projeto.