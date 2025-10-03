# Integração do Léxico Político Hierarquizado

## 📋 Resumo da Integração

Este documento descreve a integração do arquivo `lexico_politico_hierarquizado.json` no sistema de análise em lote (batch_analyzer).

## 🎯 Objetivos Alcançados

1. **Carregamento Dinâmico**: O léxico político agora é carregado dinamicamente do arquivo JSON
2. **Estrutura Hierárquica**: Suporte para macrotemas (9) e subtemas (28)
3. **Compatibilidade**: Mantida compatibilidade com código existente através de transformação
4. **Fallback**: Sistema usa keywords hardcoded se arquivo não for encontrado

## 🏗️ Arquitetura

### Novo Fluxo de Dados

```
lexico_politico_hierarquizado.json
         ↓
    load_political_lexicon()
         ↓
    _transform_lexicon()
         ↓
    ┌────────────────────┬─────────────────────┐
    │ political_keywords │ transversal_keywords │
    │   (28 subtemas)    │    (9 macrotemas)    │
    └────────────────────┴─────────────────────┘
         ↓
    _heuristic_political_classification()
```

### Estrutura do JSON

```json
{
  "metadata": {...},
  "lexico": {
    "macrotema_1": {
      "nome": "...",
      "subtemas": {
        "subtema_1": {
          "palavras": ["palavra1", "palavra2", ...]
        }
      }
    }
  }
}
```

## 💻 Implementação

### 1. Classe BatchConfig

Adicionados novos métodos:

```python
class BatchConfig:
    LEXICON_FILE = "batch_analyzer/lexico_politico_hierarquizado.json"

    @classmethod
    def load_political_lexicon(cls):
        """Carrega léxico do arquivo JSON"""
        ...

    @classmethod
    def _transform_lexicon(cls, data):
        """Transforma estrutura hierárquica em flat"""
        ...
```

### 2. Classe IntegratedBatchAnalyzer

Modificações no `__init__`:

```python
def __init__(self):
    # Novo: carrega léxico dinamicamente
    self._load_lexicon()

    # Usa self.political_keywords ao invés de self.config.POLITICAL_KEYWORDS
    # Usa self.transversal_keywords ao invés de self.config.TRANSVERSAL_KEYWORDS
```

## 📊 Estatísticas do Léxico

### Antes (Hardcoded)
- **Political Keywords**: 10 categorias, ~300 palavras
- **Transversal Keywords**: 4 categorias, ~100 palavras

### Depois (JSON)
- **Political Keywords**: 28 subtemas, 847 palavras
- **Transversal Keywords**: 9 macrotemas, 847 palavras (agregadas)

## 🧪 Testes Realizados

### Script de Teste: `test_lexicon_integration.py`

✅ **Teste 1**: Carregamento do léxico
- 28 categorias políticas carregadas
- 9 macrotemas carregados

✅ **Teste 2**: Inicialização do analisador
- Analyzer inicializado com sucesso
- Léxico disponível em memória

✅ **Teste 3**: Classificação política
- Amostras classificadas corretamente
- Keywords detectadas

✅ **Teste 4**: Verificação de estrutura
- Arquivo JSON válido
- Metadata correta

## 🔄 Mudanças no Código

### Arquivos Modificados

1. **batch_analysis.py**
   - Adicionado: `load_political_lexicon()`, `_transform_lexicon()`, `_load_lexicon()`
   - Modificado: referências de `self.config.POLITICAL_KEYWORDS` → `self.political_keywords`
   - Modificado: referências de `self.config.TRANSVERSAL_KEYWORDS` → `self.transversal_keywords`

### Compatibilidade

- ✅ Retrocompatibilidade mantida
- ✅ Fallback para keywords hardcoded se arquivo não existir
- ✅ Estrutura de dados compatível com código existente

## 📝 Como Usar

### Uso Básico

```python
from batch_analysis import IntegratedBatchAnalyzer

# O léxico é carregado automaticamente
analyzer = IntegratedBatchAnalyzer()

# Processar dataset
result = analyzer.run_analysis("data.csv")
```

### Personalização

```python
from batch_analysis import BatchConfig

# Carregar léxico manualmente
political, transversal = BatchConfig.load_political_lexicon()

# Usar em análise customizada
for category, keywords in political.items():
    print(f"{category}: {len(keywords)} palavras")
```

## ⚠️ Considerações Importantes

1. **Caminho do Arquivo**: O arquivo `lexico_politico_hierarquizado.json` deve estar em `batch_analyzer/`
2. **Encoding**: UTF-8 obrigatório para caracteres brasileiros
3. **Performance**: Carregamento único na inicialização (sem impacto em runtime)
4. **Memória**: ~200KB adicionais em memória para o léxico completo

## 🚀 Melhorias Futuras

1. **Cache**: Implementar cache do léxico processado
2. **Validação**: Adicionar schema validation para o JSON
3. **Mapeamento**: Melhorar mapeamento subtema → macrotema
4. **Configuração**: Permitir múltiplos arquivos de léxico
5. **Hot Reload**: Recarregar léxico sem reiniciar analyzer

## 📈 Impacto na Análise

### Benefícios
- ✅ **Maior Cobertura**: 847 palavras vs ~400 anteriormente
- ✅ **Melhor Organização**: Hierarquia clara de conceitos
- ✅ **Flexibilidade**: Fácil atualização do léxico sem modificar código
- ✅ **Granularidade**: 28 subcategorias para análise detalhada

### Exemplo de Resultado

```
Texto: "Bolsonaro patriota defende valores tradicionais"
→ Categoria: valores_conservadores (subtema)
→ Macrotema: identidade_patriotica
→ Palavras detectadas: ["patriota", "valores", "tradicionais"]
→ Confiança: 0.85
```

## 📋 Checklist de Validação

- [x] Léxico carrega do JSON
- [x] Transformação para formato flat
- [x] Classificação política funciona
- [x] Fallback para hardcoded
- [x] Testes passando
- [x] Logs informativos
- [x] Documentação criada

---

**Data**: 28/09/2025
**Versão**: 1.0
**Status**: ✅ Integração Completa