# 🚀 INSTRUÇÕES - Como Implementar os Métodos Validados

## Você tem 3 OPÇÕES:

---

## OPÇÃO 1: Batch Gera Tudo (Mais Simples) ✅

### O que fazer:
```bash
# 1. Use o arquivo de configuração
cp batch_validated_methods_config.json /seu/projeto/

# 2. No seu batch, adicione:
import json

with open('batch_validated_methods_config.json', 'r') as f:
    config = json.load(f)

# 3. Para cada stage, o batch verifica a config:
if config['stages']['stage_07_topic_modeling']['methods']['topic_model']['use'] == 'bertopic':
    # Batch implementa BERTopic automaticamente
```

### Vantagem: 
- Batch se auto-configura
- Você só especifica o que quer

### Desvantagem:
- Batch precisa ter todos os métodos implementados

---

## OPÇÃO 2: Usar o Batch Adaptado (Recomendado) ⭐

### O que fazer:
```bash
# 1. Copie os arquivos gerados
cp batch_validated.py /seu/projeto/
cp batch_validated_methods_config.json /seu/projeto/
cp validated_methods_implementation.py /seu/projeto/

# 2. Instale dependências
pip install bertopic hdbscan spacy sentence-transformers pandas numpy scipy sklearn
python -m spacy download pt_core_news_lg

# 3. Execute
python batch_validated.py
```

### Vantagem:
- Pronto para usar
- Todos os métodos validados já implementados
- Citações incluídas

### Desvantagem:
- Precisa instalar várias bibliotecas

---

## OPÇÃO 3: Integração Progressiva (Mais Controle) 🔧

### O que fazer:
```python
# No seu batch existente, substitua método por método:

# ANTES (heurístico):
def _heuristic_topic_modeling(self, texts):
    # código heurístico
    pass

# DEPOIS (validado):
def _heuristic_topic_modeling(self, texts):
    """
    Roberts et al. (2014) - STM
    American Journal of Political Science
    """
    from bertopic import BERTopic
    model = BERTopic(language='portuguese')
    topics, probs = model.fit_transform(texts)
    return {'topics': topics, 'citation': 'Roberts et al. 2014'}
```

### Vantagem:
- Controle total
- Pode fazer gradualmente

### Desvantagem:
- Mais trabalho manual

---

## 📋 CHECKLIST DE IMPLEMENTAÇÃO

### Independente da opção escolhida:

- [ ] Instalar bibliotecas necessárias
- [ ] Baixar modelos de linguagem (spaCy pt)
- [ ] Fazer backup do batch original
- [ ] Testar com amostra pequena primeiro
- [ ] Verificar citações nos outputs

---

## 🎯 RECOMENDAÇÃO FINAL

**Para seu caso específico, recomendo a OPÇÃO 2:**

1. **Use o batch_validated.py gerado**
2. **Configure apenas o JSON**
3. **Execute e valide resultados**

O batch já está configurado para:
- ✅ Usar BERTimbau para embeddings
- ✅ STM para topic modeling  
- ✅ HDBSCAN para clustering
- ✅ Mann-Kendall para tendências
- ✅ Kleinberg para burst detection
- ✅ Todas as citações incluídas

---

## 💻 COMANDO RÁPIDO PARA COMEÇAR

```bash
# Copiar arquivos necessários
cp /home/claude/*.py /seu/projeto/src/
cp /home/claude/*.json /seu/projeto/

# Instalar dependências básicas
pip install pandas numpy scipy scikit-learn

# Executar
cd /seu/projeto
python src/batch_validated.py
```

---

## ⚠️ IMPORTANTE

- **NÃO misture** métodos heurísticos com validados no mesmo stage
- **SEMPRE inclua** citações nos resultados
- **DOCUMENTE** qual método foi usado em cada análise
- **VALIDE** com amostra antes de rodar completo
