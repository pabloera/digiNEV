# Tarefa: Refatoração de Referências de Arquivo de Léxico Político

## 📋 Contexto do Projeto

Esta tarefa visa consolidar e modernizar o sistema de configuração do léxico político, migrando de múltiplos arquivos de configuração para um único arquivo JSON hierarquizado.

## 🎯 Objetivo Principal

Substituir todas as referências aos arquivos antigos de configuração do léxico político pelo novo arquivo consolidado, mantendo a funcionalidade completa do sistema.

## 📁 Mapeamento de Arquivos

### Arquivos Antigos (Remover Referências)
- `config/brazilian_political_lexicon.yaml`
- `config/taxonomia_lexico_integrado.json`

### Novo Arquivo Consolidado
- `batch_analyzer/lexico_politico_hierarquizado.json`

## 📝 Instruções Detalhadas

### 1. Busca e Identificação

**Extensões de arquivo a verificar:**
- `.py` (Python)
- `.yaml`, `.yml` (Configurações YAML)
- `.json` (Configurações JSON)
- `.md` (Documentação)
- `.txt` (Arquivos de texto)
- `.sh` (Scripts Shell)
- `.bat` (Scripts Batch Windows)
- `.ipynb` (Jupyter Notebooks)
- `.toml` (Configurações TOML)
- `.cfg`, `.ini` (Arquivos de configuração)

**Padrões de busca a utilizar:**
```
brazilian_political_lexicon.yaml
taxonomia_lexico_integrado.json
config/brazilian_political_lexicon
config/taxonomia_lexico_integrado
config\\brazilian_political_lexicon  # Windows
config\\taxonomia_lexico_integrado    # Windows
```

### 2. Tipos de Referências a Atualizar

- **Importações e carregamento de arquivos**
- **Caminhos em arquivos de configuração**
- **Referências em documentação (README, docs, etc.)**
- **Scripts de execução e automação**
- **Testes unitários e de integração**
- **Arquivos de ambiente (.env, .env.example)**
- **Docker e docker-compose files**
- **CI/CD pipelines (GitHub Actions, GitLab CI, etc.)**

### 3. Regras de Substituição

#### ⚠️ IMPORTANTE - O que NÃO fazer:
- **NÃO** copiar o conteúdo do arquivo novo para outros locais
- **NÃO** deletar o arquivo `batch_analyzer/lexico_politico_hierarquizado.json`
- **NÃO** modificar o conteúdo do novo arquivo
- **NÃO** criar duplicatas do arquivo em outras pastas

#### ✅ O que FAZER:
- Atualizar apenas os caminhos/referências
- Adaptar o código de leitura se necessário (YAML → JSON)
- Manter backups dos arquivos antes de modificar
- Preservar a lógica de negócio existente

### 4. Exemplos de Substituição

#### Exemplo Python - Importação Direta
**ANTES:**
```python
import yaml

def load_lexicon():
    with open('config/brazilian_political_lexicon.yaml', 'r', encoding='utf-8') as f:
        lexicon = yaml.safe_load(f)
    return lexicon
```

**DEPOIS:**
```python
import json

def load_lexicon():
    with open('batch_analyzer/lexico_politico_hierarquizado.json', 'r', encoding='utf-8') as f:
        lexicon = json.load(f)
    return lexicon
```

#### Exemplo Python - Path Dinâmico
**ANTES:**
```python
from pathlib import Path

LEXICON_PATH = Path('config') / 'brazilian_political_lexicon.yaml'
TAXONOMY_PATH = Path('config') / 'taxonomia_lexico_integrado.json'
```

**DEPOIS:**
```python
from pathlib import Path

LEXICON_PATH = Path('batch_analyzer') / 'lexico_politico_hierarquizado.json'
# TAXONOMY_PATH removido - agora integrado no arquivo único
```

#### Exemplo Configuração - settings.py
**ANTES:**
```python
CONFIG = {
    'lexicon_file': 'config/brazilian_political_lexicon.yaml',
    'taxonomy_file': 'config/taxonomia_lexico_integrado.json',
    'processing': {...}
}
```

**DEPOIS:**
```python
CONFIG = {
    'lexicon_file': 'batch_analyzer/lexico_politico_hierarquizado.json',
    # 'taxonomy_file' removido - integrado no lexicon_file
    'processing': {...}
}
```

#### Exemplo Docker
**ANTES:**
```dockerfile
COPY config/brazilian_political_lexicon.yaml /app/config/
COPY config/taxonomia_lexico_integrado.json /app/config/
```

**DEPOIS:**
```dockerfile
COPY batch_analyzer/lexico_politico_hierarquizado.json /app/batch_analyzer/
```

### 5. Checklist de Validação

- [ ] Buscar por todas as ocorrências dos arquivos antigos
- [ ] Verificar imports de YAML que podem precisar mudança para JSON
- [ ] Atualizar documentação (README, docs, comentários)
- [ ] Verificar scripts de deployment/CI/CD
- [ ] Testar carregamento do novo arquivo
- [ ] Confirmar que estrutura de dados é compatível
- [ ] Executar testes unitários após mudanças
- [ ] Verificar logs para erros de caminho

### 6. Possíveis Incompatibilidades

#### Mudança de Formato (YAML → JSON)
- **Atenção para:** Comentários YAML que não existem em JSON
- **Solução:** Migrar comentários importantes para documentação

#### Estrutura de Dados
- **Verificar:** Se a estrutura hierárquica mudou
- **Adaptar:** Código que acessa chaves específicas

#### Encoding
- **Garantir:** UTF-8 em todas as leituras do novo arquivo

### 7. Comando de Busca Sugerido

```bash
# Linux/Mac
grep -r "brazilian_political_lexicon\|taxonomia_lexico_integrado" \
  --include="*.py" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="*.json" \
  --include="*.md" \
  --include="*.txt" \
  --include="*.sh" \
  --include="*.bat" \
  .

# Alternativa com find
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" \) \
  -exec grep -l "brazilian_political_lexicon\|taxonomia_lexico_integrado" {} \;
```

### 8. Script de Backup Recomendado

```bash
#!/bin/bash
# backup_before_refactor.sh

# Criar pasta de backup com timestamp
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Listar arquivos que serão modificados
FILES=$(grep -rl "brazilian_political_lexicon\|taxonomia_lexico_integrado" \
  --include="*.py" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="*.json" .)

# Copiar arquivos para backup
for file in $FILES; do
  cp --parents "$file" "$BACKUP_DIR/"
done

echo "Backup criado em: $BACKUP_DIR"
echo "Arquivos backupeados: $(echo "$FILES" | wc -l)"
```

## 📊 Relatório Esperado

Ao concluir a tarefa, fornecer:

### 1. Resumo Executivo
- Total de arquivos analisados
- Total de arquivos modificados
- Total de substituições realizadas

### 2. Lista Detalhada de Modificações
```
Arquivo: src/analyzer/lexicon_loader.py
  Linha 15: config/brazilian_political_lexicon.yaml → batch_analyzer/lexico_politico_hierarquizado.json
  Linha 45: yaml.safe_load() → json.load()

Arquivo: tests/test_lexicon.py
  Linha 8: config/taxonomia_lexico_integrado.json → batch_analyzer/lexico_politico_hierarquizado.json
```

### 3. Avisos e Recomendações
- Incompatibilidades encontradas
- Ajustes manuais necessários
- Sugestões de melhorias

### 4. Testes Pós-Refatoração
- [ ] Todos os imports funcionando
- [ ] Arquivo novo sendo carregado corretamente
- [ ] Testes unitários passando
- [ ] Aplicação executando sem erros

## 🚀 Execução no Claude Code

```bash
# Opção 1: Executar diretamente este arquivo
code "Read and execute the refactoring instructions in refactoring_lexicon_task.md"

# Opção 2: Com confirmação passo a passo
code "Read refactoring_lexicon_task.md and show me all files that need changes before modifying"

# Opção 3: Modo seguro com backup
code "First create backups as described in refactoring_lexicon_task.md, then perform the refactoring"
```

## ⚠️ Notas Finais

1. **Sempre fazer backup antes de iniciar**
2. **Testar em ambiente de desenvolvimento primeiro**
3. **Commitar mudanças incrementalmente**
4. **Documentar qualquer decisão de design tomada**
5. **Manter log das mudanças para auditoria**

---

*Documento criado para execução automatizada via Claude Code*
*Versão: 1.0*
*Data: 2025*