# 🚀 Como Iniciar o Dashboard

## ✅ **Dashboard Testado e Funcionando**

O dashboard foi testado com sucesso e está pronto para uso!

## 🔧 **Comandos para Iniciar**

### **Opção 1: Script Automatizado** (Recomendado)
```bash
cd src/dashboard
python start_dashboard.py
```

### **Opção 2: Streamlit Direto**
```bash
cd src/dashboard
streamlit run app.py
```

### **Opção 3: Com Configurações Específicas**
```bash
cd src/dashboard
streamlit run app.py --server.port 8501 --server.headless true
```

## 🌐 **Acessar o Dashboard**

Após executar qualquer comando acima, acesse:
- **URL Local**: http://localhost:8501
- **URL da Rede**: http://192.168.15.21:8501 (se quiser acessar de outros dispositivos)

## ⚠️ **Resolver Erro "no module named streamlit"**

Se você recebeu este erro, execute:

### **1. Verificar Instalação**
```bash
pip list | grep streamlit
```

### **2. Reinstalar se Necessário**
```bash
pip install streamlit
```

### **3. Verificar Python Path**
```bash
python -c "import streamlit; print('Streamlit OK!')"
```

### **4. Instalar Todas as Dependências**
```bash
pip install -r requirements.txt
```

## 📦 **Instalação Completa (se necessário)**

```bash
# 1. Instalar dependências básicas
pip install -r requirements.txt

# 2. Instalar visualizações avançadas
python install_advanced_viz.py

# 3. Iniciar dashboard
python start_dashboard.py
```

## 🎯 **Status Atual Confirmado**

✅ **Streamlit**: Instalado e funcionando  
✅ **Dashboard**: Código testado e operacional  
✅ **Visualizações**: Todas as bibliotecas avançadas disponíveis  
✅ **Port 8501**: Livre e acessível  

## 🚀 **Iniciar Agora**

Execute simplesmente:
```bash
cd src/dashboard
python start_dashboard.py
```

**E acesse**: http://localhost:8501

🎉 **Dashboard pronto para análise dos dados do projeto Bolsonarismo!**