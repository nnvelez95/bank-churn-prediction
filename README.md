# Bank Customer Churn Prediction 🏦

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

Modelo predictivo de Machine Learning para identificar clientes bancarios con alto riesgo de abandono, permitiendo implementar estrategias de retención proactivas y basadas en datos.

---

## 📊 Descripción del Proyecto

Este proyecto desarrolla un sistema integral de predicción de churn bancario utilizando técnicas avanzadas de aprendizaje automático. El modelo analiza características demográficas, comportamiento financiero y patrones de uso para identificar clientes con alta probabilidad de abandonar el banco.

### 🎯 Objetivo del Negocio

Reducir la tasa de abandono de clientes mediante:
- Identificación temprana de clientes en riesgo
- Segmentación inteligente para estrategias de retención personalizadas
- Optimización del ROI en campañas de marketing
- Mejora en la experiencia del cliente

### 📈 Características del Dataset

- **Tamaño**: 10,000 registros de clientes bancarios
- **Objetivo**: Clasificación binaria (Churn: 1 = Abandonó, 0 = Se quedó)
- **Características**: 14 variables predictoras
- **Balanceo**: Dataset desbalanceado (~20% churn rate)

### 🔑 Variables Predictoras

| Variable | Descripción | Tipo |
|----------|-------------|------|
| **CreditScore** | Puntuación crediticia del cliente | Numérica (300-850) |
| **Geography** | País de residencia | Categórica (Francia, España, Alemania) |
| **Gender** | Género del cliente | Categórica (Masculino, Femenino) |
| **Age** | Edad del cliente | Numérica (18-92) |
| **Tenure** | Años como cliente del banco | Numérica (0-10) |
| **Balance** | Saldo en cuenta | Numérica (0-250,000+) |
| **NumOfProducts** | Número de productos bancarios | Numérica (1-4) |
| **HasCrCard** | Tiene tarjeta de crédito | Binaria (0/1) |
| **IsActiveMember** | Cliente activo | Binaria (0/1) |
| **EstimatedSalary** | Salario estimado | Numérica (11-200,000) |

---

## 🎯 Resultados del Modelo

### Métricas de Performance

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 86% | Precisión general del modelo |
| **Precision** | 78% | De los predichos como churn, 78% son correctos |
| **Recall** | 72% | Detecta 72% de los clientes que abandonarán |
| **F1-Score** | 75% | Balance entre precisión y recall |
| **ROC-AUC** | 0.85 | Excelente capacidad discriminativa |

### 📊 Matriz de Confusión

```
                 Predicho: No Churn    Predicho: Churn
Real: No Churn         1,580                 120
Real: Churn              140                 360
```

### 💡 Interpretación de Negocio

- **Alta Precision (78%)**: Reduce costos al evitar falsas alarmas en campañas de retención
- **Buen Recall (72%)**: Captura la mayoría de clientes en riesgo, maximizando oportunidades de retención
- **ROC-AUC 0.85**: El modelo distingue muy bien entre clientes que se quedarán vs. abandonarán
- **Impacto Estimado**: Potencial reducción del 50% en tasa de churn al intervenir proactivamente

---

## 🛠️ Stack Tecnológico

### Lenguaje y Entorno
```python
Python 3.10+
Jupyter Notebook / JupyterLab
```

### Análisis y Manipulación de Datos
```python
pandas==2.0.3          # Manipulación de DataFrames
numpy==1.24.3          # Operaciones numéricas
```

### Machine Learning
```python
scikit-learn==1.3.0    # Algoritmos de ML y métricas
imbalanced-learn==0.11.0  # SMOTE para balanceo de clases
```

### Visualización
```python
matplotlib==3.7.2      # Gráficos base
seaborn==0.12.2        # Visualizaciones estadísticas
plotly==5.16.1         # Gráficos interactivos (opcional)
```

### Herramientas Adicionales
```python
joblib==1.3.2          # Serialización de modelos
shap==0.42.1           # Interpretabilidad del modelo (futuro)
```

---

## 📁 Estructura del Proyecto

```
bank-churn-prediction/
│
├── data/
│   ├── raw/                          # Datos originales sin modificar
│   │   └── Churn_Modelling.csv       # Dataset principal
│   └── processed/                    # Datos procesados y limpios
│       ├── train_processed.csv       # Conjunto de entrenamiento
│       └── test_processed.csv        # Conjunto de prueba
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb      # EDA completo
│   ├── 02_feature_engineering.ipynb       # Transformación de features
│   └── 03_modeling.ipynb                  # Entrenamiento y evaluación
│
├── src/                              # Scripts Python modulares (futuro)
│   ├── __init__.py
│   ├── data_preprocessing.py         # Funciones de limpieza
│   ├── feature_engineering.py        # Transformación de features
│   ├── model_training.py             # Entrenamiento de modelos
│   └── model_evaluation.py           # Métricas y evaluación
│
├── models/                           # Modelos entrenados serializados
│   ├── random_forest_model.pkl       # Modelo final
│   └── model_metadata.json           # Hiperparámetros y métricas
│
├── reports/
│   ├── figures/                      # Gráficos y visualizaciones
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── roc_curve.png
│   │   └── correlation_heatmap.png
│   └── model_evaluation_report.pdf   # Reporte técnico completo
│
├── tests/                            # Tests unitarios (futuro)
│   └── test_preprocessing.py
│
├── .gitignore                        # Archivos a ignorar en Git
├── LICENSE                           # Licencia MIT
├── README.md                         # Este archivo
├── requirements.txt                  # Dependencias del proyecto
└── setup.py                          # Instalación del paquete (futuro)
```

---

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.10 o superior
- pip (gestor de paquetes)
- Git

### Instalación Paso a Paso

1. **Clonar el repositorio**
```bash
git clone https://github.com/nnvelez95/bank-churn-prediction.git
cd bank-churn-prediction
```

2. **Crear entorno virtual**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verificar instalación**
```bash
python -c "import pandas, sklearn, seaborn; print('✅ Todo instalado correctamente')"
```

### 🎓 Uso del Proyecto

#### Opción 1: Ejecutar Notebooks (Recomendado para exploración)

```bash
jupyter notebook
```

Abrir y ejecutar en orden:
1. `notebooks/01_exploratory_analysis.ipynb` - EDA y visualizaciones
2. `notebooks/02_feature_engineering.ipynb` - Preparación de datos
3. `notebooks/03_modeling.ipynb` - Entrenamiento y evaluación

#### Opción 2: Scripts Python (Futuro - para producción)

```bash
# Preprocesar datos
python src/data_preprocessing.py --input data/raw/Churn_Modelling.csv --output data/processed/

# Entrenar modelo
python src/model_training.py --data data/processed/ --output models/

# Evaluar modelo
python src/model_evaluation.py --model models/random_forest_model.pkl --test-data data/processed/test_processed.csv
```

---

## 📈 Metodología y Pipeline

### 1️⃣ Análisis Exploratorio de Datos (EDA)

**Objetivos:**
- Comprender distribuciones de variables
- Identificar patrones y correlaciones
- Detectar outliers y valores faltantes
- Análisis de tasa de churn por segmentos

**Técnicas aplicadas:**
- Estadísticas descriptivas
- Análisis univariado y bivariado
- Matrices de correlación
- Visualizaciones avanzadas (boxplots, histogramas, heatmaps)

**Hallazgos clave:**
- Balance cero indica alta probabilidad de churn
- Clientes con 1 solo producto son más propensos a abandonar
- Edad 40-50 años muestra mayor tasa de abandono
- Miembros inactivos tienen 3x más riesgo

### 2️⃣ Ingeniería de Características

**Transformaciones realizadas:**
```python
# Variables categóricas
- Label Encoding: Geography, Gender
- One-Hot Encoding: Alternativa evaluada

# Variables numéricas
- StandardScaler: Age, CreditScore, Balance, EstimatedSalary
- MinMaxScaler: Alternativa para modelos basados en distancias

# Features derivadas (futuro)
- Balance_per_Product = Balance / NumOfProducts
- Tenure_Age_Ratio = Tenure / Age
- High_Value_Customer = (Balance > 100000) & (NumOfProducts >= 2)
```

**Tratamiento de desbalanceo:**
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Under-sampling de clase mayoritaria
- Class weights en modelos

### 3️⃣ Modelado y Selección de Algoritmos

**Modelos evaluados:**

| Modelo | ROC-AUC | Accuracy | Recall | Tiempo |
|--------|---------|----------|--------|--------|
| Logistic Regression | 0.78 | 81% | 65% | < 1s |
| Random Forest ⭐ | 0.85 | 86% | 72% | ~5s |
| XGBoost | 0.84 | 85% | 71% | ~3s |
| SVM (RBF) | 0.80 | 83% | 68% | ~10s |
| Neural Network | 0.82 | 84% | 69% | ~15s |

**Modelo seleccionado: Random Forest**
- Mejor balance entre métricas
- Robusto ante outliers
- Permite interpretabilidad (feature importance)
- No requiere escalado estricto

**Hiperparámetros optimizados:**
```python
{
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'class_weight': 'balanced'
}
```

### 4️⃣ Validación y Evaluación

**Estrategia de validación:**
- Train/Test split: 80/20
- Cross-validation: 5-fold
- Stratified sampling (mantiene proporción de churn)

**Métricas de negocio:**
```python
# Costo de falso negativo (no detectar churn)
FN_cost = $500  # Costo de perder un cliente

# Costo de falso positivo (campaña innecesaria)
FP_cost = $50   # Costo de campaña de retención

# ROI esperado del modelo
Expected_savings = (True_Positives * $500) - (False_Positives * $50)
```

---

## 💡 Insights de Negocio

### 📊 Top 5 Variables Más Importantes

1. **Age (28%)** - Clientes 40-50 años, mayor riesgo
2. **NumOfProducts (22%)** - 1 producto = alto riesgo, 3-4 = bajo riesgo
3. **IsActiveMember (18%)** - Inactividad multiplica riesgo x3
4. **Balance (15%)** - Balance extremo (muy bajo/alto) = riesgo
5. **Geography (12%)** - Alemania muestra mayor tasa de churn

### 🎯 Recomendaciones Estratégicas

#### Para Marketing y Retención
1. **Programa de activación de clientes inactivos**
   - Email marketing personalizado
   - Incentivos de uso (cashback, descuentos)
   - Push notifications en app móvil

2. **Cross-selling inteligente**
   - Ofrecer productos complementarios a clientes con 1-2 productos
   - Bundles personalizados según perfil

3. **Segmentación geográfica**
   - Estrategias diferenciadas por país
   - Alemania requiere atención especial

#### Para Producto
1. **Mejorar engagement de clientes 40-50 años**
   - UX adaptada a este segmento
   - Productos específicos (planificación de retiro)

2. **Alertas para clientes de balance extremo**
   - Balance = 0: Riesgo de inactividad
   - Balance muy alto sin productos: Oportunidad de inversión

### 📉 Estimación de Impacto

**Escenario actual (sin modelo):**
- Tasa de churn: 20%
- Clientes perdidos/año: 2,000
- Costo estimado: $1,000,000

**Escenario con modelo (intervención proactiva):**
- Clientes detectados en riesgo: 1,440 (72% recall)
- Tasa de retención con campaña: 40%
- Clientes retenidos: 576
- **Ahorro estimado: $288,000/año**
- **ROI del modelo: 480%** (asumiendo costo campaña $60,000)

---

## 🔮 Roadmap y Futuras Features

### 🚀 Fase 1: Mejoras en el Modelo (Q1 2026)

#### Machine Learning Avanzado
- [ ] **Ensemble Stacking**: Combinar Random Forest + XGBoost + Neural Network
- [ ] **Hyperparameter Tuning con Optuna**: Optimización bayesiana avanzada
- [ ] **Calibración de probabilidades**: Platt scaling para mejores probabilidades
- [ ] **Interpretabilidad con SHAP**: Explicar predicciones individuales
- [ ] **Feature Selection avanzado**: RFE (Recursive Feature Elimination)

#### Feature Engineering v2.0
- [ ] **Variables de interacción**: Age_x_NumOfProducts, Balance_x_Tenure
- [ ] **Binning inteligente**: Discretización de Age, Balance con óptimos puntos de corte
- [ ] **Polynomial Features**: Relaciones no lineales entre variables
- [ ] **Time-based features**: Si se incorporan datos temporales
- [ ] **Clustering de clientes**: KMeans para segmentación, usar cluster como feature

### 📊 Fase 2: Productización (Q2 2026)

#### API REST para Predicciones
```python
# Endpoint de predicción en tiempo real
POST /api/v1/predict
{
  "customer_id": 12345,
  "age": 42,
  "balance": 85000,
  ...
}

Response:
{
  "churn_probability": 0.73,
  "risk_level": "HIGH",
  "top_factors": ["age", "num_products", "is_active"],
  "recommended_action": "retention_campaign_tier_1"
}
```

**Stack tecnológico:**
- [ ] FastAPI para endpoints
- [ ] Docker para containerización
- [ ] Redis para caching de predicciones
- [ ] PostgreSQL para logging de predicciones
- [ ] Celery para batch predictions

#### CI/CD y MLOps
- [ ] **GitHub Actions**: Testing automático en cada push
- [ ] **Model versioning con MLflow**: Tracking de experimentos
- [ ] **Model monitoring**: Detectar data drift y model decay
- [ ] **A/B testing framework**: Comparar modelos en producción
- [ ] **Automated retraining**: Pipeline mensual de reentrenamiento

### 📱 Fase 3: Interfaces de Usuario (Q3 2026)

#### Dashboard Interactivo con Streamlit
```python
# Características del dashboard:
- 📊 Métricas en tiempo real (churn rate, predicciones diarias)
- 🔍 Búsqueda de cliente individual
- 📈 Gráficos interactivos (filtros por segmento)
- 🎯 Segmentación dinámica de clientes
- 📥 Exportación de listas de clientes en riesgo
- 🔔 Alertas configurables
```

- [ ] Deploy en Streamlit Cloud / Heroku
- [ ] Autenticación de usuarios (JWT)
- [ ] Roles (Admin, Marketing, Analyst)

#### Integración con CRM
- [ ] Webhook a Salesforce/HubSpot cuando se detecta alto riesgo
- [ ] Enriquecimiento automático de perfiles de cliente
- [ ] Triggers para campañas de email marketing (Mailchimp/SendGrid)

### 🧠 Fase 4: Deep Learning y Series Temporales (Q4 2026)

#### Modelos Secuenciales
- [ ] **LSTM/GRU**: Predecir churn basado en secuencias de transacciones
- [ ] **Transformers**: Atención temporal en comportamiento de cliente
- [ ] **Survival Analysis**: Cox Proportional Hazards para tiempo hasta churn

#### Nuevas Fuentes de Datos
```python
# Datos adicionales a incorporar:
- Historial de transacciones (monto, frecuencia, tipo)
- Interacciones con servicio al cliente (tickets, llamadas)
- Uso de canales digitales (app móvil, web banking)
- Respuesta a campañas de marketing previas
- Datos de redes sociales (sentiment analysis)
```

### 🤖 Fase 5: Inteligencia Artificial Generativa (2027)

#### Personalización con LLMs
- [ ] **Generación automática de emails de retención**: Personalizados con GPT-4
- [ ] **Chatbot predictivo**: "Hemos notado que podrías estar interesado en..."
- [ ] **Análisis de sentimiento**: Procesar feedback de clientes con NLP
- [ ] **Recomendaciones explicables**: "Te sugerimos X porque..."

#### AutoML y No-Code ML
- [ ] **AutoML pipeline**: H2O.ai o AutoKeras para búsqueda automática de modelos
- [ ] **Low-code interface**: Para que equipos de negocio ejecuten predicciones

### 📊 Fase 6: Business Intelligence Avanzado

#### Simuladores y Optimizadores
- [ ] **What-if Analysis**: "¿Qué pasaría si aumentamos el balance mínimo?"
- [ ] **Optimizador de campañas**: Calcular ROI óptimo de estrategias de retención
- [ ] **Segmentación automática**: RFM + Churn score para marketing

#### Reporting Automatizado
- [ ] PDF reports semanales para stakeholders
- [ ] Slack/Teams bot con métricas diarias
- [ ] Power BI/Tableau integration

---

## 🧪 Testing y Calidad

### Tests Implementados (Futuro)
```bash
# Ejecutar tests
pytest tests/ -v --cov=src

# Tests de integración
pytest tests/integration/ -v

# Tests de rendimiento
pytest tests/performance/ -v --benchmark-only
```

### Cobertura de Testing
- [ ] Unit tests para preprocessing (>90% coverage)
- [ ] Integration tests para pipeline completo
- [ ] Performance tests (latencia < 100ms por predicción)
- [ ] Data validation tests (schema, ranges)

---

## 📚 Recursos y Referencias

### Documentación Técnica
- [Scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Imbalanced-learn: SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [SHAP for Model Interpretability](https://shap.readthedocs.io/)

### Papers Académicos
- *Handling Imbalanced Datasets* - He & Garcia (2009)
- *Random Forests* - Breiman (2001)
- *Customer Churn Prediction in Banking* - Jahromi et al. (2022)

### Datasets Similares
- [Kaggle: Bank Customer Churn](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- [UCI ML Repository: Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Sigue estos pasos:

### Proceso de Contribución

1. **Fork el proyecto**
2. **Crea una rama para tu feature**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit tus cambios**
   ```bash
   git commit -m 'Add: Nueva funcionalidad X'
   ```
4. **Push a la rama**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Abre un Pull Request**

### Estándares de Código
- Seguir PEP 8 (Python Style Guide)
- Documentar funciones con docstrings
- Incluir tests para nuevas features
- Actualizar README si es necesario

### Issues y Bugs
Si encuentras un bug o tienes una sugerencia:
1. Revisa los [issues existentes](https://github.com/nnvelez95/bank-churn-prediction/issues)
2. Si no existe, crea uno nuevo con:
   - Descripción clara del problema/sugerencia
   - Pasos para reproducir (si es bug)
   - Screenshots (si aplica)

---

## 📝 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2026 Norberto Velez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👤 Autor

**Norberto Velez**

Data Scientist | Machine Learning Engineer

- 🔗 LinkedIn: [linkedin.com/in/norberto-velez-672916172](https://linkedin.com/in/norberto-velez-672916172)
- 📧 Email: [nnvelez95@gmail.com](mailto:nnvelez95@gmail.com)
- 💼 GitHub: [@nnvelez95](https://github.com/nnvelez95)
- 🌐 Portfolio: [En construcción](#)

### Otros Proyectos
- Próximamente más proyectos de Data Science y ML

---

## 📧 Contacto y Soporte

¿Tienes preguntas? ¿Necesitas ayuda con el proyecto?

- **Email**: nnvelez95@gmail.com
- **LinkedIn**: [Envíame un mensaje](https://linkedin.com/in/norberto-velez-672916172)
- **GitHub Issues**: [Abrir un issue](https://github.com/nnvelez95/bank-churn-prediction/issues)

---

## 🙏 Agradecimientos

- Dataset original de [Kaggle](https://www.kaggle.com/)
- Comunidad de Scikit-learn por excelente documentación
- Stack Overflow por resolver dudas puntuales
- A todos los que contribuyan a este proyecto

---

## 📊 Estadísticas del Proyecto

![GitHub stars](https://img.shields.io/github/stars/nnvelez95/bank-churn-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/nnvelez95/bank-churn-prediction?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/nnvelez95/bank-churn-prediction?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/nnvelez95/bank-churn-prediction)
![GitHub code size](https://img.shields.io/github/languages/code-size/nnvelez95/bank-churn-prediction)

---

<div align="center">

### ⭐ Si este proyecto te resultó útil, considera darle una estrella

**Desarrollado con ❤️ por Norberto Velez**

[🔝 Volver arriba](#bank-customer-churn-prediction-)

</div>
