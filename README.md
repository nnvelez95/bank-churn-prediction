# Bank Customer Churn Prediction 🏦

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

Modelo predictivo de Machine Learning para identificar clientes bancarios con alto riesgo de abandono, permitiendo implementar estrategias de retención proactivas.

## 📊 Descripción del Proyecto

Este proyecto desarrolla un sistema de predicción de churn bancario utilizando técnicas de aprendizaje automático. El modelo analiza características demográficas, comportamiento financiero y patrones de uso para identificar clientes con probabilidad de abandonar el banco.

### Características Principales

- **Dataset**: 10,000 registros de clientes bancarios
- **Objetivo**: Clasificación binaria (Churn: Sí/No)
- **Mejor Modelo**: Random Forest Classifier
- **Performance**: ROC-AUC de 0.85

### Variables Predictoras Clave

- **Edad**: Comportamiento por grupo etario
- **Balance**: Saldo en cuenta
- **NumOfProducts**: Número de productos contratados
- **IsActiveMember**: Estado de actividad del cliente
- **Geography**: Ubicación geográfica
- **Gender**: Género del cliente
- **CreditScore**: Puntuación crediticia

## 🎯 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 86% |
| **Precision** | 78% |
| **Recall** | 72% |
| **F1-Score** | 75% |
| **ROC-AUC** | 0.85 |

### Interpretación de Resultados

- **Alta precisión**: El modelo minimiza falsos positivos, reduciendo costos en estrategias de retención innecesarias
- **Buen recall**: Identifica correctamente el 72% de los clientes que efectivamente abandonarán
- **ROC-AUC sólido**: Excelente capacidad de discriminación entre clases

## 🛠️ Tecnologías y Librerías

### Tecnologías Core
- **Python 3.10**: Lenguaje de programación principal
- **Jupyter Notebook**: Entorno de desarrollo interactivo

### Análisis y Procesamiento de Datos
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas y arrays

### Machine Learning
- **Scikit-learn**: Modelado y evaluación de algoritmos
  - Random Forest
  - Logistic Regression
  - SVM
  - Gradient Boosting

### Visualización
- **Matplotlib**: Gráficos estáticos
- **Seaborn**: Visualizaciones estadísticas avanzadas

## 📁 Estructura del Proyecto

```
bank-churn-prediction/
│
├── data/
│   ├── raw/                    # Datos originales sin procesar
│   └── processed/              # Datos limpios y transformados
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    # EDA
│   ├── 02_feature_engineering.ipynb     # Ingeniería de características
│   └── 03_modeling.ipynb                # Entrenamiento de modelos
│
├── src/
│   ├── data_preprocessing.py   # Scripts de limpieza
│   ├── feature_engineering.py  # Transformación de features
│   └── model_training.py       # Entrenamiento de modelos
│
├── reports/
│   ├── figures/                # Visualizaciones y gráficos
│   └── model_evaluation.pdf    # Reporte de métricas
│
├── requirements.txt            # Dependencias del proyecto
├── .gitignore
└── README.md
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/nnvelez95/bank-churn-prediction.git
cd bank-churn-prediction
```

2. **Crear entorno virtual (recomendado)**
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar Jupyter Notebook**
```bash
jupyter notebook
```

5. **Abrir y ejecutar los notebooks** en el siguiente orden:
   - `01_exploratory_analysis.ipynb`
   - `02_feature_engineering.ipynb`
   - `03_modeling.ipynb`

## 📈 Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Análisis de distribuciones
- Detección de valores faltantes y outliers
- Correlaciones entre variables
- Segmentación de clientes

### 2. Ingeniería de Características
- Encoding de variables categóricas
- Normalización de variables numéricas
- Creación de features derivadas
- Tratamiento de desbalance de clases (SMOTE)

### 3. Modelado
- Selección de algoritmos candidatos
- Validación cruzada (K-Fold)
- Optimización de hiperparámetros (Grid Search)
- Evaluación y selección del modelo final

### 4. Interpretabilidad
- Feature importance
- SHAP values
- Análisis de casos límite

## 💡 Insights del Negocio

### Principales Hallazgos

1. **Clientes con 3-4 productos** tienen menor tasa de churn
2. **Balance cero o muy alto** correlaciona con mayor abandono
3. **Clientes inactivos** tienen 3x más probabilidad de churn
4. **Edad 40-50 años** presenta mayor riesgo
5. **Geografía**: Diferencias significativas por país

### Recomendaciones Estratégicas

- Implementar programa de activación para clientes inactivos
- Ofrecer productos adicionales a clientes con 1-2 productos
- Crear segmentos de retención por grupo etario
- Personalizar comunicación según geografía

## 🔄 Próximos Pasos

- [ ] Implementar modelo en producción con API REST
- [ ] Crear dashboard interactivo con Streamlit
- [ ] Incorporar datos temporales (series de tiempo)
- [ ] Explorar modelos de deep learning (LSTM, Transformers)
- [ ] Desarrollar sistema de alertas tempranas
- [ ] A/B testing de estrategias de retención

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**Norberto Velez**

- LinkedIn:[LinkedIn](https://linkedin.com/in/norberto-velez-672916172)
- Email: nnvelez95@gmail.com

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o colaboraciones, no dudes en contactarme a través de [LinkedIn](https://linkedin.com/in/norberto-velez-672916172).

---

⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub
