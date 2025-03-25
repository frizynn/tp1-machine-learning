# Machine Learning - TP1

## Descripción

Este repositorio contiene la implementación del Trabajo Práctico 1 de Aprendizaje Automático, enfocado en el análisis de datos inmobiliarios mediante técnicas de regresión y clustering. El proyecto incluye modelos para predecir precios de viviendas y agrupar propiedades según sus características.

## Estructura del Proyecto

```
tp1-machine-learning/
│
├── data/
│   ├── raw/         # Datos originales sin procesar
│   └── processed/   # Datos procesados listos para el análisis
│
├── src/
│   ├── models/
│   │   ├── regression/     # Modelos de regresión
│   │   ├── clustering/     # Modelos de clustering
│   │   └── loss/           # Funciones de pérdida
│   │
│   ├── utils/              # Utilidades y herramientas compartidas
│   └── Entrega_Tp1.ipynb   # Notebook principal con análisis y resultados
│
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Esta documentación
```

## Características Implementadas

### Modelos de Regresión
- **Regresión Lineal**: Implementación de regresión lineal simple
- **Regresión Polinomial**: Implementación de regresión polinomial con diferentes grados

### Modelos de Clustering
- **K-Means**: Implementación del algoritmo de clustering K-Means

## Requisitos

Para ejecutar este proyecto necesitarás Python 3.10+ y las siguientes bibliotecas:

```
numpy
matplotlib
pandas
seaborn
```

Puedes instalar todas las dependencias con:

```bash
pip install -r requirements.txt
```

## Uso

### Notebook Principal

El análisis completo se puede encontrar en el notebook `src/Entrega_Tp1.ipynb`. Allí se detallan:

1. Exploración y visualización de los datos
2. Preprocesamiento y transformación
3. Entrenamiento de modelos
4. Evaluación de resultados
5. Predicciones sobre nuevos datos (caso de Amanda)

### Ejecución de Modelos

Para usar los modelos de regresión:

```python
from src.models.regression.linear_regressor import LinearRegressor
from src.models.regression.polynomial_regressor import PolynomialRegressor

# Regresión lineal
model = LinearRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Regresión polinomial
poly_model = PolynomialRegressor(degree=3)
poly_model.fit(X_train, y_train)
poly_predictions = poly_model.predict(X_test)
```

Para usar el modelo de clustering:

```python
from src.models.clustering.kmeans import KMeans

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)
```

## Caso de Estudio: Predicción para Amanda

El proyecto incluye un caso de estudio para predecir el precio de una vivienda específica (Amanda) utilizando los modelos entrenados. Los datos de entrada se encuentran en `vivienda_Amanda.csv` y los resultados de la predicción se detallan en el notebook principal.

## Contribuciones

Este proyecto fue desarrollado como parte del curso de Aprendizaje Automático. Las contribuciones o mejoras son bienvenidas mediante pull requests.

## Licencia

Este proyecto está disponible como código abierto bajo los términos de la licencia MIT. 