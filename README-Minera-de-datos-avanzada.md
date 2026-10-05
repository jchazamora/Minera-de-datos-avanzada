# Pronóstico de pasajeros salientes por región — Aeropuerto Internacional Juan Santamaría (SJO)

Aplicación en Streamlit que modela la serie mensual de pasajeros salientes del Aeropuerto Internacional
Juan Santamaría por región. Proyecto del curso de Minería de Datos Avanzada, LEAD University.

Responde dos preguntas:

1. **¿Qué modelo pronostica mejor los pasajeros de cada región?** Se comparan tres modelos con backtesting.
2. **¿La estacionalidad cambia entre regiones?** Se comparan los perfiles mensuales y se aplica una prueba
   estadística.

## Qué hace

| Etapa | Detalle |
| --- | --- |
| Datos | Agrega el CSV diario a series mensuales por región y rellena con 0 los meses sin registros. |
| Modelos | Seasonal naive (baseline), Holt-Winters (ETS) y SARIMAX con búsqueda de parámetros por AIC. |
| Evaluación | Backtesting rolling-origin con RMSE, MAE y MAPE; los modelos se ordenan por RMSE promedio. |
| Estacionalidad | Índice estacional mensual, fuerza estacional con descomposición STL y prueba de Kruskal-Wallis entre regiones. |

La app tiene tres páginas: **Datos** (serie mensual por región), **Modelos** (ranking de modelos con
horizonte, número de cortes y largo de la estacionalidad configurables) y **Estacionalidad** (perfiles
comparados entre las regiones elegidas).

## Estructura

| Archivo | Contenido |
| --- | --- |
| `main.py` | App de Streamlit y navegación. |
| `ts_generator.py` | Carga del CSV y armado de las series mensuales por región. |
| `ts_models.py` | Seasonal naive, Holt-Winters y SARIMAX con búsqueda de parámetros. |
| `ts_evaluator.py` | Backtesting y métricas (RMSE, MAE, MAPE). |
| `visualizer.py` | Gráficos en Plotly, índice estacional, fuerza STL y prueba de Kruskal-Wallis. |

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
pip install streamlit streamlit-option-menu pandas numpy statsmodels scikit-learn scipy plotly
streamlit run main.py
```

El dataset no se incluye en el repositorio. La app lee `Data.csv` (la ruta se cambia en la barra lateral)
con estas columnas:

| Columna | Ejemplo |
| --- | --- |
| `Year` | 2024 |
| `Month` | December |
| `Day` | 15 |
| `Sum of Departure_Total Departing Pax Count` | 1250 |
| `Departure_Region` | North America |

## Tecnologías

Python · pandas · NumPy · statsmodels · scikit-learn · SciPy · Plotly · Streamlit
