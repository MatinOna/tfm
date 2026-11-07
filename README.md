# Clasificación de géneros musicales a partir de la batería

Código del Trabajo Fin de Máster *Comparación de modelos supervisados para la clasificación de géneros musicales a partir de características rítmicas de la batería* (Máster Universitario en Inteligencia Artificial, UNIR).

El trabajo compara clasificadores supervisados (SVM, Random Forest y XGBoost, junto con tres clasificadores de referencia: clase mayoritaria, regresión logística y k-NN) para distinguir cuatro géneros (**funk, jazz, latin y rock**) a partir de características extraídas de los archivos MIDI de batería del **Groove MIDI Dataset** (Gillick et al., 2019).

## Estructura del repositorio

```
├── notebooks/              Notebooks del experimento, en orden de ejecución
│   ├── 01_carga.ipynb
│   ├── 02_exploracion.ipynb
│   ├── 03_caracteristicas.ipynb
│   ├── 04_entrenamiento.ipynb
│   └── 05_evaluacion.ipynb
├── src/
│   ├── features.py         Extracción de características (apartado 4.2)
│   └── models.py           Entrenamiento, validación cruzada y evaluación (apartados 4.3 a 4.5)
├── results/                Tablas (CSV), figuras y modelos generados por los notebooks
│   ├── figures/
│   └── models/
└── data/groove/            Corpus (no incluido en el repositorio, ver más abajo)
```

## Correspondencia con la memoria

Cada sección de los notebooks lleva el mismo número que el apartado de la memoria al que respalda, y cada tabla o figura se imprime con el mismo número que tiene en la memoria.

| Apartado de la memoria | Notebook |
|---|---|
| 4.1.1. Corpus utilizado | `01_carga.ipynb` |
| 4.1.2 a 4.1.5. Selección y preparación de los datos | `02_exploracion.ipynb` |
| 4.2. Extracción de características | `03_caracteristicas.ipynb` |
| 4.3. Entrenamiento de los modelos | `04_entrenamiento.ipynb` |
| 4.4. Validación y evaluación | `05_evaluacion.ipynb` |
| 4.5. Comparación de resultados | `05_evaluacion.ipynb` |

## Datos

El corpus no se incluye en este repositorio. Para reproducir el experimento:

1. Descargar la versión **solo MIDI** del Groove MIDI Dataset desde su página oficial: <https://magenta.tensorflow.org/datasets/groove>.
2. Descomprimirla en `data/groove/`, de modo que exista el archivo `data/groove/info.csv` junto con las carpetas `drummer1`, `drummer2`, etc.

## Entorno

El experimento se desarrolló con Python 3.13 en un entorno de conda. Bibliotecas utilizadas:

- `pandas`, `numpy`, `scipy`, `matplotlib`
- `pretty_midi` (lectura de los archivos MIDI)
- `scikit-learn` y `xgboost` (modelos)

Las versiones exactas de las principales bibliotecas se imprimen al comienzo de los notebooks 01 y 04.

## Ejecución

Los notebooks deben ejecutarse **en orden (01 → 05)**, ya que cada uno lee los resultados que guarda el anterior en `results/`. Todos los procesos aleatorios usan una semilla fija (`SEED = 42` en `src/models.py`), por lo que una nueva ejecución en el mismo entorno reproduce los mismos resultados. Con otras versiones de las bibliotecas o en otro equipo, los resultados de Random Forest pueden variar ligeramente en los últimos decimales.

## Referencia del corpus

Gillick, J., Roberts, A., Engel, J., Eck, D., y Bamman, D. (2019). Learning to groove with inverse sequence transformations. *Proceedings of the 36th International Conference on Machine Learning*, 2269–2279.
