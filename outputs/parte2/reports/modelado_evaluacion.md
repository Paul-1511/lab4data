# Ejercicios 4 y 5 — Modelado y evaluacion

> Resultados del perfil **standard**.

Generado: 2026-08-23 22:15:28 | Semilla: 42 | Dataset 2.0 (hash 65825179def16792)  
Respuesta: `high_cyano_8` (>= 8 ug/L) | Predictores: B02, B03, B07, B08, B8A, B11, B12, NDWI

---

## 4.2 Division 70/30

- Entrenamiento: **2,629,557** filas, 42,102 positivos (1.6011 %)
- Prueba: **1,126,953** filas, 18,044 positivos (1.6011 %)
- Firma del conjunto de prueba: `67bc81d40aad20b0` — identica para los tres modelos, lo que demuestra que la comparacion es justa.
- El test conserva la prevalencia natural: no se balanceo, no se submuestreo y no intervino en el ajuste de hiperparametros ni en la eleccion del umbral.

> **Advertencia.** Esta division mezcla pixeles vecinos del mismo bloque de 1 km, que son casi identicos entre si. Por eso sus resultados son **optimistas**. Se incluye porque el enunciado la exige (4.2) y sirve de contraste frente a la validacion espacial.

## 4.3 Ajuste de hiperparametros

Busqueda aleatoria sobre una muestra determinista tomada solo del entrenamiento. Metrica de seleccion: **average_precision (PR-AUC)**, nunca Accuracy.

| Modelo | Configuraciones | Folds | Mejores hiperparametros | PR-AUC tuning | Segundos |
|---|---|---|---|---|---|
| `logistic` | 12 | 3 | {'modelo__l1_ratio': 0.0, 'modelo__C': 0.01} | 0.9836 | 337.2 |
| `random_forest` | 12 | 3 | {'min_samples_leaf': 5, 'max_features': 'sqrt', 'max_depth': None} | 0.9962 | 263.3 |
| `xgboost` | 12 | 3 | {'subsample': 0.6, 'min_child_weight': 10, 'max_depth': 10, 'learning_rate': 0.03, 'colsample_bytree': 0.8} | 0.9972 | 120.6 |

Espacio de hiperparametros explorado:

- `logistic`: {'modelo__C': ['0.01', '0.1', '0.5', '1.0', '5.0', '10.0'], 'modelo__l1_ratio': ['0.0', '0.25', '0.75', '1.0']}
- `random_forest`: {'max_depth': ['10', '14', '18', '24', 'None'], 'min_samples_leaf': ['5', '10', '20', '50'], 'max_features': ['sqrt', 'log2', '0.5']}
- `xgboost`: {'max_depth': ['4', '6', '8', '10'], 'learning_rate': ['0.03', '0.05', '0.1', '0.2'], 'subsample': ['0.6', '0.8', '1.0'], 'colsample_bytree': ['0.6', '0.8', '1.0'], 'min_child_weight': ['1', '5', '10', '20']}

> **Nota sobre Regresion Logistica.** Se migro de `penalty` (deprecado en scikit-learn 1.8) a `l1_ratio`: `l1_ratio=0` equivale a L2 y `l1_ratio=1` a L1. La equivalencia se comprobo numericamente: con el mismo C, solver y semilla los coeficientes coinciden exactamente. Por eso la configuracion seleccionada, antes `penalty='l2', C=0.01`, aparece ahora como `l1_ratio=0.0, C=0.01`, y sus metricas son identicas.

## 5.1 Metricas en el conjunto de prueba

| modelo | etiqueta | umbral | pr_auc | roc_auc | recall | precision | f1 | f2 | mcc | balanced_accuracy | specificity | brier | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | umbral_0.5 | 0.5000 | 0.9814 | 0.9996 | 0.9644 | 0.8550 | 0.9064 | 0.9403 | 0.9065 | 0.9809 | 0.9973 | 0.0023 | 17401 | 643 | 2950 | 1105959 |
| random_forest | umbral_operacional | 0.9408 | 0.9814 | 0.9996 | 0.8079 | 0.9839 | 0.8873 | 0.8379 | 0.8901 | 0.9038 | 0.9998 | 0.0023 | 14578 | 3466 | 238 | 1108671 |
| xgboost | umbral_0.5 | 0.5000 | 0.9826 | 0.9997 | 0.9885 | 0.7505 | 0.8532 | 0.9295 | 0.8588 | 0.9916 | 0.9947 | 0.0043 | 17836 | 208 | 5931 | 1102978 |
| xgboost | umbral_operacional | 0.9965 | 0.9826 | 0.9997 | 0.8050 | 0.9857 | 0.8863 | 0.8357 | 0.8893 | 0.9024 | 0.9998 | 0.0043 | 14526 | 3518 | 210 | 1108699 |
| logistic | umbral_0.5 | 0.5000 | 0.9135 | 0.9979 | 0.9947 | 0.4678 | 0.6364 | 0.8118 | 0.6758 | 0.9882 | 0.9816 | 0.0141 | 17949 | 95 | 20419 | 1088490 |
| logistic | umbral_operacional | 0.9722 | 0.9135 | 0.9979 | 0.8012 | 0.8852 | 0.8411 | 0.8167 | 0.8397 | 0.8997 | 0.9983 | 0.0141 | 14456 | 3588 | 1874 | 1107035 |

**Accuracy no se usa como metrica principal**: con 1.601 % de positivos, predecir siempre la clase mayoritaria daria 98.399 % sin detectar nada.

## Umbral 0.5 frente a umbral operacional

| Modelo | Umbral | Criterio | Recall (val) | Precision (val) |
|---|---|---|---|---|
| `logistic` | 0.9722 | Recall >= 0.80 y, entre esos, maxima Precision | 0.8000 | 0.8860 |
| `random_forest` | 0.9408 | Recall >= 0.80 y, entre esos, maxima Precision | 0.8001 | 0.9874 |
| `xgboost` | 0.9965 | Recall >= 0.80 y, entre esos, maxima Precision | 0.8012 | 0.9887 |

El umbral se eligio **exigiendo un Recall minimo de 0.80** y maximizando la Precision entre los que lo cumplen, usando **solo la validacion interna del entrenamiento**. El conjunto de prueba nunca intervino.

## Importancia de las variables

| Variable | logistic | random_forest | xgboost |
|---|---|---|---|
| `B02` | -3.6165 | 0.0418 | 0.0205 |
| `B03` | 1.7803 | 0.2207 | 0.2356 |
| `B07` | 4.0020 | 0.3239 | 0.6227 |
| `B08` | -0.5975 | 0.1408 | 0.0359 |
| `B8A` | 0.6937 | 0.1492 | 0.0330 |
| `B11` | -0.7245 | 0.0246 | 0.0172 |
| `B12` | -0.9730 | 0.0204 | 0.0229 |
| `NDWI` | 0.2039 | 0.0786 | 0.0123 |

## 5.2 Comparacion

Segun PR-AUC en el mismo conjunto de prueba, el mejor modelo de esta division es **`xgboost`**. La comparacion definitiva, sin embargo, debe hacerse con la validacion espacial.

## 5.3 Interpretacion ambiental de los errores

**Falso positivo (FP):** se marca como zona de alta cianobacteria un area que no lo es. Coste: una inspeccion de campo innecesaria, alarma infundada, posible perdida de confianza si se repite.

**Falso negativo (FN):** una zona con floracion alta pasa desapercibida. Coste: no se emite aviso, puede haber exposicion recreativa o consumo de agua sin advertencia, y se pierde la ventana de intervencion temprana.

**Cual importa mas reducir.** En vigilancia ambiental el **falso negativo es el error mas grave**: un aviso de mas cuesta una visita; un aviso de menos puede costar salud publica. Por eso:

- La metrica prioritaria es el **Recall** de la clase positiva.
- Se compara con **PR-AUC**, que resume el compromiso Precision-Recall en datos desbalanceados mucho mejor que ROC-AUC.
- Se reporta **F2**, que pondera el Recall por encima de la Precision, coherente con esa prioridad.

> **Alcance.** Esto es una herramienta de **cribado**, no una medicion confirmatoria. No hubo validacion in situ; el algoritmo CyanoLakes reporta MAPE 42.3 % y RMSE relativo 95.8 %, y fue calibrado para *Microcystis aeruginosa* sobre datos simulados. **No permite diagnosticar toxicidad**: la clorofila mide biomasa, no toxinas ni especies.
