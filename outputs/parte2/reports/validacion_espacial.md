# Ejercicio 6 — Validacion espacial

> Resultados del perfil **standard**.

Generado: 2026-08-23 22:15:28 | Semilla: 42 | Dataset 2.0 (hash 65825179def16792)  
Respuesta: `high_cyano_8` (>= 8 ug/L) | Predictores: B02, B03, B07, B08, B8A, B11, B12, NDWI

---

## 6.1-6.2 Bloques de 1000 m y folds

| Fold | Bloques train | Bloques val | Compartidos | Obs. train | Obs. val | Positivos val | Prevalencia val | Lagos val |
|---|---|---|---|---|---|---|---|---|
| 1 | 267 | 41 | **0** | 3,005,284 | 751,226 | 11,856 | 1.5782 % | Amatitlan,Atitlan |
| 2 | 249 | 59 | **0** | 3,004,625 | 751,885 | 12,662 | 1.6840 % | Amatitlan,Atitlan |
| 3 | 234 | 74 | **0** | 3,005,456 | 751,054 | 11,797 | 1.5707 % | Amatitlan,Atitlan |
| 4 | 240 | 68 | **0** | 3,005,401 | 751,109 | 11,852 | 1.5779 % | Amatitlan,Atitlan |
| 5 | 242 | 66 | **0** | 3,005,274 | 751,236 | 11,979 | 1.5946 % | Amatitlan,Atitlan |

Se uso `StratifiedGroupKFold` sobre `spatial_block_1km`: **ningun bloque aparece a la vez en entrenamiento y validacion**, y la proporcion de la clase positiva se conserva en cada fold.

## 6.4 Metricas por fold

| fold | modelo | pr_auc | roc_auc | recall | precision | f2 | mcc | TP | FN |
|---|---|---|---|---|---|---|---|---|---|
| 1 | random_forest | 0.9728 | 0.9993 | 0.7534 | 0.9846 | 0.7905 | 0.8594 | 8932 | 2924 |
| 1 | xgboost | 0.9750 | 0.9995 | 0.7889 | 0.9813 | 0.8211 | 0.8782 | 9353 | 2503 |
| 2 | random_forest | 0.9677 | 0.9993 | 0.7092 | 0.9761 | 0.7502 | 0.8297 | 8980 | 3682 |
| 2 | xgboost | 0.9724 | 0.9994 | 0.7247 | 0.9850 | 0.7651 | 0.8427 | 9176 | 3486 |
| 3 | random_forest | 0.9939 | 0.9999 | 0.8961 | 0.9942 | 0.9141 | 0.9430 | 10571 | 1226 |
| 3 | xgboost | 0.9934 | 0.9999 | 0.8971 | 0.9927 | 0.9147 | 0.9428 | 10583 | 1214 |
| 4 | random_forest | 0.9615 | 0.9992 | 0.6329 | 0.9827 | 0.6814 | 0.7862 | 7501 | 4351 |
| 4 | xgboost | 0.9532 | 0.9992 | 0.6647 | 0.9563 | 0.7079 | 0.7948 | 7878 | 3974 |
| 5 | random_forest | 0.9829 | 0.9996 | 0.7972 | 0.9889 | 0.8294 | 0.8864 | 9550 | 2429 |
| 5 | xgboost | 0.9843 | 0.9997 | 0.8199 | 0.9857 | 0.8484 | 0.8975 | 9821 | 2158 |
| 1 | logistic | 0.9430 | 0.9984 | 0.8498 | 0.9132 | 0.8617 | 0.8791 | 10075 | 1781 |
| 2 | logistic | 0.8618 | 0.9971 | 0.7430 | 0.8649 | 0.7646 | 0.7986 | 9408 | 3254 |
| 3 | logistic | 0.9636 | 0.9987 | 0.9365 | 0.9275 | 0.9347 | 0.9309 | 11048 | 749 |
| 4 | logistic | 0.8285 | 0.9970 | 0.5858 | 0.8695 | 0.6267 | 0.7101 | 6943 | 4909 |
| 5 | logistic | 0.9160 | 0.9980 | 0.8212 | 0.8141 | 0.8197 | 0.8146 | 9837 | 2142 |

## Agregado por modelo

| Modelo | PR-AUC medio | Desv. | Min | Max | Recall medio |
|---|---|---|---|---|---|
| `logistic` | 0.9026 | 0.0563 | 0.8285 | 0.9636 | 0.7873 |
| `random_forest` | 0.9758 | 0.0128 | 0.9615 | 0.9939 | 0.7578 |
| `xgboost` | 0.9757 | 0.0150 | 0.9532 | 0.9934 | 0.7790 |

**Modelo mas robusto bajo validacion espacial: `random_forest`** (mayor PR-AUC medio). Es el que se usa en el Ejercicio 7.

## 6.5-6.6 Comparacion con la division aleatoria

| Modelo | PR-AUC aleatorio | PR-AUC espacial | Caida absoluta | Caida relativa |
|---|---|---|---|---|
| `logistic` | 0.9135 | 0.9026 | +0.0109 | -1.2 % |
| `random_forest` | 0.9814 | 0.9758 | +0.0056 | -0.6 % |
| `xgboost` | 0.9826 | 0.9757 | +0.0069 | -0.7 % |

**Por que cae el desempeno.** La division aleatoria coloca pixeles vecinos del mismo bloque en entrenamiento y prueba a la vez. Como la clorofila esta espacialmente autocorrelacionada, el modelo puede reconocer un vecino casi identico en lugar de generalizar. La validacion espacial elimina esa ventaja al separar bloques enteros, asi que **estima mucho mejor la capacidad de predecir en una zona nueva**.
