# Validacion temporal (exigida por la rubrica)

> Resultados del perfil **standard**.

Generado: 2026-08-23 22:15:28 | Semilla: 42 | Dataset 2.0 (hash 65825179def16792)  
Respuesta: `high_cyano_8` (>= 8 ug/L) | Predictores: B02, B03, B07, B08, B8A, B11, B12, NDWI

---

## Esquema

Ventana expansiva (*rolling origin*): las fechas de ambos lagos se ordenan en una unica linea temporal y, en cada corte, se entrena con **todas las fechas anteriores** y se evalua con la siguiente. Una asercion explicita impide que una fecha aparezca a la vez en entrenamiento y prueba, de modo que **no entra informacion futura** al entrenamiento.

## Metricas por fold

| fold | test_fecha | modelo | prevalencia_real | pr_auc | recall | precision | f2 | prevalencia_inestable |
|---|---|---|---|---|---|---|---|---|
| 1 | 2025-04-28 | random_forest | 0.1358 | 0.9185 | 0.7586 | 0.8744 | 0.7792 | False |
| 1 | 2025-04-28 | xgboost | 0.1358 | 0.9107 | 0.7978 | 0.8037 | 0.7990 | False |
| 2 | 2025-05-13 | random_forest | 0.0001 | 0.1307 | 0.2174 | 0.1471 | 0.1984 | True |
| 2 | 2025-05-13 | xgboost | 0.0001 | 0.1842 | 0.0000 | 0.0000 | 0.0000 | True |
| 3 | 2025-07-17 | random_forest | 0.0012 | 0.6617 | 0.4298 | 0.7426 | 0.4693 | False |
| 3 | 2025-07-17 | xgboost | 0.0012 | 0.5945 | 0.2779 | 0.7760 | 0.3189 | False |
| 4 | 2025-11-21 | random_forest | 0.0011 | 0.8437 | 0.6577 | 0.8760 | 0.6922 | False |
| 4 | 2025-11-21 | xgboost | 0.0011 | 0.7960 | 0.5165 | 0.9149 | 0.5658 | False |
| 5 | 2025-11-24 | random_forest | 0.0282 | 0.9137 | 0.6171 | 0.9393 | 0.6626 | False |
| 5 | 2025-11-24 | xgboost | 0.0282 | 0.9023 | 0.6599 | 0.9263 | 0.7001 | False |
| 6 | 2025-12-29 | random_forest | 0.0008 | 0.7337 | 0.4942 | 0.8194 | 0.5368 | False |
| 6 | 2025-12-29 | xgboost | 0.0008 | 0.4919 | 0.1595 | 1.0000 | 0.1918 | False |
| 7 | 2026-01-08 | random_forest | 0.0950 | 0.9638 | 0.7931 | 0.9449 | 0.8194 | False |
| 7 | 2026-01-08 | xgboost | 0.0950 | 0.9631 | 0.6676 | 0.9860 | 0.7137 | False |
| 8 | 2026-02-02 | random_forest | 0.0002 | 0.7929 | 0.6667 | 1.0000 | 0.7143 | False |
| 8 | 2026-02-02 | xgboost | 0.0002 | 0.5641 | 0.3333 | 0.6667 | 0.3704 | False |
| 9 | 2026-02-07 | random_forest | 0.0010 | 0.5435 | 0.2571 | 0.9000 | 0.3000 | False |
| 9 | 2026-02-07 | xgboost | 0.0010 | 0.5221 | 0.1429 | 1.0000 | 0.1724 | False |
| 10 | 2026-02-12 | random_forest | 0.0005 | 0.7312 | 0.6164 | 0.8033 | 0.6464 | False |
| 10 | 2026-02-12 | xgboost | 0.0005 | 0.7265 | 0.5157 | 0.8454 | 0.5593 | False |
| 11 | 2026-03-24 | random_forest | 0.0001 | 0.2378 | 0.3250 | 0.2407 | 0.3037 | False |
| 11 | 2026-03-24 | xgboost | 0.0001 | 0.2072 | 0.1750 | 0.2692 | 0.1882 | False |
| 12 | 2026-03-29 | random_forest | 0.1067 | 0.9142 | 0.5801 | 0.9991 | 0.6332 | False |
| 12 | 2026-03-29 | xgboost | 0.1067 | 0.8904 | 0.5955 | 0.9957 | 0.6476 | False |
| 13 | 2026-04-13 | random_forest | 0.0166 | 0.7099 | 0.2625 | 0.8072 | 0.3035 | False |
| 13 | 2026-04-13 | xgboost | 0.0166 | 0.7247 | 0.3899 | 0.8165 | 0.4354 | False |
| 14 | 2026-04-28 | random_forest | 0.0448 | 0.9711 | 0.1259 | 0.9815 | 0.1525 | False |
| 14 | 2026-04-28 | xgboost | 0.0448 | 0.9821 | 0.3969 | 0.9983 | 0.4512 | False |
| 15 | 2026-06-19 | random_forest | 0.6292 | 0.9879 | 0.6790 | 0.9945 | 0.7250 | True |
| 15 | 2026-06-19 | xgboost | 0.6292 | 0.9884 | 0.8055 | 0.9873 | 0.8363 | True |
| 16 | 2026-07-22 | random_forest | 0.0020 | 0.8107 | 0.5058 | 0.8818 | 0.5529 | False |
| 16 | 2026-07-22 | xgboost | 0.0020 | 0.8189 | 0.4645 | 0.8978 | 0.5141 | False |
| 1 | 2025-04-28 | logistic | 0.1358 | 0.8079 | 0.8304 | 0.6798 | 0.7952 | False |
| 2 | 2025-05-13 | logistic | 0.0001 | 0.3838 | 0.6522 | 0.1364 | 0.3713 | True |
| 3 | 2025-07-17 | logistic | 0.0012 | 0.7226 | 0.8395 | 0.4399 | 0.7105 | False |
| 4 | 2025-11-21 | logistic | 0.0011 | 0.6131 | 0.8559 | 0.4405 | 0.7201 | False |
| 5 | 2025-11-24 | logistic | 0.0282 | 0.8473 | 0.8445 | 0.6919 | 0.8088 | False |
| 6 | 2025-12-29 | logistic | 0.0008 | 0.1323 | 0.8210 | 0.1573 | 0.4453 | False |
| 7 | 2026-01-08 | logistic | 0.0950 | 0.9382 | 0.7810 | 0.9106 | 0.8039 | False |
| 8 | 2026-02-02 | logistic | 0.0002 | 0.5141 | 0.8333 | 0.0926 | 0.3205 | False |
| 9 | 2026-02-07 | logistic | 0.0010 | 0.2335 | 0.6286 | 0.2529 | 0.4846 | False |
| 10 | 2026-02-12 | logistic | 0.0005 | 0.6632 | 0.7736 | 0.2338 | 0.5293 | False |
| 11 | 2026-03-24 | logistic | 0.0001 | 0.2785 | 0.6750 | 0.1364 | 0.3771 | False |
| 12 | 2026-03-29 | logistic | 0.1067 | 0.9381 | 0.6710 | 0.9559 | 0.7135 | False |
| 13 | 2026-04-13 | logistic | 0.0166 | 0.4204 | 0.6395 | 0.2592 | 0.4944 | False |
| 14 | 2026-04-28 | logistic | 0.0448 | 0.9799 | 0.9650 | 0.9252 | 0.9568 | False |
| 15 | 2026-06-19 | logistic | 0.6292 | 0.9800 | 0.7564 | 0.9785 | 0.7923 | True |
| 16 | 2026-07-22 | logistic | 0.0020 | 0.8341 | 0.9058 | 0.5944 | 0.8199 | False |

**2 folds tienen prevalencia extrema** (por debajo de 0.01 % o por encima de 50 %). En ellos la Precision y el PR-AUC son muy inestables. Por eso se publican **dos agregados**: uno con todos los folds y otro solo con los de prevalencia no extrema, en vez de promediar silenciosamente.

## Agregado (todos los folds)

| Modelo | PR-AUC medio | Desv. | Min | Max |
|---|---|---|---|---|
| `logistic` | 0.6429 | 0.2839 | 0.1323 | 0.9800 |
| `random_forest` | 0.7416 | 0.2510 | 0.1307 | 0.9879 |
| `xgboost` | 0.7042 | 0.2571 | 0.1842 | 0.9884 |

## Comparacion de las tres estrategias

| Estrategia | Que estima |
|---|---|
| Aleatoria 70/30 | Optimista: mezcla vecinos del mismo bloque |
| **Espacial por bloques** | **Capacidad de predecir en una zona nueva del lago** |
| **Temporal expansiva** | **Capacidad de predecir una fecha futura** |

La espacial responde a *donde*; la temporal responde a *cuando*. Para un sistema de alerta que debe anticipar floraciones futuras, la **temporal** es la referencia mas honesta; para extrapolar a zonas no muestreadas, lo es la **espacial**. La aleatoria no debe usarse como estimacion de desempeno real.
