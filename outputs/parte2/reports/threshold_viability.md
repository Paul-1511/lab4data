# Viabilidad estadistica de los umbrales — Laboratorio 4, Parte 2

Generado: 2026-08-23 19:24:46  
Dataset: version 2.0, 3,756,510 observaciones reales.  
Respuesta principal: **`high_cyano_8`** (clorofila-a >= 8 ug/L)

## Criterios aplicados

Un umbral no se declara viable por tener un pixel positivo. Se exige:

- Ambas clases presentes.
- Para **entrenar en un lago**: >= 1000 positivos y >= 0.05 % de ese lago.
- Para **validacion temporal**: >= 3 fechas con ambas clases.
- Para **GroupKFold espacial** con 5 folds: >= 10 bloques con positivos y numero efectivo de bloques >= 5, para que ningun fold quede sin clase positiva.

El *numero efectivo de bloques* es el inverso del indice de Herfindahl sobre el reparto de positivos: vale N si los positivos se distribuyen por igual entre N bloques y 1 si estan todos concentrados en uno. Mide dispersion real, no simple conteo.

## Resumen por umbral

| Umbral | Positivos | % | Desbalance | Fechas ambas clases | Bloques con positivos | N.o efectivo | Stratified | GroupKFold | Temporal | Entre lagos |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 **(principal)** | 60,146 | 1.6011 % | 1:61 | 22/22 | 120/308 | 20.5 | Si | Si | Si | Si |
| 20 | 9,832 | 0.2617 % | 1:381 | 15/22 | 53/308 | 11.9 | Si | Si | Si | No |
| 25 | 5,195 | 0.1383 % | 1:722 | 14/22 | 42/308 | 9.7 | Si | Si | Si | No |
| 50 | 608 | 0.0162 % | 1:6177 | 5/22 | 13/308 | 5.0 | Si | Si | Si | No |

## Por lago

| Umbral | Lago | Positivos | Negativos | % | Entrenable | Evaluable |
|---|---|---|---|---|---|---|
| 8 | Amatitlan | 57,996 | 341,650 | 14.5118 % | Si | Si |
| 8 | Atitlan | 2,150 | 3,354,714 | 0.0640 % | Si | Si |
| 20 | Amatitlan | 9,768 | 389,878 | 2.4442 % | Si | Si |
| 20 | Atitlan | 64 | 3,356,800 | 0.0019 % | No | Si |
| 25 | Amatitlan | 5,181 | 394,465 | 1.2964 % | Si | Si |
| 25 | Atitlan | 14 | 3,356,850 | 0.0004 % | No | Si |
| 50 | Amatitlan | 604 | 399,042 | 0.1511 % | No | Si |
| 50 | Atitlan | 4 | 3,356,860 | 0.0001 % | No | Si |

## Concentracion de los positivos

| Umbral | % de positivos en las 5 fechas principales | % en los 5 bloques principales | Bloques positivos | N.o efectivo |
|---|---|---|---|---|
| 8 | 87.6 % | 37.7 % | 120 | 20.5 |
| 20 | 96.8 % | 54.4 % | 53 | 11.9 |
| 25 | 98.2 % | 58.0 % | 42 | 9.7 |
| 50 | 100.0 % | 86.2 % | 13 | 5.0 |

## Desbalance de la respuesta principal (8 ug/L)

- **Desbalance global:** 60,146 positivos frente a 3,696,364 negativos (1.601 %), razon **1:61**.
- **Amatitlan:** 57,996 positivos (14.5118 %). Entrenable: si.
- **Atitlan:** 2,150 positivos (0.0640 %). Entrenable: si.
- **Diferencia entre lagos:** la prevalencia no es comparable entre ambos, asi que un modelo entrenado en uno vera una frecuencia de la clase positiva muy distinta a la del otro.
- **Diferencia entre fechas:** solo 22 de 22 combinaciones lago-fecha contienen ambas clases.
- **Dependencia espacial:** el 37.7 % de los positivos se concentra en solo 5 bloques de 1 km. Los positivos no son independientes entre si: estan agrupados espacialmente.

### Riesgos derivados

- **Accuracy enganosa.** Un clasificador que prediga siempre la clase mayoritaria acertaria el 98.399 % sin detectar ni una sola floracion.
- **Folds sin positivos.** Con particion aleatoria simple algunos pliegues podrian quedarse sin clase positiva; con agrupacion espacial el riesgo aumenta porque los positivos estan concentrados.
- **Optimismo por autocorrelacion espacial.** Sin agrupar por bloque, pixeles vecinos casi identicos caerian en entrenamiento y prueba a la vez.
- **Colapso a la clase mayoritaria** durante el entrenamiento si no se compensa el desbalance.

### Recomendaciones para el Ejercicio 4 (no implementadas todavia)

- `class_weight="balanced"` en Regresion Logistica y Random Forest.
- `scale_pos_weight = n_negativos / n_positivos` en XGBoost.
- Reportar **PR-AUC** ademas de Accuracy, Precision, Recall, F1 y ROC-AUC: con esta prevalencia la curva ROC resulta demasiado optimista.
- Reportar **Recall y F1 de la clase positiva** por separado, no solo macro.
- Ajustar el **umbral de decision** usando solo entrenamiento/validacion, nunca el conjunto de prueba.
- **No aplicar SMOTE ni sobremuestreo antes de separar los grupos**: generaria vecinos sinteticos a partir de pixeles que luego caerian en el conjunto de prueba, inflando el desempeno.
- Conservar los grupos espaciales (`spatial_block_1km`) y temporales (`date`) intactos durante toda la particion.
