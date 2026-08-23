# Parte 1 regenerada con datos Sentinel-2 L1C reales

Generado: 2026-08-23 17:11:10  
Fuente: 22 GeoTIFF reales (SENTINEL2_L1C), EPSG:32615, 20 m.  
Area por pixel derivada de la transformacion afin: **0.0400 ha** (400 m2).

> Ningun valor de este documento procede de `synthetic_bands()`.

---

## Verificacion del evalscript CyanoLakes

- Escena verificada: Amatitlan 2025-01-28
- Pixeles comparados uno a uno: 490
- Discrepancias en la mascara de agua: 0
- Diferencias maximas frente al calculo literal del evalscript:

| Magnitud | Diferencia maxima |
|---|---|
| NDVI | 0.000e+00 |
| NDWI | 0.000e+00 |
| FAI | 0.000e+00 |
| NDCI | 0.000e+00 |
| CHL | 0.000e+00 |

Resultado: **COINCIDE** con tolerancia 1e-06.

### Limitaciones del algoritmo (documentadas, no estimadas aqui)

- Opera sobre **reflectancia de tope de atmosfera (L1C)**, sin correccion atmosferica.
- El polinomio fue **calibrado sobre datos simulados**, no sobre muestras de estos lagos.
- Calibracion especifica para **Microcystis aeruginosa**; otras especies responden distinto.
- Errores reportados por la fuente: **MAPE 42.3 %** y **RMSE relativo 95.8 %**.
- Dominio de calibracion del NDCI: **1-60 mg/m3** (Mishra & Mishra 2012).
- La coleccion `SENTINEL2_L1C` de openEO **no expone CLM, CLP, dataMask, SCL ni QA60**: no existe mascara de nubes por pixel. El unico control es la nubosidad oficial por escena y los filtros de validez espectral.
- Estos valores son **estimaciones satelitales**, no mediciones de laboratorio. No se realizo ninguna validacion in situ.

---

## Actividad 4: analisis temporal

| Lago | Fecha | Agua (ha) | Media | Mediana | Desv. est. | p95 | Max |
|---|---|---|---|---|---|---|---|
| Amatitlan | 2025-01-28 | 1,462 | 4.42 | 4.48 | 0.83 | 4.82 | 23.56 |
| Amatitlan | 2025-04-15 | 1,447 | 4.54 | 4.58 | 1.35 | 6.53 | 26.28 |
| Amatitlan | 2025-04-28 | 1,447 | 5.77 | 4.88 | 2.96 | 11.51 | 42.44 |
| Amatitlan | 2025-11-24 | 1,457 | 4.64 | 4.01 | 5.49 | 6.23 | 113.83 |
| Amatitlan | 2026-01-08 | 1,467 | 6.65 | 4.89 | 8.56 | 13.23 | 130.12 |
| Amatitlan | 2026-02-02 | 1,445 | 4.28 | 4.39 | 0.65 | 4.78 | 16.49 |
| Amatitlan | 2026-02-07 | 1,462 | 4.30 | 4.45 | 0.82 | 4.80 | 17.76 |
| Amatitlan | 2026-03-29 | 1,464 | 6.44 | 5.25 | 3.96 | 15.38 | 52.00 |
| Amatitlan | 2026-04-13 | 1,471 | 6.77 | 6.40 | 1.75 | 9.95 | 36.59 |
| Amatitlan | 2026-04-28 | 1,396 | 9.92 | 6.44 | 7.50 | 26.10 | 47.14 |
| Amatitlan | 2026-06-19 | 1,466 | 11.50 | 10.66 | 6.55 | 22.46 | 81.33 |
| Atitlan | 2025-01-18 | 12,316 | 0.33 | 0.33 | 1.33 | 2.08 | 30.94 |
| Atitlan | 2025-04-13 | 12,213 | 1.75 | 1.73 | 0.81 | 2.91 | 15.98 |
| Atitlan | 2025-05-13 | 12,212 | 1.26 | 1.25 | 1.13 | 2.94 | 18.38 |
| Atitlan | 2025-07-17 | 11,948 | 1.06 | 1.12 | 1.87 | 3.00 | 135.86 |
| Atitlan | 2025-11-21 | 12,212 | 0.22 | 0.23 | 1.37 | 2.01 | 29.51 |
| Atitlan | 2025-12-29 | 12,376 | 0.49 | 0.47 | 1.31 | 2.19 | 26.01 |
| Atitlan | 2026-02-12 | 12,245 | 0.87 | 0.88 | 0.99 | 2.21 | 32.88 |
| Atitlan | 2026-03-24 | 12,217 | 1.24 | 1.22 | 0.86 | 2.46 | 17.32 |
| Atitlan | 2026-04-13 | 12,187 | 2.10 | 2.18 | 1.03 | 3.49 | 15.38 |
| Atitlan | 2026-04-28 | 12,185 | 1.83 | 1.87 | 0.93 | 3.20 | 14.73 |
| Atitlan | 2026-07-22 | 12,164 | 1.11 | 1.10 | 1.42 | 3.02 | 25.14 |

Valores en ug/L de clorofila-a sobre pixeles de agua.

### Fechas criticas (calculadas, no escritas a mano)

| Lago | Fecha | Media | Umbral del criterio |
|---|---|---|---|
| Amatitlan | 2026-04-28 | 9.92 | 8.71 |
| Amatitlan | 2026-06-19 | 11.50 | 8.71 |
| Atitlan | 2025-04-13 | 1.75 | 1.73 |
| Atitlan | 2026-04-13 | 2.10 | 1.73 |
| Atitlan | 2026-04-28 | 1.83 | 1.73 |

---

## Actividad 6: correlaciones

| Lago | Par | Pearson |
|---|---|---|
| Amatitlan | NDVI - NDWI | -0.935 |
| Amatitlan | NDCI - NDVI | 0.756 |
| Amatitlan | NDCI - NDWI | -0.575 |
| Amatitlan | FAI - NDVI | 0.851 |
| Amatitlan | FAI - NDWI | -0.743 |
| Amatitlan | FAI - NDCI | 0.815 |
| Amatitlan | Clorofila-a - NDVI | 0.755 |
| Amatitlan | Clorofila-a - NDWI | -0.613 |
| Amatitlan | Clorofila-a - NDCI | 0.776 |
| Amatitlan | Clorofila-a - FAI | 0.882 |
| Atitlan | NDVI - NDWI | -0.938 |
| Atitlan | NDCI - NDVI | 0.666 |
| Atitlan | NDCI - NDWI | -0.633 |
| Atitlan | FAI - NDVI | 0.441 |
| Atitlan | FAI - NDWI | -0.308 |
| Atitlan | FAI - NDCI | 0.544 |
| Atitlan | Clorofila-a - NDVI | 0.620 |
| Atitlan | Clorofila-a - NDWI | -0.600 |
| Atitlan | Clorofila-a - NDCI | 0.890 |
| Atitlan | Clorofila-a - FAI | 0.419 |

**Advertencia de circularidad.** La correlacion entre NDCI y clorofila-a no es evidencia ecologica: la clorofila se calcula como un polinomio del NDCI, asi que la relacion es una identidad matematica y su valor alto era inevitable. Lo mismo aplica parcialmente al FAI y al NDVI, que comparten la banda B04 con el NDCI. Las unicas relaciones interpretables como senal ambiental son las que implican NDWI, que no participa en la cadena de calculo de la clorofila.

---

## Actividad 7: comparacion entre lagos

| Indicador | Amatitlan | Atitlan |
|---|---|---|
| chl_media_periodo | 6.30 | 1.11 |
| chl_mediana_periodo | 4.88 | 1.12 |
| chl_max_observada | 130.12 | 135.86 |
| variabilidad_entre_fechas_std | 2.41 | 0.62 |
| coef_variacion | 0.38 | 0.55 |
| area_agua_media_km2 | 14.53 | 122.07 |
| fecha_mas_critica | 2026-06-19 | 2026-04-13 |
| fecha_menos_critica | 2026-02-02 | 2025-11-21 |

---

## Actividades 8 y 10: analisis de umbrales

### Global

| Umbral | Positivos | % del agua | Area (ha) | Fechas con positivos |
|---|---|---|---|---|
| 8 ug/L | 60,146 | 1.6011 % | 2,405.8 | 22/22 |
| 20 ug/L | 9,832 | 0.2617 % | 393.3 | 15/22 |
| 25 ug/L | 5,195 | 0.1383 % | 207.8 | 14/22 |
| 50 ug/L | 608 | 0.0162 % | 24.3 | 5/22 |

### Por lago

| Umbral | Lago | Positivos | % | Area (ha) | Fechas con positivos | Fechas con ambas clases |
|---|---|---|---|---|---|---|
| 8 | Amatitlan | 57,996 | 14.5118 % | 2,319.8 | 11/11 | 11 |
| 8 | Atitlan | 2,150 | 0.0640 % | 86.0 | 11/11 | 11 |
| 20 | Amatitlan | 9,768 | 2.4442 % | 390.7 | 9/11 | 9 |
| 20 | Atitlan | 64 | 0.0019 % | 2.6 | 6/11 | 6 |
| 25 | Amatitlan | 5,181 | 1.2964 % | 207.2 | 8/11 | 8 |
| 25 | Atitlan | 14 | 0.0004 % | 0.6 | 6/11 | 6 |
| 50 | Amatitlan | 604 | 0.1511 % | 24.2 | 4/11 | 4 |
| 50 | Atitlan | 4 | 0.0001 % | 0.2 | 1/11 | 1 |

### Significado ambiental

- **8 ug/L**: Frontera mesotrofico -> eutrofico (OECD 1982). Marca el inicio de la condicion eutrofica.
- **20 ug/L**: Umbral usado en la primera version de la Parte 1. Cae dentro de la banda eutrofica de OECD (8-25) y del rango de Alerta 1 de la OMS (12-24), pero NO es una frontera publicada por si mismo.
- **25 ug/L**: Frontera eutrofico -> hipertrofico (OECD 1982); coincide practicamente con el techo de la Alerta 1 de la OMS (24 ug/L con dominancia de cianobacterias).
- **50 ug/L**: Escenario severo para analisis de sensibilidad. NO aparece como valor de clorofila-a en OECD 1982 ni en las guias de la OMS 2021.

### Viabilidad para clasificacion binaria

| Umbral | % positivo | Positivos por lago | Entrenar por lago | Evaluar entre lagos | Observacion |
|---|---|---|---|---|---|
| 8 | 1.6011 % | Amatitlan=57996;Atitlan=2150 | False | False | Entrenable solo en Amatitlan. Atitlan solo 2150 positivos (0.0640 %): el experimento entre lagos queda degenerado en ese sentido. |
| 20 | 0.2617 % | Amatitlan=9768;Atitlan=64 | False | False | Entrenable solo en Amatitlan. Atitlan solo 64 positivos (0.0019 %): el experimento entre lagos queda degenerado en ese sentido. |
| 25 | 0.1383 % | Amatitlan=5181;Atitlan=14 | False | False | Entrenable solo en Amatitlan. Atitlan solo 14 positivos (0.0004 %): el experimento entre lagos queda degenerado en ese sentido. |
| 50 | 0.0162 % | Amatitlan=604;Atitlan=4 | False | False | Todos los lagos quedan por debajo del minimo practico (1000 positivos y 0.1 %): el umbral no sostiene un modelo binario. |

Criterio de viabilidad: >= 1000 positivos y >= 0.1 % por lago. Tener uno o dos pixeles positivos no hace viable un modelo.

### Recomendacion

Se recomienda **25 ug/L** como umbral principal por su significado ambiental: es la frontera eutrofico-hipertrofico de OECD (1982) y coincide con el techo de la Alerta 1 de la OMS (24 ug/L). La eleccion se toma por significado ambiental y **no** por balance de clases. La viabilidad estadistica se discute por separado en la tabla anterior: el desbalance resultante es severo y condiciona el modelado de la Parte 2.

Como analisis de sensibilidad se recomienda repetir con **8 ug/L** (inicio de la condicion eutrofica), que es el umbral con mas positivos y por tanto el unico que podria sostener un experimento entre lagos.
