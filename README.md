# lab4data

Resumen
=======

Este repositorio contiene utilidades para el analisis de cianobacterias sobre imagenes Sentinel-2.

Archivos
- `mainlab4.py`: traduccion de funciones del script CyanoLakes Chlorophyll-a NDCI L1C a Python, mas utilidades para conectar openEO, recortar por lago, calcular NDVI/NDWI y resumir series temporales.
- `requirements.txt`: dependencias recomendadas.

Datos oficiales del PDF
- Lagos: Atitlan y Amatitlan.
- Coordenadas: incluidas en `LAKE_BBOXES`.
- Fechas oficiales: incluidas en `OFFICIAL_DATES`.
- Estas coordenadas y fechas aplican desde las actividades 1-4. El PDF indica que deben usarse exclusivamente esas fechas para reducir descargas y asegurar que todos trabajen con la misma base de imagenes.

Como usar
1. Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

2. Ejecutar un ejemplo sintetico:

```bash
python mainlab4.py
```

3. Para analisis real:
- Establezca la conexion con `connect_to_openeo_backend()`.
- Use `load_official_lake_date_cube()` para trabajar con un lago y una fecha oficial del PDF, o use `load_sentinel2_cube()` si quiere pasar manualmente el bbox/GeoJSON. Por defecto usa `SENTINEL2_L1C`, porque el script provisto es CyanoLakes Chlorophyll-a NDCI L1C.
- Para calcular localmente los indices, cargue las bandas como arreglos NumPy con `load_bands_from_geotiffs()` y use:
  - `NDVI(B04, B08)`
  - `NDWI(B03, B08)`
  - `compute_cyan_index_for_date(...)` para mascara de agua, FAI, NDCI y clorofila/cianobacteria.
- Para usar Sentinel Hub directamente, es mejor adaptar el script para que descargue valores numericos y no solo la paleta RGB. En `mainlab4.py` esta `CYANOLAKES_NUMERIC_EVALSCRIPT`, que retorna: cianobacteria/clorofila, NDCI, FAI, NDVI, NDWI y mascara de agua.
- Para el analisis temporal use `temporal_summary(...)`, `detect_bloom_peaks(...)` y `plot_temporal_evolution(...)`.

Estado frente a las actividades 1-4
- Actividad 1: incluida como funcion, pero requiere autenticar una cuenta openEO/Copernicus o Sentinel Hub.
- Actividad 2: incluida como carga/descarga filtrada por lago, fecha y bandas.
- Actividad 3: NDVI y NDWI estan implementados; el indice de cianobacteria esta reproducido localmente con las bandas minimas del script L1C traducido. Si se usa Sentinel Hub, conviene descargar el resultado numerico del evalscript y no la imagen coloreada.
- Actividad 4: incluidas funciones para promedio por lago/fecha, grafico temporal y deteccion de picos. Falta ejecutar con datos reales y redactar la interpretacion final de patrones.
