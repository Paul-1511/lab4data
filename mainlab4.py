import numpy as np
import matplotlib.pyplot as plt
import rasterio
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Para divisiones seguras

S2_L1C_COLLECTION = "SENTINEL2_L1C"
CYANO_REQUIRED_BANDS = ["B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12"]

LAKE_BBOXES = {
    "Atitlan": {
        "west": -91.326256,
        "east": -91.07151,
        "south": 14.5948,
        "north": 14.750979,
    },
    "Amatitlan": {
        "west": -90.638065,
        "east": -90.512924,
        "south": 14.412347,
        "north": 14.493799,
    },
}

OFFICIAL_DATES = {
    "Amatitlan": [
        "2025-01-28",
        "2025-04-15",
        "2025-04-28",
        "2025-11-24",
        "2026-01-08",
        "2026-02-02",
        "2026-02-07",
        "2026-03-29",
        "2026-04-13",
        "2026-04-28",
        "2026-06-19",
    ],
    "Atitlan": [
        "2025-01-18",
        "2025-04-13",
        "2025-05-13",
        "2025-07-17",
        "2025-11-21",
        "2025-12-29",
        "2026-02-12",
        "2026-03-24",
        "2026-04-13",
        "2026-04-28",
        "2026-07-22",
    ],
}

OFFICIAL_CLOUD_COVER = {
    "Amatitlan": {
        "2025-01-28": 0.06,
        "2025-04-15": 0.09,
        "2025-04-28": 1.03,
        "2025-11-24": 0.50,
        "2026-01-08": 0.77,
        "2026-02-02": 0.39,
        "2026-02-07": 0.02,
        "2026-03-29": 0.01,
        "2026-04-13": 0.09,
        "2026-04-28": 4.96,
        "2026-06-19": 13.00,
    },
    "Atitlan": {
        "2025-01-18": 0.02,
        "2025-04-13": 0.54,
        "2025-05-13": 4.37,
        "2025-07-17": 3.57,
        "2025-11-21": 3.15,
        "2025-12-29": 3.17,
        "2026-02-12": 0.04,
        "2026-03-24": 3.17,
        "2026-04-13": 0.01,
        "2026-04-28": 4.96,
        "2026-07-22": 4.02,
    },
}

CYANOLAKES_NUMERIC_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12", "dataMask"],
    output: {
      bands: 6,
      sampleType: "FLOAT32"
    }
  };
}

function safeDiv(a, b) {
  return b === 0 ? 0 : a / b;
}

function evaluatePixel(s) {
  let ndvi = safeDiv(s.B08 - s.B04, s.B08 + s.B04);
  let ndwi = safeDiv(s.B03 - s.B08, s.B03 + s.B08);
  let mndwi = safeDiv(s.B03 - s.B11, s.B03 + s.B11);
  let ndwiLeaves = safeDiv(s.B08 - s.B11, s.B08 + s.B11);
  let aweish = s.B02 + 2.5 * s.B03 - 1.5 * (s.B08 + s.B11) - 0.25 * s.B12;
  let aweinsh = 4 * (s.B03 - s.B11) - (0.25 * s.B08 + 2.75 * s.B11);
  let dbsi = safeDiv(s.B11 - s.B03, s.B11 + s.B03) - ndvi;

  let water = (
    mndwi > 0.42 || ndwi > 0.4 || aweinsh > 0.1879 ||
    aweish > 0.1112 || ndvi < -0.2 || ndwiLeaves > 1
  ) ? 1 : 0;
  if (water === 1 && (aweinsh <= -0.03 || dbsi > 0)) {
    water = 0;
  }

  let fai = s.B07 - s.B04 - (s.B8A - s.B04) * (783 - 665) / (865 - 665);
  let ndci = safeDiv(s.B05 - s.B04, s.B05 + s.B04);
  let chl = 826.57 * Math.pow(ndci, 3) - 176.43 * Math.pow(ndci, 2) + 19 * ndci + 4.071;
  let cyano = water === 1 ? chl : NaN;

  return [cyano, ndci, fai, ndvi, ndwi, water * s.dataMask];
}
"""

# ============================================================
# 1. FUNCIONES AUXILIARES
# ============================================================

def safe_div(a, b):
    """División segura: devuelve 0 cuando el denominador es 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

def normalize_band(band_array, scale=10000.0):
    """
    Convierte bandas en enteros (0-10000) a reflectancia (0-1).
    Si ya están en float 0-1, las devuelve sin cambios.
    """
    if band_array.dtype.kind in 'iu':  # integer
        return band_array.astype(np.float32) / scale
    else:
        return band_array.astype(np.float32)

def NDVI(B04, B08):
    """Normalized Difference Vegetation Index: (B08 - B04) / (B08 + B04)."""
    return safe_div(B08.astype(float) - B04.astype(float),
                    B08.astype(float) + B04.astype(float))

def NDWI(B03, B08):
    """Normalized Difference Water Index: (B03 - B08) / (B03 + B08)."""
    return safe_div(B03.astype(float) - B08.astype(float),
                    B03.astype(float) + B08.astype(float))

def bbox_from_geojson(geojson):
    """
    Obtiene west/south/east/north desde un GeoJSON Polygon, MultiPolygon o Feature.
    El CRS esperado para openEO/Sentinel Hub es EPSG:4326.
    """
    bbox_keys = {"west", "south", "east", "north"}
    if bbox_keys.issubset(geojson):
        return {key: geojson[key] for key in bbox_keys}

    if geojson.get("type") == "Feature":
        geojson = geojson["geometry"]

    coords = []

    def collect(points):
        if isinstance(points[0], (int, float)):
            coords.append(points[:2])
        else:
            for item in points:
                collect(item)

    collect(geojson["coordinates"])
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return {"west": min(xs), "south": min(ys), "east": max(xs), "north": max(ys)}

# ============================================================
# 2. DETECCIÓN DE AGUA (WBI)
# ============================================================

def wbi_vectorized(B04, B03, B02, B08, B11, B12,
                   MNDWI_threshold=0.42, NDWI_threshold=0.4,
                   filter_UABS=True):
    """
    Máscara de agua (Water Body Index) vectorizada.
    Parámetros: arrays de reflectancia (0-1) para las bandas:
        B04 (rojo), B03 (verde), B02 (azul),
        B08 (NIR), B11 (SWIR1), B12 (SWIR2).
    Retorna: booleano (True = agua).
    """
    r = B04.astype(float)
    g = B03.astype(float)
    b = B02.astype(float)
    nir = B08.astype(float)
    swir1 = B11.astype(float)
    swir2 = B12.astype(float)

    # Índices
    ndvi = safe_div(nir - r, nir + r)
    mndwi = safe_div(g - swir1, g + swir1)
    ndwi = safe_div(g - nir, g + nir)
    ndwi_leaves = safe_div(nir - swir1, nir + swir1)
    aweish = b + 2.5 * g - 1.5 * (nir + swir1) - 0.25 * swir2
    aweinsh = 4 * (g - swir1) - (0.25 * nir + 2.75 * swir1)
    dbsi = safe_div(swir1 - g, swir1 + g) - ndvi

    # Máscara inicial de agua
    ws = ((mndwi > MNDWI_threshold) |
          (ndwi > NDWI_threshold) |
          (aweinsh > 0.1879) |
          (aweish > 0.1112) |
          (ndvi < -0.2) |
          (ndwi_leaves > 1))

    # Filtro urbano / suelo desnudo
    if filter_UABS:
        ws = np.where(((aweinsh <= -0.03) | (dbsi > 0)), False, ws)

    return ws.astype(bool)

# ============================================================
# 3. ÍNDICES DE FLORACIÓN (FAI, NDCI, CLOROFILA)
# ============================================================

def FAI(B04, B07, B8A):
    """
    Floating Algae Index (FAI).
    B04: 665 nm, B07: 783 nm, B8A: 865 nm.
    """
    a = B04.astype(float)
    b = B07.astype(float)
    c = B8A.astype(float)
    return b - a - (c - a) * (783 - 665) / (865 - 665)

def NDCI(B04, B05):
    """
    Normalized Difference Chlorophyll Index (NDCI).
    B04: 665 nm, B05: 705 nm.
    """
    a = B04.astype(float)
    b = B05.astype(float)
    return safe_div(b - a, b + a)

def chl_from_ndci(ndci):
    """
    Estima la concentración de Clorofila-a (μg/L) a partir del NDCI.
    Polinomio ajustado con datos simulados (script original).
    """
    return 826.57 * ndci ** 3 - 176.43 * ndci ** 2 + 19 * ndci + 4.071

# ============================================================
# 4. GENERACIÓN DE IMAGEN RGB (PALETA DE COLORES COMPLETA)
# ============================================================

def generate_rgb_palette(water_mask, FAIv, chl):
    """
    Genera una imagen RGB (float 0..1) aplicando la paleta exacta del script original.
    Útil para visualizar mapas de clorofila al estilo Copernicus Browser.
    """
    # Crear array RGB vacío (fondo negro para no-agua)
    rgb = np.zeros(chl.shape + (3,), dtype=float)
    mask = water_mask

    # Definir todas las condiciones y colores asociados
    # Prioridad: FAI > 0.08 tiene máxima prioridad (vegetación flotante)
    conds = []
    colors = []

    # 1. Vegetación flotante (sobrescribe todo lo demás)
    conds.append(mask & (FAIv > 0.08))
    colors.append((233/255, 72/255, 21/255))  # Naranja

    # 2. Rangos de clorofila (solo si NO es vegetación flotante)
    base = mask & (FAIv <= 0.08)
    chl_ranges = [
        (0, 0.5, (0, 0, 1.0)),
        (0.5, 1, (0, 0, 1.0)),
        (1, 2.5, (0, 59/255, 1)),
        (2.5, 3.5, (0, 98/255, 1)),
        (3.5, 5, (15/255, 113/255, 141/255)),
        (5, 7, (14/255, 141/255, 120/255)),
        (7, 8, (13/255, 141/255, 103/255)),
        (8, 10, (30/255, 226/255, 28/255)),
        (10, 14, (42/255, 226/255, 28/255)),
        (14, 18, (68/255, 226/255, 28/255)),
        (18, 20, (68/255, 226/255, 28/255)),
        (20, 24, (134/255, 247/255, 0)),
        (24, 28, (140/255, 247/255, 0)),
        (28, 30, (205/255, 237/255, 0)),
        (30, 38, (208/255, 240/255, 0)),
        (38, 45, (208/255, 240/255, 0)),
        (45, 50, (251/255, 210/255, 3/255)),
        (50, 75, (248/255, 207/255, 2/255)),
        (75, 90, (134/255, 247/255, 0)),
        (90, 100, (245/255, 164/255, 9/255)),
        (100, 150, (240/255, 159/255, 8/255)),
        (150, 250, (237/255, 157/255, 7/255)),
        (250, 300, (239/255, 118/255, 15/255)),
        (300, 350, (239/255, 101/255, 15/255)),
        (350, 450, (239/255, 100/255, 14/255)),
        (450, 500, (233/255, 72/255, 21/255)),
        (500, np.inf, (233/255, 72/255, 21/255))  # >=500
    ]

    for low, high, color in chl_ranges:
        conds.append(base & (chl >= low) & (chl < high))
        colors.append(color)

    # Aplicar selección por canal (más eficiente que múltiples np.where)
    for i in range(3):
        choices = [c[i] for c in colors]
        rgb[..., i] = np.select(conds, choices, default=0.0)

    return np.clip(rgb, 0, 1)

# ============================================================
# 5. EJEMPLO DE USO CON DATOS REALES (rasterio)
# ============================================================

def load_bands_from_geotiffs(band_paths):
    """
    Carga bandas desde archivos GeoTIFF individuales.
    band_paths: dict con claves 'B02','B03','B04','B05','B07','B08','B8A','B11','B12'
    Retorna: dict con arrays normalizados (0-1).
    """
    bands = {}
    for key, path in band_paths.items():
        with rasterio.open(path) as src:
            arr = src.read(1)
            bands[key] = normalize_band(arr)
    return bands

def connect_to_openeo_backend(
        backend_url="https://openeo.dataspace.copernicus.eu",
        oidc_provider=None,
        auth_method=None,
        username=None,
        password=None):
    """
    Establece conexion con un backend openEO compatible con Sentinel-2.
    Por defecto usa OIDC interactivo. Para credenciales por entorno, defina:
    OPENEO_AUTH_METHOD=basic, OPENEO_USERNAME y OPENEO_PASSWORD.
    """
    import os
    import openeo

    connection = openeo.connect(backend_url)
    auth_method = (auth_method or os.getenv("OPENEO_AUTH_METHOD") or "oidc").lower()

    if auth_method == "basic":
        username = username or os.getenv("OPENEO_USERNAME") or os.getenv("COPERNICUS_USERNAME")
        password = password or os.getenv("OPENEO_PASSWORD") or os.getenv("COPERNICUS_PASSWORD")
        if not username or not password:
            raise ValueError(
                "Para auth_method='basic' defina OPENEO_USERNAME y OPENEO_PASSWORD."
            )
        return connection.authenticate_basic(username=username, password=password)

    if auth_method == "device":
        return connection.authenticate_oidc_device(provider_id=oidc_provider)

    if oidc_provider:
        return connection.authenticate_oidc(provider_id=oidc_provider)
    return connection.authenticate_oidc()

def load_sentinel2_cube(connection, lake_geojson, temporal_extent,
                        bands=None, max_cloud_cover=40,
                        collection=S2_L1C_COLLECTION):
    """
    Crea un cubo Sentinel-2 filtrado por lago, fechas, nubosidad y bandas.
    No ejecuta la descarga hasta llamar a download/execute_batch en el cubo.
    """
    if bands is None:
        bands = CYANO_REQUIRED_BANDS

    bbox = bbox_from_geojson(lake_geojson)
    return connection.load_collection(
        collection,
        spatial_extent=bbox,
        temporal_extent=temporal_extent,
        bands=bands,
        properties={"eo:cloud_cover": lambda v: v <= max_cloud_cover}
    )

def date_to_temporal_extent(date):
    """Convierte una fecha oficial YYYY-MM-DD en un intervalo de un dia."""
    from datetime import date as date_type, datetime, timedelta

    if isinstance(date, str):
        start = datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, date_type):
        start = date
    else:
        raise TypeError("date debe ser str YYYY-MM-DD o datetime.date")

    end = start + timedelta(days=1)
    return [start.isoformat(), end.isoformat()]

def load_official_lake_date_cube(connection, lake_name, date,
                                 bands=None, collection=S2_L1C_COLLECTION):
    """
    Crea un cubo Sentinel-2 para una fecha oficial del laboratorio y un lago oficial.
    """
    if lake_name not in LAKE_BBOXES:
        raise KeyError(f"Lago desconocido: {lake_name}")
    if date not in OFFICIAL_DATES[lake_name]:
        raise ValueError(f"{date} no esta en las fechas oficiales de {lake_name}")

    max_cloud_cover = OFFICIAL_CLOUD_COVER[lake_name][date] + 0.01
    return load_sentinel2_cube(
        connection=connection,
        lake_geojson=LAKE_BBOXES[lake_name],
        temporal_extent=date_to_temporal_extent(date),
        bands=bands,
        max_cloud_cover=max_cloud_cover,
        collection=collection,
    )

def download_lake_bands(connection, lake_geojson, temporal_extent, output_path,
                        bands=None, collection=S2_L1C_COLLECTION):
    """
    Descarga solo el recorte y las bandas requeridas para el lago/periodo indicado.
    El formato GeoTIFF sirve cuando se trabaja por fecha; NetCDF suele ser mejor
    para series temporales con varias fechas.
    """
    cube = load_sentinel2_cube(
        connection=connection,
        lake_geojson=lake_geojson,
        temporal_extent=temporal_extent,
        bands=bands,
        collection=collection
    )
    cube.download(output_path)
    return output_path

def compute_cyan_index_for_date(band_dict):
    """
    Procesa una fecha completa y devuelve: máscara de agua, FAI, NDCI, Clorofila.
    band_dict debe contener: B02, B03, B04, B05, B07, B08, B8A, B11, B12.
    """
    # Extraer bandas
    B02 = band_dict['B02']
    B03 = band_dict['B03']
    B04 = band_dict['B04']
    B05 = band_dict['B05']
    B07 = band_dict['B07']
    B08 = band_dict['B08']
    B8A = band_dict['B8A']
    B11 = band_dict['B11']
    B12 = band_dict['B12']

    # Calcular todo
    water = wbi_vectorized(B04, B03, B02, B08, B11, B12)
    faiv = FAI(B04, B07, B8A)
    ndci = NDCI(B04, B05)
    ndvi = NDVI(B04, B08)
    ndwi = NDWI(B03, B08)
    chl = chl_from_ndci(ndci)

    # Enmascarar fuera del agua (para análisis)
    chl_water = np.where(water, chl, np.nan)
    faiv_water = np.where(water, faiv, np.nan)

    return {
        'water_mask': water,
        'FAI': faiv_water,
        'NDCI': ndci,
        'NDVI': ndvi,
        'NDWI': ndwi,
        'chlorophyll': chl_water,
        'mean_cyano_index': np.nanmean(chl_water),
        'mean_chl': np.nanmean(chl_water),
        'max_chl': np.nanmax(chl_water),
        'percent_water': np.mean(water) * 100
    }

def summarize_lake_date(lake_name, date, band_dict):
    """Devuelve los indicadores promedio para un lago en una fecha."""
    result = compute_cyan_index_for_date(band_dict)
    return {
        "lake": lake_name,
        "date": date,
        "mean_cyano_index": result["mean_cyano_index"],
        "max_cyano_index": result["max_chl"],
        "mean_ndvi": np.nanmean(np.where(result["water_mask"], result["NDVI"], np.nan)),
        "mean_ndwi": np.nanmean(np.where(result["water_mask"], result["NDWI"], np.nan)),
        "percent_water": result["percent_water"],
    }

def temporal_summary(lake_date_bands):
    """
    Calcula una tabla temporal.
    lake_date_bands: iterable de tuplas (lake_name, date, band_dict).
    """
    return [
        summarize_lake_date(lake_name, date, band_dict)
        for lake_name, date, band_dict in lake_date_bands
    ]

def detect_bloom_peaks(rows, threshold=None):
    """
    Identifica fechas criticas por lago.
    Si no se da umbral, usa media + 1 desviacion estandar por lago.
    """
    peaks = []
    lake_names = sorted({row["lake"] for row in rows})

    for lake in lake_names:
        lake_rows = [row for row in rows if row["lake"] == lake]
        values = np.array([row["mean_cyano_index"] for row in lake_rows], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        lake_threshold = threshold if threshold is not None else values.mean() + values.std()
        for row in lake_rows:
            if row["mean_cyano_index"] >= lake_threshold:
                peak = dict(row)
                peak["threshold"] = lake_threshold
                peaks.append(peak)

    return peaks

def plot_temporal_evolution(rows, output_path="evolucion_temporal_cianobacteria.png"):
    """Grafica la evolucion temporal del indice promedio de cianobacteria por lago."""
    lake_names = sorted({row["lake"] for row in rows})
    fig, ax = plt.subplots(figsize=(10, 5))

    for lake in lake_names:
        lake_rows = sorted(
            [row for row in rows if row["lake"] == lake],
            key=lambda row: row["date"]
        )
        dates = [row["date"] for row in lake_rows]
        values = [row["mean_cyano_index"] for row in lake_rows]
        ax.plot(dates, values, marker="o", label=lake)

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Indice promedio de cianobacteria")
    ax.set_title("Evolucion temporal por lago")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    return output_path

# ============================================================
# 6. EJEMPLO PRÁCTICO (simulación con datos sintéticos)
# ============================================================

if __name__ == '__main__':
    print("=== EJEMPLO CON DATOS SINTÉTICOS ===\n")

    # Crear un escenario de prueba (100x100 píxeles)
    shape = (100, 100)
    rng = np.random.default_rng(42)  # semilla fija para reproducibilidad

    # Simular reflectancias (0-1) para todas las bandas
    bands = {
        'B02': rng.random(shape),  # azul
        'B03': rng.random(shape),  # verde
        'B04': rng.random(shape),  # rojo
        'B05': rng.random(shape),  # borde rojo 1 (705nm)
        'B07': rng.random(shape),  # borde rojo 3 (783nm)
        'B08': rng.random(shape),  # NIR
        'B8A': rng.random(shape),  # NIR estrecho (865nm)
        'B11': rng.random(shape),  # SWIR1
        'B12': rng.random(shape)   # SWIR2
    }

    # Simular una zona de agua en el centro
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    water_circle = (x**2 + y**2) < 0.6  # círculo central
    # Forzar que B04 sea baja y B08 alta en el agua (para que la máscara funcione)
    bands['B04'] = np.where(water_circle, 0.05, bands['B04'])
    bands['B08'] = np.where(water_circle, 0.3, bands['B08'])
    bands['B03'] = np.where(water_circle, 0.08, bands['B03'])
    bands['B11'] = np.where(water_circle, 0.02, bands['B11'])

    # Procesar
    results = compute_cyan_index_for_date(bands)
    chl = results['chlorophyll']
    water = results['water_mask']

    print(f"Porcentaje de agua en la escena: {results['percent_water']:.2f}%")
    print(f"Clorofila promedio (solo agua): {results['mean_chl']:.2f} μg/L")
    print(f"Clorofila máxima: {results['max_chl']:.2f} μg/L")

    # Visualizar resultados
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(water, cmap='Blues', interpolation='nearest')
    axes[0].set_title("Máscara de agua")
    axes[0].axis('off')

    im = axes[1].imshow(chl, cmap='viridis', interpolation='nearest')
    axes[1].set_title("Clorofila-a (μg/L)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Generar RGB con paleta completa
    rgb_img = generate_rgb_palette(water, results['FAI'], chl)
    axes[2].imshow(rgb_img, interpolation='nearest')
    axes[2].set_title("Paleta Copernicus (RGB)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('resultado_ejemplo_cyanolakes.png', dpi=150)
    print("\nImagen guardada como 'resultado_ejemplo_cyanolakes.png'")
    plt.show()
