"""
Preparacion y auditoria del dataset de Machine Learning (Laboratorio 4, Parte 2).

Cubre exclusivamente los ejercicios 1, 2 y 3 del PDF de la Parte 2:
    1. Construccion del dataset por pixel a partir de los 22 GeoTIFF reales.
    2. Construccion y justificacion de la variable respuesta binaria.
    3. Seleccion de variables predictoras sin fuga de informacion.

NO entrena modelos, NO ejecuta SHAP, NO genera mapas predictivos.
NO utiliza datos sinteticos bajo ninguna circunstancia.

Modos:
    python preparar_dataset_ml.py --dry-run
    python preparar_dataset_ml.py --build
    python preparar_dataset_ml.py --validate
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mainlab4 import (  # noqa: E402
    OFFICIAL_DATES,
    OFFICIAL_CLOUD_COVER,
    CYANO_REQUIRED_BANDS,
    S2_L1C_COLLECTION,
    wbi_vectorized,
    NDVI as calcular_ndvi,
    NDWI as calcular_ndwi,
    NDCI as calcular_ndci,
    FAI as calcular_fai,
    chl_from_ndci,
)

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

RASTER_DIR = ROOT / "outputs" / "rasters"
MANIFEST_RASTER = ROOT / "outputs" / "manifests" / "raster_manifest.csv"

BASE = ROOT / "outputs" / "parte2"
DATA_DIR = BASE / "data"
PIXELS_DIR = DATA_DIR / "pixels"
EDA_DIR = BASE / "eda"
TARGET_DIR = BASE / "target"
REPORTS_DIR = BASE / "reports"
LOG_DIR = BASE / "logs"

CRS_ESPERADO = "EPSG:32615"
RESOLUCION_ESPERADA = 20.0
NODATA = -9999.0

# Escala oficial declarada por la coleccion SENTINEL2_L1C en Copernicus Data
# Space: raster:bands -> {"scale": 0.0001, "offset": 0}. Los GeoTIFF descargados
# conservan los numeros digitales (DN), NO reflectancia. Se verifica en tiempo de
# ejecucion antes de aplicarla, para no dividir dos veces.
ESCALA_REFLECTANCIA = 0.0001
# Si la mediana de una banda supera este valor, los datos estan en DN.
UMBRAL_DETECCION_DN = 2.0

# --- Variable respuesta -------------------------------------------------------
# Umbrales candidatos de clorofila-a (ug/L). Ver justificacion en el reporte.
UMBRALES_CANDIDATOS = [8.0, 20.0, 25.0, 50.0]

# Respuesta principal. 8 ug/L es la frontera mesotrofico -> eutrofico de la
# clasificacion trofica de OECD (1982): operacionaliza "alta presencia" como el
# punto en que el cuerpo de agua entra en condicion eutrofica, es decir cuando la
# biomasa algal deja de ser propia de un lago equilibrado.
#
# Los otros tres se conservan como analisis de sensibilidad, no como alternativas
# equivalentes, porque describen situaciones ambientales distintas:
#   - 20 ug/L: continuidad con el umbral usado en la Parte 1.
#   - 25 ug/L: frontera eutrofico -> hipertrofico (OECD 1982); condicion severa.
#   - 50 ug/L: escenario extremo, sin respaldo directo en OECD ni en la OMS.
#
# La mayor cantidad de observaciones positivas que produce 8 ug/L es una
# CONSECUENCIA de haber elegido la transicion eutrofica, no el motivo de la
# eleccion. El criterio es ambiental; la viabilidad estadistica se documenta
# aparte, en el analisis de viabilidad.
#
# Nota de prudencia: la respuesta representa clorofila-a ALTA ESTIMADA POR
# SATELITE. No es una confirmacion in situ de presencia de cianobacterias ni de
# toxicidad; la clorofila mide biomasa fotosintetica, no taxonomia ni toxinas.
TARGET_COLUMN = "high_cyano_8"
TARGET_THRESHOLD_UG_L = 8.0
UMBRAL_RECOMENDADO = TARGET_THRESHOLD_UG_L

SIGNIFICADO_UMBRAL = {
    8.0: "Transicion aproximada hacia condicion eutrofica (frontera mesotrofico -> "
         "eutrofico, OECD 1982). Respuesta principal.",
    20.0: "Continuidad con el umbral empleado en la Parte 1. Dentro de la banda "
          "eutrofica de OECD (8-25) y del rango de Alerta 1 de la OMS (12-24).",
    25.0: "Transicion aproximada hacia condicion hipertrofica (OECD 1982); coincide "
          "con el techo de la Alerta 1 de la OMS (24 ug/L).",
    50.0: "Escenario extremo para analisis de sensibilidad. No aparece como valor de "
          "clorofila-a en OECD 1982 ni en las guias de la OMS 2021.",
}

# --- Version del esquema ------------------------------------------------------
# Cambia cuando cambian las columnas o la definicion de la respuesta. Las
# particiones escritas con una version anterior se consideran obsoletas y se
# reconstruyen, para no mezclar esquemas dentro del mismo dataset.
DATASET_VERSION = "2.0"

# Tamano nominal del bloque espacial para validacion agrupada (metros, EPSG:32615).
BLOQUE_ESPACIAL_M = 1000.0

# Dominio de calibracion del modelo NDCI->clorofila (Mishra & Mishra 2012:
# 1-60 mg/m3 en datos simulados). Fuera de este rango la estimacion se reporta
# pero se marca como extrapolacion.
CALIB_MIN = 1.0
CALIB_MAX = 60.0

MUESTRA_EDA = 200_000

# --- Conjuntos de variables (ejercicio 3) ------------------------------------
# Cadena de construccion de la respuesta:
#   B04, B05 -> NDCI -> chlorophyll -> high_cyano_*
# Todo lo que participe en esa cadena queda excluido del conjunto predictor.
PREDICTORES_PRINCIPALES = ["B02", "B03", "B07", "B08", "B8A", "B11", "B12", "NDWI"]

EXCLUIDAS_POR_FUGA = {
    "B04": "insumo directo de NDCI, que genera la clorofila y la respuesta",
    "B05": "insumo directo de NDCI, que genera la clorofila y la respuesta",
    "NDCI": "indice del que se deriva la clorofila y por tanto la respuesta",
    "chlorophyll": "variable de la que se deriva directamente la respuesta",
    "FAI": "utiliza B04; comparte insumo con NDCI (fuga indirecta)",
    "NDVI": "utiliza B04; se conserva porque el enunciado lo exige, pero solo "
            "es admisible en un analisis de sensibilidad etiquetado como fuga indirecta",
    "high_cyano_8": "es la propia respuesta principal",
    "high_cyano_20": "es una respuesta candidata (sensibilidad)",
    "high_cyano_25": "es una respuesta candidata (sensibilidad)",
    "high_cyano_50": "es una respuesta candidata (sensibilidad)",
    "water_mask": "es un filtro de construccion del dataset, no un predictor",
    "valid_data": "es un filtro de construccion del dataset, no un predictor",
    "fuera_calibracion": "es un diagnostico derivado de la clorofila",
    "lake": "identifica el lago: usarlo permitiria memorizar diferencias entre "
            "cuerpos de agua en vez de aprender la senal espectral",
    "date": "memorizacion temporal: el modelo aprenderia la fecha, no el fenomeno",
    "year": "memorizacion temporal",
    "month": "memorizacion temporal",
    "season": "memorizacion temporal",
    "row": "coordenada de rejilla: memorizacion espacial",
    "col": "coordenada de rejilla: memorizacion espacial",
    "x_utm": "coordenada: memorizacion espacial",
    "y_utm": "coordenada: memorizacion espacial",
    "longitude": "coordenada: memorizacion espacial",
    "latitude": "coordenada: memorizacion espacial",
    "spatial_block_1km": "identificador de bloque: es una variable de agrupacion "
                         "para la validacion, no un predictor",
}

COLUMNAS_TRAZABILIDAD = {
    "lake": "agrupacion, validacion entre lagos y estratificacion",
    "date": "validacion temporal y estratificacion",
    "row": "reconstruccion espacial del raster",
    "col": "reconstruccion espacial del raster",
    "x_utm": "bloques espaciales de 1x1 km en EPSG:32615",
    "y_utm": "bloques espaciales de 1x1 km en EPSG:32615",
    "longitude": "mapas y trazabilidad geografica",
    "latitude": "mapas y trazabilidad geografica",
    "year": "validacion temporal",
    "month": "estacionalidad y validacion temporal",
    "season": "estacionalidad (seca / lluviosa)",
    "spatial_block_1km": "agrupacion para GroupKFold espacial (bloques de 1 km)",
}

# Variables auxiliares creadas para la validacion. NO entran al conjunto
# predictor principal: solo sirven para agrupar y estratificar.
COLUMNAS_AUXILIARES = ["year", "month", "season", "spatial_block_1km"]

BANDAS = list(CYANO_REQUIRED_BANDS)
INDICES = ["NDVI", "NDWI", "NDCI", "FAI", "chlorophyll"]


def _verificar_predictores_sin_fuga():
    """
    Impide que una columna prohibida entre al conjunto predictor.

    Se ejecuta al importar el modulo para que el error salte antes de construir
    nada, no despues de horas de calculo.
    """
    intrusas = [c for c in PREDICTORES_PRINCIPALES if c in EXCLUIDAS_POR_FUGA]
    assert not intrusas, (
        f"FUGA DE INFORMACION: {intrusas} estan en PREDICTORES_PRINCIPALES pero "
        "figuran como excluidas.")
    respuestas = [f"high_cyano_{int(u)}" for u in UMBRALES_CANDIDATOS]
    solapan = [c for c in PREDICTORES_PRINCIPALES if c in respuestas]
    assert not solapan, f"FUGA: {solapan} son columnas de respuesta."
    aux = [c for c in PREDICTORES_PRINCIPALES if c in COLUMNAS_AUXILIARES]
    assert not aux, f"Las auxiliares {aux} no deben ser predictores."
    assert TARGET_COLUMN in respuestas, (
        f"TARGET_COLUMN={TARGET_COLUMN} no coincide con ningun umbral candidato.")


_verificar_predictores_sin_fuga()

LOGGER = logging.getLogger("preparar_dataset_ml")


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def configurar_logging(verbose: bool = False) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = LOG_DIR / f"dataset_{marca}.log"

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()

    consola = logging.StreamHandler(sys.stdout)
    consola.setLevel(logging.DEBUG if verbose else logging.INFO)
    consola.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(consola)

    archivo = logging.FileHandler(ruta, encoding="utf-8")
    archivo.setLevel(logging.DEBUG)
    archivo.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(archivo)
    return ruta


def combinaciones_oficiales(lago=None, fecha=None):
    pares = [(l, f) for l in sorted(OFFICIAL_DATES) for f in OFFICIAL_DATES[l]]
    if lago:
        coincide = [l for l in OFFICIAL_DATES if l.lower() == lago.lower()]
        if not coincide:
            raise SystemExit(f"ERROR: lago desconocido '{lago}'.")
        pares = [p for p in pares if p[0] == coincide[0]]
    if fecha:
        pares = [p for p in pares if p[1] == fecha]
        if not pares:
            raise SystemExit(f"ERROR: fecha '{fecha}' no oficial para la seleccion.")
    return pares


def ruta_raster(lago: str, fecha: str) -> Path:
    return RASTER_DIR / lago / f"{lago}_{fecha}.tif"


def ruta_particion(lago: str, fecha: str) -> Path:
    return PIXELS_DIR / f"lake={lago}" / f"date={fecha}" / "part-0.parquet"


def columnas_esperadas() -> list:
    """Esquema canonico de una particion (sin lake/date, que van en la ruta Hive)."""
    respuestas = [f"high_cyano_{int(u)}" for u in UMBRALES_CANDIDATOS]
    return (["row", "col", "x_utm", "y_utm", "longitude", "latitude"]
            + BANDAS + INDICES + ["water_mask", "valid_data"]
            + respuestas + ["fuera_calibracion"] + COLUMNAS_AUXILIARES)


def hash_esquema() -> str:
    """
    Huella de la configuracion que determina el contenido de una particion.

    Cambia si cambian las columnas, los umbrales, la respuesta principal o el
    tamano del bloque espacial. Se guarda en el manifiesto para poder detectar
    particiones obsoletas sin abrirlas todas.
    """
    import hashlib
    firma = json.dumps({
        "version": DATASET_VERSION,
        "columnas": columnas_esperadas(),
        "umbrales": UMBRALES_CANDIDATOS,
        "target": TARGET_COLUMN,
        "bloque_m": BLOQUE_ESPACIAL_M,
        "calib": [CALIB_MIN, CALIB_MAX],
    }, sort_keys=True)
    return hashlib.sha256(firma.encode("utf-8")).hexdigest()[:16]


def particion_vigente(lago: str, fecha: str) -> tuple[bool, str]:
    """
    Determina si una particion existente corresponde al esquema actual.

    Una particion escrita con un esquema anterior NO se considera valida: se
    reconstruye. Asi se evita mezclar particiones de versiones distintas dentro
    del mismo dataset.
    """
    ruta = ruta_particion(lago, fecha)
    if not ruta.exists():
        return False, "no existe"
    try:
        import pyarrow.parquet as pq
        esquema = pq.read_schema(ruta)
    except Exception as exc:
        return False, f"ilegible ({type(exc).__name__})"

    presentes = set(esquema.names)
    faltan = [c for c in columnas_esperadas() if c not in presentes]
    if faltan:
        return False, f"esquema anterior; faltan {faltan[:4]}" + (
            f" y {len(faltan) - 4} mas" if len(faltan) > 4 else "")

    metadatos = esquema.metadata or {}
    version = metadatos.get(b"dataset_version", b"").decode("utf-8", "ignore")
    firma = metadatos.get(b"schema_hash", b"").decode("utf-8", "ignore")
    if version != DATASET_VERSION or firma != hash_esquema():
        return False, (f"version {version or 'desconocida'} != {DATASET_VERSION} "
                       "o configuracion distinta")

    if pq.read_metadata(ruta).num_rows == 0:
        return False, "particion vacia"
    return True, "vigente"


def escribir_particion(tabla: pd.DataFrame, lago: str, fecha: str) -> Path:
    """
    Escribe una particion de forma atomica: primero a un archivo temporal, se
    valida, y solo entonces se reemplaza la particion definitiva. Si algo falla,
    la particion anterior permanece intacta.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    destino = ruta_particion(lago, fecha)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".tmp.parquet")
    if temporal.exists():
        temporal.unlink()

    # lake y date no se guardan dentro del archivo: viajan en la ruta Hive.
    cuerpo = tabla.drop(columns=["lake", "date"])
    faltan = [c for c in columnas_esperadas() if c not in cuerpo.columns]
    if faltan:
        raise ValueError(f"{lago} {fecha}: faltan columnas {faltan} antes de escribir")

    tabla_arrow = pa.Table.from_pandas(cuerpo, preserve_index=False)
    tabla_arrow = tabla_arrow.replace_schema_metadata({
        "dataset_version": DATASET_VERSION,
        "schema_hash": hash_esquema(),
        "target_column": TARGET_COLUMN,
        "target_threshold_ug_L": str(TARGET_THRESHOLD_UG_L),
        "lake": lago, "date": fecha,
        "generado": datetime.now().isoformat(timespec="seconds"),
    })
    pq.write_table(tabla_arrow, temporal, compression="snappy")

    # Se valida el temporal ANTES de sustituir la particion buena.
    verificacion = pq.read_metadata(temporal)
    if verificacion.num_rows != len(cuerpo):
        temporal.unlink(missing_ok=True)
        raise ValueError(f"{lago} {fecha}: el temporal tiene "
                         f"{verificacion.num_rows} filas, se esperaban {len(cuerpo)}")

    temporal.replace(destino)
    return destino


def leer_particion(lago: str, fecha: str) -> pd.DataFrame:
    """
    Lee una particion y reconstruye lake/date.

    Esas dos columnas no se guardan dentro del archivo: ya viajan en la ruta
    Hive (lake=.../date=...). Escribirlas ademas como columna provoca un choque
    de tipos al leer el dataset completo, porque pyarrow infiere la clave de
    particion como diccionario con indices distintos a los del archivo.
    """
    df = pd.read_parquet(ruta_particion(lago, fecha), engine="pyarrow")
    df.insert(0, "lake", lago)
    df.insert(1, "date", fecha)
    return df


def barra(iterable, total, descripcion):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=descripcion, unit="raster", ncols=88)
    except ImportError:
        return iterable


# ----------------------------------------------------------------------------
# 1. INSPECCION DE LOS RASTERES
# ----------------------------------------------------------------------------
def inspeccionar_raster(lago: str, fecha: str) -> dict:
    """
    Verifica estructura y georreferenciacion de un GeoTIFF. No asume el orden de
    las bandas: lo lee de las descripciones y lo contrasta con el tag 'bandas'.
    """
    import rasterio

    ruta = ruta_raster(lago, fecha)
    if not ruta.exists():
        raise FileNotFoundError(f"Falta el raster de {lago} {fecha}: {ruta}")

    with rasterio.open(ruta) as src:
        descripciones = [d for d in src.descriptions]
        tags = src.tags()
        tag_bandas = [b for b in (tags.get("bandas") or "").split(",") if b]

        if any(d is None for d in descripciones):
            raise ValueError(f"{ruta.name}: hay bandas sin descripcion; no se puede "
                             "determinar el orden de forma fiable.")

        faltantes = [b for b in BANDAS if b not in descripciones]
        if faltantes:
            raise ValueError(f"{ruta.name}: faltan las bandas {faltantes}. "
                             f"Presentes: {descripciones}")

        if tag_bandas and tag_bandas != descripciones:
            raise ValueError(f"{ruta.name}: el tag 'bandas' {tag_bandas} no coincide "
                             f"con las descripciones {descripciones}.")

        crs = str(src.crs) if src.crs else None
        if crs != CRS_ESPERADO:
            raise ValueError(f"{ruta.name}: CRS {crs}, se esperaba {CRS_ESPERADO}.")

        if not (np.isclose(src.res[0], RESOLUCION_ESPERADA)
                and np.isclose(src.res[1], RESOLUCION_ESPERADA)):
            raise ValueError(f"{ruta.name}: resolucion {src.res}, se esperaba "
                             f"{RESOLUCION_ESPERADA} m en ambos ejes.")

        if len(set(src.dtypes)) != 1:
            raise ValueError(f"{ruta.name}: dtypes heterogeneos {src.dtypes}.")

        return {
            "lago": lago,
            "fecha": fecha,
            "ruta": ruta,
            "descripciones": descripciones,
            "indice_banda": {n: i + 1 for i, n in enumerate(descripciones)},
            "ancho": src.width,
            "alto": src.height,
            "pixeles": src.width * src.height,
            "crs": crs,
            "transform": src.transform,
            "res": src.res,
            "bounds": tuple(src.bounds),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "tags": tags,
        }


def detectar_escala(muestra: np.ndarray) -> tuple[float, str]:
    """
    Decide si los valores estan en DN (requieren x0.0001) o ya en reflectancia.

    Regla: la reflectancia TOA de agua y tierra vive en [0, ~1.5]. Una mediana
    por encima de UMBRAL_DETECCION_DN solo es compatible con numeros digitales.
    """
    finitos = muestra[np.isfinite(muestra)]
    if finitos.size == 0:
        raise ValueError("No hay valores finitos para determinar la escala.")
    mediana = float(np.median(np.abs(finitos)))
    if mediana > UMBRAL_DETECCION_DN:
        return ESCALA_REFLECTANCIA, (
            f"numeros digitales (mediana |v|={mediana:.1f}); se aplica escala "
            f"{ESCALA_REFLECTANCIA} declarada por la coleccion (offset 0)"
        )
    return 1.0, (
        f"ya en reflectancia (mediana |v|={mediana:.4f}); NO se vuelve a dividir"
    )


# ----------------------------------------------------------------------------
# 2-3. CONSTRUCCION POR PIXEL
# ----------------------------------------------------------------------------
def construir_tabla_pixeles(info: dict) -> tuple[pd.DataFrame, dict]:
    """
    Convierte un GeoTIFF en una tabla de una fila por pixel de agua valido.

    Devuelve (dataframe, diagnostico_de_limpieza). Procesa un solo raster para
    mantener acotado el uso de memoria.
    """
    import rasterio
    from rasterio.transform import xy
    from pyproj import Transformer

    with rasterio.open(info["ruta"]) as src:
        datos = src.read().astype(np.float64)
        mascaras = src.read_masks()
        transform = src.transform

    idx = info["indice_banda"]
    bruto = {n: datos[idx[n] - 1] for n in BANDAS}

    # --- Escala a reflectancia (verificada, no asumida) ---
    escala, motivo_escala = detectar_escala(bruto["B08"])
    banda = {n: bruto[n] * escala for n in BANDAS}

    total_pixeles = int(datos.shape[1] * datos.shape[2])
    diagnostico = {"total_pixeles": total_pixeles, "motivo_escala": motivo_escala,
                   "escala_aplicada": escala}

    # --- Filtros de validez, acumulativos y contabilizados ---
    con_datos = (mascaras > 0).all(axis=0)
    for n in BANDAS:
        con_datos &= (bruto[n] != NODATA)
    diagnostico["descartados_nodata"] = int(total_pixeles - con_datos.sum())

    finitos = np.ones_like(con_datos)
    for n in BANDAS:
        finitos &= np.isfinite(banda[n])
    diagnostico["descartados_no_finitos"] = int((con_datos & ~finitos).sum())
    valido = con_datos & finitos

    # Regla fisica documentada: la reflectancia TOA es una fraccion de la
    # radiacion incidente y no puede ser <= 0. Valores negativos aparecen en
    # aguas muy oscuras por efectos atmosfericos y de remuestreo; se descartan
    # porque invalidan los cocientes normalizados (NDCI, NDWI, NDVI).
    # NO se impone ningun limite superior arbitrario: los valores > 1 se
    # conservan y solo se reportan como diagnostico.
    positivos = np.ones_like(valido)
    for n in BANDAS:
        positivos &= (banda[n] > 0)
    diagnostico["descartados_reflectancia_no_positiva"] = int((valido & ~positivos).sum())
    valido_fisico = valido & positivos

    saturados = np.zeros_like(valido)
    for n in BANDAS:
        saturados |= (banda[n] > 1.0)
    diagnostico["diagnostico_reflectancia_mayor_1"] = int((valido_fisico & saturados).sum())

    # --- Mascara de agua WBI (unica separacion agua/tierra disponible) ---
    agua = wbi_vectorized(banda["B04"], banda["B03"], banda["B02"],
                          banda["B08"], banda["B11"], banda["B12"])
    diagnostico["descartados_no_agua"] = int((valido_fisico & ~agua).sum())
    seleccion = valido_fisico & agua
    diagnostico["pixeles_agua"] = int(seleccion.sum())

    if seleccion.sum() == 0:
        raise ValueError(f"{info['ruta'].name}: la mascara WBI no dejo ningun pixel "
                         "de agua; revise el raster antes de continuar.")

    # --- Indices espectrales ---
    ndvi = calcular_ndvi(banda["B04"], banda["B08"])
    ndwi = calcular_ndwi(banda["B03"], banda["B08"])
    ndci = calcular_ndci(banda["B04"], banda["B05"])
    fai = calcular_fai(banda["B04"], banda["B07"], banda["B8A"])
    chl = chl_from_ndci(ndci)

    indices_finitos = (np.isfinite(ndvi) & np.isfinite(ndwi)
                       & np.isfinite(ndci) & np.isfinite(fai) & np.isfinite(chl))
    diagnostico["descartados_indice_no_finito"] = int((seleccion & ~indices_finitos).sum())
    seleccion &= indices_finitos

    filas, columnas = np.nonzero(seleccion)

    # --- Coordenadas desde la transformacion afin real ---
    xs, ys = xy(transform, filas, columnas, offset="center")
    x_utm = np.asarray(xs, dtype=np.float64)
    y_utm = np.asarray(ys, dtype=np.float64)

    transformador = Transformer.from_crs(CRS_ESPERADO, "EPSG:4326", always_xy=True)
    lon, lat = transformador.transform(x_utm, y_utm)

    tabla = pd.DataFrame({
        "lake": pd.Categorical([info["lago"]] * filas.size),
        "date": pd.Categorical([info["fecha"]] * filas.size),
        "row": filas.astype(np.int32),
        "col": columnas.astype(np.int32),
        "x_utm": x_utm.astype(np.float64),
        "y_utm": y_utm.astype(np.float64),
        "longitude": np.asarray(lon, dtype=np.float64),
        "latitude": np.asarray(lat, dtype=np.float64),
    })
    for n in BANDAS:
        tabla[n] = banda[n][seleccion].astype(np.float32)
    tabla["NDVI"] = ndvi[seleccion].astype(np.float32)
    tabla["NDWI"] = ndwi[seleccion].astype(np.float32)
    tabla["NDCI"] = ndci[seleccion].astype(np.float32)
    tabla["FAI"] = fai[seleccion].astype(np.float32)
    tabla["chlorophyll"] = chl[seleccion].astype(np.float32)

    # Se conservan explicitamente aunque sean constantes tras el filtrado: el
    # enunciado exige ambas columnas en el dataset.
    tabla["water_mask"] = True
    tabla["valid_data"] = True

    # --- Respuestas candidatas (diagnostico) ---
    for umbral in UMBRALES_CANDIDATOS:
        tabla[f"high_cyano_{int(umbral)}"] = (
            tabla["chlorophyll"] >= umbral
        ).astype(np.int8)

    # --- Dominio de calibracion ---
    # Los valores negativos NO se recortan ni se llevan a cero: son resultado de
    # evaluar el polinomio NDCI->clorofila fuera de su dominio y se conservan tal
    # cual para trazabilidad. Se marcan aqui para poder excluirlos en el analisis
    # sin alterar el dato original.
    tabla["fuera_calibracion"] = (
        (tabla["chlorophyll"] < CALIB_MIN) | (tabla["chlorophyll"] > CALIB_MAX)
    )
    diagnostico["fuera_calibracion"] = int(tabla["fuera_calibracion"].sum())
    diagnostico["chl_negativa"] = int((tabla["chlorophyll"] < 0).sum())

    # --- Variables auxiliares para validacion (no son predictores) ---
    fecha_dt = pd.Timestamp(info["fecha"])
    tabla["year"] = np.int16(fecha_dt.year)
    tabla["month"] = np.int8(fecha_dt.month)
    # En Guatemala la estacion seca va de noviembre a abril y la lluviosa de mayo
    # a octubre. Se codifica como texto para que sea legible en las tablas.
    tabla["season"] = "seca" if fecha_dt.month in (11, 12, 1, 2, 3, 4) else "lluviosa"

    # Bloques de 1x1 km derivados de las coordenadas UTM (EPSG:32615, en metros).
    # El identificador incluye el lago para que dos lagos no compartan bloque.
    bloque_col = np.floor(tabla["x_utm"].to_numpy() / BLOQUE_ESPACIAL_M).astype(np.int32)
    bloque_fila = np.floor(tabla["y_utm"].to_numpy() / BLOQUE_ESPACIAL_M).astype(np.int32)
    tabla["spatial_block_1km"] = [
        f"{info['lago']}_{c}_{f}" for c, f in zip(bloque_col, bloque_fila)
    ]

    antes = len(tabla)
    tabla = tabla.drop_duplicates(subset=["lake", "date", "row", "col"])
    diagnostico["descartados_duplicados"] = int(antes - len(tabla))
    diagnostico["filas_finales"] = int(len(tabla))

    return tabla.reset_index(drop=True), diagnostico


# ----------------------------------------------------------------------------
# MODO --dry-run
# ----------------------------------------------------------------------------
def ejecutar_dry_run(args) -> int:
    LOGGER.info("=" * 84)
    LOGGER.info("MODO DRY-RUN: inspeccion de rasteres y estimacion (no escribe el dataset)")
    LOGGER.info("=" * 84)

    if not MANIFEST_RASTER.exists():
        LOGGER.error("No existe %s. Ejecute antes descargar_rasters.py --validate",
                     MANIFEST_RASTER)
        return 2

    manifiesto = {(f["lago"], f["fecha"]): f
                  for f in csv.DictReader(open(MANIFEST_RASTER, encoding="utf-8"))}
    LOGGER.info("Manifiesto de rasteres: %d registros", len(manifiesto))

    pares = combinaciones_oficiales(args.lake, args.date)
    LOGGER.info("Combinaciones oficiales a inspeccionar: %d", len(pares))
    LOGGER.info("")

    infos, errores = [], []
    for lago, fecha in barra(pares, len(pares), "Inspeccionando"):
        try:
            info = inspeccionar_raster(lago, fecha)
            if (lago, fecha) not in manifiesto:
                raise ValueError("no aparece en raster_manifest.csv")
            infos.append(info)
        except Exception as exc:
            errores.append((lago, fecha, f"{type(exc).__name__}: {exc}"))

    if errores:
        LOGGER.error("")
        LOGGER.error("Se detuvieron %d combinaciones por error:", len(errores))
        for lago, fecha, det in errores:
            LOGGER.error("  - %s %s -> %s", lago, fecha, det)
        return 2

    # Escala y estimacion de agua sobre una muestra real (un raster por lago)
    LOGGER.info("")
    LOGGER.info("Verificacion de escala radiometrica y estimacion de agua:")
    fraccion_agua, motivo_por_lago = {}, {}
    import rasterio
    for lago in sorted({i["lago"] for i in infos}):
        info = next(i for i in infos if i["lago"] == lago)
        with rasterio.open(info["ruta"]) as src:
            idx = info["indice_banda"]
            datos = src.read().astype(np.float64)
            masc = (src.read_masks() > 0).all(axis=0)
        escala, motivo = detectar_escala(datos[idx["B08"] - 1])
        b = {n: datos[idx[n] - 1] * escala for n in BANDAS}
        agua = wbi_vectorized(b["B04"], b["B03"], b["B02"], b["B08"], b["B11"], b["B12"])
        pos = np.ones_like(masc)
        for n in BANDAS:
            pos &= (b[n] > 0)
        sel = agua & masc & pos
        fraccion_agua[lago] = float(sel.sum()) / sel.size
        motivo_por_lago[lago] = motivo
        LOGGER.info("  %-11s %s", lago, motivo)
        LOGGER.info("  %-11s agua valida estimada: %.2f%% (%s km2 a 20 m)",
                    "", 100 * fraccion_agua[lago],
                    f"{sel.sum() * 400 / 1e6:.1f}")

    filas_estimadas = sum(
        int(i["pixeles"] * fraccion_agua[i["lago"]]) for i in infos
    )
    n_columnas = (len(COLUMNAS_TRAZABILIDAD) + len(BANDAS) + len(INDICES)
                  + 2 + len(UMBRALES_CANDIDATOS) + 1)
    bytes_fila = (len(BANDAS) + len(INDICES)) * 4 + 4 * 8 + 2 * 4 + 8
    mem_mb = filas_estimadas * bytes_fila / 1e6

    LOGGER.info("")
    LOGGER.info("ESTIMACION DEL DATASET")
    LOGGER.info("  Rasteres a procesar        : %d", len(infos))
    LOGGER.info("  Pixeles totales (bruto)    : %s",
                f"{sum(i['pixeles'] for i in infos):,}")
    LOGGER.info("  Filas de agua estimadas    : %s", f"{filas_estimadas:,}")
    LOGGER.info("  Columnas del esquema       : %d", n_columnas)
    LOGGER.info("  Memoria en RAM (estimada)  : %.0f MB si se cargara completo", mem_mb)
    LOGGER.info("  Pico real por raster       : ~%.0f MB (se procesa de uno en uno)",
                max(i["pixeles"] for i in infos) * 9 * 8 / 1e6)
    LOGGER.info("  Parquet en disco (estimado): %.0f-%.0f MB (snappy, 3-5x)",
                mem_mb / 5, mem_mb / 3)
    LOGGER.info("  Muestra EDA                : %s filas (semilla %d)",
                f"{MUESTRA_EDA:,}", SEED)

    LOGGER.info("")
    LOGGER.info("Coherencia geoespacial por lago:")
    for lago in sorted({i["lago"] for i in infos}):
        grupo = [i for i in infos if i["lago"] == lago]
        dims = {(i["ancho"], i["alto"]) for i in grupo}
        crs = {i["crs"] for i in grupo}
        res = {i["res"] for i in grupo}
        tr = {tuple(np.round(list(i["transform"])[:6], 6)) for i in grupo}
        LOGGER.info("  %-11s n=%d dims_unicas=%s crs_unico=%s res_unica=%s transform_unica=%s",
                    lago, len(grupo), len(dims) == 1, len(crs) == 1,
                    len(res) == 1, len(tr) == 1)
        if len(tr) != 1:
            LOGGER.warning("    Transformaciones distintas dentro del lago: %d", len(tr))

    faltan = set(combinaciones_oficiales()) - {(i["lago"], i["fecha"]) for i in infos}
    LOGGER.info("")
    if not args.lake and not args.date:
        LOGGER.info("Cobertura: %d/22 combinaciones oficiales", len(infos))
        if faltan:
            LOGGER.error("FALTANTES: %s", sorted(faltan))
            return 1

    LOGGER.info("")
    LOGGER.info("DRY-RUN correcto. No se escribio ningun dataset.")
    LOGGER.info("Siguiente paso: python preparar_dataset_ml.py --build")
    return 0


# ----------------------------------------------------------------------------
# MODO --build
# ----------------------------------------------------------------------------
def ejecutar_build(args) -> int:
    LOGGER.info("=" * 84)
    LOGGER.info("MODO BUILD: construccion del dataset por pixel (datos reales)")
    LOGGER.info("=" * 84)

    for carpeta in (DATA_DIR, PIXELS_DIR, EDA_DIR, TARGET_DIR, REPORTS_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)

    pares = combinaciones_oficiales(args.lake, args.date)
    LOGGER.info("Combinaciones a procesar: %d", len(pares))
    LOGGER.info("Semilla fija: %d", SEED)
    LOGGER.info("")

    registros, diagnosticos, errores = [], [], []
    inicio_global = time.time()

    LOGGER.info("Version del dataset: %s (hash de esquema %s)",
                DATASET_VERSION, hash_esquema())
    LOGGER.info("Respuesta principal: %s (>= %.1f ug/L)",
                TARGET_COLUMN, TARGET_THRESHOLD_UG_L)
    LOGGER.info("")

    for lago, fecha in barra(pares, len(pares), "Construyendo"):
        etiqueta = f"{lago} {fecha}"

        # Reanudable: solo se reutiliza una particion del esquema ACTUAL. Una
        # escrita con un esquema anterior se reconstruye, para no mezclar
        # versiones dentro del mismo dataset.
        if not args.force:
            vigente, motivo = particion_vigente(lago, fecha)
            if vigente:
                import pyarrow.parquet as pq
                n = pq.read_metadata(ruta_particion(lago, fecha)).num_rows
                LOGGER.info("[OMITIR] %s vigente (%s filas).", etiqueta, f"{n:,}")
                registros.append({"lake": lago, "date": fecha, "filas": n,
                                  "estado": "reutilizado",
                                  "dataset_version": DATASET_VERSION,
                                  "schema_hash": hash_esquema(),
                                  "archivo": str(ruta_particion(lago, fecha).relative_to(ROOT))})
                continue
            if ruta_particion(lago, fecha).exists():
                LOGGER.info("[MIGRAR] %s: %s", etiqueta, motivo)

        try:
            info = inspeccionar_raster(lago, fecha)
            t0 = time.time()
            tabla, diag = construir_tabla_pixeles(info)
            destino = escribir_particion(tabla, lago, fecha)

            vigente, motivo = particion_vigente(lago, fecha)
            if not vigente:
                raise ValueError(f"la particion escrita no supero la verificacion "
                                 f"({motivo})")

            diag.update({"lake": lago, "date": fecha,
                         "nubosidad_oficial_pct": OFFICIAL_CLOUD_COVER[lago][fecha]})
            diagnosticos.append(diag)
            registros.append({"lake": lago, "date": fecha, "filas": len(tabla),
                              "estado": "construido",
                              "dataset_version": DATASET_VERSION,
                              "schema_hash": hash_esquema(),
                              "archivo": str(destino.relative_to(ROOT))})
            LOGGER.info("[OK] %-22s %8s filas de agua  (%.1f s, %.1f MB)",
                        etiqueta, f"{len(tabla):,}", time.time() - t0,
                        destino.stat().st_size / 1e6)
        except Exception as exc:
            LOGGER.error("[ERROR] %s: %s: %s", etiqueta, type(exc).__name__, exc)
            errores.append((etiqueta, str(exc)))

    if errores:
        LOGGER.error("")
        LOGGER.error("Errores en %d combinaciones; no se generan reportes parciales.",
                     len(errores))
        for e, d in errores:
            LOGGER.error("  - %s -> %s", e, d)
        return 1

    total_filas = sum(r["filas"] for r in registros)
    LOGGER.info("")
    LOGGER.info("Particiones: %d | Filas totales: %s | Tiempo: %.1f s",
                len(registros), f"{total_filas:,}", time.time() - inicio_global)

    if diagnosticos:
        pd.DataFrame(diagnosticos).to_csv(
            DATA_DIR / "limpieza_diagnostico.csv", index=False)

    pd.DataFrame(registros).to_csv(DATA_DIR / "dataset_manifest.csv", index=False)
    LOGGER.info("Manifiesto: %s", (DATA_DIR / "dataset_manifest.csv").relative_to(ROOT))

    LOGGER.info("")
    LOGGER.info("Generando estadisticas agregadas, muestra EDA y figuras...")
    generar_analisis(registros, diagnosticos)

    LOGGER.info("")
    LOGGER.info("BUILD completado. Siguiente paso:")
    LOGGER.info("  python preparar_dataset_ml.py --validate")
    return 0


# ----------------------------------------------------------------------------
# Estadisticas, muestra y figuras
# ----------------------------------------------------------------------------
def cargar_columnas(columnas):
    """Lee el dataset completo restringido a unas columnas (memoria acotada)."""
    return pd.read_parquet(PIXELS_DIR, columns=columnas, engine="pyarrow")


# ----------------------------------------------------------------------------
# Analisis de viabilidad estadistica de cada umbral
# ----------------------------------------------------------------------------
# Criterios explicitos y prudentes. Un solo pixel positivo NO hace viable un
# experimento: se exige que la clase rara este presente, repartida y con
# suficientes grupos para los folds propuestos.
MIN_POSITIVOS_ENTRENAR = 1000     # minimo absoluto de la clase rara
MIN_PCT_ENTRENAR = 0.05           # % minimo de la clase rara
MIN_FECHAS_CON_AMBAS = 3          # fechas con ambas clases para validacion temporal
N_FOLDS_PROPUESTOS = 5
MIN_BLOQUES_POSITIVOS = 2 * N_FOLDS_PROPUESTOS  # >=2 bloques positivos por fold


def analizar_viabilidad(completo: pd.DataFrame) -> dict:
    """
    Evalua, para cada umbral, si sostiene los experimentos de la Parte 2.

    Se calcula sobre el dataset COMPLETO, no sobre la muestra.
    """
    respuestas = [f"high_cyano_{int(u)}" for u in UMBRALES_CANDIDATOS]
    lagos = sorted(completo["lake"].unique())
    total = len(completo)

    f_global, f_lago, f_fecha, f_bloque, f_conc = [], [], [], [], []

    for umbral, col in zip(UMBRALES_CANDIDATOS, respuestas):
        pos = int(completo[col].sum())
        neg = total - pos
        pct = 100.0 * pos / total if total else 0.0

        por_fecha = completo.groupby(["lake", "date"])[col].agg(["sum", "count"])
        por_fecha["pos"] = por_fecha["sum"].astype(int)
        por_fecha["neg"] = (por_fecha["count"] - por_fecha["sum"]).astype(int)
        fechas_ambas = int(((por_fecha["pos"] > 0) & (por_fecha["neg"] > 0)).sum())
        fechas_con_pos = int((por_fecha["pos"] > 0).sum())

        por_bloque = completo.groupby("spatial_block_1km")[col].agg(["sum", "count"])
        por_bloque["pos"] = por_bloque["sum"].astype(int)
        por_bloque["neg"] = (por_bloque["count"] - por_bloque["sum"]).astype(int)
        bloques_totales = len(por_bloque)
        bloques_con_pos = int((por_bloque["pos"] > 0).sum())
        bloques_ambas = int(((por_bloque["pos"] > 0) & (por_bloque["neg"] > 0)).sum())

        # Concentracion: que fraccion de los positivos vive en los 5 grupos mayores.
        top5_fechas = por_fecha["pos"].sort_values(ascending=False).head(5)
        top5_bloques = por_bloque["pos"].sort_values(ascending=False).head(5)
        conc_fechas = 100.0 * top5_fechas.sum() / pos if pos else 0.0
        conc_bloques = 100.0 * top5_bloques.sum() / pos if pos else 0.0

        # Numero efectivo de grupos positivos (inverso del indice de Herfindahl):
        # si los positivos estuvieran repartidos por igual entre N bloques daria N;
        # si estuvieran todos en uno daria 1. Mide dispersion real, no conteo bruto.
        p = por_bloque.loc[por_bloque["pos"] > 0, "pos"].to_numpy(dtype=float)
        efectivo = float((p.sum() ** 2) / (p ** 2).sum()) if p.size else 0.0

        # --- Criterios de viabilidad ---
        estratificado = pos > 0 and neg > 0 and pos >= N_FOLDS_PROPUESTOS
        groupkfold = (bloques_con_pos >= MIN_BLOQUES_POSITIVOS
                      and bloques_ambas >= 1 and efectivo >= N_FOLDS_PROPUESTOS)
        temporal = fechas_ambas >= MIN_FECHAS_CON_AMBAS

        entrenable, evaluable = {}, {}
        for lago in lagos:
            g = completo[completo["lake"] == lago]
            p_l = int(g[col].sum())
            n_l = len(g) - p_l
            pct_l = 100.0 * p_l / len(g) if len(g) else 0.0
            entrenable[lago] = (p_l >= MIN_POSITIVOS_ENTRENAR
                                and pct_l >= MIN_PCT_ENTRENAR and n_l > 0)
            # Para EVALUAR basta con que ambas clases existan, aunque sean pocas.
            evaluable[lago] = p_l > 0 and n_l > 0
            f_lago.append({
                "umbral_ug_L": umbral, "lake": lago, "n_total": len(g),
                "positivos": p_l, "negativos": n_l, "pct_positivo": pct_l,
                "entrenable": entrenable[lago], "evaluable": evaluable[lago],
                "criterio_entrenable": (f">= {MIN_POSITIVOS_ENTRENAR} positivos y "
                                        f">= {MIN_PCT_ENTRENAR} %"),
            })

        for (lago, fecha), fila in por_fecha.iterrows():
            f_fecha.append({
                "umbral_ug_L": umbral, "lake": lago, "date": fecha,
                "n_total": int(fila["count"]), "positivos": int(fila["pos"]),
                "negativos": int(fila["neg"]),
                "pct_positivo": 100.0 * fila["pos"] / fila["count"],
                "ambas_clases": bool(fila["pos"] > 0 and fila["neg"] > 0),
            })

        # Solo se listan los bloques con positivos: los vacios son decenas de miles.
        for bloque, fila in por_bloque[por_bloque["pos"] > 0].iterrows():
            f_bloque.append({
                "umbral_ug_L": umbral, "spatial_block_1km": bloque,
                "n_total": int(fila["count"]), "positivos": int(fila["pos"]),
                "negativos": int(fila["neg"]),
                "pct_positivo": 100.0 * fila["pos"] / fila["count"],
                "ambas_clases": bool(fila["neg"] > 0),
            })

        f_conc.append({
            "umbral_ug_L": umbral, "positivos_totales": pos,
            "top5_fechas": ";".join(f"{a}|{b}={int(v)}" for (a, b), v in top5_fechas.items()),
            "pct_positivos_en_top5_fechas": conc_fechas,
            "top5_bloques": ";".join(f"{k}={int(v)}" for k, v in top5_bloques.items()),
            "pct_positivos_en_top5_bloques": conc_bloques,
            "bloques_positivos": bloques_con_pos,
            "n_efectivo_bloques_positivos": efectivo,
        })

        f_global.append({
            "umbral_ug_L": umbral, "es_respuesta_principal": (col == TARGET_COLUMN),
            "n_total": total, "positivos": pos, "negativos": neg,
            "pct_positivo": pct,
            "razon_desbalance": (neg / pos) if pos else np.inf,
            "fechas_totales": len(por_fecha), "fechas_con_positivos": fechas_con_pos,
            "fechas_con_ambas_clases": fechas_ambas,
            "bloques_totales": bloques_totales, "bloques_con_positivos": bloques_con_pos,
            "bloques_con_ambas_clases": bloques_ambas,
            "n_efectivo_bloques_positivos": efectivo,
            "pct_positivos_en_top5_fechas": conc_fechas,
            "pct_positivos_en_top5_bloques": conc_bloques,
            "viable_stratified_70_30": estratificado,
            "viable_groupkfold_bloque": groupkfold,
            "viable_validacion_temporal": temporal,
            "entrenable_por_lago": ";".join(f"{l}={entrenable[l]}" for l in lagos),
            "evaluable_por_lago": ";".join(f"{l}={evaluable[l]}" for l in lagos),
            "viable_generalizacion_entre_lagos": all(entrenable.values()),
            "significado_ambiental": SIGNIFICADO_UMBRAL[umbral],
        })

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(f_global).to_csv(TARGET_DIR / "threshold_viability_global.csv", index=False)
    pd.DataFrame(f_lago).to_csv(TARGET_DIR / "threshold_viability_by_lake.csv", index=False)
    pd.DataFrame(f_fecha).to_csv(TARGET_DIR / "threshold_viability_by_date.csv", index=False)
    pd.DataFrame(f_bloque).to_csv(TARGET_DIR / "threshold_viability_by_block.csv", index=False)
    pd.DataFrame(f_conc).to_csv(TARGET_DIR / "threshold_group_concentration.csv", index=False)

    return {"global": pd.DataFrame(f_global), "lago": pd.DataFrame(f_lago),
            "concentracion": pd.DataFrame(f_conc)}


def escribir_reporte_viabilidad(via: dict, completo: pd.DataFrame) -> None:
    g = via["global"]
    lago_df = via["lago"]
    conc = via["concentracion"]
    lagos = sorted(completo["lake"].unique())
    principal = g[g["umbral_ug_L"] == TARGET_THRESHOLD_UG_L].iloc[0]

    ruta = REPORTS_DIR / "threshold_viability.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# Viabilidad estadistica de los umbrales — Laboratorio 4, Parte 2\n\n")
        w(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}  \n")
        w(f"Dataset: version {DATASET_VERSION}, {len(completo):,} observaciones reales.  \n")
        w(f"Respuesta principal: **`{TARGET_COLUMN}`** "
          f"(clorofila-a >= {TARGET_THRESHOLD_UG_L:.0f} ug/L)\n\n")

        w("## Criterios aplicados\n\n")
        w("Un umbral no se declara viable por tener un pixel positivo. Se exige:\n\n")
        w(f"- Ambas clases presentes.\n")
        w(f"- Para **entrenar en un lago**: >= {MIN_POSITIVOS_ENTRENAR} positivos y "
          f">= {MIN_PCT_ENTRENAR} % de ese lago.\n")
        w(f"- Para **validacion temporal**: >= {MIN_FECHAS_CON_AMBAS} fechas con ambas clases.\n")
        w(f"- Para **GroupKFold espacial** con {N_FOLDS_PROPUESTOS} folds: "
          f">= {MIN_BLOQUES_POSITIVOS} bloques con positivos y numero efectivo de "
          f"bloques >= {N_FOLDS_PROPUESTOS}, para que ningun fold quede sin clase positiva.\n\n")
        w("El *numero efectivo de bloques* es el inverso del indice de Herfindahl sobre "
          "el reparto de positivos: vale N si los positivos se distribuyen por igual "
          "entre N bloques y 1 si estan todos concentrados en uno. Mide dispersion "
          "real, no simple conteo.\n\n")

        w("## Resumen por umbral\n\n")
        w("| Umbral | Positivos | % | Desbalance | Fechas ambas clases | Bloques con "
          "positivos | N.o efectivo | Stratified | GroupKFold | Temporal | Entre lagos |\n")
        w("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in g.iterrows():
            marca = " **(principal)**" if r["es_respuesta_principal"] else ""
            w(f"| {r['umbral_ug_L']:.0f}{marca} | {int(r['positivos']):,} | "
              f"{r['pct_positivo']:.4f} % | 1:{r['razon_desbalance']:.0f} | "
              f"{int(r['fechas_con_ambas_clases'])}/{int(r['fechas_totales'])} | "
              f"{int(r['bloques_con_positivos'])}/{int(r['bloques_totales'])} | "
              f"{r['n_efectivo_bloques_positivos']:.1f} | "
              f"{'Si' if r['viable_stratified_70_30'] else 'No'} | "
              f"{'Si' if r['viable_groupkfold_bloque'] else 'No'} | "
              f"{'Si' if r['viable_validacion_temporal'] else 'No'} | "
              f"{'Si' if r['viable_generalizacion_entre_lagos'] else 'No'} |\n")

        w("\n## Por lago\n\n")
        w("| Umbral | Lago | Positivos | Negativos | % | Entrenable | Evaluable |\n")
        w("|---|---|---|---|---|---|---|\n")
        for _, r in lago_df.iterrows():
            w(f"| {r['umbral_ug_L']:.0f} | {r['lake']} | {int(r['positivos']):,} | "
              f"{int(r['negativos']):,} | {r['pct_positivo']:.4f} % | "
              f"{'Si' if r['entrenable'] else 'No'} | "
              f"{'Si' if r['evaluable'] else 'No'} |\n")

        w("\n## Concentracion de los positivos\n\n")
        w("| Umbral | % de positivos en las 5 fechas principales | % en los 5 bloques "
          "principales | Bloques positivos | N.o efectivo |\n|---|---|---|---|---|\n")
        for _, r in conc.iterrows():
            w(f"| {r['umbral_ug_L']:.0f} | {r['pct_positivos_en_top5_fechas']:.1f} % | "
              f"{r['pct_positivos_en_top5_bloques']:.1f} % | "
              f"{int(r['bloques_positivos'])} | "
              f"{r['n_efectivo_bloques_positivos']:.1f} |\n")

        w(f"\n## Desbalance de la respuesta principal ({TARGET_THRESHOLD_UG_L:.0f} ug/L)\n\n")
        w(f"- **Desbalance global:** {int(principal['positivos']):,} positivos frente a "
          f"{int(principal['negativos']):,} negativos "
          f"({principal['pct_positivo']:.3f} %), razon "
          f"**1:{principal['razon_desbalance']:.0f}**.\n")
        for lago in lagos:
            r = lago_df[(lago_df["umbral_ug_L"] == TARGET_THRESHOLD_UG_L)
                        & (lago_df["lake"] == lago)].iloc[0]
            w(f"- **{lago}:** {int(r['positivos']):,} positivos "
              f"({r['pct_positivo']:.4f} %). Entrenable: "
              f"{'si' if r['entrenable'] else 'NO'}.\n")
        w(f"- **Diferencia entre lagos:** la prevalencia no es comparable entre ambos, "
          "asi que un modelo entrenado en uno vera una frecuencia de la clase positiva "
          "muy distinta a la del otro.\n")
        w(f"- **Diferencia entre fechas:** solo "
          f"{int(principal['fechas_con_ambas_clases'])} de "
          f"{int(principal['fechas_totales'])} combinaciones lago-fecha contienen ambas "
          "clases.\n")
        w(f"- **Dependencia espacial:** el "
          f"{principal['pct_positivos_en_top5_bloques']:.1f} % de los positivos se "
          "concentra en solo 5 bloques de 1 km. Los positivos no son independientes "
          "entre si: estan agrupados espacialmente.\n\n")

        w("### Riesgos derivados\n\n")
        w(f"- **Accuracy enganosa.** Un clasificador que prediga siempre la clase "
          f"mayoritaria acertaria el {100 - principal['pct_positivo']:.3f} % sin "
          "detectar ni una sola floracion.\n")
        w("- **Folds sin positivos.** Con particion aleatoria simple algunos pliegues "
          "podrian quedarse sin clase positiva; con agrupacion espacial el riesgo "
          "aumenta porque los positivos estan concentrados.\n")
        w("- **Optimismo por autocorrelacion espacial.** Sin agrupar por bloque, "
          "pixeles vecinos casi identicos caerian en entrenamiento y prueba a la vez.\n")
        w("- **Colapso a la clase mayoritaria** durante el entrenamiento si no se "
          "compensa el desbalance.\n\n")

        w("### Recomendaciones para el Ejercicio 4 (no implementadas todavia)\n\n")
        w("- `class_weight=\"balanced\"` en Regresion Logistica y Random Forest.\n")
        w("- `scale_pos_weight = n_negativos / n_positivos` en XGBoost.\n")
        w("- Reportar **PR-AUC** ademas de Accuracy, Precision, Recall, F1 y ROC-AUC: "
          "con esta prevalencia la curva ROC resulta demasiado optimista.\n")
        w("- Reportar **Recall y F1 de la clase positiva** por separado, no solo macro.\n")
        w("- Ajustar el **umbral de decision** usando solo entrenamiento/validacion, "
          "nunca el conjunto de prueba.\n")
        w("- **No aplicar SMOTE ni sobremuestreo antes de separar los grupos**: "
          "generaria vecinos sinteticos a partir de pixeles que luego caerian en el "
          "conjunto de prueba, inflando el desempeno.\n")
        w("- Conservar los grupos espaciales (`spatial_block_1km`) y temporales "
          "(`date`) intactos durante toda la particion.\n")

    LOGGER.info("  Reporte de viabilidad: %s", ruta.relative_to(ROOT))


def generar_analisis(registros, diagnosticos) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")

    columnas_numericas = BANDAS + INDICES
    columnas_respuesta = [f"high_cyano_{int(u)}" for u in UMBRALES_CANDIDATOS]

    # --- Conteos y balance sobre el DATASET COMPLETO ---
    completo = cargar_columnas(
        ["lake", "date", "chlorophyll", "fuera_calibracion",
         "spatial_block_1km", "year", "month", "season"] + columnas_respuesta
    )
    completo["lake"] = completo["lake"].astype(str)
    completo["date"] = completo["date"].astype(str)
    total = len(completo)
    LOGGER.info("  Filas totales (dataset completo): %s", f"{total:,}")

    # --- Viabilidad estadistica de cada umbral (dataset completo) ---
    LOGGER.info("  Analizando viabilidad de los umbrales...")
    viabilidad = analizar_viabilidad(completo)
    escribir_reporte_viabilidad(viabilidad, completo)

    por_lago = completo.groupby("lake").size().rename("observaciones")
    por_fecha = completo.groupby(["lake", "date"]).size().rename("observaciones")
    por_lago.to_frame().to_csv(DATA_DIR / "observaciones_por_lago.csv")
    por_fecha.to_frame().to_csv(DATA_DIR / "observaciones_por_lago_fecha.csv")

    # --- Distribuciones de la respuesta ---
    filas_global = []
    for umbral in UMBRALES_CANDIDATOS:
        col = f"high_cyano_{int(umbral)}"
        n1 = int(completo[col].sum())
        filas_global.append({
            "umbral_ug_L": umbral, "n_total": total,
            "n_alta": n1, "n_baja": total - n1,
            "pct_alta": 100.0 * n1 / total if total else 0.0,
            "razon_desbalance": (total - n1) / n1 if n1 else np.inf,
        })
    pd.DataFrame(filas_global).to_csv(
        TARGET_DIR / "target_distribution_global.csv", index=False)

    filas_lago = []
    for (lago), grupo in completo.groupby("lake"):
        for umbral in UMBRALES_CANDIDATOS:
            col = f"high_cyano_{int(umbral)}"
            n1 = int(grupo[col].sum())
            filas_lago.append({
                "lake": lago, "umbral_ug_L": umbral, "n_total": len(grupo),
                "n_alta": n1, "pct_alta": 100.0 * n1 / len(grupo) if len(grupo) else 0.0,
            })
    pd.DataFrame(filas_lago).to_csv(
        TARGET_DIR / "target_distribution_by_lake.csv", index=False)

    filas_fecha = []
    for (lago, fecha), grupo in completo.groupby(["lake", "date"]):
        registro = {"lake": lago, "date": fecha, "n_total": len(grupo),
                    "chl_mediana": float(grupo["chlorophyll"].median()),
                    "chl_p99": float(grupo["chlorophyll"].quantile(0.99)),
                    "chl_max": float(grupo["chlorophyll"].max())}
        for umbral in UMBRALES_CANDIDATOS:
            col = f"high_cyano_{int(umbral)}"
            registro[f"pct_alta_{int(umbral)}"] = 100.0 * grupo[col].mean()
        filas_fecha.append(registro)
    pd.DataFrame(filas_fecha).to_csv(
        TARGET_DIR / "target_distribution_by_date.csv", index=False)

    estad_chl = {
        "n": total,
        "min": float(completo["chlorophyll"].min()),
        "p01": float(completo["chlorophyll"].quantile(0.01)),
        "mediana": float(completo["chlorophyll"].median()),
        "p99": float(completo["chlorophyll"].quantile(0.99)),
        "max": float(completo["chlorophyll"].max()),
        "negativos": int((completo["chlorophyll"] < 0).sum()),
        "fuera_calibracion": int(completo["fuera_calibracion"].sum()),
    }
    with open(TARGET_DIR / "chlorophyll_stats.json", "w", encoding="utf-8") as fh:
        json.dump(estad_chl, fh, indent=2)

    # --- Muestra determinista para figuras ---
    muestra_n = min(MUESTRA_EDA, total)
    todas = ["lake", "date", "row", "col", "x_utm", "y_utm", "longitude", "latitude"] \
        + columnas_numericas + columnas_respuesta \
        + ["fuera_calibracion"] + COLUMNAS_AUXILIARES
    completo_full = cargar_columnas(todas)
    muestra = completo_full.sample(n=muestra_n, random_state=SEED).reset_index(drop=True)
    muestra.to_parquet(DATA_DIR / "eda_sample.parquet", engine="pyarrow",
                       compression="snappy", index=False)
    LOGGER.info("  Muestra EDA: %s filas (semilla %d)", f"{muestra_n:,}", SEED)

    # --- Esquema ---
    esquema = []
    for col in completo_full.columns:
        serie = completo_full[col]
        if col in PREDICTORES_PRINCIPALES:
            rol = "predictor_principal"
        elif col == TARGET_COLUMN:
            rol = "respuesta_principal"
        elif col in columnas_respuesta:
            rol = "respuesta_sensibilidad"
        elif col in COLUMNAS_AUXILIARES:
            rol = "auxiliar_validacion"
        elif col in EXCLUIDAS_POR_FUGA:
            rol = "excluida_por_fuga"
        elif col in COLUMNAS_TRAZABILIDAD:
            rol = "trazabilidad"
        else:
            rol = "diagnostico"
        esquema.append({
            "variable": col, "tipo": str(serie.dtype), "rol": rol,
            "faltantes_pct": 100.0 * serie.isna().mean(),
            "motivo": EXCLUIDAS_POR_FUGA.get(col, COLUMNAS_TRAZABILIDAD.get(col, "")),
        })
    pd.DataFrame(esquema).to_csv(DATA_DIR / "dataset_schema.csv", index=False)

    # --- Figuras ---
    pf = por_fecha.reset_index()
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=pf, x="date", y="observaciones", hue="lake", ax=ax)
    ax.set_title("Observaciones validas de agua por lago y fecha")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Pixeles de agua")
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(EDA_DIR / "observations_by_lake_date.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    faltantes = completo_full.isna().mean().mul(100).sort_values(ascending=False)
    sns.barplot(x=faltantes.values, y=faltantes.index, ax=ax, color="#4878a8")
    ax.set_title("Porcentaje de valores faltantes por variable (dataset completo)")
    ax.set_xlabel("% faltantes"); ax.set_xlim(0, max(1.0, faltantes.max() * 1.1))
    fig.tight_layout(); fig.savefig(EDA_DIR / "missing_values.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    for ax, col in zip(axes.ravel(), columnas_numericas):
        sns.histplot(muestra[col], bins=60, ax=ax, color="#4878a8")
        ax.set_title(col, fontsize=10); ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes.ravel()[len(columnas_numericas):]:
        ax.axis("off")
    fig.suptitle(f"Distribuciones de bandas e indices (muestra n={muestra_n:,}, semilla {SEED})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(EDA_DIR / "feature_distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for lago, grupo in muestra.groupby("lake", observed=True):
        sns.kdeplot(grupo["chlorophyll"].clip(-20, 120), ax=axes[0],
                    label=str(lago), fill=True, alpha=0.35)
    for umbral in UMBRALES_CANDIDATOS:
        axes[0].axvline(umbral, ls="--", lw=1.2, color="red")
        axes[0].text(umbral, axes[0].get_ylim()[1] * 0.9, f" {int(umbral)}",
                     color="red", fontsize=9)
    axes[0].set_title("Clorofila-a por lago (recortada a [-20,120] solo para dibujar)")
    axes[0].set_xlabel("Clorofila-a (ug/L)"); axes[0].legend()
    sns.boxplot(data=muestra, x="lake", y="chlorophyll", ax=axes[1], showfliers=False)
    axes[1].axhline(UMBRAL_RECOMENDADO, ls="--", color="red",
                    label=f"Umbral recomendado {UMBRAL_RECOMENDADO:.0f}")
    axes[1].set_title("Clorofila-a por lago"); axes[1].set_ylabel("ug/L"); axes[1].legend()
    fig.tight_layout(); fig.savefig(EDA_DIR / "chlorophyll_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(UMBRALES_CANDIDATOS),
                             figsize=(5.2 * len(UMBRALES_CANDIDATOS), 4.8))
    for ax, umbral in zip(np.atleast_1d(axes), UMBRALES_CANDIDATOS):
        col = f"high_cyano_{int(umbral)}"
        resumen = (completo.groupby("lake")[col].mean().mul(100).reset_index())
        sns.barplot(data=resumen, x="lake", y=col, ax=ax, color="#c44e52")
        pct = 100.0 * completo[col].mean()
        ax.set_title(f"Umbral {int(umbral)} ug/L\nglobal alta = {pct:.3f}%")
        ax.set_ylabel("% clase alta"); ax.set_xlabel("")
    fig.suptitle("Balance de la variable respuesta (dataset completo)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(EDA_DIR / "target_balance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    corr_pred = muestra[PREDICTORES_PRINCIPALES].corr()
    sns.heatmap(corr_pred, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, ax=axes[0], cbar_kws={"shrink": 0.7})
    axes[0].set_title("Predictores principales (SIN fuga)", fontweight="bold")
    con_fuga = [c for c in ["B04", "B05", "NDCI", "FAI", "NDVI", "chlorophyll"]
                if c in muestra.columns]
    sns.heatmap(muestra[con_fuga].corr(), annot=True, fmt=".2f", cmap="Reds",
                center=0, square=True, ax=axes[1], cbar_kws={"shrink": 0.7})
    axes[1].set_title("Variables CON fuga (excluidas del modelo)", fontweight="bold",
                      color="#a00")
    fig.suptitle(f"Matrices de correlacion (muestra n={muestra_n:,})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(EDA_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, lago in zip(axes, sorted(muestra["lake"].astype(str).unique())):
        g = muestra[muestra["lake"].astype(str) == lago]
        s = ax.scatter(g["longitude"], g["latitude"], c=g["chlorophyll"],
                       s=2, cmap="RdYlGn_r", vmin=0,
                       vmax=float(g["chlorophyll"].quantile(0.98)))
        ax.set_title(f"{lago}: distribucion espacial de la muestra")
        ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
        fig.colorbar(s, ax=ax, label="Clorofila-a (ug/L)")
    fig.tight_layout(); fig.savefig(EDA_DIR / "spatial_sample.png", dpi=150)
    plt.close(fig)

    # --- Bloques espaciales de 1 km y concentracion de la clase positiva ---
    resumen_bloques = (completo.groupby(["lake", "spatial_block_1km"])
                       .agg(n=("chlorophyll", "size"),
                            positivos=(TARGET_COLUMN, "sum"))
                       .reset_index())
    resumen_bloques["pct_positivo"] = (
        100.0 * resumen_bloques["positivos"] / resumen_bloques["n"])
    resumen_bloques.to_csv(DATA_DIR / "resumen_bloques_1km.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    conteo = resumen_bloques.groupby("lake")["spatial_block_1km"].nunique()
    axes[0].bar(conteo.index, conteo.values, color=["#4878a8", "#c44e52"])
    axes[0].set_title(f"Bloques de {BLOQUE_ESPACIAL_M:.0f} m por lago", fontweight="bold")
    axes[0].set_ylabel("N.o de bloques")
    for i, v in enumerate(conteo.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    sns.histplot(resumen_bloques["n"], bins=40, ax=axes[1], color="#4878a8")
    axes[1].set_title("Observaciones por bloque", fontweight="bold")
    axes[1].set_xlabel("Pixeles en el bloque"); axes[1].set_ylabel("N.o de bloques")

    con_pos = resumen_bloques[resumen_bloques["positivos"] > 0]
    for lago, g in muestra.groupby("lake", observed=True):
        axes[2].scatter(g["x_utm"], g["y_utm"], s=1, alpha=0.12, color="#bbbbbb")
    if not con_pos.empty:
        centros = (completo[completo[TARGET_COLUMN] == 1]
                   .groupby("spatial_block_1km")[["chlorophyll"]].size())
        pos_muestra = muestra[muestra[TARGET_COLUMN] == 1]
        axes[2].scatter(pos_muestra["x_utm"], pos_muestra["y_utm"], s=4,
                        color="#c44e52", label=f"{TARGET_COLUMN} = 1")
        axes[2].legend()
    axes[2].set_title(f"Ubicacion de la clase positiva ({TARGET_THRESHOLD_UG_L:.0f} ug/L)",
                      fontweight="bold")
    axes[2].set_xlabel("x UTM (m)"); axes[2].set_ylabel("y UTM (m)")
    fig.suptitle("Bloques espaciales para validacion agrupada (EPSG:32615)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(EDA_DIR / "spatial_blocks.png", dpi=150)
    plt.close(fig)

    escribir_reporte(completo, filas_global, filas_lago, filas_fecha,
                     estad_chl, diagnosticos, esquema, muestra_n)


def escribir_reporte(completo, filas_global, filas_lago, filas_fecha,
                     estad_chl, diagnosticos, esquema, muestra_n) -> None:
    """Redacta el informe de preparacion con los numeros reales calculados."""
    total = len(completo)
    diag = pd.DataFrame(diagnosticos) if diagnosticos else pd.DataFrame()

    def pct(umbral, lake=None):
        for f in filas_lago if lake else filas_global:
            if f["umbral_ug_L"] == umbral and (not lake or f["lake"] == lake):
                return f["pct_alta"]
        return float("nan")

    lagos = sorted(completo["lake"].unique())
    ruta = REPORTS_DIR / "preparacion_dataset.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# Preparacion del dataset de Machine Learning — Laboratorio 4, Parte 2\n\n")
        w(f"Generado: {datetime.now().isoformat(timespec='seconds')}  \n")
        w(f"Fuente: {len(filas_fecha)} GeoTIFF Sentinel-2 L1C reales "
          f"({S2_L1C_COLLECTION}), EPSG:32615, 20 m.  \n")
        w(f"Semilla fija: {SEED}\n\n")
        w("> **Origen de los datos.** Todas las cifras provienen de imagenes "
          "Sentinel-2 descargadas de Copernicus Data Space. No se utilizo ningun "
          "dato sintetico en ninguna etapa.\n\n---\n\n")

        w("## 1. Construccion y limpieza\n\n")
        w(f"- Observaciones validas de agua: **{total:,}**\n")
        for lago in lagos:
            n = int((completo['lake'] == lago).sum())
            w(f"  - {lago}: {n:,} ({100*n/total:.1f} %)\n")
        if not diag.empty:
            w("\n### Descartes por criterio (suma sobre los 22 rasteres)\n\n")
            w("| Criterio | Pixeles descartados |\n|---|---|\n")
            for col, etiqueta in [
                ("descartados_nodata", "NoData / -9999 / fuera de datos validos"),
                ("descartados_no_finitos", "NaN o infinitos"),
                ("descartados_reflectancia_no_positiva", "Reflectancia TOA <= 0 (no fisica)"),
                ("descartados_no_agua", "No es agua segun la mascara WBI"),
                ("descartados_indice_no_finito", "Indice espectral no finito"),
                ("descartados_duplicados", "Filas duplicadas (lake,date,row,col)"),
            ]:
                if col in diag:
                    w(f"| {etiqueta} | {int(diag[col].sum()):,} |\n")
            w(f"\nPixeles brutos totales: {int(diag['total_pixeles'].sum()):,}\n")
            if "diagnostico_reflectancia_mayor_1" in diag:
                w(f"\nDiagnostico (NO se descartan): {int(diag['diagnostico_reflectancia_mayor_1'].sum()):,} "
                  "pixeles conservan alguna banda con reflectancia > 1, compatible con "
                  "nubes o reflexion especular. No se impuso ningun limite superior "
                  "arbitrario.\n")

        w("\n### Escala radiometrica\n\n")
        if not diag.empty and "motivo_escala" in diag:
            w(f"- Deteccion automatica: {diag['motivo_escala'].iloc[0]}\n")
        w("- La coleccion `SENTINEL2_L1C` declara `scale = 0.0001, offset = 0`. "
          "Los GeoTIFF conservan numeros digitales, por lo que se multiplican por "
          "0.0001 una unica vez.\n")
        w("- `mainlab4.normalize_band()` **no** se utiliza aqui: solo divide cuando "
          "el dtype es entero, y estos rasteres son float32, por lo que habria "
          "dejado los DN sin escalar.\n")

        w("\n### Limitaciones documentadas\n\n")
        w("- **Separacion agua/tierra:** los GeoJSON disponibles son rectangulos "
          "identicos al bounding box oficial, no contornos reales de lago. La "
          "unica separacion efectiva es la mascara **WBI** de `mainlab4.py`. "
          "El area detectada es coherente con la realidad "
          "(Atitlan ~123 km2, Amatitlan ~15 km2), lo que respalda la mascara.\n")
        w("- **Nubes:** `SENTINEL2_L1C` no expone CLM, CLP, dataMask, SCL ni QA60. "
          "No se fabrico ninguna mascara sustituta. El control disponible es la "
          "seleccion oficial de fechas con baja nubosidad sobre el lago mas los "
          "filtros de validez espectral aplicados aqui.\n")
        w("- **Nivel L1C:** son reflectancias de tope de atmosfera, sin correccion "
          "atmosferica. El algoritmo NDCI fue disenado para L1C, pero esto anade "
          "incertidumbre a la magnitud absoluta de la clorofila.\n")
        w("- **Estimador satelital, no medicion in situ:** el algoritmo reporta "
          "**MAPE 42.3 %** y **RMSE relativo 95.8 %**, y fue calibrado para "
          "*Microcystis aeruginosa* sobre datos simulados. En este laboratorio no se "
          "realizo ninguna validacion de campo.\n")

        w("\n#### Advertencia sobre el Lago de Atitlan\n\n")
        fuera_por_lago = (completo.groupby("lake")["fuera_calibracion"]
                          .mean().mul(100).round(2).to_dict())
        w("> El estimador satelital muestra concentraciones generalmente menores en "
          "Atitlan que en Amatitlan durante las fechas estudiadas; sin embargo, una "
          "proporcion importante de sus pixeles cae fuera del dominio de calibracion, "
          "por lo que los valores absolutos requieren validacion in situ.\n\n")
        w("Porcentaje de observaciones fuera del dominio de calibracion "
          f"[{CALIB_MIN:.0f}, {CALIB_MAX:.0f}] ug/L, por lago:\n\n")
        for lago_n, pct_fuera in sorted(fuera_por_lago.items()):
            w(f"- **{lago_n}:** {pct_fuera:.2f} %\n")
        w("\nPor tanto, la conclusion defendible es **comparativa** (Atitlan presenta "
          "valores estimados menores que Amatitlan en estas fechas), no absoluta: no "
          "procede afirmar sin mas que Atitlan esta \"limpio\".\n")

        w("\n---\n\n## 2. Variable respuesta\n\n")
        w("### 2.1 Distribucion real de la clorofila-a\n\n")
        w(f"- Rango observado: {estad_chl['min']:.2f} a {estad_chl['max']:.2f} ug/L\n")
        w(f"- Mediana: {estad_chl['mediana']:.2f} ug/L | p99: {estad_chl['p99']:.2f} ug/L\n")
        w(f"- Valores negativos: {estad_chl['negativos']:,} "
          f"({100*estad_chl['negativos']/total:.2f} %)\n")
        w(f"- Fuera del dominio de calibracion [{CALIB_MIN}, {CALIB_MAX}] ug/L: "
          f"{estad_chl['fuera_calibracion']:,} ({100*estad_chl['fuera_calibracion']/total:.2f} %)\n\n")
        w("Los valores negativos carecen de sentido fisico: proceden de evaluar el "
          "polinomio NDCI->clorofila con NDCI negativo, fuera de su dominio. **No se "
          "recortaron ni transformaron**, porque hacerlo alteraria artificialmente el "
          "balance de clases; se reportan y se marcan con `fuera_calibracion`.\n")

        w("\n### 2.2 Verificacion de la bibliografia citada en `lab4-2.ipynb`\n\n")
        w("| Referencia citada | Verificacion |\n|---|---|\n")
        w("| OECD (1982), *Eutrophication of Waters*, DOI 10.1787/9789264077980-en | "
          "**Existe y es pertinente.** Sistema de fronteras fijas por clorofila-a media: "
          "ultraoligotrofico <1, oligotrofico <2.5, mesotrofico 2.5-8, **eutrofico 8-25**, "
          "**hipertrofico >25** ug/L. |\n")
        w("| WHO (2021), *Guidelines on recreational water quality, Vol. 1* | "
          "**Existe y es pertinente.** Con dominancia de cianobacterias: nivel de "
          "vigilancia 1-12 ug/L de clorofila-a; **Alerta 1: 12-24 ug/L**. La Alerta 2 "
          "se define por natas y transparencia, no por un valor de clorofila. |\n")
        w("| Mishra, S. et al. (2019), *Applicability of Sentinel-2...*, "
          "RSE 232, 111354, DOI 10.1016/j.rse.2019.111354 | "
          "**REFERENCIA INCORRECTA.** Ese DOI corresponde a Hurskainen, Adhikari, "
          "Siljander, Pellikka y Hemp (2019), *Auxiliary datasets improve accuracy of "
          "object-based land use/land cover classification in heterogeneous savanna "
          "landscapes*, RSE **233**, 111354: un articulo de cobertura del suelo en "
          "sabana, sin relacion con clorofila ni calidad de agua. |\n\n")
        w("**Sustitucion propuesta.** La referencia correcta para el indice empleado es "
          "Mishra, S. & Mishra, D. R. (2012), *Normalized difference chlorophyll index: "
          "a novel model for remote estimation of chlorophyll-a concentration in turbid "
          "productive waters*, Remote Sensing of Environment **117**, 394-406 "
          "(DOI 10.1016/j.rse.2011.10.016), que introduce el NDCI y calibra el modelo "
          f"cuadratico en un rango de **{CALIB_MIN:.0f}-{CALIB_MAX:.0f} mg/m3**.\n\n")
        w("El polinomio implementado en `mainlab4.chl_from_ndci` "
          "(`826.57*NDCI^3 - 176.43*NDCI^2 + 19*NDCI + 4.071`) proviene del script "
          "*Cyanobacteria Chlorophyll-a NDCI L1C* de Sentinel Hub, cuya documentacion "
          "indica calibracion para *Microcystis aeruginosa*, entrenamiento con "
          "clorofila < 500 ug/L y errores de **MAPE 42.3 %** y **RMSE relativo 95.8 %**. "
          "Esa incertidumbre debe acompanar cualquier conclusion.\n")

        w("\n### 2.3 Analisis de los umbrales candidatos\n\n")
        w("| Umbral | Respaldo bibliografico | % clase alta global |")
        for lago in lagos:
            w(f" % alta {lago} |")
        w("\n|---|---|---|" + "---|" * len(lagos) + "\n")
        for umbral in UMBRALES_CANDIDATOS:
            marca = " **(principal)**" if umbral == TARGET_THRESHOLD_UG_L else ""
            w(f"| {umbral:.0f} ug/L{marca} | {SIGNIFICADO_UMBRAL[umbral]} | "
              f"{pct(umbral):.3f} % |")
            for lago in lagos:
                w(f" {pct(umbral, lago):.3f} % |")
            w("\n")

        w(f"\n### 2.4 Respuesta principal: `{TARGET_COLUMN}` "
          f"(>= {TARGET_THRESHOLD_UG_L:.0f} ug/L)\n\n")
        w(f"Se adopta **{TARGET_THRESHOLD_UG_L:.0f} ug/L** como respuesta principal "
          "porque representa la **transicion aproximada hacia la condicion eutrofica**: "
          "es la frontera mesotrofico -> eutrofico de la clasificacion trofica de OECD "
          "(1982). Operacionaliza \"alta presencia\" como el punto en que la biomasa "
          "algal deja de ser la propia de un lago equilibrado y el cuerpo de agua entra "
          "en un regimen de exceso de nutrientes.\n\n")
        w("Los otros tres umbrales **no son alternativas equivalentes**: describen "
          "situaciones ambientales distintas y se conservan como analisis de "
          "sensibilidad.\n\n")
        w("- **20 ug/L**: mantiene la continuidad con el umbral usado en la Parte 1.\n")
        w("- **25 ug/L**: representa una condicion **hipertrofica**, es decir un estado "
          "mas severo que el que se quiere detectar como \"alta presencia\".\n")
        w("- **50 ug/L**: se conserva como **escenario extremo**.\n\n")
        w("La eleccion es **ambiental**. Que 8 ug/L produzca ademas mas observaciones "
          "positivas es una **consecuencia secundaria** de haber escogido la transicion "
          "eutrofica, no el criterio de seleccion: la viabilidad estadistica se analiza "
          "por separado en `threshold_viability.md` y no intervino en la definicion del "
          "umbral.\n\n")
        w("> **Prudencia en la interpretacion.** La variable respuesta indica "
          "**clorofila-a alta estimada por satelite**, no una confirmacion in situ de "
          "presencia de cianobacterias ni de toxicidad. La clorofila mide biomasa "
          "fotosintetica total: no identifica especies ni toxinas.\n")

        w("\n### 2.5 Advertencia critica sobre el desbalance\n\n")
        for f in filas_global:
            if f["umbral_ug_L"] == UMBRAL_RECOMENDADO:
                w(f"Con el umbral recomendado la clase alta representa el "
                  f"**{f['pct_alta']:.3f} %** del dataset "
                  f"({f['n_alta']:,} de {f['n_total']:,} observaciones), es decir una "
                  f"razon de desbalance de **1:{f['razon_desbalance']:.0f}**.\n\n")
        for lago in lagos:
            p = pct(UMBRAL_RECOMENDADO, lago)
            if p == 0.0:
                w(f"- **{lago} no tiene ninguna observacion de clase alta** con este "
                  "umbral. Esto compromete el ejercicio 7 (generalizacion entre lagos): "
                  f"un modelo entrenado solo con {lago} no veria nunca la clase "
                  "positiva. Es un resultado ecologico real, no un defecto del "
                  "procesamiento, y debe discutirse con el docente antes de continuar.\n")
        w("\nConsecuencias para el modelado: *accuracy* sera enganosa; deben reportarse "
          "*recall*, *precision*, F1 y sobre todo **PR-AUC**, y considerar pesos de "
          "clase o remuestreo en el entrenamiento, nunca modificando la clorofila.\n")

        w("\n---\n\n## 3. Variables predictoras\n\n")
        w("Cadena de construccion de la respuesta:\n\n")
        w("```\nB04, B05  ->  NDCI  ->  chlorophyll  ->  high_cyano_*\n```\n\n")
        w("### 3.1 Conjunto predictor principal (sin fuga)\n\n")
        w("| Variable | Tipo | Justificacion |\n|---|---|---|\n")
        justificacion = {
            "B02": ("Banda espectral", "Azul (492 nm); sensible a dispersion y turbidez."),
            "B03": ("Banda espectral", "Verde (560 nm); maximo de reflectancia de la "
                                       "biomasa algal, no interviene en el NDCI."),
            "B07": ("Banda espectral", "Borde rojo (783 nm); responde a biomasa sin usar B04/B05."),
            "B08": ("Banda espectral", "NIR (833 nm); separa agua de vegetacion y detecta natas."),
            "B8A": ("Banda espectral", "NIR estrecho (865 nm); complementa a B08."),
            "B11": ("Banda espectral", "SWIR1 (1610 nm); distingue agua de suelo y nube."),
            "B12": ("Banda espectral", "SWIR2 (2190 nm); refuerza la discriminacion de agua."),
            "NDWI": ("Indice", "(B03-B08)/(B03+B08); no usa B04 ni B05, por lo que es "
                               "independiente de la cadena de la respuesta."),
        }
        for var in PREDICTORES_PRINCIPALES:
            t, j = justificacion[var]
            w(f"| `{var}` | {t} | {j} |\n")

        w("\n### 3.2 Variables excluidas del modelo principal\n\n")
        w("| Variable | Motivo de exclusion |\n|---|---|\n")
        for var, motivo in EXCLUIDAS_POR_FUGA.items():
            w(f"| `{var}` | {motivo} |\n")
        w("\n**NDVI** se conserva en el dataset porque el enunciado lo exige "
          "explicitamente, pero **no entra en el modelo principal**: usa B04, el mismo "
          "canal que alimenta el NDCI, por lo que aporta fuga indirecta. Queda "
          "preparado como analisis de sensibilidad etiquetado.\n")

        w("\n### 3.3 Columnas de trazabilidad (no predictoras)\n\n")
        w("| Columna | Uso previsto |\n|---|---|\n")
        for col, uso in COLUMNAS_TRAZABILIDAD.items():
            w(f"| `{col}` | {uso} |\n")
        w("\nNo se usan como predictoras para evitar que el modelo memorice la "
          "geografia en lugar de aprender una senal espectral generalizable.\n")

        w("\n---\n\n## 4. Artefactos generados\n\n")
        w(f"- `outputs/parte2/data/pixels/` — Parquet particionado por lago y fecha "
          f"({total:,} filas)\n")
        w(f"- `outputs/parte2/data/eda_sample.parquet` — muestra determinista "
          f"({muestra_n:,} filas, semilla {SEED})\n")
        w("- `outputs/parte2/data/dataset_schema.csv`, `dataset_manifest.csv`, "
          "`limpieza_diagnostico.csv`\n")
        w("- `outputs/parte2/target/` — distribuciones global, por lago y por fecha\n")
        w("- `outputs/parte2/eda/` — figuras\n")

    LOGGER.info("  Reporte: %s", ruta.relative_to(ROOT))


# ----------------------------------------------------------------------------
# MODO --validate
# ----------------------------------------------------------------------------
def ejecutar_validate(args) -> int:
    LOGGER.info("=" * 84)
    LOGGER.info("MODO VALIDACION del dataset construido")
    LOGGER.info("=" * 84)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reporte = REPORTS_DIR / "validation_dataset.txt"
    lineas, criticos, avisos = [], [], []

    esperadas = combinaciones_oficiales()
    presentes = []
    for lago, fecha in esperadas:
        if ruta_particion(lago, fecha).exists():
            presentes.append((lago, fecha))
    faltan = sorted(set(esperadas) - set(presentes))

    lineas.append(f"Particiones esperadas : {len(esperadas)}")
    lineas.append(f"Particiones presentes : {len(presentes)}")
    if faltan:
        criticos.append(f"Faltan {len(faltan)} particiones: {faltan}")

    if not presentes:
        criticos.append("No existe ninguna particion. Ejecute --build primero.")
        _escribir_validacion(reporte, lineas, criticos, avisos)
        for c in criticos:
            LOGGER.error("CRITICO: %s", c)
        return 1

    esquema_ref, total, columnas_respuesta = None, 0, [
        f"high_cyano_{int(u)}" for u in UMBRALES_CANDIDATOS]

    for lago, fecha in barra(presentes, len(presentes), "Validando"):
        try:
            df = leer_particion(lago, fecha)
        except Exception as exc:
            criticos.append(f"{lago} {fecha}: particion ilegible ({exc})")
            continue

        total += len(df)
        firma = tuple(sorted(df.columns))
        if esquema_ref is None:
            esquema_ref = firma
        elif firma != esquema_ref:
            criticos.append(f"{lago} {fecha}: esquema distinto al de referencia")

        if len(df) == 0:
            criticos.append(f"{lago} {fecha}: particion vacia")
            continue

        numericas = [c for c in BANDAS + INDICES if c in df.columns]
        bloque = df[numericas]
        if not np.isfinite(bloque.to_numpy(dtype=np.float64)).all():
            criticos.append(f"{lago} {fecha}: hay NaN o infinitos en columnas numericas")
        if (bloque.to_numpy(dtype=np.float64) == NODATA).any():
            criticos.append(f"{lago} {fecha}: quedan valores NoData ({NODATA})")

        if not df["water_mask"].all():
            criticos.append(f"{lago} {fecha}: hay filas con water_mask=False")
        if not df["valid_data"].all():
            criticos.append(f"{lago} {fecha}: hay filas con valid_data=False")

        for banda in BANDAS:
            if (df[banda] <= 0).any():
                criticos.append(f"{lago} {fecha}: reflectancia <= 0 en {banda}")

        n_dup = int(df.duplicated(subset=["row", "col"]).sum())
        if n_dup:
            criticos.append(f"{lago} {fecha}: {n_dup} duplicados (row,col)")

        if str(df["lake"].iloc[0]) != lago or str(df["date"].iloc[0]) != fecha:
            criticos.append(f"{lago} {fecha}: las columnas lake/date no coinciden "
                            "con la particion")

        # Coordenadas dentro del bbox oficial del lago
        from mainlab4 import LAKE_BBOXES
        caja = LAKE_BBOXES[lago]
        fuera = int(((df["longitude"] < caja["west"] - 0.01)
                     | (df["longitude"] > caja["east"] + 0.01)
                     | (df["latitude"] < caja["south"] - 0.01)
                     | (df["latitude"] > caja["north"] + 0.01)).sum())
        if fuera:
            criticos.append(f"{lago} {fecha}: {fuera} filas con coordenadas fuera "
                            "del bounding box oficial")
        if not (df["x_utm"] > 0).all():
            criticos.append(f"{lago} {fecha}: coordenadas UTM no positivas")

        for col in columnas_respuesta:
            if col not in df.columns:
                criticos.append(f"{lago} {fecha}: falta la columna respuesta {col}")
            elif not df[col].isin([0, 1]).all():
                criticos.append(f"{lago} {fecha}: {col} contiene valores distintos de 0/1")

        for umbral in UMBRALES_CANDIDATOS:
            col = f"high_cyano_{int(umbral)}"
            if col in df.columns:
                esperado = (df["chlorophyll"] >= umbral).astype(np.int8)
                if not esperado.equals(df[col].astype(np.int8)):
                    criticos.append(f"{lago} {fecha}: {col} no es coherente con "
                                    "chlorophyll")

        # --- Version de esquema de la particion ---
        vigente, motivo = particion_vigente(lago, fecha)
        if not vigente:
            criticos.append(f"{lago} {fecha}: particion no vigente ({motivo})")

        # --- Identificadores espaciales de 1 km ---
        if "spatial_block_1km" in df.columns:
            bloques = df["spatial_block_1km"].astype(str)
            if bloques.isna().any() or (bloques == "").any():
                criticos.append(f"{lago} {fecha}: bloques espaciales vacios")
            if not bloques.str.startswith(f"{lago}_").all():
                criticos.append(f"{lago} {fecha}: hay bloques que no llevan el prefijo "
                                "del lago (riesgo de colision entre lagos)")
            # El bloque debe reproducirse exactamente desde las coordenadas UTM.
            bc = np.floor(df["x_utm"].to_numpy() / BLOQUE_ESPACIAL_M).astype(np.int32)
            bf = np.floor(df["y_utm"].to_numpy() / BLOQUE_ESPACIAL_M).astype(np.int32)
            recalculado = pd.Series([f"{lago}_{c}_{f}" for c, f in zip(bc, bf)],
                                    index=df.index)
            if not recalculado.equals(bloques):
                criticos.append(f"{lago} {fecha}: spatial_block_1km no coincide con "
                                "las coordenadas UTM")
            # Extension real de cada bloque: debe caber en 1 km.
            ext_x = df.groupby(bloques)["x_utm"].agg(lambda s: s.max() - s.min()).max()
            ext_y = df.groupby(bloques)["y_utm"].agg(lambda s: s.max() - s.min()).max()
            if ext_x >= BLOQUE_ESPACIAL_M or ext_y >= BLOQUE_ESPACIAL_M:
                criticos.append(f"{lago} {fecha}: un bloque abarca "
                                f"{max(ext_x, ext_y):.0f} m, mas de "
                                f"{BLOQUE_ESPACIAL_M:.0f} m")

        # --- Variables temporales coherentes con la fecha de la particion ---
        if {"year", "month", "season"}.issubset(df.columns):
            ts = pd.Timestamp(fecha)
            if int(df["year"].iloc[0]) != ts.year or int(df["month"].iloc[0]) != ts.month:
                criticos.append(f"{lago} {fecha}: year/month no coinciden con la fecha")
            if df["year"].nunique() != 1 or df["month"].nunique() != 1:
                criticos.append(f"{lago} {fecha}: year/month no son constantes")

    lineas.append(f"Filas totales         : {total:,}")

    # Comprobaciones globales
    if esquema_ref:
        obligatorias = (list(COLUMNAS_TRAZABILIDAD) + BANDAS + INDICES
                        + ["water_mask", "valid_data"] + columnas_respuesta)
        ausentes = [c for c in obligatorias if c not in esquema_ref]
        if ausentes:
            criticos.append(f"Faltan columnas obligatorias en el esquema: {ausentes}")
        lineas.append(f"Columnas del esquema  : {len(esquema_ref)}")

    for artefacto in [
        DATA_DIR / "dataset_manifest.csv", DATA_DIR / "dataset_schema.csv",
        DATA_DIR / "eda_sample.parquet",
        TARGET_DIR / "target_distribution_global.csv",
        TARGET_DIR / "target_distribution_by_lake.csv",
        TARGET_DIR / "target_distribution_by_date.csv",
        TARGET_DIR / "threshold_viability_global.csv",
        TARGET_DIR / "threshold_viability_by_lake.csv",
        TARGET_DIR / "threshold_viability_by_date.csv",
        TARGET_DIR / "threshold_viability_by_block.csv",
        TARGET_DIR / "threshold_group_concentration.csv",
        REPORTS_DIR / "preparacion_dataset.md",
        REPORTS_DIR / "threshold_viability.md",
        EDA_DIR / "spatial_blocks.png",
        EDA_DIR / "target_balance.png",
        EDA_DIR / "correlation_matrix.png",
    ]:
        if not artefacto.exists():
            criticos.append(f"Falta el artefacto {artefacto.relative_to(ROOT)}")

    # --- Respuesta principal y sensibilidad ---
    if esquema_ref:
        if TARGET_COLUMN not in esquema_ref:
            criticos.append(f"Falta la respuesta principal {TARGET_COLUMN}")
        else:
            lineas.append(f"Respuesta principal   : {TARGET_COLUMN} "
                          f"(>= {TARGET_THRESHOLD_UG_L:.0f} ug/L)")
        ausentes_sens = [c for c in columnas_respuesta if c not in esquema_ref]
        if ausentes_sens:
            criticos.append(f"Faltan respuestas de sensibilidad: {ausentes_sens}")
        else:
            lineas.append(f"Respuestas totales    : {len(columnas_respuesta)} "
                          f"({', '.join(columnas_respuesta)})")
        ausentes_aux = [c for c in COLUMNAS_AUXILIARES if c not in esquema_ref]
        if ausentes_aux:
            criticos.append(f"Faltan variables auxiliares: {ausentes_aux}")

    # --- Fuga de informacion ---
    prohibidas = set(EXCLUIDAS_POR_FUGA) | set(columnas_respuesta) | set(COLUMNAS_AUXILIARES)
    fuga = [c for c in PREDICTORES_PRINCIPALES if c in prohibidas]
    if fuga:
        criticos.append(f"El conjunto predictor contiene variables prohibidas: {fuga}")
    else:
        lineas.append(f"Predictores sin fuga  : {len(PREDICTORES_PRINCIPALES)} "
                      f"({', '.join(PREDICTORES_PRINCIPALES)})")
    for obligatoria in ["B04", "B05", "NDCI", "chlorophyll", "FAI", "NDVI",
                        "water_mask", "valid_data", "lake", "date", "year",
                        "month", "season", "row", "col", "x_utm", "y_utm",
                        "longitude", "latitude", "spatial_block_1km"]:
        if obligatoria in PREDICTORES_PRINCIPALES:
            criticos.append(f"{obligatoria} NO puede ser predictor principal")

    # --- El notebook usa el dataset real ---
    nb = ROOT / "lab4-2.ipynb"
    if not nb.exists():
        criticos.append("No existe lab4-2.ipynb")
    else:
        texto_nb = nb.read_text(encoding="utf-8")
        usa_real = "outputs/parte2" in texto_nb or "parte2" in texto_nb
        genera_sintetico = "synthetic_bands" in texto_nb or "rng.normal(0.0" in texto_nb
        modo_demo = "demostrativo parametrizado" in texto_nb
        lineas.append(f"Notebook con dataset real: {usa_real}")
        if not usa_real:
            criticos.append("lab4-2.ipynb no carga el dataset real de outputs/parte2")
        if genera_sintetico or modo_demo:
            criticos.append("lab4-2.ipynb conserva el generador sintetico en el flujo "
                            "principal")
        import json as _json
        try:
            nb_json = _json.loads(texto_nb)
            ejecutadas = sum(1 for c in nb_json["cells"]
                             if c["cell_type"] == "code" and c.get("execution_count"))
            errores_nb = sum(1 for c in nb_json["cells"]
                             for o in c.get("outputs", [])
                             if o.get("output_type") == "error")
            lineas.append(f"Notebook ejecutado    : {ejecutadas} celdas, "
                          f"{errores_nb} errores")
            if errores_nb:
                criticos.append(f"lab4-2.ipynb tiene {errores_nb} celdas con error")
            if ejecutadas == 0:
                criticos.append("lab4-2.ipynb no esta ejecutado")
        except Exception as exc:
            criticos.append(f"lab4-2.ipynb ilegible: {exc}")

    _escribir_validacion(reporte, lineas, criticos, avisos)

    LOGGER.info("")
    for linea in lineas:
        LOGGER.info("  %s", linea)
    if avisos:
        LOGGER.warning("")
        for a in avisos:
            LOGGER.warning("AVISO: %s", a)
    if criticos:
        LOGGER.error("")
        for c in criticos:
            LOGGER.error("CRITICO: %s", c)
        LOGGER.error("")
        LOGGER.error("Validacion FALLIDA: %d problemas criticos.", len(criticos))
        LOGGER.info("Reporte: %s", reporte.relative_to(ROOT))
        return 1

    LOGGER.info("")
    LOGGER.info("Validacion CORRECTA. Reporte: %s", reporte.relative_to(ROOT))
    return 0


def _escribir_validacion(ruta, lineas, criticos, avisos) -> None:
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("VALIDACION DEL DATASET DE MACHINE LEARNING - LABORATORIO 4 PARTE 2\n")
        fh.write("=" * 74 + "\n")
        fh.write(f"Generado: {datetime.now().isoformat(timespec='seconds')}\n\n")
        for linea in lineas:
            fh.write(linea + "\n")
        fh.write(f"\nAvisos  : {len(avisos)}\n")
        for a in avisos:
            fh.write(f"  - {a}\n")
        fh.write(f"\nCriticos: {len(criticos)}\n")
        for c in criticos:
            fh.write(f"  - {c}\n")
        fh.write("\nRESULTADO: " + ("FALLIDA" if criticos else "CORRECTA") + "\n")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preparar_dataset_ml.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Preparacion y auditoria del dataset de Machine Learning del\n"
            "Laboratorio 4, Parte 2 (ejercicios 1, 2 y 3).\n\n"
            "Construye una tabla de una fila por pixel de agua a partir de los 22\n"
            "GeoTIFF Sentinel-2 L1C reales, define la variable respuesta binaria y\n"
            "selecciona los predictores sin fuga de informacion.\n\n"
            "NO entrena modelos, NO ejecuta SHAP, NO genera mapas predictivos.\n"
            "NO utiliza datos sinteticos."
        ),
        epilog=(
            "EJEMPLOS\n"
            "  python preparar_dataset_ml.py --dry-run\n"
            "      Inspecciona los 22 rasteres y estima filas, memoria y disco.\n\n"
            "  python preparar_dataset_ml.py --build\n"
            "      Construye el Parquet particionado, las figuras y los reportes.\n\n"
            "  python preparar_dataset_ml.py --validate\n"
            "      Audita el dataset construido.\n\n"
            "CODIGOS DE SALIDA\n"
            "  0  correcto\n"
            "  1  faltan particiones o hay problemas criticos\n"
            "  2  error de entrada (rasteres ausentes o invalidos)\n"
        ),
    )
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--dry-run", action="store_true",
                      help="Inspecciona los rasteres y estima el dataset sin escribirlo.")
    modo.add_argument("--build", action="store_true",
                      help="Construye el dataset, las estadisticas y los reportes.")
    modo.add_argument("--validate", action="store_true",
                      help="Audita el dataset ya construido.")
    parser.add_argument("--lake", metavar="NOMBRE", default=None,
                        help="Procesa solo un lago (Atitlan o Amatitlan).")
    parser.add_argument("--date", metavar="AAAA-MM-DD", default=None,
                        help="Procesa solo una fecha oficial.")
    parser.add_argument("--force", action="store_true",
                        help="Reconstruye las particiones aunque ya existan.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Muestra mensajes de depuracion detallados.")
    return parser


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    configurar_logging(args.verbose)
    try:
        if args.dry_run:
            return ejecutar_dry_run(args)
        if args.build:
            return ejecutar_build(args)
        if args.validate:
            return ejecutar_validate(args)
    except KeyboardInterrupt:
        LOGGER.warning("\nInterrumpido por la usuaria.")
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        LOGGER.exception("Error no controlado: %s: %s", type(exc).__name__, exc)
        return 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
