"""
Adquisicion y validacion de rasters reales Sentinel-2 L1C para el Laboratorio 4.

Automatiza la consulta, descarga reanudable y validacion de las 22 combinaciones
oficiales lago-fecha usando Copernicus Data Space / openEO.

Modos:
    python descargar_rasters.py --dry-run
    python descargar_rasters.py --download
    python descargar_rasters.py --validate

Este script NO genera datos sinteticos bajo ninguna circunstancia.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Reutilizacion de la configuracion oficial declarada en mainlab4.py
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mainlab4 import (  # noqa: E402
    S2_L1C_COLLECTION,
    CYANO_REQUIRED_BANDS,
    LAKE_BBOXES,
    OFFICIAL_DATES,
    OFFICIAL_CLOUD_COVER,
    date_to_temporal_extent,
)

# ----------------------------------------------------------------------------
# Parametros de adquisicion
# ----------------------------------------------------------------------------
BACKEND_URL = "https://openeo.dataspace.copernicus.eu"
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# EPSG:32615 = WGS 84 / UTM zona 15N. Ambos lagos caen dentro de la zona 15N.
# Descargar directamente en este CRS deja los rasters listos para la cuadricula
# espacial de 1 km x 1 km que exige la Parte 2 sin reproyecciones posteriores.
TARGET_EPSG = 32615

# 20 m es la resolucion nativa de B05, B07, B8A, B11 y B12. Remuestrear todo a
# 20 m evita atribuir detalle de 10 m a bandas que no lo tienen. Las bandas de
# 10 m (B02, B03, B04, B08) se degradan a 20 m, que es una perdida real de
# detalle pero honesta; lo contrario inventaria informacion inexistente.
TARGET_RESOLUTION_M = 20.0
RESAMPLE_METHOD = "near"

NODATA_VALUE = -9999.0
OUTPUT_DTYPE = "float32"

# --- Filtro de nubosidad -----------------------------------------------------
# OFFICIAL_CLOUD_COVER contiene la nubosidad medida SOBRE EL AOI (el recorte del
# lago), mientras que la propiedad eo:cloud_cover del catalogo describe el TILE
# COMPLETO de 110 km x 110 km. Son magnitudes de granularidad distinta y no son
# comparables. Filtrar tiles con el umbral del AOI expulsaba escenas validas y
# dejaba huecos espaciales, porque cada lago se cubre con DOS tiles:
#   Atitlan   -> T15PXS (~98 % del bbox) + T15PYS (~31 %)
#   Amatitlan -> T15PYR (~97 % del bbox) + T15PYS (~70 %)
# Por eso el filtro por tile queda DESACTIVADO por omision: las fechas oficiales
# ya fueron elegidas porque el agua del lago esta despejada, y el recorte al bbox
# descarta la nubosidad que exista en el resto del tile. Puede reactivarse de
# forma explicita con --max-cloud.
MAX_CLOUD_COVER_DEFAULT = None

# --- Reintentos ante fallos transitorios -------------------------------------
MAX_REINTENTOS = 5
ESPERA_BASE_S = 5.0
ESPERA_MAX_S = 300.0
ESTADOS_TRANSITORIOS = {408, 429, 500, 502, 503, 504}

# Porcentaje minimo de pixeles validos para aceptar un GeoTIFF como completo.
# Un archivo por debajo de este umbral se considera incompleto y se vuelve a
# descargar, en lugar de darse por bueno con huecos.
MIN_VALIDOS_PCT_DEFAULT = 95.0

GEOJSON_FILES = {
    "Atitlan": ROOT / "Lago_Atitlan.geojson",
    "Amatitlan": ROOT / "Lago_Amatitlan.geojson",
}

RASTER_DIR = ROOT / "outputs" / "rasters"
MANIFEST_DIR = ROOT / "outputs" / "manifests"
LOG_DIR = ROOT / "outputs" / "logs"

# Bandas de mascara que se consultan en el backend antes de asumir nada.
CANDIDATE_MASK_BANDS = ["CLM", "CLP", "dataMask", "SCL", "QA60", "SNW"]

LOGGER = logging.getLogger("descargar_rasters")


# ----------------------------------------------------------------------------
# Utilidades generales
# ----------------------------------------------------------------------------
def configurar_logging(verbose: bool = False) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"adquisicion_{marca}.log"

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()

    consola = logging.StreamHandler(sys.stdout)
    consola.setLevel(logging.DEBUG if verbose else logging.INFO)
    consola.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(consola)

    archivo = logging.FileHandler(log_path, encoding="utf-8")
    archivo.setLevel(logging.DEBUG)
    archivo.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(archivo)

    return log_path


def combinaciones_oficiales(lago=None, fecha=None):
    """Devuelve la lista de tuplas (lago, fecha) oficiales, con filtros opcionales."""
    pares = []
    for nombre_lago in sorted(OFFICIAL_DATES):
        for f in OFFICIAL_DATES[nombre_lago]:
            pares.append((nombre_lago, f))

    if lago:
        coincidencias = [l for l in OFFICIAL_DATES if l.lower() == lago.lower()]
        if not coincidencias:
            raise SystemExit(
                f"ERROR: lago desconocido '{lago}'. Opciones: {', '.join(sorted(OFFICIAL_DATES))}"
            )
        pares = [p for p in pares if p[0] == coincidencias[0]]

    if fecha:
        pares = [p for p in pares if p[1] == fecha]
        if not pares:
            raise SystemExit(
                f"ERROR: la fecha '{fecha}' no es oficial para la seleccion indicada."
            )

    return pares


def ruta_destino(lago: str, fecha: str) -> Path:
    return RASTER_DIR / lago / f"{lago}_{fecha}.tif"


def configurar_reintentos_http(conexion) -> None:
    """
    Instala reintentos a nivel de transporte en la sesion HTTP de openEO.

    urllib3 respeta de forma nativa la cabecera Retry-After que envia el backend
    en las respuestas 429, y aplica espera exponencial en 502/503/504. Esta es la
    unica capa que puede leer esa cabecera, porque las excepciones de openEO no
    la exponen.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    politica = Retry(
        total=MAX_REINTENTOS,
        connect=MAX_REINTENTOS,
        read=MAX_REINTENTOS,
        status=MAX_REINTENTOS,
        backoff_factor=2.0,
        status_forcelist=sorted(ESTADOS_TRANSITORIOS),
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adaptador = HTTPAdapter(max_retries=politica)
    conexion.session.mount("https://", adaptador)
    conexion.session.mount("http://", adaptador)
    LOGGER.debug(
        "Reintentos HTTP activos: %d intentos, backoff x2, Retry-After respetado, estados %s",
        MAX_REINTENTOS, sorted(ESTADOS_TRANSITORIOS),
    )


def clasificar_error(exc) -> tuple[bool, str]:
    """
    Determina si un error es transitorio (merece reintento) o permanente.

    Devuelve (es_transitorio, etiqueta_legible).
    """
    import requests

    codigo = getattr(exc, "http_status_code", None)
    codigo_api = getattr(exc, "code", None)  # p. ej. 'NoDataAvailable'

    if codigo in ESTADOS_TRANSITORIOS:
        return True, f"HTTP {codigo} transitorio"

    if isinstance(exc, (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.ChunkedEncodingError)):
        return True, f"fallo de red ({type(exc).__name__})"

    if codigo_api == "NoDataAvailable":
        return False, "NoDataAvailable (no hay escenas que cumplan los filtros)"

    if codigo is not None and 400 <= codigo < 500:
        return False, f"HTTP {codigo} permanente"

    # Mensajes planos sin codigo estructurado: se detecta por texto.
    texto = str(exc)
    for estado in ESTADOS_TRANSITORIOS:
        if f"[{estado}]" in texto:
            return True, f"HTTP {estado} transitorio"

    return False, f"{type(exc).__name__}"


def espera_reintento(intento: int) -> float:
    """Espera exponencial acotada: 5 s, 10 s, 20 s, 40 s, 80 s... hasta el tope."""
    return min(ESPERA_BASE_S * (2 ** intento), ESPERA_MAX_S)


def ejecutar_con_reintentos(funcion, descripcion: str, max_intentos: int = MAX_REINTENTOS):
    """
    Ejecuta `funcion` reintentando solo ante errores transitorios.

    Complementa a los reintentos de transporte: cubre los casos en que openEO
    convierte la respuesta en excepcion antes de que urllib3 pueda reintentar.
    Los errores permanentes se propagan de inmediato, sin gastar intentos.
    """
    ultimo = None
    for intento in range(max_intentos):
        try:
            return funcion()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            ultimo = exc
            transitorio, etiqueta = clasificar_error(exc)
            if not transitorio:
                raise
            if intento == max_intentos - 1:
                break
            pausa = espera_reintento(intento)
            LOGGER.warning(
                "  [REINTENTO %d/%d] %s: %s. Esperando %.0f s...",
                intento + 1, max_intentos - 1, descripcion, etiqueta, pausa,
            )
            time.sleep(pausa)

    raise RuntimeError(
        f"{descripcion}: agotados {max_intentos} intentos. Ultimo error: "
        f"{type(ultimo).__name__}: {ultimo}"
    ) from ultimo


def barra_progreso(iterable, total, descripcion):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=descripcion, unit="escena", ncols=88)
    except ImportError:  # pragma: no cover - tqdm esta instalado en este entorno
        return iterable


# ----------------------------------------------------------------------------
# Validacion de los GeoJSON de los lagos
# ----------------------------------------------------------------------------
def validar_geojson(lago: str, ruta: Path) -> dict:
    """
    Valida un GeoJSON de lago: existencia, tipo de geometria, validez topologica,
    rango de coordenadas EPSG:4326 e interseccion con el bounding box oficial.

    Devuelve un diccionario con la geometria y el diagnostico. Lanza excepcion si
    el archivo no puede usarse.
    """
    from shapely.geometry import shape, box
    from shapely.validation import explain_validity

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el GeoJSON de {lago}: {ruta}")

    with open(ruta, encoding="utf-8") as fh:
        contenido = json.load(fh)

    tipo_raiz = contenido.get("type")
    if tipo_raiz == "FeatureCollection":
        features = contenido.get("features", [])
        if not features:
            raise ValueError(f"{ruta.name}: FeatureCollection sin features.")
        geometrias = [shape(f["geometry"]) for f in features]
    elif tipo_raiz == "Feature":
        geometrias = [shape(contenido["geometry"])]
    else:
        geometrias = [shape(contenido)]

    from shapely.ops import unary_union

    geom = unary_union(geometrias) if len(geometrias) > 1 else geometrias[0]

    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"{ruta.name}: la geometria es {geom.geom_type}; se requiere Polygon o MultiPolygon."
        )

    if not geom.is_valid:
        raise ValueError(f"{ruta.name}: geometria invalida -> {explain_validity(geom)}")

    minx, miny, maxx, maxy = geom.bounds
    if not (-180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90):
        raise ValueError(
            f"{ruta.name}: coordenadas fuera del rango EPSG:4326 -> {geom.bounds}"
        )

    bbox_oficial = LAKE_BBOXES[lago]
    caja = box(
        bbox_oficial["west"], bbox_oficial["south"],
        bbox_oficial["east"], bbox_oficial["north"],
    )
    if not geom.intersects(caja):
        raise ValueError(
            f"{ruta.name}: la geometria NO intersecta el bounding box oficial de {lago}."
        )

    interseccion = geom.intersection(caja).area
    cobertura = 100.0 * interseccion / geom.area if geom.area else 0.0

    # Un poligono real de lago tiene muchos vertices y area menor que su bbox.
    n_vertices = len(geom.exterior.coords) if geom.geom_type == "Polygon" else sum(
        len(p.exterior.coords) for p in geom.geoms
    )
    razon_area = geom.area / caja.area if caja.area else 0.0
    es_rectangulo = geom.geom_type == "Polygon" and n_vertices <= 5 and razon_area > 0.999

    return {
        "lago": lago,
        "ruta": ruta,
        "geometria": geom,
        "tipo": geom.geom_type,
        "n_vertices": n_vertices,
        "bounds": geom.bounds,
        "cobertura_bbox_pct": cobertura,
        "razon_area_vs_bbox": razon_area,
        "es_bounding_box": es_rectangulo,
    }


def cargar_y_validar_geometrias() -> dict:
    resultados = {}
    LOGGER.info("Validando GeoJSON de los lagos")
    for lago, ruta in GEOJSON_FILES.items():
        info = validar_geojson(lago, ruta)
        resultados[lago] = info
        LOGGER.info(
            "  %-10s %-12s vertices=%-4d area/bbox=%.4f  interseccion_bbox=%.1f%%",
            lago, info["tipo"], info["n_vertices"],
            info["razon_area_vs_bbox"], info["cobertura_bbox_pct"],
        )
        if info["es_bounding_box"]:
            LOGGER.warning(
                "  AVISO %s: la geometria es un RECTANGULO identico al bounding box oficial, "
                "no el contorno real del lago. El recorte no eliminara tierra circundante; "
                "para aislar agua sera indispensable la mascara WBI de mainlab4.py.",
                lago,
            )
    return resultados


# ----------------------------------------------------------------------------
# Inspeccion de metadatos del backend (bandas de mascara realmente disponibles)
# ----------------------------------------------------------------------------
def inspeccionar_coleccion(conexion=None) -> dict:
    """
    Consulta los metadatos publicos de SENTINEL2_L1C y determina que bandas de
    mascara existen realmente. No se asume ninguna; lo que no este, se registra
    como limitacion.
    """
    import openeo

    if conexion is None:
        conexion = openeo.connect(BACKEND_URL)

    metadatos = conexion.describe_collection(S2_L1C_COLLECTION)
    disponibles = metadatos.get("cube:dimensions", {}).get("bands", {}).get("values", [])

    mascaras_presentes = [b for b in CANDIDATE_MASK_BANDS if b in disponibles]
    mascaras_ausentes = [b for b in CANDIDATE_MASK_BANDS if b not in disponibles]

    faltantes_requeridas = [b for b in CYANO_REQUIRED_BANDS if b not in disponibles]
    if faltantes_requeridas:
        raise RuntimeError(
            f"La coleccion {S2_L1C_COLLECTION} no expone las bandas requeridas: "
            f"{faltantes_requeridas}. Bandas disponibles: {disponibles}"
        )

    LOGGER.info("Coleccion: %s", S2_L1C_COLLECTION)
    LOGGER.info("  Bandas expuestas por el backend: %d", len(disponibles))
    LOGGER.info("  Bandas requeridas presentes: %s", ", ".join(CYANO_REQUIRED_BANDS))
    if mascaras_presentes:
        LOGGER.info("  Bandas de mascara disponibles: %s", ", ".join(mascaras_presentes))
    else:
        LOGGER.warning(
            "  LIMITACION: %s NO expone ninguna banda de mascara (%s). "
            "No se fabricara ninguna mascara sustituta. El unico control de nubes "
            "disponible es el filtro de escena eo:cloud_cover con los valores "
            "oficiales del laboratorio.",
            S2_L1C_COLLECTION, ", ".join(mascaras_ausentes),
        )

    return {
        "conexion": conexion,
        "bandas_disponibles": disponibles,
        "mascaras_presentes": mascaras_presentes,
        "mascaras_ausentes": mascaras_ausentes,
        "bandas_a_solicitar": list(CYANO_REQUIRED_BANDS) + mascaras_presentes,
    }


# ----------------------------------------------------------------------------
# Consulta de disponibilidad real de escenas (catalogo OData, sin autenticacion)
# ----------------------------------------------------------------------------
def consultar_disponibilidad(lago: str, fecha: str, geometria) -> dict:
    """
    Pregunta al catalogo de Copernicus si existe al menos un producto L1C que
    intersecte el lago en esa fecha. Es una consulta de metadatos: no descarga
    imagenes ni consume creditos de procesamiento.
    """
    import requests

    minx, miny, maxx, maxy = geometria.bounds
    wkt = (
        f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},"
        f"{minx} {maxy},{minx} {miny}))"
    )
    inicio, fin = date_to_temporal_extent(fecha)

    filtro = (
        "Collection/Name eq 'SENTINEL-2' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
        f"ContentDate/Start gt {inicio}T00:00:00.000Z and "
        f"ContentDate/Start lt {fin}T00:00:00.000Z"
    )

    try:
        respuesta = requests.get(
            ODATA_URL, params={"$filter": filtro, "$top": 50}, timeout=90
        )
        respuesta.raise_for_status()
        productos = respuesta.json().get("value", [])
    except Exception as exc:
        return {
            "estado": "ERROR_CONSULTA",
            "n_productos_l1c": 0,
            "productos": [],
            "detalle": f"{type(exc).__name__}: {exc}",
        }

    l1c = [p for p in productos if "MSIL1C" in p.get("Name", "")]
    return {
        "estado": "DISPONIBLE" if l1c else "SIN_ESCENA_L1C",
        "n_productos_l1c": len(l1c),
        "productos": [p.get("Name", "") for p in l1c],
        "detalle": "",
    }


# ----------------------------------------------------------------------------
# Construccion del cubo openEO
# ----------------------------------------------------------------------------
def construir_cubo(conexion, lago: str, fecha: str, geometria, bandas,
                   max_cloud=MAX_CLOUD_COVER_DEFAULT):
    """Cubo Sentinel-2 L1C recortado, reproyectado y remuestreado a resolucion comun."""
    minx, miny, maxx, maxy = geometria.bounds
    extension_espacial = {
        "west": minx, "south": miny, "east": maxx, "north": maxy, "crs": "EPSG:4326",
    }

    # Sin filtro por tile se cargan TODOS los tiles del dia que tocan el lago y
    # openEO los mosaica, lo que garantiza cobertura espacial completa del bbox.
    propiedades = None
    if max_cloud is not None:
        propiedades = {"eo:cloud_cover": lambda v: v <= max_cloud}

    cubo = conexion.load_collection(
        S2_L1C_COLLECTION,
        spatial_extent=extension_espacial,
        temporal_extent=date_to_temporal_extent(fecha),
        bands=bandas,
        properties=propiedades,
    )

    # Todas las bandas quedan en el mismo CRS, resolucion, grilla y extent.
    cubo = cubo.resample_spatial(
        resolution=TARGET_RESOLUTION_M,
        projection=TARGET_EPSG,
        method=RESAMPLE_METHOD,
    )

    # Recorte por la geometria del lago. Con un GeoJSON rectangular esto no
    # elimina nada adicional, pero funciona correctamente si mas adelante se
    # sustituye por el contorno real del lago.
    cubo = cubo.mask_polygon(geometria)

    # Colapsa la dimension temporal para obtener un GeoTIFF plano. Casi todas las
    # fechas tienen una sola toma, pero algunas (p. ej. Atitlan 2025-07-17) tienen
    # dos orbitas distintas el mismo dia. Se reduce con "min" porque las nubes son
    # brillantes: quedarse con el valor mas bajo escoge la observacion mas despejada
    # en lugar de la mas contaminada, que es lo que haria "max".
    cubo = cubo.reduce_dimension(dimension="t", reducer="min")

    return cubo


# ----------------------------------------------------------------------------
# Validacion y normalizacion de un GeoTIFF descargado
# ----------------------------------------------------------------------------
def inspeccionar_raster(ruta: Path) -> dict:
    """Extrae metadatos y estadisticas de un GeoTIFF. No modifica el archivo."""
    import rasterio

    info = {"ruta": str(ruta), "tamano_bytes": ruta.stat().st_size, "error": ""}
    with rasterio.open(ruta) as src:
        info.update({
            "n_bandas": src.count,
            "nombres_bandas": list(src.descriptions),
            "ancho": src.width,
            "alto": src.height,
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6],
            "res_x": src.res[0],
            "res_y": src.res[1],
            "bounds": list(src.bounds),
            "dtype": src.dtypes[0],
            "dtypes_todas": list(src.dtypes),
            "nodata": src.nodata,
            "tags": dict(src.tags()),
        })

        datos = src.read(masked=True)
        total = datos[0].size if src.count else 0
        validos_por_banda = []
        estadisticas = []
        for i in range(src.count):
            banda = datos[i]
            n_validos = int(banda.count())
            validos_por_banda.append(n_validos)
            if n_validos:
                estadisticas.append({
                    "banda": (src.descriptions[i] or f"banda_{i+1}"),
                    "min": float(banda.min()),
                    "max": float(banda.max()),
                    "media": float(banda.mean()),
                    "validos_pct": 100.0 * n_validos / total if total else 0.0,
                })
            else:
                estadisticas.append({
                    "banda": (src.descriptions[i] or f"banda_{i+1}"),
                    "min": None, "max": None, "media": None, "validos_pct": 0.0,
                })

        info["pixeles_totales"] = total
        info["pixeles_validos_min"] = min(validos_por_banda) if validos_por_banda else 0
        info["validos_pct"] = (
            100.0 * min(validos_por_banda) / total if total and validos_por_banda else 0.0
        )
        info["estadisticas"] = estadisticas
        # Alineacion interna: en un GeoTIFF todas las bandas comparten grilla por
        # construccion; se verifica la consistencia de dtype y la presencia de CRS.
        info["alineacion_interna_ok"] = (
            len(set(src.dtypes)) == 1 and src.crs is not None and src.width > 0 and src.height > 0
        )
    return info


def es_raster_valido(ruta: Path, n_bandas_esperado: int,
                     min_validos_pct: float = MIN_VALIDOS_PCT_DEFAULT) -> tuple[bool, str]:
    """
    Comprueba si un GeoTIFF esta completo y es utilizable.

    Ademas de la integridad estructural exige cobertura espacial: un archivo con
    huecos amplios (tiles ausentes) es tecnicamente legible pero inservible para
    el analisis, y debe volver a descargarse.
    """
    try:
        info = inspeccionar_raster(ruta)
    except Exception as exc:
        return False, f"no se puede abrir ({type(exc).__name__}: {exc})"

    if info["n_bandas"] != n_bandas_esperado:
        return False, f"tiene {info['n_bandas']} bandas, se esperaban {n_bandas_esperado}"
    if not info["crs"]:
        return False, "sin CRS"
    if info["ancho"] <= 0 or info["alto"] <= 0:
        return False, "dimensiones nulas"
    if info["validos_pct"] <= 0.0:
        return False, "sin pixeles validos"
    if info["validos_pct"] < min_validos_pct:
        return False, (
            f"cobertura espacial incompleta: {info['validos_pct']:.2f}% de pixeles "
            f"validos, se exige >= {min_validos_pct:.1f}% (probable tile faltante)"
        )
    return True, ""


def normalizar_geotiff(origen: Path, destino: Path, lago: str, fecha: str,
                       bandas: list, productos: list) -> None:
    """
    Reescribe el GeoTIFF descargado conservando georreferenciacion y anadiendo
    metadatos explicitos: nombres de banda, NoData, fecha, lago, coleccion,
    resolucion e identificadores de adquisicion.
    """
    import rasterio

    with rasterio.open(origen) as src:
        perfil = src.profile.copy()
        datos = src.read().astype(np.float32)
        # Los pixeles enmascarados por openEO llegan como NaN; se fijan al
        # centinela NoData declarado para que sea explicito en el archivo.
        mascara = src.read_masks()
        datos[mascara == 0] = NODATA_VALUE
        datos = np.where(np.isfinite(datos), datos, NODATA_VALUE)

    perfil.update({
        "driver": "GTiff",
        "dtype": OUTPUT_DTYPE,
        "nodata": NODATA_VALUE,
        "compress": "lzw",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "count": datos.shape[0],
    })

    etiquetas = {
        "lago": lago,
        "fecha": fecha,
        "coleccion": S2_L1C_COLLECTION,
        "fuente": "Copernicus Data Space Ecosystem / openEO",
        "crs_destino": f"EPSG:{TARGET_EPSG}",
        "resolucion_m": str(TARGET_RESOLUTION_M),
        "metodo_remuestreo": RESAMPLE_METHOD,
        "nubosidad_oficial_pct": str(OFFICIAL_CLOUD_COVER[lago][fecha]),
        "bandas": ",".join(bandas),
        "identificadores_adquisicion": ";".join(productos) if productos else "no_registrado",
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "nodata": str(NODATA_VALUE),
    }

    destino.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destino, "w", **perfil) as dst:
        dst.write(datos)
        for indice, nombre in enumerate(bandas[: datos.shape[0]], start=1):
            dst.set_band_description(indice, nombre)
        dst.update_tags(**etiquetas)


# ----------------------------------------------------------------------------
# MODO --dry-run
# ----------------------------------------------------------------------------
def ejecutar_dry_run(args) -> int:
    LOGGER.info("=" * 84)
    LOGGER.info("MODO DRY-RUN: preparacion y verificacion de solicitudes (sin descargar)")
    LOGGER.info("=" * 84)

    geometrias = cargar_y_validar_geometrias()

    LOGGER.info("")
    try:
        coleccion = inspeccionar_coleccion()
    except Exception as exc:
        LOGGER.error("No se pudieron inspeccionar los metadatos del backend: %s", exc)
        return 2

    bandas = coleccion["bandas_a_solicitar"]
    pares = combinaciones_oficiales(args.lake, args.date)

    LOGGER.info("")
    LOGGER.info("Combinaciones a preparar: %d", len(pares))
    LOGGER.info("Resolucion comun: %.0f m | CRS destino: EPSG:%d", TARGET_RESOLUTION_M, TARGET_EPSG)
    LOGGER.info("")

    filas = []
    problemas = 0
    for lago, fecha in barra_progreso(pares, len(pares), "Consultando catalogo"):
        info_geom = geometrias[lago]
        disponibilidad = consultar_disponibilidad(lago, fecha, info_geom["geometria"])
        destino = ruta_destino(lago, fecha)

        if destino.exists():
            ok, _ = es_raster_valido(destino, len(bandas), args.min_validos)
            estado_local = "YA_DESCARGADO" if ok else "LOCAL_INCOMPLETO"
        else:
            estado_local = "PENDIENTE"

        if disponibilidad["estado"] != "DISPONIBLE":
            problemas += 1

        filas.append({
            "lago": lago,
            "fecha": fecha,
            "coleccion": S2_L1C_COLLECTION,
            "geometria": info_geom["ruta"].name,
            "tipo_geometria": info_geom["tipo"],
            "geometria_es_bbox": info_geom["es_bounding_box"],
            "bandas": ",".join(bandas),
            "n_bandas": len(bandas),
            "nubosidad_oficial_pct": OFFICIAL_CLOUD_COVER[lago][fecha],
            "resolucion_m": TARGET_RESOLUTION_M,
            "crs_destino": f"EPSG:{TARGET_EPSG}",
            "estado_catalogo": disponibilidad["estado"],
            "n_escenas_l1c": disponibilidad["n_productos_l1c"],
            "identificadores": ";".join(disponibilidad["productos"]),
            "estado_local": estado_local,
            "archivo_destino": str(destino.relative_to(ROOT)),
            "detalle": disponibilidad["detalle"],
        })

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifiesto = MANIFEST_DIR / "dry_run_manifest.csv"
    with open(manifiesto, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    LOGGER.info("")
    LOGGER.info("%-11s %-12s %-8s %-16s %-15s %s", "LAGO", "FECHA", "NUBES%", "CATALOGO", "LOCAL", "ESCENAS")
    LOGGER.info("-" * 84)
    for fila in filas:
        LOGGER.info(
            "%-11s %-12s %-8.2f %-16s %-15s %d",
            fila["lago"], fila["fecha"], fila["nubosidad_oficial_pct"],
            fila["estado_catalogo"], fila["estado_local"], fila["n_escenas_l1c"],
        )

    LOGGER.info("")
    LOGGER.info("Manifiesto generado: %s", manifiesto.relative_to(ROOT))

    # Verificacion de completitud oficial (solo si no hubo filtros)
    codigo = 0
    if not args.lake and not args.date:
        LOGGER.info("")
        LOGGER.info("Verificacion de completitud:")
        for lago in sorted(OFFICIAL_DATES):
            n = sum(1 for f in filas if f["lago"] == lago)
            marca = "OK" if n == 11 else "FALTAN"
            LOGGER.info("  %-11s %2d/11 fechas  [%s]", lago, n, marca)
            if n != 11:
                codigo = 1
        total = len(filas)
        LOGGER.info("  %-11s %2d/22 combinaciones  [%s]", "TOTAL", total, "OK" if total == 22 else "FALTAN")
        if total != 22:
            codigo = 1

    if problemas:
        LOGGER.error("")
        LOGGER.error("%d combinacion(es) sin escena L1C disponible o con error de consulta.", problemas)
        codigo = 1

    if codigo == 0:
        LOGGER.info("")
        LOGGER.info("DRY-RUN correcto. Ninguna descarga fue realizada.")
    return codigo


# ----------------------------------------------------------------------------
# MODO --download
# ----------------------------------------------------------------------------
def ejecutar_download(args) -> int:
    LOGGER.info("=" * 84)
    LOGGER.info("MODO DESCARGA: adquisicion de rasters Sentinel-2 L1C reales")
    LOGGER.info("=" * 84)

    import openeo

    geometrias = cargar_y_validar_geometrias()

    LOGGER.info("")
    LOGGER.info("Iniciando autenticacion OIDC contra %s", BACKEND_URL)
    LOGGER.info("Se mostrara un enlace y un codigo. Abra el enlace en su navegador,")
    LOGGER.info("inicie sesion con su cuenta de Copernicus y autorice el acceso.")
    LOGGER.info("Este script no solicita ni almacena usuario, contrasena ni tokens.")
    LOGGER.info("")

    try:
        conexion = openeo.connect(BACKEND_URL)
        configurar_reintentos_http(conexion)
        conexion.authenticate_oidc()
        LOGGER.info("Autenticacion completada.")
    except Exception as exc:
        LOGGER.error("Fallo la autenticacion OIDC: %s: %s", type(exc).__name__, exc)
        LOGGER.error("Sin autenticacion no es posible descargar datos reales. Se detiene.")
        return 2

    try:
        coleccion = inspeccionar_coleccion(conexion)
    except Exception as exc:
        LOGGER.error("No se pudieron inspeccionar los metadatos: %s", exc)
        return 2

    bandas = coleccion["bandas_a_solicitar"]
    pares = combinaciones_oficiales(args.lake, args.date)

    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("")
    LOGGER.info("Combinaciones a procesar: %d", len(pares))
    if args.max_cloud is None:
        LOGGER.info("Filtro de nubosidad por tile: DESACTIVADO (se cargan todos los")
        LOGGER.info("  tiles del dia y se mosaican; el recorte al bbox descarta el resto).")
    else:
        LOGGER.info("Filtro de nubosidad por tile: eo:cloud_cover <= %.2f%%", args.max_cloud)
    LOGGER.info("Cobertura minima exigida: %.1f%% de pixeles validos", args.min_validos)
    LOGGER.info("")

    exitos, omitidos, errores = [], [], []

    for lago, fecha in barra_progreso(pares, len(pares), "Descargando"):
        destino = ruta_destino(lago, fecha)
        etiqueta = f"{lago} {fecha}"

        # Reanudacion: no se sobrescribe un archivo ya valido.
        if destino.exists():
            ok, motivo = es_raster_valido(destino, len(bandas), args.min_validos)
            if ok:
                LOGGER.info("[OMITIR] %s ya descargado y valido.", etiqueta)
                omitidos.append(etiqueta)
                continue
            LOGGER.warning("[REEMPLAZAR] %s existe pero es invalido (%s).", etiqueta, motivo)

        temporal = destino.parent / f".{destino.stem}.descargando.tif"
        temporal.parent.mkdir(parents=True, exist_ok=True)
        if temporal.exists():
            temporal.unlink()

        try:
            disponibilidad = ejecutar_con_reintentos(
                lambda: consultar_disponibilidad(lago, fecha, geometrias[lago]["geometria"]),
                f"consulta de catalogo {etiqueta}",
            )
            if disponibilidad["estado"] != "DISPONIBLE":
                raise RuntimeError(
                    f"el catalogo no reporta escenas L1C ({disponibilidad['estado']})"
                )

            cubo = construir_cubo(conexion, lago, fecha, geometrias[lago]["geometria"],
                                  bandas, max_cloud=args.max_cloud)

            inicio = time.time()
            ejecutar_con_reintentos(
                lambda: cubo.download(str(temporal), format="GTiff"),
                f"descarga {etiqueta}",
            )
            duracion = time.time() - inicio

            ok, motivo = es_raster_valido(temporal, len(bandas), args.min_validos)
            if not ok:
                raise RuntimeError(f"la descarga no supero la validacion ({motivo})")

            normalizar_geotiff(temporal, destino, lago, fecha, bandas,
                               disponibilidad["productos"])

            ok_final, motivo_final = es_raster_valido(destino, len(bandas), args.min_validos)
            if not ok_final:
                if destino.exists():
                    destino.unlink()
                raise RuntimeError(f"el archivo final no es valido ({motivo_final})")

            temporal.unlink(missing_ok=True)
            cobertura = inspeccionar_raster(destino)["validos_pct"]
            LOGGER.info(
                "[OK] %s -> %s (%.1f s, %.1f MB, %.2f%% valido)",
                etiqueta, destino.relative_to(ROOT), duracion,
                destino.stat().st_size / 1e6, cobertura,
            )
            exitos.append(etiqueta)

        except KeyboardInterrupt:
            LOGGER.warning("")
            LOGGER.warning("Interrumpido por la usuaria. El progreso valido se conserva;")
            LOGGER.warning("vuelva a ejecutar --download para reanudar.")
            temporal.unlink(missing_ok=True)
            return 130
        except Exception as exc:
            temporal.unlink(missing_ok=True)
            LOGGER.error("[ERROR] %s: %s: %s", etiqueta, type(exc).__name__, exc)
            errores.append((etiqueta, f"{type(exc).__name__}: {exc}"))

    LOGGER.info("")
    LOGGER.info("=" * 84)
    LOGGER.info("RESUMEN DE DESCARGA")
    LOGGER.info("  Descargados en esta ejecucion: %d", len(exitos))
    LOGGER.info("  Omitidos (ya validos):         %d", len(omitidos))
    LOGGER.info("  Errores:                       %d", len(errores))
    if errores:
        LOGGER.error("")
        LOGGER.error("Combinaciones con error:")
        for etiqueta, detalle in errores:
            LOGGER.error("  - %s -> %s", etiqueta, detalle)
        LOGGER.error("")
        LOGGER.error("No se genero ningun dato sustituto ni sintetico para estas fechas.")
        LOGGER.error("Reintente una combinacion concreta con --lake y --date.")
        return 1

    LOGGER.info("")
    LOGGER.info("Ejecute ahora la validacion: python descargar_rasters.py --validate")
    return 0


# ----------------------------------------------------------------------------
# MODO --validate
# ----------------------------------------------------------------------------
def ejecutar_validate(args) -> int:
    LOGGER.info("=" * 84)
    LOGGER.info("MODO VALIDACION: revision de los GeoTIFF presentes en outputs/rasters/")
    LOGGER.info("=" * 84)

    pares = combinaciones_oficiales(args.lake, args.date)
    esperados = {(l, f): ruta_destino(l, f) for l, f in pares}

    archivos = sorted(RASTER_DIR.rglob("*.tif")) if RASTER_DIR.exists() else []
    archivos = [a for a in archivos if not a.name.startswith(".")]

    if not archivos:
        LOGGER.error("")
        LOGGER.error("No se encontro ningun GeoTIFF en %s", RASTER_DIR.relative_to(ROOT))
        LOGGER.error("Cobertura: 0/%d combinaciones oficiales.", len(esperados))
        LOGGER.error("Ejecute primero: python descargar_rasters.py --download")
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        reporte = MANIFEST_DIR / "validation_report.txt"
        with open(reporte, "w", encoding="utf-8") as fh:
            fh.write("REPORTE DE VALIDACION DE RASTERES\n")
            fh.write(f"Generado: {datetime.now().isoformat(timespec='seconds')}\n\n")
            fh.write("No se encontro ningun GeoTIFF.\n")
            fh.write(f"Cobertura: 0/{len(esperados)} combinaciones oficiales.\n")
            fh.write("\nFaltantes:\n")
            for (lago, fecha) in sorted(esperados):
                fh.write(f"  - {lago} {fecha}\n")
        LOGGER.info("Reporte generado: %s", reporte.relative_to(ROOT))
        return 1

    filas, lineas, errores, incompletos = [], [], [], []
    encontrados = set()
    firmas = {}

    for ruta in barra_progreso(archivos, len(archivos), "Validando"):
        try:
            info = inspeccionar_raster(ruta)
        except Exception as exc:
            errores.append(f"{ruta.name}: CORRUPTO O ILEGIBLE ({type(exc).__name__}: {exc})")
            continue

        tags = info.get("tags", {})
        lago = tags.get("lago") or ruta.parent.name
        fecha = tags.get("fecha") or ruta.stem.split("_")[-1]
        encontrados.add((lago, fecha))
        firmas.setdefault((lago, fecha), []).append(ruta.name)

        completo = info["validos_pct"] >= args.min_validos
        if not completo:
            incompletos.append((lago, fecha, info["validos_pct"]))

        filas.append({
            "archivo": str(ruta.relative_to(ROOT)),
            "lago": lago,
            "fecha": fecha,
            "tamano_MB": round(info["tamano_bytes"] / 1e6, 3),
            "n_bandas": info["n_bandas"],
            "nombres_bandas": ",".join(n or "" for n in info["nombres_bandas"]),
            "ancho": info["ancho"],
            "alto": info["alto"],
            "crs": info["crs"],
            "transform": ",".join(f"{v:.6f}" for v in info["transform"]),
            "res_x": info["res_x"],
            "res_y": info["res_y"],
            "bounds": ",".join(f"{v:.3f}" for v in info["bounds"]),
            "dtype": info["dtype"],
            "nodata": info["nodata"],
            "pixeles_totales": info["pixeles_totales"],
            "pixeles_validos": info["pixeles_validos_min"],
            "validos_pct": round(info["validos_pct"], 3),
            "alineacion_interna_ok": info["alineacion_interna_ok"],
            "cobertura_completa": completo,
            "coleccion": tags.get("coleccion", ""),
            "identificadores_adquisicion": tags.get("identificadores_adquisicion", ""),
        })

        lineas.append(f"\n--- {ruta.relative_to(ROOT)} ---")
        lineas.append(f"  Lago/Fecha      : {lago} / {fecha}")
        lineas.append(f"  Tamano          : {info['tamano_bytes']/1e6:.2f} MB")
        lineas.append(f"  Bandas ({info['n_bandas']})     : {list(info['nombres_bandas'])}")
        lineas.append(f"  Dimensiones     : {info['ancho']} x {info['alto']}")
        lineas.append(f"  CRS             : {info['crs']}")
        lineas.append(f"  Transformacion  : {info['transform']}")
        lineas.append(f"  Resolucion      : {info['res_x']} x {info['res_y']}")
        lineas.append(f"  Bounds          : {info['bounds']}")
        lineas.append(f"  dtype / NoData  : {info['dtype']} / {info['nodata']}")
        lineas.append(f"  Pixeles validos : {info['pixeles_validos_min']:,} de "
                      f"{info['pixeles_totales']:,} ({info['validos_pct']:.2f}%)")
        lineas.append(f"  Alineacion OK   : {info['alineacion_interna_ok']}")
        lineas.append("  Estadisticas por banda:")
        for est in info["estadisticas"]:
            if est["min"] is None:
                lineas.append(f"    {est['banda']:>10}: sin pixeles validos")
            else:
                lineas.append(
                    f"    {est['banda']:>10}: min={est['min']:12.4f}  "
                    f"max={est['max']:12.4f}  media={est['media']:12.4f}  "
                    f"validos={est['validos_pct']:.1f}%"
                )

    faltantes = sorted(set(esperados) - encontrados)
    duplicados = {k: v for k, v in firmas.items() if len(v) > 1}

    # Coherencia entre archivos del mismo lago
    coherencia = []
    for lago in sorted({f["lago"] for f in filas}):
        grupo = [f for f in filas if f["lago"] == lago]
        crs_set = {f["crs"] for f in grupo}
        res_set = {(f["res_x"], f["res_y"]) for f in grupo}
        dim_set = {(f["ancho"], f["alto"]) for f in grupo}
        coherencia.append({
            "lago": lago, "n": len(grupo),
            "crs_unico": len(crs_set) == 1, "crs": crs_set,
            "res_unica": len(res_set) == 1, "res": res_set,
            "dim_unica": len(dim_set) == 1, "dim": dim_set,
        })

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifiesto = MANIFEST_DIR / "raster_manifest.csv"
    if filas:
        with open(manifiesto, "w", newline="", encoding="utf-8") as fh:
            escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            escritor.writeheader()
            escritor.writerows(filas)

    conteo_lago = {}
    for fila in filas:
        conteo_lago[fila["lago"]] = conteo_lago.get(fila["lago"], 0) + 1

    reporte = MANIFEST_DIR / "validation_report.txt"
    with open(reporte, "w", encoding="utf-8") as fh:
        fh.write("REPORTE DE VALIDACION DE RASTERES SENTINEL-2 L1C\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"Generado: {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"Directorio: {RASTER_DIR}\n\n")
        fh.write(f"Archivos encontrados      : {len(archivos)}\n")
        fh.write(f"Archivos validados        : {len(filas)}\n")
        fh.write(f"Archivos corruptos        : {len(errores)}\n")
        fh.write(f"Cobertura oficial         : {len(encontrados & set(esperados))}/{len(esperados)}\n\n")
        fh.write("Conteo por lago:\n")
        for lago, n in sorted(conteo_lago.items()):
            fh.write(f"  {lago:<12}: {n}\n")
        fh.write("\nCoherencia entre archivos del mismo lago:\n")
        for c in coherencia:
            fh.write(f"  {c['lago']:<12} n={c['n']:<3} CRS unico={c['crs_unico']} "
                     f"Resolucion unica={c['res_unica']} Dimensiones unicas={c['dim_unica']}\n")
            if not c["crs_unico"]:
                fh.write(f"      CRS distintos: {c['crs']}\n")
            if not c["res_unica"]:
                fh.write(f"      Resoluciones distintas: {c['res']}\n")
            if not c["dim_unica"]:
                fh.write(f"      Dimensiones distintas: {c['dim']}\n")
        if faltantes:
            fh.write(f"\nFALTANTES ({len(faltantes)}):\n")
            for lago, fecha in faltantes:
                fh.write(f"  - {lago} {fecha}\n")
        else:
            fh.write("\nFALTANTES: ninguno\n")
        if incompletos:
            fh.write(f"\nCOBERTURA ESPACIAL INCOMPLETA (< {args.min_validos:.1f}% valido) "
                     f"({len(incompletos)}):\n")
            for lago, fecha, pct in sorted(incompletos):
                fh.write(f"  - {lago} {fecha}: {pct:.2f}% de pixeles validos\n")
            fh.write("  Causa habitual: falta uno de los dos tiles que cubren el lago.\n")
            fh.write("  Solucion: volver a descargar sin filtro de nubosidad por tile.\n")
        else:
            fh.write("\nCOBERTURA ESPACIAL INCOMPLETA: ninguna\n")
        if duplicados:
            fh.write(f"\nDUPLICADOS ({len(duplicados)}):\n")
            for clave, nombres in duplicados.items():
                fh.write(f"  - {clave}: {nombres}\n")
        else:
            fh.write("\nDUPLICADOS: ninguno\n")
        if errores:
            fh.write(f"\nCORRUPTOS ({len(errores)}):\n")
            for e in errores:
                fh.write(f"  - {e}\n")
        fh.write("\n" + "=" * 70 + "\n")
        fh.write("DETALLE POR ARCHIVO\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(lineas))
        fh.write("\n")

    LOGGER.info("")
    LOGGER.info("Archivos encontrados : %d", len(archivos))
    LOGGER.info("Archivos validados   : %d", len(filas))
    LOGGER.info("Archivos corruptos   : %d", len(errores))
    for lago, n in sorted(conteo_lago.items()):
        LOGGER.info("  %-11s : %d", lago, n)
    LOGGER.info("Cobertura oficial    : %d/%d", len(encontrados & set(esperados)), len(esperados))
    if faltantes:
        LOGGER.warning("Faltantes (%d):", len(faltantes))
        for lago, fecha in faltantes:
            LOGGER.warning("  - %s %s", lago, fecha)
    if incompletos:
        LOGGER.warning("")
        LOGGER.warning("COBERTURA ESPACIAL INCOMPLETA (%d archivos, umbral %.1f%%):",
                       len(incompletos), args.min_validos)
        for lago, fecha, pct in sorted(incompletos):
            LOGGER.warning("  - %-11s %-12s %6.2f%% valido", lago, fecha, pct)
        LOGGER.warning("  Causa: falta uno de los dos tiles que cubren el lago.")
        LOGGER.warning("  Vuelva a ejecutar --download para regenerarlos completos.")
    if duplicados:
        LOGGER.warning("Duplicados: %s", duplicados)
    if errores:
        for e in errores:
            LOGGER.error("  %s", e)

    LOGGER.info("")
    LOGGER.info("Manifiesto: %s", manifiesto.relative_to(ROOT) if filas else "(no generado)")
    LOGGER.info("Reporte   : %s", reporte.relative_to(ROOT))

    return 0 if (not faltantes and not errores and not duplicados and not incompletos) else 1


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="descargar_rasters.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Adquisicion y validacion de rasters Sentinel-2 L1C reales para el\n"
            "Laboratorio 4 (lagos Atitlan y Amatitlan, 22 combinaciones oficiales).\n\n"
            "Descarga desde Copernicus Data Space mediante openEO, recorta con el\n"
            "GeoJSON de cada lago, homogeneiza CRS/resolucion y valida cada GeoTIFF.\n"
            "NUNCA genera datos sinteticos: si no hay datos reales, se detiene."
        ),
        epilog=(
            "EJEMPLOS\n"
            "  python descargar_rasters.py --dry-run\n"
            "      Prepara y verifica las 22 solicitudes sin descargar nada.\n\n"
            "  python descargar_rasters.py --download\n"
            "      Autentica por OIDC y descarga las 22 combinaciones (reanudable).\n\n"
            "  python descargar_rasters.py --validate\n"
            "      Revisa los GeoTIFF ya descargados y genera los manifiestos.\n\n"
            "  python descargar_rasters.py --download --lake Atitlan --date 2025-01-18\n"
            "      Reintenta una unica combinacion.\n\n"
            "CODIGOS DE SALIDA\n"
            "  0  todo correcto\n"
            "  1  faltan combinaciones, hay errores o archivos invalidos\n"
            "  2  fallo de autenticacion o de acceso al backend\n"
            "  130 interrumpido por la usuaria\n"
        ),
    )

    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--dry-run", action="store_true",
                      help="Prepara y verifica las 22 solicitudes sin descargar rasteres.")
    modo.add_argument("--download", action="store_true",
                      help="Autentica por OIDC y descarga los rasteres reales (reanudable).")
    modo.add_argument("--validate", action="store_true",
                      help="Valida los GeoTIFF descargados y genera los manifiestos.")

    parser.add_argument("--lake", metavar="NOMBRE", default=None,
                        help="Procesa solo un lago (Atitlan o Amatitlan).")
    parser.add_argument("--date", metavar="AAAA-MM-DD", default=None,
                        help="Procesa solo una fecha oficial.")
    parser.add_argument("--max-cloud", type=float, default=MAX_CLOUD_COVER_DEFAULT,
                        metavar="PCT", dest="max_cloud",
                        help=("Filtra tiles por eo:cloud_cover <= PCT. Desactivado por "
                              "omision: ese valor describe el tile completo (110x110 km), "
                              "no el lago, y filtrarlo deja huecos porque cada lago se "
                              "cubre con dos tiles."))
    parser.add_argument("--min-validos", type=float, default=MIN_VALIDOS_PCT_DEFAULT,
                        metavar="PCT", dest="min_validos",
                        help=(f"Porcentaje minimo de pixeles validos para aceptar un "
                              f"GeoTIFF (por omision {MIN_VALIDOS_PCT_DEFAULT:.0f}%%). "
                              "Los archivos por debajo se consideran incompletos y se "
                              "vuelven a descargar. Use 0 para aceptar cualquier cobertura."))
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Muestra mensajes de depuracion detallados.")
    return parser


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    log_path = configurar_logging(args.verbose)
    LOGGER.debug("Registro en %s", log_path)

    try:
        if args.dry_run:
            return ejecutar_dry_run(args)
        if args.download:
            return ejecutar_download(args)
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
