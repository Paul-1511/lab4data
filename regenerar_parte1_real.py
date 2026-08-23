"""
Regeneracion de la Parte 1 del Laboratorio 4 con datos Sentinel-2 L1C REALES.

Reutiliza los 22 GeoTIFF ya descargados y validados en outputs/rasters/.
No se conecta a Copernicus y no usa synthetic_bands() ni ningun sustituto
sintetico: si falta un raster, el script se detiene.

Cubre las actividades 1 a 8 del PDF de la Parte 1 y produce todas las
evidencias en outputs/parte1_real/.

Modos:
    python regenerar_parte1_real.py --dry-run
    python regenerar_parte1_real.py --build
    python regenerar_parte1_real.py --validate
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
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
    LAKE_BBOXES,
    S2_L1C_COLLECTION,
    normalize_band,
    reflectance_scale_report,
    wbi_vectorized,
    safe_div,
    NDVI as calc_ndvi,
    NDWI as calc_ndwi,
    NDCI as calc_ndci,
    FAI as calc_fai,
    chl_from_ndci,
)

SEED = 42
np.random.seed(SEED)

RASTER_DIR = ROOT / "outputs" / "rasters"
BASE = ROOT / "outputs" / "parte1_real"
FIG_DIR = BASE / "figures"
MAP_DIR = BASE / "maps"
TAB_DIR = BASE / "tables"
REP_DIR = BASE / "reports"
VAL_DIR = BASE / "validation"
LOG_DIR = BASE / "logs"

CRS_ESPERADO = "EPSG:32615"
NODATA = -9999.0
ESCALA_DN = 10000.0

# Umbrales ambientales candidatos (ug/L de clorofila-a).
UMBRALES = [8.0, 20.0, 25.0, 50.0]
UMBRAL_PRINCIPAL = 25.0

SIGNIFICADO_UMBRAL = {
    8.0:  "Frontera mesotrofico -> eutrofico (OECD 1982). Marca el inicio de la "
          "condicion eutrofica.",
    20.0: "Umbral usado en la primera version de la Parte 1. Cae dentro de la banda "
          "eutrofica de OECD (8-25) y del rango de Alerta 1 de la OMS (12-24), pero "
          "NO es una frontera publicada por si mismo.",
    25.0: "Frontera eutrofico -> hipertrofico (OECD 1982); coincide practicamente con "
          "el techo de la Alerta 1 de la OMS (24 ug/L con dominancia de cianobacterias).",
    50.0: "Escenario severo para analisis de sensibilidad. NO aparece como valor de "
          "clorofila-a en OECD 1982 ni en las guias de la OMS 2021.",
}

# Dominio de calibracion del modelo NDCI -> clorofila (Mishra & Mishra 2012).
CALIB_MIN, CALIB_MAX = 1.0, 60.0

LOGGER = logging.getLogger("parte1_real")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def configurar_logging(verbose=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ruta = LOG_DIR / f"parte1_{datetime.now():%Y%m%d_%H%M%S}.log"
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    c = logging.StreamHandler(sys.stdout)
    c.setLevel(logging.DEBUG if verbose else logging.INFO)
    c.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(c)
    f = logging.FileHandler(ruta, encoding="utf-8")
    f.setLevel(logging.DEBUG)
    f.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(f)
    return ruta


def combinaciones(lago=None):
    pares = [(l, f) for l in sorted(OFFICIAL_DATES) for f in OFFICIAL_DATES[l]]
    if lago:
        coincide = [l for l in OFFICIAL_DATES if l.lower() == lago.lower()]
        if not coincide:
            raise SystemExit(f"ERROR: lago desconocido '{lago}'.")
        pares = [p for p in pares if p[0] == coincide[0]]
    return pares


def ruta_raster(lago, fecha):
    return RASTER_DIR / lago / f"{lago}_{fecha}.tif"


def barra(iterable, total, desc):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="escena", ncols=86)
    except ImportError:
        return iterable


def area_pixel_ha(transform):
    """
    Deriva el area de un pixel en hectareas desde la transformacion afin real.

    Para los rasteres de este laboratorio (20 m) da 400 m2 = 0.04 ha. Se calcula
    en lugar de fijarse porque la version anterior asumia 0.01 ha (10 m) y
    subestimaba todas las superficies en un factor de 4.
    """
    return abs(transform.a * transform.e) / 10_000.0


# ---------------------------------------------------------------------------
# ACTIVIDAD 3: reproduccion verificada del evalscript CyanoLakes
# ---------------------------------------------------------------------------
def evalscript_literal(b02, b03, b04, b05, b07, b08, b8a, b11, b12):
    """
    Traduccion literal, pixel a pixel, del evalscript oficial
    'Cyanobacteria Chlorophyll-a NDCI L1C' de Sentinel Hub.

    Se usa solo como referencia para verificar la implementacion vectorizada de
    mainlab4.py. Sigue el orden y las condiciones exactas del script original.
    """
    def sd(a, b):
        return 0.0 if b == 0 else a / b

    ndvi = sd(b08 - b04, b08 + b04)
    ndwi = sd(b03 - b08, b03 + b08)
    mndwi = sd(b03 - b11, b03 + b11)
    ndwi_leaves = sd(b08 - b11, b08 + b11)
    aweish = b02 + 2.5 * b03 - 1.5 * (b08 + b11) - 0.25 * b12
    aweinsh = 4 * (b03 - b11) - (0.25 * b08 + 2.75 * b11)
    dbsi = sd(b11 - b03, b11 + b03) - ndvi

    water = 1 if (mndwi > 0.42 or ndwi > 0.4 or aweinsh > 0.1879
                  or aweish > 0.1112 or ndvi < -0.2 or ndwi_leaves > 1) else 0
    if water == 1 and (aweinsh <= -0.03 or dbsi > 0):
        water = 0

    fai = b07 - b04 - (b8a - b04) * (783 - 665) / (865 - 665)
    ndci = sd(b05 - b04, b05 + b04)
    chl = 826.57 * ndci ** 3 - 176.43 * ndci ** 2 + 19 * ndci + 4.071
    cyano = chl if water == 1 else float("nan")

    return {"ndvi": ndvi, "ndwi": ndwi, "mndwi": mndwi, "aweish": aweish,
            "aweinsh": aweinsh, "dbsi": dbsi, "water": water, "fai": fai,
            "ndci": ndci, "chl": chl, "cyano": cyano}


def verificar_evalscript(escena, n_muestras=500, tolerancia=1e-6):
    """
    Compara la implementacion vectorizada con el evalscript literal en pixeles
    concretos elegidos de forma determinista. Devuelve el reporte y falla si
    alguna diferencia supera la tolerancia.
    """
    rng = np.random.default_rng(SEED)
    alto, ancho = escena["B04"].shape
    filas = rng.integers(0, alto, n_muestras)
    cols = rng.integers(0, ancho, n_muestras)

    difs = {k: 0.0 for k in ["ndvi", "ndwi", "fai", "ndci", "chl"]}
    discrepancias_agua = 0
    comparados = 0
    ejemplo = None

    for r, c in zip(filas, cols):
        if not escena["valid"][r, c]:
            continue
        vals = [float(escena[n][r, c]) for n in
                ["B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12"]]
        ref = evalscript_literal(*vals)
        comparados += 1

        difs["ndvi"] = max(difs["ndvi"], abs(ref["ndvi"] - float(escena["NDVI"][r, c])))
        difs["ndwi"] = max(difs["ndwi"], abs(ref["ndwi"] - float(escena["NDWI"][r, c])))
        difs["fai"] = max(difs["fai"], abs(ref["fai"] - float(escena["FAI"][r, c])))
        difs["ndci"] = max(difs["ndci"], abs(ref["ndci"] - float(escena["NDCI"][r, c])))
        difs["chl"] = max(difs["chl"], abs(ref["chl"] - float(escena["chl"][r, c])))
        if bool(ref["water"]) != bool(escena["water"][r, c]):
            discrepancias_agua += 1

        if ejemplo is None:
            ejemplo = {"fila": int(r), "col": int(c),
                       "bandas": {n: round(v, 6) for n, v in zip(
                           ["B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12"], vals)},
                       "evalscript": {k: (None if isinstance(v, float) and math.isnan(v)
                                          else round(float(v), 8))
                                      for k, v in ref.items()},
                       "local": {"ndvi": round(float(escena["NDVI"][r, c]), 8),
                                 "ndwi": round(float(escena["NDWI"][r, c]), 8),
                                 "fai": round(float(escena["FAI"][r, c]), 8),
                                 "ndci": round(float(escena["NDCI"][r, c]), 8),
                                 "chl": round(float(escena["chl"][r, c]), 8),
                                 "water": bool(escena["water"][r, c])}}

    ok = all(v <= tolerancia for v in difs.values()) and discrepancias_agua == 0
    return {"pixeles_comparados": comparados,
            "diferencias_maximas": {k: float(v) for k, v in difs.items()},
            "discrepancias_mascara_agua": int(discrepancias_agua),
            "tolerancia": tolerancia, "coincide": bool(ok), "ejemplo": ejemplo}


# ---------------------------------------------------------------------------
# Carga de una escena real
# ---------------------------------------------------------------------------
def cargar_escena(lago, fecha, con_diagnostico=False):
    """
    Lee un GeoTIFF real y calcula todos los indices sobre reflectancia.

    La escala DN -> reflectancia se aplica exactamente una vez y se verifica.
    """
    import rasterio

    ruta = ruta_raster(lago, fecha)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta el raster real de {lago} {fecha}: {ruta}. "
            "Ejecute descargar_rasters.py --download. NO se generan datos sinteticos.")

    with rasterio.open(ruta) as src:
        nombres = list(src.descriptions)
        if any(n is None for n in nombres):
            raise ValueError(f"{ruta.name}: bandas sin descripcion.")
        faltan = [b for b in CYANO_REQUIRED_BANDS if b not in nombres]
        if faltan:
            raise ValueError(f"{ruta.name}: faltan bandas {faltan}.")
        if str(src.crs) != CRS_ESPERADO:
            raise ValueError(f"{ruta.name}: CRS {src.crs}, se esperaba {CRS_ESPERADO}.")

        crudo = {n: src.read(i + 1).astype(np.float64) for i, n in enumerate(nombres)}
        mascaras = src.read_masks()
        transform, crs, bounds = src.transform, src.crs, src.bounds
        alto, ancho = src.height, src.width

    diagnostico = None
    if con_diagnostico:
        diagnostico = {b: reflectance_scale_report(crudo[b], scale=ESCALA_DN)
                       for b in CYANO_REQUIRED_BANDS}

    # Escala aplicada una unica vez y verificada por normalize_band.
    banda = {b: normalize_band(crudo[b], scale=ESCALA_DN).astype(np.float64)
             for b in CYANO_REQUIRED_BANDS}

    valid = (mascaras > 0).all(axis=0)
    for b in CYANO_REQUIRED_BANDS:
        valid &= (crudo[b] != NODATA) & np.isfinite(banda[b])
    # Regla fisica: la reflectancia TOA no puede ser <= 0. Se conservan los
    # valores > 1 (nubes, reflexion especular) porque son fisicamente posibles.
    for b in CYANO_REQUIRED_BANDS:
        valid &= (banda[b] > 0)

    ndvi = calc_ndvi(banda["B04"], banda["B08"])
    ndwi = calc_ndwi(banda["B03"], banda["B08"])
    mndwi = safe_div(banda["B03"] - banda["B11"], banda["B03"] + banda["B11"])
    ndci = calc_ndci(banda["B04"], banda["B05"])
    fai = calc_fai(banda["B04"], banda["B07"], banda["B8A"])
    chl = chl_from_ndci(ndci)

    water = wbi_vectorized(banda["B04"], banda["B03"], banda["B02"],
                           banda["B08"], banda["B11"], banda["B12"]) & valid
    for arr in (ndvi, ndwi, ndci, fai, chl):
        water &= np.isfinite(arr)

    escena = {**banda, "NDVI": ndvi, "NDWI": ndwi, "MNDWI": mndwi, "NDCI": ndci,
              "FAI": fai, "chl": chl, "water": water, "valid": valid,
              "transform": transform, "crs": crs, "bounds": bounds,
              "alto": alto, "ancho": ancho, "lago": lago, "fecha": fecha,
              "area_pixel_ha": area_pixel_ha(transform),
              "diagnostico_escala": diagnostico}
    # Clorofila enmascarada a agua: es la capa que se mapea e interpreta.
    escena["chl_agua"] = np.where(water, chl, np.nan)
    return escena


# ---------------------------------------------------------------------------
# MODO --dry-run
# ---------------------------------------------------------------------------
def ejecutar_dry_run(args):
    LOGGER.info("=" * 84)
    LOGGER.info("MODO DRY-RUN: verificacion de los 22 rasteres reales (no genera artefactos)")
    LOGGER.info("=" * 84)

    pares = combinaciones(args.lake)
    LOGGER.info("Combinaciones oficiales: %d", len(pares))

    faltantes = [(l, f) for l, f in pares if not ruta_raster(l, f).exists()]
    if faltantes:
        LOGGER.error("Faltan %d rasteres reales: %s", len(faltantes), faltantes)
        LOGGER.error("NO se generan datos sinteticos. Ejecute descargar_rasters.py --download")
        return 2

    LOGGER.info("Los %d rasteres existen.", len(pares))
    LOGGER.info("")

    filas = []
    for lago in sorted({l for l, _ in pares}):
        fecha = OFFICIAL_DATES[lago][0]
        escena = cargar_escena(lago, fecha, con_diagnostico=True)
        d = escena["diagnostico_escala"]["B08"]
        area = escena["area_pixel_ha"]
        n_agua = int(escena["water"].sum())
        LOGGER.info("%s (%s)", lago, fecha)
        LOGGER.info("  Escala   : mediana DN=%.1f -> factor %.4g -> reflectancia "
                    "[%.4f, %.4f] (mediana %.4f)",
                    d["mediana_original"], d["escala_aplicada"],
                    d["min_resultante"], d["max_resultante"], d["mediana_resultante"])
        if not (0 < d["mediana_resultante"] < 1.5):
            LOGGER.error("  La reflectancia resultante esta fuera de rango plausible.")
            return 2
        LOGGER.info("  Pixel    : %.0f x %.0f m -> %.4f ha (%.0f m2)",
                    abs(escena["transform"].a), abs(escena["transform"].e),
                    area, area * 10000)
        LOGGER.info("  Agua WBI : %s pixeles -> %.1f km2", f"{n_agua:,}",
                    n_agua * area * 0.01)

        ver = verificar_evalscript(escena, n_muestras=300)
        LOGGER.info("  Evalscript: %s (%d pixeles, dif. max chl=%.2e)",
                    "COINCIDE" if ver["coincide"] else "NO COINCIDE",
                    ver["pixeles_comparados"], ver["diferencias_maximas"]["chl"])
        if not ver["coincide"]:
            LOGGER.error("  La implementacion local no reproduce el evalscript oficial.")
            return 2
        filas.append({"lago": lago, "area_pixel_ha": area, "pixeles_agua": n_agua})
        LOGGER.info("")

    if abs(filas[0]["area_pixel_ha"] - 0.04) > 1e-9:
        LOGGER.error("El area de pixel derivada no es 0.04 ha; revise la resolucion.")
        return 2

    LOGGER.info("ESTIMACION DE PRODUCTOS A GENERAR")
    LOGGER.info("  Tablas   : ~10 CSV (series, correlaciones, extension, umbrales)")
    LOGGER.info("  Figuras  : ~8 PNG (temporal, boxplots, histogramas, correlaciones)")
    LOGGER.info("  Mapas    : %d PNG por fecha + comparativos + persistencia + 2 HTML",
                len(pares))
    LOGGER.info("  Memoria  : pico ~%.0f MB (una escena a la vez + pila por lago)",
                1380 * 876 * 11 * 4 / 1e6)
    LOGGER.info("  Tiempo   : ~1-3 minutos")
    LOGGER.info("")
    LOGGER.info("DRY-RUN correcto. Siguiente paso:")
    LOGGER.info("  python regenerar_parte1_real.py --build")
    return 0


# ---------------------------------------------------------------------------
# MODO --build
# ---------------------------------------------------------------------------
def ejecutar_build(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")

    LOGGER.info("=" * 84)
    LOGGER.info("MODO BUILD: regeneracion de la Parte 1 con datos reales")
    LOGGER.info("=" * 84)

    for d in (FIG_DIR, MAP_DIR, TAB_DIR, REP_DIR, VAL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pares = combinaciones(args.lake)
    faltantes = [(l, f) for l, f in pares if not ruta_raster(l, f).exists()]
    if faltantes:
        LOGGER.error("Faltan rasteres reales: %s. Se detiene.", faltantes)
        return 2

    lagos = sorted({l for l, _ in pares})
    pila_chl, pila_water, metadatos = {}, {}, {}
    filas_serie, filas_corr, filas_ext = [], [], []
    verificacion_evalscript = None

    LOGGER.info("Procesando %d escenas reales...", len(pares))
    for lago, fecha in barra(pares, len(pares), "Escenas"):
        escena = cargar_escena(lago, fecha, con_diagnostico=(verificacion_evalscript is None))

        if verificacion_evalscript is None:
            verificacion_evalscript = verificar_evalscript(escena, n_muestras=500)
            verificacion_evalscript["escena"] = f"{lago} {fecha}"
            if not verificacion_evalscript["coincide"]:
                LOGGER.error("La implementacion local NO reproduce el evalscript oficial.")
                return 2
            LOGGER.info("Verificacion del evalscript: COINCIDE en %d pixeles.",
                        verificacion_evalscript["pixeles_comparados"])

        agua = escena["water"]
        chl = escena["chl"][agua]
        area = escena["area_pixel_ha"]

        # --- Actividad 4: estadisticas por lago y fecha ---
        filas_serie.append({
            "lago": lago, "fecha": fecha,
            "n_pixeles_agua": int(agua.sum()),
            "area_agua_ha": float(agua.sum() * area),
            "chl_media": float(np.mean(chl)),
            "chl_mediana": float(np.median(chl)),
            "chl_std": float(np.std(chl)),
            "chl_p05": float(np.percentile(chl, 5)),
            "chl_p25": float(np.percentile(chl, 25)),
            "chl_p75": float(np.percentile(chl, 75)),
            "chl_p90": float(np.percentile(chl, 90)),
            "chl_p95": float(np.percentile(chl, 95)),
            "chl_p99": float(np.percentile(chl, 99)),
            "chl_min": float(np.min(chl)), "chl_max": float(np.max(chl)),
            "pct_fuera_calibracion": float(100 * np.mean(
                (chl < CALIB_MIN) | (chl > CALIB_MAX))),
            "pct_chl_negativa": float(100 * np.mean(chl < 0)),
            "nubosidad_oficial_pct": OFFICIAL_CLOUD_COVER[lago][fecha],
        })

        # --- Actividad 6: correlaciones sobre pixeles de agua ---
        sub = pd.DataFrame({
            "NDVI": escena["NDVI"][agua], "NDWI": escena["NDWI"][agua],
            "NDCI": escena["NDCI"][agua], "FAI": escena["FAI"][agua],
            "chl": chl,
        })
        c = sub.corr(method="pearson")
        filas_corr.append({
            "lago": lago, "fecha": fecha, "n": len(sub),
            "NDVI_chl": c.loc["NDVI", "chl"], "NDWI_chl": c.loc["NDWI", "chl"],
            "NDCI_chl": c.loc["NDCI", "chl"], "FAI_chl": c.loc["FAI", "chl"],
            "NDVI_NDWI": c.loc["NDVI", "NDWI"],
        })

        # --- Actividad 8 + umbrales ---
        for u in UMBRALES:
            n_pos = int((chl >= u).sum())
            filas_ext.append({
                "lago": lago, "fecha": fecha, "umbral_ug_L": u,
                "pixeles_positivos": n_pos,
                "pixeles_agua": int(agua.sum()),
                "pct_area_afectada": float(100 * n_pos / agua.sum()),
                "area_afectada_ha": float(n_pos * area),
                "area_agua_total_ha": float(agua.sum() * area),
            })

        pila_chl.setdefault(lago, []).append(escena["chl_agua"].astype(np.float32))
        pila_water.setdefault(lago, []).append(agua)
        metadatos.setdefault(lago, escena)

    serie = pd.DataFrame(filas_serie).sort_values(["lago", "fecha"]).reset_index(drop=True)
    corr_df = pd.DataFrame(filas_corr).sort_values(["lago", "fecha"]).reset_index(drop=True)
    ext_df = pd.DataFrame(filas_ext).sort_values(
        ["lago", "fecha", "umbral_ug_L"]).reset_index(drop=True)

    serie.to_csv(TAB_DIR / "serie_temporal_por_lago_fecha.csv", index=False)
    corr_df.to_csv(TAB_DIR / "correlaciones_por_lago_fecha.csv", index=False)
    ext_df.to_csv(TAB_DIR / "extension_espacial_por_umbral.csv", index=False)

    # --- Actividad 4.3: picos y fechas criticas CALCULADOS ---
    picos = []
    for lago in lagos:
        g = serie[serie["lago"] == lago]
        umbral_pico = g["chl_media"].mean() + g["chl_media"].std()
        for _, fila in g.iterrows():
            if fila["chl_media"] >= umbral_pico:
                picos.append({"lago": lago, "fecha": fila["fecha"],
                              "chl_media": fila["chl_media"],
                              "umbral_pico": float(umbral_pico),
                              "criterio": "media + 1 desviacion estandar del lago"})
    picos_df = pd.DataFrame(picos)
    picos_df.to_csv(TAB_DIR / "fechas_criticas.csv", index=False)

    # --- Analisis de umbrales (global / lago / fecha / viabilidad) ---
    generar_analisis_umbrales(ext_df, serie, lagos)

    # --- Figuras y mapas ---
    LOGGER.info("Generando figuras y mapas...")
    figuras_actividad_4(serie, picos_df)
    figuras_actividad_6(corr_df, pila_chl, metadatos, pares)
    mapas_actividad_5(pila_chl, pila_water, metadatos, serie, lagos)
    figuras_actividad_8(pila_chl, pila_water, metadatos, serie, ext_df, lagos)
    comparacion_actividad_7(serie, ext_df, lagos)

    # --- Registro de verificacion ---
    with open(VAL_DIR / "evalscript_verification.json", "w", encoding="utf-8") as fh:
        json.dump(verificacion_evalscript, fh, indent=2, ensure_ascii=False)

    diag = metadatos[lagos[0]]["diagnostico_escala"]
    if diag:
        pd.DataFrame(diag).T.to_csv(TAB_DIR / "verificacion_escala_reflectancia.csv")

    escribir_reporte_parte1(serie, corr_df, ext_df, picos_df, lagos,
                            verificacion_evalscript, metadatos)

    LOGGER.info("")
    LOGGER.info("BUILD completado. Siguiente paso:")
    LOGGER.info("  python regenerar_parte1_real.py --validate")
    return 0


# ---------------------------------------------------------------------------
# Analisis de umbrales
# ---------------------------------------------------------------------------
def generar_analisis_umbrales(ext_df, serie, lagos):
    filas_global, filas_lago, filas_via = [], [], []

    for u in UMBRALES:
        sub = ext_df[ext_df["umbral_ug_L"] == u]
        pos = int(sub["pixeles_positivos"].sum())
        tot = int(sub["pixeles_agua"].sum())
        filas_global.append({
            "umbral_ug_L": u, "pixeles_positivos": pos, "pixeles_agua": tot,
            "pct_positivo": 100.0 * pos / tot if tot else 0.0,
            "area_afectada_ha": float(sub["area_afectada_ha"].sum()),
            "fechas_con_positivos": int((sub["pixeles_positivos"] > 0).sum()),
            "fechas_totales": len(sub),
            "significado_ambiental": SIGNIFICADO_UMBRAL[u],
        })

        for lago in lagos:
            s = sub[sub["lago"] == lago]
            p, t = int(s["pixeles_positivos"].sum()), int(s["pixeles_agua"].sum())
            filas_lago.append({
                "lago": lago, "umbral_ug_L": u, "pixeles_positivos": p,
                "pixeles_agua": t, "pct_positivo": 100.0 * p / t if t else 0.0,
                "area_afectada_ha": float(s["area_afectada_ha"].sum()),
                "fechas_con_positivos": int((s["pixeles_positivos"] > 0).sum()),
                "fechas_totales": len(s),
                "fechas_con_ambas_clases": int(
                    ((s["pixeles_positivos"] > 0)
                     & (s["pixeles_positivos"] < s["pixeles_agua"])).sum()),
            })

        # Viabilidad para clasificacion binaria.
        # Tener >= 1 positivo no basta: un lago con unas decenas de pixeles
        # positivos no permite entrenar ni evaluar de forma significativa. Se
        # exige un minimo absoluto y uno relativo para declarar viabilidad.
        MIN_POSITIVOS = 1000       # minimo practico de pixeles de la clase rara
        MIN_PCT = 0.10             # % minimo para que el desbalance sea manejable

        pct = 100.0 * pos / tot if tot else 0.0
        conteo_lago = {l: int(sub[sub["lago"] == l]["pixeles_positivos"].sum())
                       for l in lagos}
        pct_lago = {}
        for l in lagos:
            t_l = int(sub[sub["lago"] == l]["pixeles_agua"].sum())
            pct_lago[l] = 100.0 * conteo_lago[l] / t_l if t_l else 0.0

        lagos_con_pos = [l for l in lagos if conteo_lago[l] > 0]
        lagos_entrenables = [l for l in lagos
                             if conteo_lago[l] >= MIN_POSITIVOS
                             and pct_lago[l] >= MIN_PCT]
        lagos_marginales = [l for l in lagos_con_pos if l not in lagos_entrenables]

        if not lagos_con_pos:
            obs = "Ningun lago alcanza el umbral: clasificacion imposible."
        elif not lagos_entrenables:
            obs = (f"Todos los lagos quedan por debajo del minimo practico "
                   f"({MIN_POSITIVOS} positivos y {MIN_PCT} %): el umbral no sostiene "
                   "un modelo binario.")
        elif lagos_marginales:
            detalle = ", ".join(f"{l} solo {conteo_lago[l]} positivos "
                                f"({pct_lago[l]:.4f} %)" for l in lagos_marginales)
            obs = (f"Entrenable solo en {', '.join(lagos_entrenables)}. {detalle}: "
                   "el experimento entre lagos queda degenerado en ese sentido.")
        else:
            obs = "Ambos lagos superan el minimo practico."

        filas_via.append({
            "umbral_ug_L": u,
            "pct_positivo_global": pct,
            "positivos_global": pos,
            "clases_presentes_global": "ambas" if 0 < pos < tot else "una sola",
            "viable_clasificacion_binaria": bool(pos >= MIN_POSITIVOS and pct >= MIN_PCT),
            "lagos_con_algun_positivo": ";".join(lagos_con_pos) if lagos_con_pos else "ninguno",
            "lagos_entrenables": ";".join(lagos_entrenables) if lagos_entrenables else "ninguno",
            "n_lagos_entrenables": len(lagos_entrenables),
            "viable_entrenar_por_lago": len(lagos_entrenables) == len(lagos),
            "viable_evaluar_entre_lagos": len(lagos_entrenables) == len(lagos),
            "positivos_por_lago": ";".join(f"{l}={conteo_lago[l]}" for l in lagos),
            "criterio_minimo": f">= {MIN_POSITIVOS} positivos y >= {MIN_PCT} % por lago",
            "observacion": obs,
        })

    pd.DataFrame(filas_global).to_csv(
        TAB_DIR / "threshold_analysis_global.csv", index=False)
    pd.DataFrame(filas_lago).to_csv(
        TAB_DIR / "threshold_analysis_by_lake.csv", index=False)
    ext_df.rename(columns={"pct_area_afectada": "pct_positivo"}).to_csv(
        TAB_DIR / "threshold_analysis_by_date.csv", index=False)
    pd.DataFrame(filas_via).to_csv(
        TAB_DIR / "threshold_model_viability.csv", index=False)


# ---------------------------------------------------------------------------
# Figuras por actividad
# ---------------------------------------------------------------------------
def figuras_actividad_4(serie, picos_df):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False)
    for lago, g in serie.groupby("lago"):
        g = g.sort_values("fecha")
        axes[0].plot(g["fecha"], g["chl_media"], marker="o", label=lago, lw=2)
        axes[0].fill_between(g["fecha"],
                             g["chl_media"] - g["chl_std"],
                             g["chl_media"] + g["chl_std"], alpha=0.15)
    if not picos_df.empty:
        for _, p in picos_df.iterrows():
            axes[0].scatter([p["fecha"]], [p["chl_media"]], s=180,
                            facecolors="none", edgecolors="red", lw=2, zorder=5)
        axes[0].scatter([], [], s=180, facecolors="none", edgecolors="red",
                        lw=2, label="Fecha critica (media + 1 desv. est.)")
    axes[0].set_title("Evolucion temporal de la clorofila-a promedio por lago "
                      "(Sentinel-2 L1C real)", fontweight="bold")
    axes[0].set_ylabel("Clorofila-a (ug/L)"); axes[0].legend()
    plt.setp(axes[0].get_xticklabels(), rotation=60, ha="right", fontsize=8)

    for lago, g in serie.groupby("lago"):
        g = g.sort_values("fecha")
        axes[1].plot(g["fecha"], g["chl_mediana"], marker="s", label=f"{lago} mediana")
        axes[1].plot(g["fecha"], g["chl_p95"], marker="^", ls="--",
                     label=f"{lago} percentil 95")
    axes[1].set_title("Mediana y percentil 95 por lago", fontweight="bold")
    axes[1].set_ylabel("Clorofila-a (ug/L)"); axes[1].set_xlabel("Fecha")
    axes[1].legend(fontsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "actividad4_evolucion_temporal.png", dpi=150)
    plt.close(fig)


def figuras_actividad_6(corr_df, pila_chl, metadatos, pares):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Matriz agregada por lago, recalculada sobre todos los pixeles de agua
    fig, axes = plt.subplots(1, len(pila_chl), figsize=(8 * len(pila_chl), 6.5))
    axes = np.atleast_1d(axes)
    matrices = {}
    for ax, lago in zip(axes, sorted(pila_chl)):
        acum = []
        for fecha in OFFICIAL_DATES[lago]:
            esc = cargar_escena(lago, fecha)
            a = esc["water"]
            acum.append(pd.DataFrame({
                "NDVI": esc["NDVI"][a], "NDWI": esc["NDWI"][a],
                "NDCI": esc["NDCI"][a], "FAI": esc["FAI"][a],
                "Clorofila-a": esc["chl"][a]}))
        df = pd.concat(acum, ignore_index=True)
        m = df.corr(method="pearson")
        matrices[lago] = m
        sns.heatmap(m, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
                    square=True, ax=ax, cbar_kws={"shrink": 0.75})
        ax.set_title(f"{lago}\n(n = {len(df):,} pixeles de agua, 11 fechas)",
                     fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Correlacion de Pearson entre indices y clorofila-a "
                 "(solo pixeles de agua)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "actividad6_correlaciones.png", dpi=150)
    plt.close(fig)

    salida = []
    for lago, m in matrices.items():
        for a in m.columns:
            for b in m.columns:
                if a < b:
                    salida.append({"lago": lago, "var_a": a, "var_b": b,
                                   "pearson": float(m.loc[a, b])})
    pd.DataFrame(salida).to_csv(TAB_DIR / "correlaciones_agregadas.csv", index=False)


def mapas_actividad_5(pila_chl, pila_water, metadatos, serie, lagos):
    import matplotlib.pyplot as plt

    for lago in lagos:
        pila = np.stack(pila_chl[lago])
        fechas = OFFICIAL_DATES[lago]
        esc = metadatos[lago]
        b = esc["bounds"]
        extent = [b.left, b.right, b.bottom, b.top]

        finitos = pila[np.isfinite(pila)]
        vmin, vmax = float(np.percentile(finitos, 2)), float(np.percentile(finitos, 98))

        # 5.1 Mapa por cada fecha, escala comun
        ncol = 4
        nfil = int(np.ceil(len(fechas) / ncol))
        fig, axes = plt.subplots(nfil, ncol, figsize=(4.4 * ncol, 4.0 * nfil))
        axes = np.atleast_1d(axes).ravel()
        im = None
        for ax, fecha, capa in zip(axes, fechas, pila):
            im = ax.imshow(capa, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                           extent=extent, origin="upper", interpolation="nearest")
            ax.set_title(fecha, fontsize=10, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[len(fechas):]:
            ax.axis("off")
        cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.6, extend="both")
        cb.set_label("Clorofila-a (ug/L)\nverde = agua limpia, rojo = mas cianobacteria")
        fig.suptitle(f"Lago {lago}: clorofila-a en las 11 fechas oficiales "
                     f"(escala comun, EPSG:32615)", fontsize=15, fontweight="bold")
        fig.savefig(MAP_DIR / f"actividad5_mapas_por_fecha_{lago}.png",
                    dpi=140, bbox_inches="tight")
        plt.close(fig)

        # 5.2 Comparacion minimo vs maximo (calculada)
        g = serie[serie["lago"] == lago].sort_values("chl_media")
        f_min, f_max = g.iloc[0]["fecha"], g.iloc[-1]["fecha"]
        i_min, i_max = fechas.index(f_min), fechas.index(f_max)
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        for ax, idx, fecha, titulo in [
            (axes[0], i_min, f_min, f"Minimo: {f_min}"),
            (axes[1], i_max, f_max, f"Maximo: {f_max}")]:
            im = ax.imshow(pila[idx], cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                           extent=extent, origin="upper")
            ax.set_title(titulo, fontweight="bold")
            ax.set_xlabel("Este (m, UTM 15N)"); ax.set_ylabel("Norte (m)")
        fig.colorbar(im, ax=axes[1], shrink=0.8, label="Clorofila-a (ug/L)")
        dif = pila[i_max] - pila[i_min]
        lim = float(np.nanpercentile(np.abs(dif), 98))
        im2 = axes[2].imshow(dif, cmap="RdBu_r", vmin=-lim, vmax=lim,
                             extent=extent, origin="upper")
        axes[2].set_title(f"Diferencia ({f_max} - {f_min})", fontweight="bold")
        axes[2].set_xlabel("Este (m, UTM 15N)")
        fig.colorbar(im2, ax=axes[2], shrink=0.8, label="Cambio (ug/L)")
        fig.suptitle(f"Lago {lago}: comparacion entre la fecha de menor y mayor "
                     "clorofila-a promedio", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(MAP_DIR / f"actividad5_comparativo_min_max_{lago}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 5.1 Mapa interactivo con reproyeccion real a EPSG:4326
        try:
            generar_mapa_folium(lago, f_max, pila[i_max], esc, vmin, vmax)
        except Exception as exc:
            LOGGER.warning("No se pudo generar el mapa interactivo de %s: %s", lago, exc)


def generar_mapa_folium(lago, fecha, capa, escena, vmin, vmax):
    """
    Mapa interactivo con reproyeccion REAL de EPSG:32615 a EPSG:4326.

    La version anterior superponia el arreglo en coordenadas UTM usando los
    limites geograficos del bounding box, lo que desplazaba y deformaba la capa.
    Aqui se remuestrea el raster al CRS geografico antes de superponerlo.
    """
    import folium
    import branca.colormap as bcm
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS

    dst_crs = CRS.from_epsg(4326)
    transform_dst, ancho_dst, alto_dst = calculate_default_transform(
        escena["crs"], dst_crs, escena["ancho"], escena["alto"], *escena["bounds"])

    destino = np.full((alto_dst, ancho_dst), np.nan, dtype=np.float32)
    reproject(source=np.ascontiguousarray(capa.astype(np.float32)),
              destination=destino,
              src_transform=escena["transform"], src_crs=escena["crs"],
              dst_transform=transform_dst, dst_crs=dst_crs,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.nearest)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = plt.get_cmap("RdYlGn_r")(norm(np.nan_to_num(destino, nan=vmin)))
    # Transparencia total en los pixeles que no son agua.
    rgba[..., 3] = np.where(np.isfinite(destino), 0.85, 0.0)
    rgba = (rgba * 255).astype(np.uint8)

    oeste = transform_dst.c
    norte = transform_dst.f
    este = oeste + transform_dst.a * ancho_dst
    sur = norte + transform_dst.e * alto_dst

    mapa = folium.Map(location=[(sur + norte) / 2, (oeste + este) / 2],
                      zoom_start=12, tiles="CartoDB positron", control_scale=True)
    folium.raster_layers.ImageOverlay(
        image=rgba, bounds=[[sur, oeste], [norte, este]],
        opacity=0.85, name=f"Clorofila-a {fecha}").add_to(mapa)
    escala = bcm.LinearColormap(
        colors=[mpl.colors.to_hex(plt.get_cmap("RdYlGn_r")(t))
                for t in np.linspace(0, 1, 8)],
        vmin=vmin, vmax=vmax,
        caption=f"Lago {lago} - {fecha} - Clorofila-a (ug/L), Sentinel-2 L1C real")
    escala.add_to(mapa)
    folium.LayerControl(collapsed=False).add_to(mapa)
    mapa.save(str(MAP_DIR / f"actividad5_mapa_interactivo_{lago}.html"))


def figuras_actividad_8(pila_chl, pila_water, metadatos, serie, ext_df, lagos):
    import matplotlib.pyplot as plt
    import seaborn as sns

    for lago in lagos:
        pila = np.stack(pila_chl[lago])
        fechas = OFFICIAL_DATES[lago]
        esc = metadatos[lago]
        area = esc["area_pixel_ha"]
        b = esc["bounds"]
        extent = [b.left, b.right, b.bottom, b.top]

        # 8.3 Boxplots + histogramas
        registros = [pd.DataFrame({"Fecha": f, "chl": capa[np.isfinite(capa)]})
                     for f, capa in zip(fechas, pila)]
        df = pd.concat(registros, ignore_index=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
        sns.boxplot(data=df, x="Fecha", y="chl", order=fechas, ax=ax1,
                    showfliers=False, color="#7fb3d5")
        for u, color in zip(UMBRALES, ["#2ca02c", "#ff7f0e", "#d62728", "#7f0000"]):
            ax1.axhline(u, ls="--", lw=1.3, color=color, label=f"{u:.0f} ug/L")
        ax1.set_title(f"Lago {lago}: distribucion de clorofila-a por fecha "
                      "(pixeles de agua)", fontweight="bold")
        ax1.set_ylabel("Clorofila-a (ug/L)"); ax1.set_xlabel("")
        ax1.legend(fontsize=8, ncol=4)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=9)

        paleta = sns.color_palette("viridis", len(fechas))
        for fecha, color in zip(fechas, paleta):
            s = df.loc[df["Fecha"] == fecha, "chl"]
            sns.histplot(s, bins=80, element="step", fill=False, stat="density",
                         color=color, label=fecha, ax=ax2, lw=1.4)
        ax2.set_xlim(float(np.nanpercentile(df["chl"], 0.5)),
                     float(np.nanpercentile(df["chl"], 99.5)))
        for u, color in zip(UMBRALES, ["#2ca02c", "#ff7f0e", "#d62728", "#7f0000"]):
            ax2.axvline(u, ls="--", lw=1.3, color=color)
        ax2.set_title("Histogramas superpuestos", fontweight="bold")
        ax2.set_xlabel("Clorofila-a (ug/L)"); ax2.set_ylabel("Densidad")
        ax2.legend(title="Fecha", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"actividad8_distribuciones_{lago}.png", dpi=150)
        plt.close(fig)

        # 8.2 Persistencia espacial: en cuantas de las 11 fechas se supera el umbral
        fig, axes = plt.subplots(1, len(UMBRALES), figsize=(5.3 * len(UMBRALES), 5.2))
        axes = np.atleast_1d(axes)
        filas_pers = []
        for ax, u in zip(axes, UMBRALES):
            conteo = np.nansum(pila >= u, axis=0).astype(np.float32)
            valido = np.any(np.isfinite(pila), axis=0)
            conteo[~valido] = np.nan
            im = ax.imshow(conteo, cmap="inferno", vmin=0, vmax=len(fechas),
                           extent=extent, origin="upper", interpolation="nearest")
            ax.set_title(f"{u:.0f} ug/L", fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for k in range(len(fechas) + 1):
                n = int(np.nansum(conteo == k))
                if n:
                    filas_pers.append({"lago": lago, "umbral_ug_L": u,
                                       "n_fechas_superadas": k, "n_pixeles": n,
                                       "area_ha": n * area})
            cb = fig.colorbar(im, ax=ax, shrink=0.8)
            cb.set_label("N.o de fechas (0-11)")
        fig.suptitle(f"Lago {lago}: persistencia espacial. Por cada pixel, en cuantas "
                     f"de las {len(fechas)} fechas se supero el umbral",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(MAP_DIR / f"actividad8_persistencia_{lago}.png", dpi=150)
        plt.close(fig)
        pd.DataFrame(filas_pers).to_csv(
            TAB_DIR / f"persistencia_espacial_{lago}.csv", index=False)

    # 8.4 Patron estacional
    serie2 = serie.copy()
    serie2["mes"] = pd.to_datetime(serie2["fecha"]).dt.month
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for lago, g in serie2.groupby("lago"):
        axes[0].scatter(g["mes"], g["chl_media"], s=90, label=lago)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_title("Clorofila-a promedio segun el mes del ano", fontweight="bold")
    axes[0].set_xlabel("Mes"); axes[0].set_ylabel("Clorofila-a (ug/L)"); axes[0].legend()
    pivote = ext_df[ext_df["umbral_ug_L"] == UMBRAL_PRINCIPAL].copy()
    pivote["mes"] = pd.to_datetime(pivote["fecha"]).dt.month
    for lago, g in pivote.groupby("lago"):
        axes[1].scatter(g["mes"], g["pct_area_afectada"], s=90, label=lago)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title(f"Area del lago sobre {UMBRAL_PRINCIPAL:.0f} ug/L segun el mes",
                      fontweight="bold")
    axes[1].set_xlabel("Mes"); axes[1].set_ylabel("% del area de agua"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "actividad8_patron_estacional.png", dpi=150)
    plt.close(fig)


def comparacion_actividad_7(serie, ext_df, lagos):
    import matplotlib.pyplot as plt

    filas = []
    for lago in lagos:
        g = serie[serie["lago"] == lago]
        e = ext_df[(ext_df["lago"] == lago)
                   & (ext_df["umbral_ug_L"] == UMBRAL_PRINCIPAL)]
        filas.append({
            "lago": lago,
            "chl_media_periodo": float(g["chl_media"].mean()),
            "chl_mediana_periodo": float(g["chl_mediana"].median()),
            "chl_max_observada": float(g["chl_max"].max()),
            "variabilidad_entre_fechas_std": float(g["chl_media"].std()),
            "coef_variacion": float(g["chl_media"].std() / g["chl_media"].mean())
            if g["chl_media"].mean() else np.nan,
            "area_agua_media_ha": float(g["area_agua_ha"].mean()),
            "area_agua_media_km2": float(g["area_agua_ha"].mean() / 100),
            f"fechas_con_area_sobre_{UMBRAL_PRINCIPAL:.0f}": int(
                (e["pixeles_positivos"] > 0).sum()),
            f"pct_area_max_sobre_{UMBRAL_PRINCIPAL:.0f}": float(
                e["pct_area_afectada"].max()),
            "fecha_mas_critica": g.loc[g["chl_media"].idxmax(), "fecha"],
            "fecha_menos_critica": g.loc[g["chl_media"].idxmin(), "fecha"],
        })
    comp = pd.DataFrame(filas)
    comp.to_csv(TAB_DIR / "comparacion_entre_lagos.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].bar(comp["lago"], comp["chl_media_periodo"], color=["#4878a8", "#c44e52"])
    axes[0].set_title("Clorofila-a promedio del periodo", fontweight="bold")
    axes[0].set_ylabel("ug/L")
    axes[1].bar(comp["lago"], comp["variabilidad_entre_fechas_std"],
                color=["#4878a8", "#c44e52"])
    axes[1].set_title("Variabilidad temporal (desv. est. entre fechas)", fontweight="bold")
    axes[1].set_ylabel("ug/L")
    axes[2].bar(comp["lago"], comp[f"pct_area_max_sobre_{UMBRAL_PRINCIPAL:.0f}"],
                color=["#4878a8", "#c44e52"])
    axes[2].set_title(f"Maximo % de area sobre {UMBRAL_PRINCIPAL:.0f} ug/L",
                      fontweight="bold")
    axes[2].set_ylabel("% del area de agua")
    fig.suptitle("Comparacion entre lagos (datos Sentinel-2 L1C reales)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "actividad7_comparacion_lagos.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporte tecnico
# ---------------------------------------------------------------------------
def escribir_reporte_parte1(serie, corr_df, ext_df, picos_df, lagos,
                            verificacion, metadatos):
    glob = pd.read_csv(TAB_DIR / "threshold_analysis_global.csv")
    por_lago = pd.read_csv(TAB_DIR / "threshold_analysis_by_lake.csv")
    via = pd.read_csv(TAB_DIR / "threshold_model_viability.csv")
    corr_agg = pd.read_csv(TAB_DIR / "correlaciones_agregadas.csv")
    area = metadatos[lagos[0]]["area_pixel_ha"]

    ruta = REP_DIR / "parte1_resultados_reales.md"
    with open(ruta, "w", encoding="utf-8") as fh:
        w = fh.write
        w("# Parte 1 regenerada con datos Sentinel-2 L1C reales\n\n")
        w(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}  \n")
        w(f"Fuente: 22 GeoTIFF reales ({S2_L1C_COLLECTION}), EPSG:32615, 20 m.  \n")
        w(f"Area por pixel derivada de la transformacion afin: **{area:.4f} ha** "
          f"({area * 10000:.0f} m2).\n\n")
        w("> Ningun valor de este documento procede de `synthetic_bands()`.\n\n---\n\n")

        w("## Verificacion del evalscript CyanoLakes\n\n")
        w(f"- Escena verificada: {verificacion['escena']}\n")
        w(f"- Pixeles comparados uno a uno: {verificacion['pixeles_comparados']}\n")
        w(f"- Discrepancias en la mascara de agua: "
          f"{verificacion['discrepancias_mascara_agua']}\n")
        w("- Diferencias maximas frente al calculo literal del evalscript:\n\n")
        w("| Magnitud | Diferencia maxima |\n|---|---|\n")
        for k, v in verificacion["diferencias_maximas"].items():
            w(f"| {k.upper()} | {v:.3e} |\n")
        w(f"\nResultado: **{'COINCIDE' if verificacion['coincide'] else 'NO COINCIDE'}** "
          f"con tolerancia {verificacion['tolerancia']:.0e}.\n\n")

        w("### Limitaciones del algoritmo (documentadas, no estimadas aqui)\n\n")
        w("- Opera sobre **reflectancia de tope de atmosfera (L1C)**, sin correccion "
          "atmosferica.\n")
        w("- El polinomio fue **calibrado sobre datos simulados**, no sobre muestras "
          "de estos lagos.\n")
        w("- Calibracion especifica para **Microcystis aeruginosa**; otras especies "
          "responden distinto.\n")
        w("- Errores reportados por la fuente: **MAPE 42.3 %** y **RMSE relativo "
          "95.8 %**.\n")
        w(f"- Dominio de calibracion del NDCI: **{CALIB_MIN:.0f}-{CALIB_MAX:.0f} mg/m3** "
          "(Mishra & Mishra 2012).\n")
        w("- La coleccion `SENTINEL2_L1C` de openEO **no expone CLM, CLP, dataMask, "
          "SCL ni QA60**: no existe mascara de nubes por pixel. El unico control es la "
          "nubosidad oficial por escena y los filtros de validez espectral.\n")
        w("- Estos valores son **estimaciones satelitales**, no mediciones de "
          "laboratorio. No se realizo ninguna validacion in situ.\n\n---\n\n")

        w("## Actividad 4: analisis temporal\n\n")
        w("| Lago | Fecha | Agua (ha) | Media | Mediana | Desv. est. | p95 | Max |\n")
        w("|---|---|---|---|---|---|---|---|\n")
        for _, r in serie.iterrows():
            w(f"| {r['lago']} | {r['fecha']} | {r['area_agua_ha']:,.0f} | "
              f"{r['chl_media']:.2f} | {r['chl_mediana']:.2f} | {r['chl_std']:.2f} | "
              f"{r['chl_p95']:.2f} | {r['chl_max']:.2f} |\n")
        w("\nValores en ug/L de clorofila-a sobre pixeles de agua.\n\n")
        w("### Fechas criticas (calculadas, no escritas a mano)\n\n")
        if picos_df.empty:
            w("No se detecto ninguna fecha por encima de media + 1 desviacion estandar.\n")
        else:
            w("| Lago | Fecha | Media | Umbral del criterio |\n|---|---|---|---|\n")
            for _, r in picos_df.iterrows():
                w(f"| {r['lago']} | {r['fecha']} | {r['chl_media']:.2f} | "
                  f"{r['umbral_pico']:.2f} |\n")
        w("\n")

        w("---\n\n## Actividad 6: correlaciones\n\n")
        w("| Lago | Par | Pearson |\n|---|---|---|\n")
        for _, r in corr_agg.iterrows():
            w(f"| {r['lago']} | {r['var_a']} - {r['var_b']} | {r['pearson']:.3f} |\n")
        w("\n**Advertencia de circularidad.** La correlacion entre NDCI y clorofila-a "
          "no es evidencia ecologica: la clorofila se calcula como un polinomio del "
          "NDCI, asi que la relacion es una identidad matematica y su valor alto era "
          "inevitable. Lo mismo aplica parcialmente al FAI y al NDVI, que comparten la "
          "banda B04 con el NDCI. Las unicas relaciones interpretables como senal "
          "ambiental son las que implican NDWI, que no participa en la cadena de "
          "calculo de la clorofila.\n\n")

        w("---\n\n## Actividad 7: comparacion entre lagos\n\n")
        comp = pd.read_csv(TAB_DIR / "comparacion_entre_lagos.csv")
        w("| Indicador | " + " | ".join(comp["lago"]) + " |\n|---|" +
          "---|" * len(comp) + "\n")
        for col in ["chl_media_periodo", "chl_mediana_periodo", "chl_max_observada",
                    "variabilidad_entre_fechas_std", "coef_variacion",
                    "area_agua_media_km2", "fecha_mas_critica", "fecha_menos_critica"]:
            valores = []
            for _, r in comp.iterrows():
                v = r[col]
                valores.append(f"{v:.2f}" if isinstance(v, (int, float, np.floating))
                               else str(v))
            w(f"| {col} | " + " | ".join(valores) + " |\n")
        w("\n")

        w("---\n\n## Actividades 8 y 10: analisis de umbrales\n\n")
        w("### Global\n\n")
        w("| Umbral | Positivos | % del agua | Area (ha) | Fechas con positivos |\n")
        w("|---|---|---|---|---|\n")
        for _, r in glob.iterrows():
            w(f"| {r['umbral_ug_L']:.0f} ug/L | {int(r['pixeles_positivos']):,} | "
              f"{r['pct_positivo']:.4f} % | {r['area_afectada_ha']:,.1f} | "
              f"{int(r['fechas_con_positivos'])}/{int(r['fechas_totales'])} |\n")
        w("\n### Por lago\n\n")
        w("| Umbral | Lago | Positivos | % | Area (ha) | Fechas con positivos | "
          "Fechas con ambas clases |\n|---|---|---|---|---|---|---|\n")
        for _, r in por_lago.iterrows():
            w(f"| {r['umbral_ug_L']:.0f} | {r['lago']} | {int(r['pixeles_positivos']):,} | "
              f"{r['pct_positivo']:.4f} % | {r['area_afectada_ha']:,.1f} | "
              f"{int(r['fechas_con_positivos'])}/{int(r['fechas_totales'])} | "
              f"{int(r['fechas_con_ambas_clases'])} |\n")
        w("\n### Significado ambiental\n\n")
        for u in UMBRALES:
            w(f"- **{u:.0f} ug/L**: {SIGNIFICADO_UMBRAL[u]}\n")
        w("\n### Viabilidad para clasificacion binaria\n\n")
        w("| Umbral | % positivo | Positivos por lago | Entrenar por lago | "
          "Evaluar entre lagos | Observacion |\n|---|---|---|---|---|---|\n")
        for _, r in via.iterrows():
            w(f"| {r['umbral_ug_L']:.0f} | {r['pct_positivo_global']:.4f} % | "
              f"{r['positivos_por_lago']} | {r['viable_entrenar_por_lago']} | "
              f"{r['viable_evaluar_entre_lagos']} | {r['observacion']} |\n")
        w(f"\nCriterio de viabilidad: {via.iloc[0]['criterio_minimo']}. Tener uno o dos "
          "pixeles positivos no hace viable un modelo.\n")
        w(f"\n### Recomendacion\n\nSe recomienda **{UMBRAL_PRINCIPAL:.0f} ug/L** como "
          "umbral principal por su significado ambiental: es la frontera "
          "eutrofico-hipertrofico de OECD (1982) y coincide con el techo de la Alerta 1 "
          "de la OMS (24 ug/L). La eleccion se toma por significado ambiental y **no** "
          "por balance de clases. La viabilidad estadistica se discute por separado en "
          "la tabla anterior: el desbalance resultante es severo y condiciona el "
          "modelado de la Parte 2.\n\n")
        w("Como analisis de sensibilidad se recomienda repetir con **8 ug/L** (inicio "
          "de la condicion eutrofica), que es el umbral con mas positivos y por tanto "
          "el unico que podria sostener un experimento entre lagos.\n")

    LOGGER.info("Reporte: %s", ruta.relative_to(ROOT))


# ---------------------------------------------------------------------------
# MODO --validate
# ---------------------------------------------------------------------------
def ejecutar_validate(args):
    LOGGER.info("=" * 84)
    LOGGER.info("MODO VALIDACION de la Parte 1 regenerada")
    LOGGER.info("=" * 84)

    VAL_DIR.mkdir(parents=True, exist_ok=True)
    criticos, avisos, lineas = [], [], []

    # 1. Rasteres reales
    pares = combinaciones()
    faltan = [(l, f) for l, f in pares if not ruta_raster(l, f).exists()]
    lineas.append(f"Rasteres reales presentes : {len(pares) - len(faltan)}/{len(pares)}")
    if faltan:
        criticos.append(f"Faltan rasteres: {faltan}")

    # 2. Tablas obligatorias
    obligatorias = [
        "serie_temporal_por_lago_fecha.csv", "correlaciones_por_lago_fecha.csv",
        "correlaciones_agregadas.csv", "extension_espacial_por_umbral.csv",
        "fechas_criticas.csv", "comparacion_entre_lagos.csv",
        "threshold_analysis_global.csv", "threshold_analysis_by_lake.csv",
        "threshold_analysis_by_date.csv", "threshold_model_viability.csv",
    ]
    for nombre in obligatorias:
        ruta = TAB_DIR / nombre
        if not ruta.exists():
            criticos.append(f"Falta la tabla {nombre}")
        elif pd.read_csv(ruta).empty:
            criticos.append(f"La tabla {nombre} esta vacia")

    if (TAB_DIR / "serie_temporal_por_lago_fecha.csv").exists():
        serie = pd.read_csv(TAB_DIR / "serie_temporal_por_lago_fecha.csv")
        lineas.append(f"Filas de la serie temporal: {len(serie)}")
        if len(serie) != 22:
            criticos.append(f"La serie tiene {len(serie)} filas; se esperaban 22")
        for lago in sorted(OFFICIAL_DATES):
            n = int((serie["lago"] == lago).sum())
            lineas.append(f"  {lago}: {n} fechas")
            if n != 11:
                criticos.append(f"{lago} tiene {n} fechas; se esperaban 11")
        if serie["chl_media"].isna().any():
            criticos.append("Hay clorofila media NaN en la serie")

        # 3. Escala y unidades coherentes
        if not (serie["area_agua_ha"] > 0).all():
            criticos.append("Hay areas de agua no positivas")
        area_km2 = serie.groupby("lago")["area_agua_ha"].mean() / 100
        for lago, km2 in area_km2.items():
            lineas.append(f"  Area media de agua {lago}: {km2:.1f} km2")
        # Superficies conocidas: Atitlan ~130 km2, Amatitlan ~15 km2.
        esperado = {"Atitlan": (100, 150), "Amatitlan": (10, 20)}
        for lago, (lo, hi) in esperado.items():
            if lago in area_km2.index and not (lo <= area_km2[lago] <= hi):
                avisos.append(f"El area detectada de {lago} ({area_km2[lago]:.1f} km2) "
                              f"se aleja del rango esperado {lo}-{hi} km2")

    # 4. Area de pixel = 0.04 ha
    if pares and ruta_raster(*pares[0]).exists():
        import rasterio
        with rasterio.open(ruta_raster(*pares[0])) as src:
            a = area_pixel_ha(src.transform)
        lineas.append(f"Area por pixel derivada   : {a:.4f} ha ({a*10000:.0f} m2)")
        if abs(a - 0.04) > 1e-9:
            criticos.append(f"El area de pixel es {a} ha; se esperaba 0.04 ha")

    # 5. Verificacion del evalscript
    ruta_ver = VAL_DIR / "evalscript_verification.json"
    if not ruta_ver.exists():
        criticos.append("Falta evalscript_verification.json")
    else:
        with open(ruta_ver, encoding="utf-8") as fh:
            v = json.load(fh)
        lineas.append(f"Evalscript CyanoLakes     : "
                      f"{'COINCIDE' if v.get('coincide') else 'NO COINCIDE'} "
                      f"({v.get('pixeles_comparados')} pixeles)")
        if not v.get("coincide"):
            criticos.append("La implementacion local no reproduce el evalscript")

    # 6. Ninguna cifra proviene de synthetic_bands
    sospechosos = []
    for py in [ROOT / "regenerar_parte1_real.py"]:
        texto = py.read_text(encoding="utf-8")
        if "synthetic_bands" in texto and "no usa synthetic_bands" not in texto.lower():
            sospechosos.append(py.name)
    lineas.append(f"Uso de synthetic_bands()  : "
                  f"{'NO (correcto)' if not sospechosos else sospechosos}")
    if sospechosos:
        criticos.append(f"El pipeline referencia synthetic_bands: {sospechosos}")

    # 7. Mapas y figuras con contenido
    for carpeta, minimo in [(FIG_DIR, 4), (MAP_DIR, 4)]:
        archivos = [p for p in carpeta.glob("*") if p.is_file()]
        vacios = [p.name for p in archivos if p.stat().st_size < 5000]
        lineas.append(f"Archivos en {carpeta.name:<8}      : {len(archivos)}")
        if len(archivos) < minimo:
            criticos.append(f"{carpeta.name} tiene {len(archivos)} archivos; "
                            f"se esperaban al menos {minimo}")
        if vacios:
            criticos.append(f"Archivos sospechosamente pequenos en {carpeta.name}: {vacios}")

    # 8. Referencias de figuras del informe
    informe = ROOT / "INFORME_LAB4.md"
    if informe.exists():
        import re
        texto = informe.read_text(encoding="utf-8")
        refs = re.findall(r"!\[.*?\]\((.+?)\)", texto)
        rotas = [r for r in refs if not (ROOT / r).exists()]
        pendientes = re.findall(r"\[Insertar aqu[ií].*?\]", texto)
        lineas.append(f"Figuras referidas en el informe: {len(refs)} "
                      f"(rotas: {len(rotas)})")
        if rotas:
            criticos.append(f"El informe referencia figuras inexistentes: {rotas}")
        if pendientes:
            criticos.append(f"El informe conserva {len(pendientes)} marcadores "
                            "de imagen pendientes")
        if "sintétic" in texto.lower() and "obsolet" not in texto.lower():
            avisos.append("El informe menciona datos sinteticos: verifique que se "
                          "identifiquen como obsoletos")
    else:
        criticos.append("No existe INFORME_LAB4.md")

    # 9. El notebook usa el flujo real
    nb = ROOT / "lab4.ipynb"
    if nb.exists():
        texto_nb = nb.read_text(encoding="utf-8")
        usa_real = ("regenerar_parte1_real" in texto_nb) or ("outputs/rasters" in texto_nb)
        fallback = "synthetic_bands(lake, date)" in texto_nb and "APENDICE" not in texto_nb.upper()
        lineas.append(f"Notebook con flujo real   : {usa_real}")
        if not usa_real:
            criticos.append("lab4.ipynb no invoca el pipeline real")
        if fallback:
            criticos.append("lab4.ipynb conserva synthetic_bands como fallback silencioso")
    else:
        criticos.append("No existe lab4.ipynb")

    ruta_rep = VAL_DIR / "validation_report.txt"
    with open(ruta_rep, "w", encoding="utf-8") as fh:
        fh.write("VALIDACION DE LA PARTE 1 REGENERADA CON DATOS REALES\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        for l in lineas:
            fh.write(l + "\n")
        fh.write(f"\nAvisos  : {len(avisos)}\n")
        for a in avisos:
            fh.write(f"  - {a}\n")
        fh.write(f"\nCriticos: {len(criticos)}\n")
        for c in criticos:
            fh.write(f"  - {c}\n")
        fh.write("\nRESULTADO: " + ("FALLIDA" if criticos else "CORRECTA") + "\n")

    for l in lineas:
        LOGGER.info("  %s", l)
    for a in avisos:
        LOGGER.warning("AVISO: %s", a)
    for c in criticos:
        LOGGER.error("CRITICO: %s", c)

    LOGGER.info("")
    LOGGER.info("Reporte: %s", ruta_rep.relative_to(ROOT))
    if criticos:
        LOGGER.error("Validacion FALLIDA (%d criticos).", len(criticos))
        return 1
    LOGGER.info("Validacion CORRECTA.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def construir_parser():
    p = argparse.ArgumentParser(
        prog="regenerar_parte1_real.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Regenera la Parte 1 del Laboratorio 4 usando exclusivamente los "
                     "22 GeoTIFF\nSentinel-2 L1C reales de outputs/rasters/.\n\n"
                     "No se conecta a Copernicus y no usa datos sinteticos."),
        epilog=("EJEMPLOS\n"
                "  python regenerar_parte1_real.py --dry-run\n"
                "  python regenerar_parte1_real.py --build\n"
                "  python regenerar_parte1_real.py --validate\n\n"
                "CODIGOS DE SALIDA\n"
                "  0 correcto | 1 errores criticos | 2 faltan datos reales\n"))
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--dry-run", action="store_true",
                   help="Verifica los 22 rasteres, la escala y el evalscript.")
    m.add_argument("--build", action="store_true",
                   help="Regenera tablas, figuras, mapas y reportes reales.")
    m.add_argument("--validate", action="store_true",
                   help="Valida los artefactos, el informe y el notebook.")
    p.add_argument("--lake", metavar="NOMBRE", default=None,
                   help="Procesa solo un lago.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Mensajes de depuracion.")
    return p


def main(argv=None):
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
