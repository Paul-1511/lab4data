import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

CMAP_CYANO = "RdYlGn_r"
UNIDAD_CYANO = "Clorofila-a (µg/L)"


def _rango_robusto(*rasters, pmin=2, pmax=98):
    juntos = np.concatenate([r[np.isfinite(r)].ravel() for r in rasters])
    if juntos.size == 0:
        return 0.0, 1.0
    return float(np.percentile(juntos, pmin)), float(np.percentile(juntos, pmax))




# __ ACTIVIDAD 5 - ANALISIS ESPACIAL ____________________________________________

def mapa_calor_comparativo(raster_fecha_a, raster_fecha_b, fecha_a, fecha_b,
                           nombre_lago, cmap=CMAP_CYANO, vmin=None, vmax=None,
                           unidad=UNIDAD_CYANO, guardar_en=None):
    if vmin is None or vmax is None:
        rmin, rmax = _rango_robusto(raster_fecha_a, raster_fecha_b)
        vmin = rmin if vmin is None else vmin
        vmax = rmax if vmax is None else vmax

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    im = None
    for ax, raster, fecha in zip(axes, (raster_fecha_a, raster_fecha_b), (fecha_a, fecha_b)):
        im = ax.imshow(raster, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{fecha}", fontsize=15, fontweight="bold")
        ax.set_xlabel("Oeste  →  Este")
        ax.set_ylabel("Norte  →  Sur")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, extend="both")
    cbar.set_label(f"{unidad}\n(verde = agua sana, rojo = mayor cianobacteria)", fontsize=12)
    fig.suptitle(f"Lago {nombre_lago}: comparación de cianobacteria entre dos fechas",
                 fontsize=17, fontweight="bold")
    if guardar_en:
        fig.savefig(guardar_en, dpi=150, bbox_inches="tight")
    return fig


def _raster_a_rgba(raster_data, cmap=CMAP_CYANO, vmin=None, vmax=None, opacidad=0.85):
    if vmin is None or vmax is None:
        rmin, rmax = _rango_robusto(raster_data)
        vmin = rmin if vmin is None else vmin
        vmax = rmax if vmax is None else vmax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = plt.get_cmap(cmap)(norm(np.nan_to_num(raster_data, nan=vmin)))
    rgba[..., 3] = np.where(np.isfinite(raster_data), opacidad, 0.0)
    return (rgba * 255).astype(np.uint8), vmin, vmax


def mapa_interactivo_cianobacteria(raster_data, lake_bbox, fecha, nombre_lago,
                                   cmap=CMAP_CYANO, vmin=None, vmax=None,
                                   unidad=UNIDAD_CYANO, guardar_en=None):
    import folium
    import branca.colormap as bcm

    rgba, vmin, vmax = _raster_a_rgba(raster_data, cmap, vmin, vmax)
    south, west = lake_bbox["south"], lake_bbox["west"]
    north, east = lake_bbox["north"], lake_bbox["east"]
    centro = [(south + north) / 2, (west + east) / 2]

    mapa = folium.Map(location=centro, zoom_start=12, tiles="CartoDB positron",
                      control_scale=True)
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        opacity=0.85,
        name=f"Cianobacteria {fecha}",
    ).add_to(mapa)

    escala = bcm.LinearColormap(
        colors=[mpl.colors.to_hex(plt.get_cmap(cmap)(t)) for t in np.linspace(0, 1, 8)],
        vmin=vmin, vmax=vmax,
        caption=f"Lago {nombre_lago} — {fecha} — {unidad}",
    )
    escala.add_to(mapa)
    folium.LayerControl(collapsed=False).add_to(mapa)
    if guardar_en:
        mapa.save(guardar_en)
    return mapa





# __ ACTIVIDAD 6 - CORRELACION ____________________________________________

def matriz_correlacion(ndvi_data, ndwi_data, cyano_data, lake_mask=None,
                       nombre_lago="", guardar_en=None):
    valido = np.isfinite(ndvi_data) & np.isfinite(ndwi_data) & np.isfinite(cyano_data)
    if lake_mask is not None:
        valido &= lake_mask.astype(bool)

    df = pd.DataFrame({
        "NDVI (vegetación)": ndvi_data[valido].ravel(),
        "NDWI (agua)": ndwi_data[valido].ravel(),
        "Cianobacteria": cyano_data[valido].ravel(),
    })
    corr = df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, annot_kws={"size": 14},
                cbar_kws={"label": "Correlación de Pearson (-1 a 1)"}, ax=ax)
    titulo = "Relación entre vegetación, agua y cianobacteria"
    if nombre_lago:
        titulo += f" — Lago {nombre_lago}"
    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=14)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    if guardar_en:
        fig.savefig(guardar_en, dpi=150, bbox_inches="tight")
    return corr, fig




# __ ACTIVIDAD 8 - ANALISIS EXPLORATORIO ____________________________________________

def extension_espacial(raster_data, umbral, lake_mask=None, area_pixel_ha=None):
    if lake_mask is not None:
        valores = raster_data[lake_mask.astype(bool)]
    else:
        valores = raster_data
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        return {"porcentaje": 0.0, "pixeles_afectados": 0, "pixeles_totales": 0}

    afectados = int(np.sum(valores > umbral))
    total = int(valores.size)
    resultado = {
        "umbral": umbral,
        "porcentaje": afectados / total * 100,
        "pixeles_afectados": afectados,
        "pixeles_totales": total,
    }
    if area_pixel_ha is not None:
        resultado["area_afectada_ha"] = afectados * area_pixel_ha
        resultado["area_total_ha"] = total * area_pixel_ha
    return resultado


def comparar_distribuciones(cyano_por_fecha, nombre_lago, umbral=None,
                            unidad=UNIDAD_CYANO, guardar_en=None):
    orden = list(cyano_por_fecha.keys())
    registros = []
    for fecha in orden:
        vals = cyano_por_fecha[fecha]
        vals = vals[np.isfinite(vals)].ravel()
        registros.append(pd.DataFrame({"Fecha": fecha, "valor": vals}))
    df = pd.concat(registros, ignore_index=True)

    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(14, 11))

    sns.boxplot(data=df, x="Fecha", y="valor", order=orden, ax=ax_box,
                palette="YlOrRd", showfliers=False, width=0.6)
    ax_box.set_title(f"Lago {nombre_lago}: distribución de cianobacteria por fecha",
                     fontsize=15, fontweight="bold")
    ax_box.set_xlabel("")
    ax_box.set_ylabel(unidad)
    plt.setp(ax_box.get_xticklabels(), rotation=40, ha="right", fontsize=11)
    if umbral is not None:
        ax_box.axhline(umbral, color="red", linestyle="--", linewidth=1.5,
                       label=f"Umbral de alerta ({umbral} µg/L)")
        ax_box.legend(loc="upper right")

    paleta = sns.color_palette("viridis", len(orden))
    for fecha, color in zip(orden, paleta):
        serie = df.loc[df["Fecha"] == fecha, "valor"]
        sns.histplot(serie, bins=40, element="step", fill=False, stat="density",
                     color=color, label=fecha, ax=ax_hist, linewidth=1.6)
    ax_hist.set_title("Histogramas superpuestos (patrones estacionales)",
                      fontsize=15, fontweight="bold")
    ax_hist.set_xlabel(unidad)
    ax_hist.set_ylabel("Densidad de píxeles")
    ax_hist.legend(title="Fecha", ncol=2, fontsize=9, title_fontsize=10)
    if umbral is not None:
        ax_hist.axvline(umbral, color="red", linestyle="--", linewidth=1.5)

    fig.tight_layout()
    if guardar_en:
        fig.savefig(guardar_en, dpi=150, bbox_inches="tight")
    return fig
