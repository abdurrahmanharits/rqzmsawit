"""Dashboard Kebun Sawit — sumber data: db/data-okta.gpkg.

Semua kolom kategori/numerik untuk filter, peta, dan pembentukan zona RQZM
dideteksi otomatis dari skema layer yang sedang dibuka (bukan nama kolom
yang di-hardcode), supaya halaman ini tetap berfungsi walau skema gpkg
berubah atau diganti layer/dataset lain.
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from shapely import wkb as shapely_wkb
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore", message=r".*choropleth_mapbox.*deprecated.*", category=DeprecationWarning)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from nav import render_sidebar

ROOT = Path(__file__).resolve().parents[1]
GPKG_PATH = ROOT / "db" / "data-okta.gpkg"
LAYER_NAME = "joinkarakteristiksawitokta"

# Kolom bookkeeping generik ala GIS/OGR (bukan variabel domain) — selalu dikecualikan
# dari pool kandidat, apa pun nama layer/atribut aslinya.
TECHNICAL_COLS = {"id_blok", "OBJECTID", "JobID", "QLevel", "SHAPE_Leng", "SHAPE_Area", "fid"}

_GPKG_ENVELOPE_SIZES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _parse_gpkg_geometry(blob: bytes):
    """Decode a GeoPackage Binary (GPB) blob to a shapely geometry.

    Avoids requiring fiona/pyogrio (and their GDAL system dependency) just to
    read a .gpkg — the format is documented (OGC GeoPackage spec) and is a
    thin header wrapping a standard WKB geometry.
    """
    flags = blob[3]
    if (flags >> 4) & 0x01:  # empty-geometry flag
        return None
    envelope_indicator = (flags >> 1) & 0x07
    header_len = 8 + _GPKG_ENVELOPE_SIZES.get(envelope_indicator, 0)
    return shapely_wkb.loads(blob[header_len:])


def _read_gpkg_layer(path: str, layer: str):
    """Read a GeoPackage layer into a GeoDataFrame using stdlib sqlite3 + shapely only."""
    import geopandas as gpd

    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name=?", (layer,))
        geom_col, srs_id = cur.fetchone()
        cur.execute("SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=?", (srs_id,))
        crs_wkt = cur.fetchone()[0]
        df = pd.read_sql_query(f'SELECT * FROM "{layer}"', con)
    finally:
        con.close()

    geoms = df[geom_col].apply(_parse_gpkg_geometry)
    return gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geoms, crs=crs_wkt)


def detect_numeric_columns(df: pd.DataFrame, min_numeric_frac: float = 0.5) -> list[str]:
    cols = []
    for c in df.columns:
        if c in TECHNICAL_COLS:
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() > 0 and coerced.notna().mean() >= min_numeric_frac:
            cols.append(c)
    return cols


def detect_categorical_columns(df: pd.DataFrame, min_unique: int = 2, max_unique: int = 30) -> list[str]:
    cols = []
    for c in df.columns:
        if c in TECHNICAL_COLS or df[c].dtype.kind not in "OU":
            continue
        nun = df[c].nunique(dropna=True)
        if min_unique <= nun <= max_unique:
            cols.append(c)
    return cols


@st.cache_data(show_spinner="Memuat data-okta.gpkg...")
def load_data(path: str, layer: str) -> tuple[pd.DataFrame, dict, tuple[float, float], str]:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(), {}, (111.4, -0.05), f"File tidak ditemukan: {p}"
    try:
        import geopandas as gpd
    except Exception as exc:
        return pd.DataFrame(), {}, (111.4, -0.05), f"geopandas belum terpasang: {exc}"

    gdf = _read_gpkg_layer(str(p), layer)
    gdf["id_blok"] = np.arange(len(gdf))  # kunci join ke geojson feature id

    centroid_proj = gdf.geometry.centroid  # dihitung di CRS proyeksi (UTM) sebelum reproject
    center_point = gpd.GeoSeries(centroid_proj, crs=gdf.crs).to_crs(4326)
    center = (float(center_point.x.mean()), float(center_point.y.mean()))

    gdf_ll = gdf.to_crs(4326)
    geojson_obj = gdf_ll.__geo_interface__

    df = pd.DataFrame(gdf.drop(columns="geometry"))
    for c in detect_numeric_columns(df):  # kolom numerik gpkg bisa datang sebagai String (mis. YOP)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, geojson_obj, center, ""


st.set_page_config(page_title="Kebun Sawit — OKTA", layout="wide")
render_sidebar(__file__)

st.title("🌴 Dashboard RQZM-Sawit")
st.caption(f"Sumber: `db/data-okta.gpkg` — layer `{LAYER_NAME}`.")

df, geojson_obj, map_center, err = load_data(str(GPKG_PATH), LAYER_NAME)
if err:
    st.warning(err)
    st.stop()

numeric_candidates = detect_numeric_columns(df)
categorical_candidates = detect_categorical_columns(df)

# ---- Filter (kolom kategori dideteksi otomatis) ---- #
st.sidebar.markdown("### Filter")
active_filters: dict[str, list[str]] = {}
for col in categorical_candidates:
    options = sorted(df[col].dropna().astype(str).unique().tolist())
    sel = st.sidebar.multiselect(col, options=options, default=[])
    if sel:
        active_filters[col] = sel

filtered = df.copy()
for col, sel in active_filters.items():
    filtered = filtered[filtered[col].astype(str).isin(sel)]

for col in numeric_candidates:
    if filtered[col].notna().any():
        v_min, v_max = float(filtered[col].min()), float(filtered[col].max())
        if v_min < v_max and st.sidebar.checkbox(f"Filter rentang: {col}", value=False):
            lo, hi = st.sidebar.slider(col, min_value=v_min, max_value=v_max, value=(v_min, v_max))
            filtered = filtered[filtered[col].between(lo, hi)]

selected_ids = set(filtered["id_blok"].tolist())
st.caption(f"{len(filtered):,} dari {len(df):,} blok sesuai filter saat ini.")

# ---- Ringkasan numerik (dinamis atas semua kolom numerik terdeteksi) ---- #
st.markdown("### Ringkasan")
c1, c2 = st.columns([1, 3])
c1.metric("Jumlah blok", f"{len(filtered):,}")
if numeric_candidates:
    summary_tbl = filtered[numeric_candidates].agg(["sum", "mean", "min", "max"]).T
    summary_tbl.index.name = "variabel"
    c2.dataframe(summary_tbl.reset_index(), width="stretch", hide_index=True)

# ---- Peta ---- #
st.markdown("### Peta Blok Kebun")
if geojson_obj and len(filtered) and numeric_candidates:
    color_col = st.selectbox("Warnai peta berdasarkan variabel", options=numeric_candidates)
    map_geojson = {
        "type": "FeatureCollection",
        "features": [f for f in geojson_obj["features"] if f["properties"].get("id_blok") in selected_ids],
    }
    fig_map = px.choropleth_mapbox(
        filtered, geojson=map_geojson, locations="id_blok", featureidkey="properties.id_blok",
        color=color_col, color_continuous_scale="YlGn",
        hover_data=categorical_candidates[:4],
        mapbox_style="carto-positron", zoom=11.5,
        center={"lon": map_center[0], "lat": map_center[1]},
        opacity=0.75, height=520,
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, width="stretch")
else:
    st.info("Tidak ada data/variabel numerik untuk ditampilkan pada peta dengan filter saat ini.")

# ---- Tabel ---- #
st.markdown("### Tabel Atribut")
show_cols = categorical_candidates + numeric_candidates
st.dataframe(filtered[show_cols].reset_index(drop=True), width="stretch")
st.download_button(
    "Download CSV (data terfilter)",
    data=filtered[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="okta_sawit_filtered.csv",
    mime="text/csv",
)

# ---- Agregasi kategori x numerik (dipilih pengguna, bukan hardcode) ---- #
if categorical_candidates and numeric_candidates:
    st.markdown("### Agregasi per Kategori")
    ag1, ag2 = st.columns(2)
    with ag1:
        agg_cat = st.selectbox("Kelompokkan berdasarkan", options=categorical_candidates)
    with ag2:
        agg_num = st.selectbox("Jumlahkan variabel", options=numeric_candidates)
    agg_df = filtered.groupby(agg_cat, dropna=False)[agg_num].sum().reset_index().sort_values(agg_num, ascending=False)
    fig_bar = px.bar(agg_df, x=agg_cat, y=agg_num, title=f"Total {agg_num} per {agg_cat}")
    st.plotly_chart(fig_bar, width="stretch")

# =============================================================================
# Pembentukan Zona (RQZM) — variabel pembangun dipilih dinamis oleh pengguna
# =============================================================================
st.markdown("---")
st.markdown("## Pembentukan Zona (RQZM)")
st.caption(
    "Pilih sendiri variabel numerik yang menjadi dimensi zonasi (bukan kolom tetap). "
    "Metode: standardisasi (Z-score) -> jarak Euclidean -> Ward linkage -> potong k klaster."
)

rqzm_vars = st.multiselect(
    "Variabel pembangun RQZM",
    options=numeric_candidates,
    default=numeric_candidates[: min(3, len(numeric_candidates))],
)

if len(rqzm_vars) < 2:
    st.info("Pilih minimal 2 variabel numerik untuk membentuk zona.")
else:
    work = filtered.dropna(subset=rqzm_vars).copy()
    k = st.slider("Jumlah zona (k)", min_value=2, max_value=8, value=3, step=1)

    if len(work) < k:
        st.warning(f"Data valid ({len(work)}) lebih sedikit dari k={k}. Kurangi k atau longgarkan filter.")
    else:
        X = StandardScaler().fit_transform(work[rqzm_vars].to_numpy(dtype=float))
        Z = linkage(X, method="ward")
        work["zona"] = fcluster(Z, t=int(k), criterion="maxclust").astype(int)

        counts = work.groupby("zona").size().rename("n").reset_index()
        counts["pct"] = (counts["n"] / counts["n"].sum() * 100).round(1)

        zc1, zc2 = st.columns([1, 2])
        with zc1:
            fig_zone_bar = px.bar(counts, x="zona", y="n", title="Jumlah Blok per Zona")
            fig_zone_bar.update_xaxes(type="category")
            st.plotly_chart(fig_zone_bar, width="stretch")
        with zc2:
            profile = work.groupby("zona", dropna=False)[rqzm_vars].mean().reset_index()
            st.markdown("**Profil rata-rata per zona (skala asli)**")
            st.dataframe(profile, width="stretch", hide_index=True)

        long_profile = profile.melt(id_vars="zona", value_vars=rqzm_vars, var_name="variabel", value_name="nilai")
        long_profile["zscore"] = long_profile.groupby("variabel")["nilai"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0)
        )
        long_profile["zona"] = long_profile["zona"].astype(str)
        fig_profile = px.line(long_profile, x="variabel", y="zscore", color="zona", markers=True,
                              title="Profil Relatif Antar Zona (Z-score)")
        st.plotly_chart(fig_profile, width="stretch")

        if geojson_obj:
            zone_geojson = {
                "type": "FeatureCollection",
                "features": [f for f in geojson_obj["features"] if f["properties"].get("id_blok") in set(work["id_blok"])],
            }
            work["zona_str"] = work["zona"].astype(str)
            fig_zone_map = px.choropleth_mapbox(
                work, geojson=zone_geojson, locations="id_blok", featureidkey="properties.id_blok",
                color="zona_str", hover_data=[c for c in categorical_candidates[:3]] + rqzm_vars,
                mapbox_style="carto-positron", zoom=11.5,
                center={"lon": map_center[0], "lat": map_center[1]},
                opacity=0.8, height=520, title="Peta Zona RQZM",
            )
            fig_zone_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title_text="Zona")
            st.plotly_chart(fig_zone_map, width="stretch")

        st.markdown("**Sebaran variabel terpilih per zona**")
        box_long = work.melt(id_vars="zona_str", value_vars=rqzm_vars, var_name="variabel", value_name="nilai")
        fig_box = px.box(box_long, x="variabel", y="nilai", color="zona_str", points=False,
                         title="Distribusi Variabel per Zona")
        st.plotly_chart(fig_box, width="stretch")

        dl_cols = categorical_candidates + rqzm_vars + ["zona"]
        st.download_button(
            "Download CSV (hasil zonasi RQZM)",
            data=work[dl_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"okta_rqzm_zona_k{k}.csv",
            mime="text/csv",
        )
