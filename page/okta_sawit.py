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
from collections import deque
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
TECHNICAL_COLS = {
    "id_blok", "OBJECTID", "JobID", "QLevel", "SHAPE_Leng", "SHAPE_Area", "fid",
    "_x_utm", "_y_utm",  # centroid blok (CRS proyeksi asli gpkg) — dipakai suku kontiguitas RQZM
}

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
    df["_x_utm"] = centroid_proj.x.to_numpy()
    df["_y_utm"] = centroid_proj.y.to_numpy()
    for c in detect_numeric_columns(df):  # kolom numerik gpkg bisa datang sebagai String (mis. YOP)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, geojson_obj, center, ""


# =============================================================================
# Kontiguitas spasial (RQZM-II) — bobot tetangga, sliver merge, diagnostik.
# =============================================================================
def _build_spatial_weights(coords: np.ndarray, knn_k: int = 5):
    """KNN weights: dipakai untuk struktur ketetanggaan, diagnostik, dan merge sliver."""
    try:
        import libpysal
    except Exception:
        return None
    if len(coords) < 2:
        return None
    k_use = min(max(1, int(knn_k)), len(coords) - 1)
    w = libpysal.weights.KNN.from_array(coords, k=k_use, silence_warnings=True)
    w.transform = "R"
    return w


def _build_neighbor_dict(w) -> dict[int, list[int]]:
    return {int(i): [int(x) for x in neigh] for i, neigh in w.neighbors.items()}


def _connected_components_for_cluster(labels: np.ndarray, neighbors: dict[int, list[int]], target_cluster: int) -> list[list[int]]:
    members = set(np.where(labels == target_cluster)[0].tolist())
    visited: set[int] = set()
    comps: list[list[int]] = []
    for node in members:
        if node in visited:
            continue
        q = deque([node])
        visited.add(node)
        comp: list[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in neighbors.get(cur, []):
                if nb in members and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def _merge_small_components(
    work: pd.DataFrame,
    feature_used: list[str],
    w,
    cluster_col: str = "cluster_raw",
    min_component_size: int = 10,
    min_component_share: float = 0.02,
) -> pd.DataFrame:
    """Gabungkan 'sliver' spasial kecil ke zona tetangga ber-atribut terdekat (pasca-hoc)."""
    out = work.copy()
    labels = out[cluster_col].to_numpy().copy()
    neighbors = _build_neighbor_dict(w)

    changed = True
    iter_count = 0
    while changed and iter_count < 10:
        changed = False
        iter_count += 1
        unique_clusters = sorted(pd.unique(labels).tolist())

        centroids: dict[int, np.ndarray] = {}
        cluster_sizes: dict[int, int] = {}
        for c in unique_clusters:
            mask = labels == c
            cluster_sizes[c] = int(mask.sum())
            centroids[c] = out.loc[mask, feature_used].mean(numeric_only=True).to_numpy(dtype=float)

        for c in unique_clusters:
            for comp in _connected_components_for_cluster(labels, neighbors, c):
                comp_size = len(comp)
                if comp_size == 0:
                    continue
                share = comp_size / max(cluster_sizes.get(c, 1), 1)
                if comp_size >= int(min_component_size) and share >= float(min_component_share):
                    continue

                border_clusters: set[int] = set()
                for node in comp:
                    for nb in neighbors.get(node, []):
                        if labels[nb] != c:
                            border_clusters.add(int(labels[nb]))
                if not border_clusters:
                    continue

                comp_vec = out.iloc[comp][feature_used].mean(numeric_only=True).to_numpy(dtype=float)
                best_cluster, best_dist = None, np.inf
                for bc in sorted(border_clusters):
                    d = float(np.linalg.norm(comp_vec - centroids[bc]))
                    if d < best_dist:
                        best_dist, best_cluster = d, bc
                if best_cluster is not None:
                    labels[np.array(comp, dtype=int)] = int(best_cluster)
                    changed = True

    out["cluster_clean"] = labels.astype(int)
    return out


def _fragmentation_metrics(res_df: pd.DataFrame, w, cluster_col: str = "cluster") -> pd.DataFrame:
    if cluster_col not in res_df.columns or w is None:
        return pd.DataFrame()
    labels = res_df[cluster_col].to_numpy(dtype=int)
    neighbors = _build_neighbor_dict(w)
    rows = []
    for c in sorted(pd.unique(labels).tolist()):
        comps = _connected_components_for_cluster(labels, neighbors, c)
        comp_sizes = sorted([len(x) for x in comps], reverse=True)
        n_members = int(np.sum(labels == c))
        rows.append(
            {
                "zona": int(c),
                "n_members": n_members,
                "n_components": len(comps),
                "largest_component": comp_sizes[0] if comp_sizes else 0,
                "largest_component_share": (comp_sizes[0] / n_members) if n_members > 0 and comp_sizes else np.nan,
                "fragmentation_index": (len(comps) / n_members) if n_members > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("zona").reset_index(drop=True)


def _morans_I_vec(values: np.ndarray, w) -> float:
    if w is None:
        return np.nan
    W = w.sparse
    s0 = float(W.sum())
    z = np.asarray(values, dtype=float)
    z = z - z.mean()
    den = s0 * float(z @ z)
    if den == 0:
        return np.nan
    return float(len(z) * float(z @ (W @ z)) / den)


def _augmented_matrix(work: pd.DataFrame, z_cols: list[str], xy_cols: list[str], beta: float) -> np.ndarray:
    Xa = work[z_cols].to_numpy(dtype=float)
    if beta and beta > 0:
        Xc = float(beta) * work[xy_cols].to_numpy(dtype=float)
        return np.hstack([Xa, Xc])
    return Xa


def _cluster_labels(work: pd.DataFrame, z_cols: list[str], xy_cols: list[str], beta: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    X = _augmented_matrix(work, z_cols, xy_cols, beta)
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=int(k), criterion="maxclust").astype(int)
    return labels, Z


def _avg_cv_multivar(labels: np.ndarray, values: pd.DataFrame) -> float:
    """CV rata-rata antar semua variabel RQZM per zona (generalisasi dari CV skor tunggal)."""
    tmp = values.copy()
    tmp["__cluster"] = labels
    per_var_cv = []
    for col in values.columns:
        stats = tmp.groupby("__cluster")[col].agg(["mean", "std"])
        cv = (stats["std"] / stats["mean"].replace(0, np.nan)).mean()
        per_var_cv.append(cv)
    return float(np.nanmean(per_var_cv))


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
st.markdown("## Pembentukan Zona")
st.caption(
    "Pilih sendiri variabel numerik yang menjadi dimensi zonasi (bukan kolom tetap). "
    "Metode: standardisasi (Z-score) -> Ward linkage -> potong k klaster, opsional ditambah "
    "bobot kontiguitas spasial (posisi blok) supaya zona kompak & bertetangga (RQZM-II)."
)

rqzm_vars = st.multiselect(
    "Variabel pembangun RQZM",
    options=numeric_candidates,
    default=numeric_candidates[: min(3, len(numeric_candidates))],
)

if len(rqzm_vars) < 2:
    st.info("Pilih minimal 2 variabel numerik untuk membentuk zona.")
else:
    st.markdown("**Pengaturan Jumlah Zona**")
    k = st.slider("Number of zones (k)", min_value=2, max_value=8, value=3, step=1)
    st.caption(f"Zona dihitung dari `{LAYER_NAME}` ({len(rqzm_vars)} variabel terstandardisasi + suku kontiguitas).")

    st.markdown("**Bobot Kontiguitas Spasial (β) & Struktur Ketetanggaan**")
    st.caption("β = 0 setara Non-Contiguous; β makin besar = lokasi makin menarik tetangga ke zona yang sama.")
    beta = st.select_slider("Bobot kontiguitas β", options=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0], value=0.5)
    knn_k = st.slider("Tetangga KNN (struktur kontiguitas & diagnostik)", min_value=3, max_value=10, value=5, step=1)

    st.markdown("**Pembersihan Sliver (pasca-hoc, opsional)**")
    do_merge = st.checkbox("Gabungkan sliver spasial kecil ke zona tetangga ber-atribut terdekat", value=True)
    cmid_l, cmid_r = st.columns(2)
    with cmid_l:
        min_component_size = st.slider("Min ukuran patch (blok)", min_value=3, max_value=30, value=10, step=1)
    with cmid_r:
        min_component_share = st.slider("Min share patch dalam zona", min_value=0.01, max_value=0.10, value=0.02, step=0.01)

    work = filtered.dropna(subset=rqzm_vars + ["_x_utm", "_y_utm"]).reset_index(drop=True)

    if len(work) < k:
        st.warning(f"Data valid ({len(work)}) lebih sedikit dari k={k}. Kurangi k atau longgarkan filter.")
    else:
        z_cols = [f"__z__{c}" for c in rqzm_vars]
        work[z_cols] = StandardScaler().fit_transform(work[rqzm_vars].to_numpy(dtype=float))
        xy_cols = ["__z__x_utm", "__z__y_utm"]
        work[xy_cols] = StandardScaler().fit_transform(work[["_x_utm", "_y_utm"]].to_numpy(dtype=float))

        coords = work[["_x_utm", "_y_utm"]].to_numpy(dtype=float)
        w_knn = _build_spatial_weights(coords, knn_k=int(knn_k))
        if w_knn is None:
            st.info("libpysal belum terpasang: diagnostik kontiguitas & sliver-merge dilewati (zonasi tetap jalan).")

        # ---- Sweep β (langkah pendefinisi kontiguitas RQZM-II) ---- #
        st.markdown("### Pemilihan β")
        beta_options = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
        beta_rows = []
        for b in beta_options:
            labels_b, _ = _cluster_labels(work, z_cols, xy_cols, b, int(k))
            avg_cv = _avg_cv_multivar(labels_b, work[rqzm_vars])
            if w_knn is not None:
                frag_b = _fragmentation_metrics(pd.DataFrame({"zona": labels_b}), w_knn, cluster_col="zona")
                mean_share = float(frag_b["largest_component_share"].mean()) if len(frag_b) else np.nan
                total_comp = int(frag_b["n_components"].sum()) if len(frag_b) else -1
            else:
                mean_share, total_comp = np.nan, -1
            beta_rows.append({"beta": b, "avg_cv": avg_cv, "mean_largest_comp_share": mean_share, "total_components": total_comp})
        beta_df = pd.DataFrame(beta_rows)
        beta_df["rank_cv"] = beta_df["avg_cv"].rank(ascending=True, method="min")
        pos_rows = beta_df[beta_df["beta"] > 0]
        best_beta_row = (pos_rows if len(pos_rows) else beta_df).sort_values(["rank_cv", "beta"]).iloc[0]
        st.info(
            f"Rekomendasi β (di antara β>0): β={best_beta_row['beta']:g} — CV rata-rata terendah pada k={int(k)}. "
            f"Periksa juga `mean_largest_comp_share` (makin tinggi makin kontigu) sebelum memutuskan."
        )
        st.dataframe(beta_df, width="stretch", hide_index=True)
        fig_beta = px.line(beta_df, x="beta", y="avg_cv", markers=True,
                           title=f"Avg within-zone CV vs β (k={int(k)}) — semakin rendah semakin homogen")
        fig_beta.update_traces(line={"width": 3}, marker={"size": 10})
        fig_beta.update_layout(height=320, xaxis_title="β (bobot kontiguitas)", yaxis_title="Avg CV (variabel RQZM)")
        st.plotly_chart(fig_beta, width="stretch")
        st.caption(f"β yang dipakai untuk zona final = {beta:g} (atur lewat slider di atas).")

        # ---- Pemilihan k pada β terpilih ---- #
        st.markdown(f"**Pemilihan k pada β={beta:g}**")
        k_sweep_values = tuple(v for v in range(2, 9) if v <= len(work))
        k_rows = []
        for kk in k_sweep_values:
            labels_k, _ = _cluster_labels(work, z_cols, xy_cols, float(beta), kk)
            k_rows.append({"k": kk, "avg_cv": _avg_cv_multivar(labels_k, work[rqzm_vars])})
        k_df = pd.DataFrame(k_rows)
        if len(k_df):
            k_df["rank_cv"] = k_df["avg_cv"].rank(ascending=True, method="min")
            best_k_row = k_df.sort_values(["rank_cv", "k"]).iloc[0]
            st.info(f"Auto recommendation: k={int(best_k_row['k'])} (CV rata-rata dalam-zona terendah).")
            st.dataframe(k_df, width="stretch", hide_index=True)

        # ---- Zonasi final: Ward atas [atribut | beta*koordinat] + sliver merge opsional ---- #
        labels, Z = _cluster_labels(work, z_cols, xy_cols, float(beta), int(k))
        work["cluster_raw"] = labels

        if do_merge and w_knn is not None:
            work = _merge_small_components(
                work, feature_used=z_cols, w=w_knn, cluster_col="cluster_raw",
                min_component_size=int(min_component_size), min_component_share=float(min_component_share),
            )
            work["zona"] = work["cluster_clean"].astype(int)
        else:
            work["zona"] = work["cluster_raw"].astype(int)

        st.caption(
            f"Fitur zona: [{', '.join(rqzm_vars)} | β×koordinat blok] dengan β={beta:g}. "
            f"Merge sliver: {'ON' if do_merge else 'OFF'}."
        )

        counts = work.groupby("zona").size().rename("n").reset_index()
        counts["pct"] = (counts["n"] / counts["n"].sum() * 100).round(1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total blok", f"{len(work):,}")
        c2.metric("Jumlah zona", f"{work['zona'].nunique():,}")
        c3.metric("Rata-rata per zona", f"{len(work) / max(work['zona'].nunique(), 1):.1f}")

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
                opacity=0.8, height=520, title=f"Peta Zona RQZM (β={beta:g})",
            )
            fig_zone_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title_text="Zona")
            st.plotly_chart(fig_zone_map, width="stretch")

        st.markdown("**Sebaran variabel terpilih per zona**")
        box_long = work.melt(id_vars="zona_str", value_vars=rqzm_vars, var_name="variabel", value_name="nilai")
        fig_box = px.box(box_long, x="variabel", y="nilai", color="zona_str", points=False,
                         title="Distribusi Variabel per Zona")
        st.plotly_chart(fig_box, width="stretch")

        # ---- Diagnostik kontiguitas / kompaksitas ---- #
        st.markdown("**Diagnostik Kontiguitas / Kompaksitas per Zona**")
        st.caption(
            "n_components rendah & largest_component_share tinggi (mendekati 1) = zona makin kontigu/kompak. "
            "Bandingkan dengan β=0 (Non-Contiguous) untuk melihat efek suku kontiguitas."
        )
        if w_knn is not None:
            frag_tbl = _fragmentation_metrics(work, w_knn, cluster_col="zona")
            st.dataframe(frag_tbl, width="stretch", hide_index=True)

            moran_rows = []
            for zc in sorted(work["zona"].unique().tolist()):
                z_ind = (work["zona"] == zc).astype(float).to_numpy()
                moran_rows.append({"zona": int(zc), "morans_I": _morans_I_vec(z_ind, w_knn)})
            st.markdown("**Moran's I per Zona (diagnostik pasca-hoc)**")
            st.dataframe(pd.DataFrame(moran_rows), width="stretch", hide_index=True)

            zona_vals = pd.to_numeric(work["zona"], errors="coerce").to_numpy(dtype=float)
            if np.nanvar(zona_vals) > 0:
                st.metric("Moran's I (label zona, diagnostik)", f"{_morans_I_vec(zona_vals, w_knn):.4f}")
        else:
            st.info("Spatial weights tidak tersedia (libpysal belum terpasang).")

        dl_cols = categorical_candidates + rqzm_vars + ["zona"]
        st.download_button(
            "Download CSV (hasil zonasi RQZM)",
            data=work[dl_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"okta_rqzm_zona_k{k}_b{beta:g}.csv",
            mime="text/csv",
        )
