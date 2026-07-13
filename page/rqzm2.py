"""
5b. Spatial Clustering — RQZM Contiguous II (RQZM-II)

Pembanding untuk page RQZM Non-Contiguous. Pipeline-nya SENGAJA dibuat identik
dengan versi NC (standardisasi 5 dimensi SLA -> jarak Euclidean -> Ward ->
CV memilih k), KECUALI satu hal: ditambahkan BOBOT KONTIGUITAS SPASIAL (beta).

Mekanisme RQZM-II (Contiguous II / C''):
- Ruang fitur klaster = [ atribut SLA terstandardisasi | beta * koordinat
  terstandardisasi (x_utm, y_utm) ]. Lokasi ikut menarik tetangga ke zona yang
  sama, sehingga zona cenderung KOMPAK dan KONTIGU.
- beta = 0  -> identik dengan Non-Contiguous (lokasi tidak berperan).
  beta naik -> pengaruh lokasi makin kuat -> zona makin kompak/kontigu.
- beta DIPILIH lewat CV dalam-klaster (skala asli 0-100) — konvensi RQZM
  (CV terendah = anggota paling homogen), sambil memperhatikan tingkat
  kontiguitas (share komponen terbesar). Nilai beta yang diuji: {0; 0,25; 0,5; 1}.
- deprivation_score TIDAK dipakai sebagai fitur (hanya untuk label/CV/cutoff).
- Opsional: merge "sliver" spasial kecil ke zona tetangga ber-atribut terdekat
  (pembersihan pasca-hoc agar zona lebih rapi).

Karena hanya berbeda pada suku beta, selisih hasil NC vs Contiguous dapat
diatribusikan ke faktor kontiguitas. Luaran di sini ditafsirkan sebagai ZONA
(contiguous policy zones), berbeda dari TIPOLOGI pada versi NC.
"""

from __future__ import annotations

from pathlib import Path
from collections import deque
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if "LOKY_MAX_CPU_COUNT" not in os.environ and os.cpu_count():
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

warnings.filterwarnings("ignore", message=r".*scatter_mapbox.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*scattermapbox.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*Could not find the number of physical cores.*", category=UserWarning)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from nav import render_sidebar

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


BASE_DIR = Path(r"C:\Users\USER\Downloads\TESIS\TESIS\Perhitungan\py")
OUT = BASE_DIR / "outputs"
BATA_PADUKUHAN_SHP = BASE_DIR / "db" / "Batas_Padukuhan.shp"
RESULT_PATH = OUT / "spatial_clustering_results.csv"
RUNINFO_PATH = OUT / "spatial_clustering_runinfo.txt"
MPI_HH_PATH = OUT / "mpi_household.parquet"

DIMENSION_FEATURES = ["dim_human", "dim_physical", "dim_natural", "dim_social", "dim_financial"]
SCORE_COL = "deprivation_score"
SUMMARY_FEATURES = [SCORE_COL] + DIMENSION_FEATURES
LINKAGE_METHOD = "ward"
DEFAULT_BETAS = (0.0, 0.25, 0.5, 1.0)   # bobot kontiguitas spasial yang diuji
ZONE_PREFIX = "Zona"


# --------------------------------------------------------------------------- #
# Helper tampilan
# --------------------------------------------------------------------------- #
def _with_no_index(data) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        tbl = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        tbl = data.copy()
    else:
        tbl = pd.DataFrame(data)
    tbl = tbl.reset_index(drop=True)
    tbl.index = np.arange(1, len(tbl) + 1)
    tbl.index.name = "No"
    return tbl


def show_df(data, show_no: bool = True, **kwargs):
    kwargs.setdefault("width", "stretch")
    if show_no:
        st.dataframe(_with_no_index(data), **kwargs)
    else:
        kwargs.setdefault("hide_index", True)
        if isinstance(data, pd.Series):
            tbl = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            tbl = data.copy()
        else:
            tbl = pd.DataFrame(data)
        st.dataframe(tbl.reset_index(drop=True), **kwargs)


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _label_num(s: str) -> int:
    """Ambil angka pertama dari label (mis. 'Zona 2 - Dominan ...' -> 2)."""
    for tok in str(s).replace("-", " ").split():
        if tok.isdigit():
            return int(tok)
    return 0


@st.cache_data(show_spinner=False)
def load_padukuhan_boundary_overlay(shp_path: str) -> dict:
    p = Path(shp_path)
    empty = pd.DataFrame(columns=["dusun", "lat", "lng"])
    if not p.exists():
        return {"geojson": None, "lines": [], "labels": empty, "error": "Shapefile tidak ditemukan."}
    try:
        import geopandas as gpd
    except Exception as exc:
        return {"geojson": None, "lines": [], "labels": empty, "error": f"geopandas belum terpasang: {exc}"}

    try:
        gdf = gpd.read_file(p)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:32749", allow_override=True)
        gdf_metric = gdf if not gdf.crs.is_geographic else gdf.to_crs("EPSG:32749")
        gdf = gdf_metric.to_crs("EPSG:4326")

        name_col = "Dusun" if "Dusun" in gdf.columns else ("dusun" if "dusun" in gdf.columns else None)
        if name_col is None:
            gdf["Dusun"] = [f"Dusun {i+1}" for i in range(len(gdf))]
            name_col = "Dusun"

        geojson_obj = gdf.__geo_interface__
        lines: list[dict] = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
            for poly in polys:
                x, y = poly.exterior.xy
                lines.append({"lon": list(x), "lat": list(y)})

        rep_metric = gdf_metric.geometry.representative_point()
        rep_ll = gpd.GeoSeries(rep_metric, crs=gdf_metric.crs).to_crs("EPSG:4326")
        labels = gdf[[name_col]].copy()
        labels["lng"] = rep_ll.x.values
        labels["lat"] = rep_ll.y.values
        labels = labels[[name_col, "lat", "lng"]].rename(columns={name_col: "dusun"})
        labels["dusun"] = labels["dusun"].astype(str)
        return {"geojson": geojson_obj, "lines": lines, "labels": labels, "error": None}
    except Exception as exc:
        return {"geojson": None, "lines": [], "labels": empty, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Spatial weights (struktur ketetanggaan) + diagnostik kontiguitas
# --------------------------------------------------------------------------- #
def _build_spatial_weights(coords: np.ndarray, knn_k: int = 5):
    """KNN weights: dipakai untuk diagnostik Moran, kontiguitas, dan merge sliver."""
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
                "cluster": int(c),
                "n_members": n_members,
                "n_components": len(comps),
                "largest_component": comp_sizes[0] if comp_sizes else 0,
                "largest_component_share": (comp_sizes[0] / n_members) if n_members > 0 and comp_sizes else np.nan,
                "fragmentation_index": (len(comps) / n_members) if n_members > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


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


# --------------------------------------------------------------------------- #
# Inti: fitur terstandardisasi (atribut + koordinat) lalu Ward atas [attr | beta*coord]
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _features_and_coords(
    hh_path: str,
    dim_features: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str], list[str], str]:
    p = Path(hh_path)
    if not p.exists():
        return pd.DataFrame(), [], [], f"File tidak ditemukan: {p}"

    hh = pd.read_parquet(p)
    miss = [c for c in dim_features if c not in hh.columns]
    if miss:
        return pd.DataFrame(), [], [], f"Dimensi tidak ditemukan di data: {miss}"
    if SCORE_COL not in hh.columns:
        return pd.DataFrame(), [], [], f"Kolom {SCORE_COL} wajib ada (untuk label/CV/cutoff)."
    if not {"x_utm", "y_utm"}.issubset(hh.columns):
        return pd.DataFrame(), [], [], "Kolom x_utm/y_utm wajib ada (untuk suku kontiguitas & peta)."

    add_cols = [
        c for c in [
            "id_keluarga", "dusun", "hh_size", "status_pekerjaan", "ijazah_kepala",
            "monthly_expense", "mpi_poor", "lat", "lng",
        ] if c in hh.columns
    ]
    needed = list(dim_features) + [SCORE_COL, "x_utm", "y_utm"] + add_cols
    work = hh[list(dict.fromkeys(needed))].copy()
    work = _safe_numeric(work, list(dim_features) + [SCORE_COL, "x_utm", "y_utm", "lat", "lng"])
    work = work.dropna(subset=list(dim_features) + ["x_utm", "y_utm"]).reset_index(drop=True)
    if len(work) == 0:
        return pd.DataFrame(), [], [], "Tidak ada baris valid setelah cleaning."

    if "lng" in work.columns and "lon" not in work.columns:
        work["lon"] = pd.to_numeric(work["lng"], errors="coerce")

    # Standardisasi atribut SLA
    z_cols = [f"{c}_z" for c in dim_features]
    work[z_cols] = StandardScaler().fit_transform(work[list(dim_features)].to_numpy(dtype=float))

    # Standardisasi koordinat (agar beta = bobot relatif lokasi vs atribut)
    xy_cols = ["x_z", "y_z"]
    work[xy_cols] = StandardScaler().fit_transform(work[["x_utm", "y_utm"]].to_numpy(dtype=float))

    return work, z_cols, xy_cols, ""


def _augmented_matrix(work: pd.DataFrame, z_cols: list[str], xy_cols: list[str], beta: float) -> np.ndarray:
    Xa = work[z_cols].to_numpy(dtype=float)
    if beta and beta > 0:
        Xc = float(beta) * work[xy_cols].to_numpy(dtype=float)
        return np.hstack([Xa, Xc])
    return Xa


def _cluster_labels(work, z_cols, xy_cols, beta: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    X = _augmented_matrix(work, z_cols, xy_cols, beta)
    Z = linkage(X, method=LINKAGE_METHOD)
    labels = fcluster(Z, t=int(k), criterion="maxclust").astype(int) - 1
    return labels, Z


@st.cache_data(show_spinner=False)
def _run_contiguous(
    hh_path: str,
    k: int,
    dim_features: tuple[str, ...],
    beta: float,
    knn_k: int,
    do_merge: bool,
    min_component_size: int,
    min_component_share: float,
) -> tuple[pd.DataFrame, str, list[str], list[str], object | None, object | None]:
    work, z_cols, xy_cols, err = _features_and_coords(hh_path, dim_features)
    if err:
        return pd.DataFrame(), err, [], [], None, None
    if len(work) < k:
        return pd.DataFrame(), f"Data valid ({len(work)}) lebih kecil dari k={k}.", [], [], None, None

    work = work.copy()
    labels, Z = _cluster_labels(work, z_cols, xy_cols, float(beta), int(k))
    work["cluster_raw"] = labels.astype(int)

    coords = work[["x_utm", "y_utm"]].to_numpy(dtype=float)
    w = _build_spatial_weights(coords, knn_k=int(knn_k))

    if do_merge and w is not None:
        work = _merge_small_components(
            work, feature_used=list(z_cols), w=w, cluster_col="cluster_raw",
            min_component_size=int(min_component_size), min_component_share=float(min_component_share),
        )
        work["cluster"] = work["cluster_clean"].astype(int)
    else:
        work["cluster"] = work["cluster_raw"].astype(int)

    return work, "", z_cols, xy_cols, Z, w


# --------------------------------------------------------------------------- #
# Evaluasi: pemilihan beta (langkah pendefinisi RQZM-II) & pemilihan k
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _beta_sweep_by_cv(
    hh_path: str,
    dim_features: tuple[str, ...],
    k: int,
    beta_values: tuple[float, ...],
    knn_k: int,
) -> tuple[pd.DataFrame, str]:
    """Untuk k tetap, uji beberapa beta. CV terendah = lebih homogen (kriteria utama RQZM)."""
    work, z_cols, xy_cols, err = _features_and_coords(hh_path, dim_features)
    if err:
        return pd.DataFrame(), err
    if len(work) < k:
        return pd.DataFrame(), f"Data valid ({len(work)}) lebih kecil dari k={k}."

    y = pd.to_numeric(work.get(SCORE_COL), errors="coerce")
    coords = work[["x_utm", "y_utm"]].to_numpy(dtype=float)
    w = _build_spatial_weights(coords, knn_k=int(knn_k))

    rows: list[dict] = []
    for beta in beta_values:
        labels, _ = _cluster_labels(work, z_cols, xy_cols, float(beta), int(k))
        tmp = pd.DataFrame({"cluster": labels, "y": y})
        stats = tmp.groupby("cluster")["y"].agg(["mean", "std"]).reset_index()
        stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
        avg_cv = float(stats["cv"].mean())

        if w is not None:
            # frag butuh kolom cluster dengan index 0..n-1 sejajar urutan coords pada w
            frag = _fragmentation_metrics(pd.DataFrame({"cluster": labels}), w, cluster_col="cluster")
            mean_share = float(frag["largest_component_share"].mean()) if len(frag) else np.nan
            total_comp = int(frag["n_components"].sum()) if len(frag) else -1
        else:
            mean_share, total_comp = np.nan, -1

        rows.append({
            "beta": float(beta),
            "avg_cv": avg_cv,
            "mean_largest_comp_share": mean_share,
            "total_components": total_comp,
        })

    df = pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)
    df["rank_cv"] = df["avg_cv"].rank(ascending=True, method="min")
    return df, ""


@st.cache_data(show_spinner=False)
def _evaluate_cv_by_k(
    hh_path: str,
    dim_features: tuple[str, ...],
    k_values: tuple[int, ...],
    beta: float,
) -> tuple[pd.DataFrame, str]:
    work, z_cols, xy_cols, err = _features_and_coords(hh_path, dim_features)
    if err:
        return pd.DataFrame(), err
    if len(work) < max(k_values):
        return pd.DataFrame(), f"Data valid ({len(work)}) lebih kecil dari k maksimum ({max(k_values)})."

    y = pd.to_numeric(work.get(SCORE_COL), errors="coerce")
    rows: list[dict] = []
    for k in k_values:
        labels, _ = _cluster_labels(work, z_cols, xy_cols, float(beta), int(k))
        tmp = pd.DataFrame({"cluster": labels, "y": y})
        stats = tmp.groupby("cluster")["y"].agg(["mean", "std"]).reset_index()
        stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
        rows.append({"k": int(k), "avg_cv": float(stats["cv"].mean())})

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    df["rank_cv"] = df["avg_cv"].rank(ascending=True, method="min")
    return df, ""


@st.cache_data(show_spinner=False)
def _moran_per_class(
    hh_path: str,
    dim_features: tuple[str, ...],
    k: int,
    beta: float,
    diag_k: int,
) -> tuple[pd.DataFrame, str]:
    work, z_cols, xy_cols, err = _features_and_coords(hh_path, dim_features)
    if err:
        return pd.DataFrame(), err
    coords = work[["x_utm", "y_utm"]].to_numpy(dtype=float)
    w_diag = _build_spatial_weights(coords, knn_k=int(diag_k))
    if w_diag is None:
        return pd.DataFrame(), "libpysal belum terpasang: diagnostik Moran dilewati."
    labels, _ = _cluster_labels(work, z_cols, xy_cols, float(beta), int(k))
    rows = []
    for cls in sorted(np.unique(labels).tolist()):
        z = (labels == cls).astype(float)
        rows.append({"class": int(cls + 1), "morans_I": _morans_I_vec(z, w_diag)})
    return pd.DataFrame(rows), ""


def _moran_labels(res_df: pd.DataFrame, w) -> tuple[float | None, str]:
    if "cluster_id" not in res_df.columns or w is None:
        return None, "Spatial weights / cluster_id tidak tersedia."
    z = pd.to_numeric(res_df["cluster_id"], errors="coerce").to_numpy(dtype=float)
    if np.allclose(np.nanvar(z), 0.0):
        return None, "Varians label = 0."
    return _morans_I_vec(z, w), ""


# --------------------------------------------------------------------------- #
# Profil & pelabelan zona
# --------------------------------------------------------------------------- #
def _build_cluster_profile_labels(summary_df, res_df, cluster_col: str = "cluster"):
    if len(summary_df) and cluster_col in summary_df.columns:
        prof = summary_df.copy()
    else:
        num_cols = [c for c in res_df.select_dtypes(include=["number"]).columns if c != cluster_col]
        prof = res_df.groupby(cluster_col, dropna=False)[num_cols].mean(numeric_only=True).reset_index()
        prof = prof.rename(columns={c: f"mean_{c}" for c in num_cols})

    dep_col = "mean_deprivation_score" if "mean_deprivation_score" in prof.columns else None
    if dep_col is None:
        candidates = [c for c in prof.columns if c.startswith("mean_")]
        dep_col = candidates[0] if candidates else None
        if dep_col is None:
            prof["cluster_label"] = prof[cluster_col].astype(str)
            return prof, dict(zip(prof[cluster_col], prof["cluster_label"]))

    prof = prof.sort_values(dep_col, ascending=False).reset_index(drop=True)
    dim_map = {
        "mean_dim_human": "Human", "mean_dim_physical": "Physical", "mean_dim_natural": "Natural",
        "mean_dim_social": "Social", "mean_dim_financial": "Financial",
    }
    dim_cols = [c for c in dim_map if c in prof.columns]
    dominant = prof[dim_cols].idxmax(axis=1).map(dim_map).fillna("Campuran") if dim_cols else pd.Series(["Campuran"] * len(prof))
    prof["cluster_label"] = [f"{ZONE_PREFIX} {i + 1} - Dominan {dominant.iloc[i]}" for i in range(len(prof))]
    return prof, dict(zip(prof[cluster_col], prof["cluster_label"]))


def _summary_from_res(res_df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in SUMMARY_FEATURES if c in res_df.columns]
    if len(num_cols) == 0:
        return pd.DataFrame(columns=["cluster"])
    s = res_df.groupby("cluster", dropna=False)[num_cols].mean(numeric_only=True).reset_index()
    return s.rename(columns={c: f"mean_{c}" for c in num_cols})


def _compute_cutoff_table(res_df, label_map, score_col=SCORE_COL, cluster_col="cluster"):
    stats = (
        res_df.groupby(cluster_col, dropna=False)[score_col]
        .agg(["size", "mean", "min", "median", "max"]).reset_index()
        .rename(columns={"size": "n"}).sort_values("mean").reset_index(drop=True)
    )
    stats["label"] = stats[cluster_col].map(label_map).fillna(stats[cluster_col].astype(str))
    means = stats["mean"].to_numpy(dtype=float)
    clusters = stats[cluster_col].to_list()
    cutoffs = [float((means[i] + means[i + 1]) / 2.0) for i in range(len(means) - 1)]
    bounds = [-np.inf] + cutoffs + [np.inf]
    intervals = []
    for i in range(len(clusters)):
        lo, hi = bounds[i], bounds[i + 1]
        if np.isneginf(lo):
            intervals.append(f"<= {hi:.4f}")
        elif np.isposinf(hi):
            intervals.append(f"> {lo:.4f}")
        else:
            intervals.append(f"> {lo:.4f} s.d. <= {hi:.4f}")
    stats["cutoff_interval"] = intervals
    pred = np.zeros(len(res_df), dtype=object)
    vals = res_df[score_col].to_numpy(dtype=float)
    for idx, v in enumerate(vals):
        j = 0
        while j < len(cutoffs) and v > cutoffs[j]:
            j += 1
        pred[idx] = clusters[j]
    cutoff_acc = float(np.mean(pred == res_df[cluster_col].to_numpy()))
    return stats, cutoffs, cutoff_acc


def _severity_rank(label: str) -> int:
    txt = str(label).lower()
    for key, val in [("sangat rentan", 0), ("rentan", 1), ("menengah", 2), ("relatif ringan", 3), ("paling ringan", 4)]:
        if key in txt:
            return val
    return 99


def _build_color_map(labels: list[str]) -> tuple[dict, list[str]]:
    ordered = sorted(list(dict.fromkeys(labels)), key=_label_num)
    palette = ["#b30000", "#e34a33", "#fdbb84", "#a1d99b", "#31a354"]
    return {lab: palette[min(i, len(palette) - 1)] for i, lab in enumerate(ordered)}, ordered


def _build_display_cluster_map(profile_df: pd.DataFrame) -> dict:
    if len(profile_df) == 0 or "cluster" not in profile_df.columns:
        return {}
    tmp = profile_df.copy()
    if "cluster_label" not in tmp.columns:
        tmp["cluster_label"] = tmp["cluster"].astype(str)
    tmp["_sev"] = tmp["cluster_label"].astype(str).map(_severity_rank)
    if "mean_deprivation_score" in tmp.columns:
        tmp = tmp.sort_values(["_sev", "mean_deprivation_score"], ascending=[True, False])
    else:
        tmp = tmp.sort_values(["_sev", "cluster"], ascending=[True, True])
    return {int(c): i for i, c in enumerate(tmp["cluster"].tolist(), start=1)}


def _cluster_kk_dusun_tables(res_df, label_map):
    if len(res_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), "Hasil clustering kosong."
    if not {"id_keluarga", "dusun"}.issubset(res_df.columns):
        return pd.DataFrame(), pd.DataFrame(), "Kolom id_keluarga/dusun tidak tersedia di hasil."
    work = res_df[["cluster", "id_keluarga", "dusun"]].copy()
    work["cluster_label"] = work["cluster"].map(label_map).fillna(work["cluster"].astype(str))
    work["id_keluarga"] = work["id_keluarga"].astype(str)
    work["dusun"] = work["dusun"].astype(str).str.strip()
    work.loc[work["dusun"].isin(["", "nan", "None"]), "dusun"] = "Tidak diketahui"
    summary = (
        work.groupby(["cluster", "cluster_label"], dropna=False)
        .agg(jumlah_kk=("id_keluarga", "nunique"), jumlah_dusun=("dusun", "nunique"))
        .reset_index().sort_values("cluster")
    )
    dusun_list = (
        work.groupby(["cluster", "cluster_label"], dropna=False)["dusun"]
        .apply(lambda s: ", ".join(sorted(set(s.dropna().astype(str))))).reset_index(name="daftar_dusun")
    )
    summary = summary.merge(dusun_list, on=["cluster", "cluster_label"], how="left")
    detail = (
        work.groupby(["cluster", "cluster_label", "dusun"], dropna=False)["id_keluarga"]
        .nunique().reset_index(name="jumlah_kk")
        .sort_values(["cluster", "jumlah_kk", "dusun"], ascending=[True, False, True])
    )
    return summary, detail, ""


def _mode_text(s: pd.Series) -> str:
    t = s.astype(str).str.strip()
    t = t[~t.isin(["", "nan", "None"])]
    if len(t) == 0:
        return "-"
    m = t.mode(dropna=True)
    return str(m.iloc[0]) if len(m) else "-"


def _cluster_household_characteristics(res_df, label_map):
    if len(res_df) == 0:
        return pd.DataFrame(), "Hasil clustering kosong."
    work = res_df.copy()
    if "id_keluarga" not in work.columns:
        work["id_keluarga"] = np.arange(1, len(work) + 1).astype(str)
    for c in ["hh_size", "monthly_expense", "mpi_poor"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    rows = []
    for c, g in work.groupby("cluster", dropna=False):
        rows.append({
            "cluster": int(c),
            "cluster_label": label_map.get(c, str(c)),
            "jumlah_kk": int(g["id_keluarga"].astype(str).nunique()),
            "rata2_anggota_keluarga": float(g["hh_size"].mean()) if "hh_size" in g.columns else np.nan,
            "median_pengeluaran_bulanan": float(g["monthly_expense"].median()) if "monthly_expense" in g.columns else np.nan,
            "persen_mpi_poor": float(g["mpi_poor"].mean() * 100.0) if "mpi_poor" in g.columns else np.nan,
            "pekerjaan_dominan": _mode_text(g["status_pekerjaan"]) if "status_pekerjaan" in g.columns else "-",
            "ijazah_kepala_dominan": _mode_text(g["ijazah_kepala"]) if "ijazah_kepala" in g.columns else "-",
        })
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True), ""


def _cluster_variable_profile_table(profile_df, display_cluster_map):
    var_label_map = {
        "mean_dim_human": "Human", "mean_dim_physical": "Physical", "mean_dim_natural": "Natural",
        "mean_dim_social": "Social", "mean_dim_financial": "Financial",
    }
    use_cols = [c for c in var_label_map if c in profile_df.columns]
    if len(use_cols) == 0:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for _, r in profile_df.iterrows():
        c = int(r["cluster"])
        for col in use_cols:
            rows.append({
                "cluster": c, "cluster_id": int(display_cluster_map.get(c, c)),
                "cluster_label": r.get("cluster_label", str(c)),
                "variabel": var_label_map[col], "nilai_mean": float(r[col]),
            })
    long_df = pd.DataFrame(rows)
    variable_order = [var_label_map[c] for c in
                      ["mean_dim_financial", "mean_dim_human", "mean_dim_natural", "mean_dim_physical", "mean_dim_social"]
                      if c in use_cols]
    long_df["variabel"] = pd.Categorical(long_df["variabel"], categories=variable_order, ordered=True)
    long_df["zscore"] = long_df.groupby("variabel", observed=False)["nilai_mean"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0)
    )
    long_df = long_df.sort_values(["cluster_id", "variabel"]).reset_index(drop=True)
    pivot = (
        long_df.pivot_table(index=["cluster_id", "cluster_label"], columns="variabel", values="zscore",
                            aggfunc="first", observed=False).reset_index().sort_values("cluster_id")
    )
    ordered_cols = ["cluster_id", "cluster_label"] + [v for v in variable_order if v in pivot.columns]
    return long_df, pivot.reindex(columns=ordered_cols)


def _cluster_mean_profile(res_df, dim_features, label_map, display_cluster_map):
    feat_cols = [c for c in dim_features if c in res_df.columns]
    if len(feat_cols) == 0:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for c, g in res_df.groupby("cluster", dropna=False):
        mean_vec = g[feat_cols].mean(numeric_only=True)
        for f in feat_cols:
            rows.append({
                "cluster": int(c), "cluster_id": int(display_cluster_map.get(c, c)),
                "cluster_label": label_map.get(c, str(c)),
                "variabel": f.replace("dim_", "").title(), "mean_value": float(mean_vec[f]),
            })
    long_df = pd.DataFrame(rows).sort_values(["cluster_id", "variabel"]).reset_index(drop=True)
    pivot = (
        long_df.pivot_table(index=["cluster_id", "cluster_label"], columns="variabel", values="mean_value",
                            aggfunc="first").reset_index().sort_values("cluster_id")
    )
    return long_df, pivot


def _cv_by_cluster(res_df, label_map, display_cluster_map):
    if SCORE_COL not in res_df.columns:
        return pd.DataFrame()
    rows = []
    for c, g in res_df.groupby("cluster", dropna=False):
        y = pd.to_numeric(g[SCORE_COL], errors="coerce").dropna()
        if len(y) == 0:
            continue
        mean = float(y.mean())
        std = float(y.std(ddof=0))
        rows.append({
            "cluster": int(c), "cluster_id": int(display_cluster_map.get(c, c)),
            "cluster_label": label_map.get(c, str(c)),
            "mean_deprivation_score": mean, "std_deprivation_score": std,
            "cv_deprivation_score": (std / mean) if mean != 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("cluster_id")


def _render_dendrogram_from_Z(Z, target_k: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return None, f"matplotlib belum tersedia: {exc}"
    n_obs = int(Z.shape[0] + 1)
    k = int(max(1, min(int(target_k), n_obs)))
    if k == 1:
        cut_h = float(Z[-1, 2]) + 1e-9
    else:
        idx_upper = n_obs - k
        d_upper = float(Z[idx_upper, 2])
        d_lower = float(Z[idx_upper - 1, 2]) if idx_upper - 1 >= 0 else 0.0
        cut_h = float((d_lower + d_upper) / 2.0) if d_upper > d_lower else float(np.nextafter(d_upper, -np.inf))
    labels = fcluster(Z, t=cut_h, criterion="distance")
    k_realized = int(pd.Series(labels).nunique())
    fig, ax = plt.subplots(figsize=(16, 7))
    dendrogram(Z, truncate_mode="level", p=30, no_labels=True, color_threshold=cut_h,
               above_threshold_color="#6b7280", ax=ax)
    ax.axhline(y=cut_h, color="#dc2626", linestyle="--", linewidth=2)
    ax.set_title(f"Hierarchical Dendrogram (Ward, [atribut | beta*koordinat]) — cut @ k={k}")
    ax.set_xlabel("Indeks rumah tangga (truncated)")
    ax.set_ylabel("Jarak (Ward)")
    fig.tight_layout()
    return (fig, k, k_realized, cut_h), ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Spatial Clustering — RQZM Contiguous", layout="wide")
render_sidebar(__file__)

st.title("5b. Spatial Clustering — RQZM Contiguous II")

st.info(
    "Metode: **RQZM varian Contiguous II (RQZM-II)**. Pipeline identik dengan versi "
    "Non-Contiguous (standardisasi 5 dimensi SLA -> Euclidean -> Ward -> CV memilih k), "
    "**ditambah bobot kontiguitas spasial β**: ruang fitur = [atribut SLA terstandardisasi | "
    "β × koordinat terstandardisasi]. **β = 0 ⇒ persis NC**; β makin besar ⇒ zona makin "
    "**kompak/kontigu**. β dipilih lewat **CV** (CV terendah = paling homogen), sambil melihat "
    "tingkat kontiguitas. Luaran = **zona** (contiguous policy zones)."
)

analysis_mode = st.radio("Opsi Analisis", options=["Default (k dinamis)", "RQZM (3/5 zona)"], index=0, horizontal=True)

st.markdown("**Pengaturan Jumlah Zona**")
if analysis_mode == "RQZM (3/5 zona)":
    k_dynamic = st.selectbox("Jumlah zona (k)", options=[3, 5], index=0)
    st.info(f"Mode RQZM aktif: jumlah zona dikunci ke k={int(k_dynamic)}.")
else:
    k_dynamic = st.slider("Number of zones (k)", min_value=3, max_value=6, value=3, step=1)
st.caption("Zona dihitung dari `mpi_household.parquet` (5 dimensi SLA terstandardisasi + suku kontiguitas).")
eval_k_values = (int(k_dynamic),) if analysis_mode == "RQZM (3/5 zona)" else (3, 4, 5, 6)

st.markdown("**Bobot Kontiguitas Spasial (β) & Struktur Ketetanggaan**")
st.caption("β = 0 setara Non-Contiguous; β makin besar = lokasi makin menarik tetangga ke zona yang sama.")
beta = st.select_slider("Bobot kontiguitas β", options=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0], value=0.5)
knn_k = st.slider("Tetangga KNN (struktur kontiguitas & diagnostik)", min_value=3, max_value=10, value=5, step=1)

st.markdown("**Pembersihan Sliver (pasca-hoc, opsional)**")
do_merge = st.checkbox("Gabungkan sliver spasial kecil ke zona tetangga ber-atribut terdekat", value=True)
cmid_l, cmid_r = st.columns(2)
with cmid_l:
    min_component_size = st.slider("Min ukuran patch (rumah tangga)", min_value=3, max_value=30, value=10, step=1)
with cmid_r:
    min_component_share = st.slider("Min share patch dalam zona", min_value=0.01, max_value=0.10, value=0.02, step=0.01)

st.caption("Catatan: perhitungan Ward dijalankan beberapa kali (sweep β & k); proses pertama bisa beberapa detik.")

# ---- Sweep β (langkah pendefinisi RQZM-II) ---- #
st.markdown("### Pemilihan β (langkah inti RQZM-II)")
beta_df, beta_err = _beta_sweep_by_cv(
    hh_path=str(MPI_HH_PATH), dim_features=tuple(DIMENSION_FEATURES),
    k=int(k_dynamic), beta_values=DEFAULT_BETAS, knn_k=int(knn_k),
)
recommended_beta = float(beta)
if beta_err:
    st.warning(beta_err)
else:
    pos = beta_df[beta_df["beta"] > 0]
    best_beta_row = (pos if len(pos) else beta_df).sort_values(["rank_cv", "beta"]).iloc[0]
    recommended_beta = float(best_beta_row["beta"])
    st.info(
        f"Rekomendasi β (di antara β>0): β={recommended_beta:g} — CV terendah pada k={int(k_dynamic)}. "
        f"Periksa juga 'mean_largest_comp_share' (makin tinggi makin kontigu) sebelum memutuskan."
    )
    show_df(beta_df[["beta", "avg_cv", "mean_largest_comp_share", "total_components", "rank_cv"]], show_no=False)
    fig_beta = px.line(beta_df, x="beta", y="avg_cv", markers=True,
                       title=f"Avg within-cluster CV vs β (k={int(k_dynamic)}) — semakin rendah semakin homogen")
    fig_beta.update_traces(line={"width": 3}, marker={"size": 12})
    fig_beta.update_layout(height=320, xaxis_title="β (bobot kontiguitas)", yaxis_title="Avg CV (deprivation_score)")
    st.plotly_chart(fig_beta, width="stretch")
    st.caption(f"β yang dipakai untuk perhitungan zona di bawah ini = {float(beta):g} (atur lewat slider di atas).")

# ---- Pemilihan k pada β terpilih ---- #
cv_eval, cv_err = _evaluate_cv_by_k(
    hh_path=str(MPI_HH_PATH), dim_features=tuple(DIMENSION_FEATURES),
    k_values=eval_k_values, beta=float(beta),
)
recommended_k_for_dendro = int(k_dynamic)
if cv_err:
    st.warning(cv_err)
else:
    best_cv = cv_eval.sort_values(["rank_cv", "k"]).iloc[0]
    recommended_k_for_dendro = int(best_cv["k"])
    st.markdown(f"**Pemilihan k pada β={float(beta):g} (CV terendah)**")
    st.info(f"Auto recommendation: k={int(best_cv['k'])} (CV dalam-klaster terendah).")
    show_df(cv_eval[["k", "avg_cv", "rank_cv"]], show_no=False)

# ---- Diagnostik Moran per kelas ---- #
mcls_df, mcls_err = _moran_per_class(
    hh_path=str(MPI_HH_PATH), dim_features=tuple(DIMENSION_FEATURES),
    k=int(k_dynamic), beta=float(beta), diag_k=int(knn_k),
)
if mcls_err:
    st.warning(mcls_err)
else:
    st.markdown("**Moran's I per Zona (diagnostik pasca-hoc)**")
    show_df(mcls_df, show_no=False)

# ---- Jalankan zonasi final ---- #
res, dyn_err, used_features, xy_cols, Z_link, w_used = _run_contiguous(
    hh_path=str(MPI_HH_PATH), k=int(k_dynamic), dim_features=tuple(DIMENSION_FEATURES),
    beta=float(beta), knn_k=int(knn_k), do_merge=bool(do_merge),
    min_component_size=int(min_component_size), min_component_share=float(min_component_share),
)
if dyn_err:
    if RESULT_PATH.exists():
        st.warning(f"{dyn_err} Menggunakan fallback hasil terakhir dari output CSV.")
        res = pd.read_csv(RESULT_PATH)
        used_features = [f"{c}_z" for c in DIMENSION_FEATURES if f"{c}_z" in res.columns]
        Z_link, w_used = None, None
    else:
        st.warning(dyn_err)
        st.stop()
else:
    st.caption(
        f"Fitur zona: [{', '.join(used_features)} | β×{', '.join(xy_cols)}] dengan β={float(beta):g}. "
        f"Merge sliver: {'ON' if do_merge else 'OFF'}."
    )

if "cluster" not in res.columns:
    st.warning("Kolom `cluster` tidak ditemukan di hasil.")
    st.stop()

# ---- Profil, label, warna ---- #
summary = _summary_from_res(res)
cluster_counts = res.groupby("cluster", dropna=False).size().rename("n").reset_index().sort_values("cluster")
cluster_counts["pct"] = (cluster_counts["n"] / cluster_counts["n"].sum() * 100.0).round(2)

profile_df, label_map = _build_cluster_profile_labels(summary, res, cluster_col="cluster")
display_cluster_map = _build_display_cluster_map(profile_df)

cluster_counts["label"] = cluster_counts["cluster"].map(label_map).fillna(cluster_counts["cluster"].astype(str))
cluster_counts["cluster_id"] = cluster_counts["cluster"].map(display_cluster_map).fillna(cluster_counts["cluster"]).astype(int)
cluster_counts = cluster_counts.sort_values("cluster_id")
cluster_counts["cluster_legend"] = cluster_counts["cluster_id"].astype(int).astype(str)

res["cluster_label"] = res["cluster"].map(label_map).fillna(res["cluster"].astype(str))
res["cluster_id"] = res["cluster"].map(display_cluster_map).fillna(res["cluster"]).astype(int)
res["cluster_legend"] = res["cluster_id"].astype(int).astype(str)

moran_I, moran_err = _moran_labels(res, w_used)

label_color_map, ordered_labels = _build_color_map(res["cluster_label"].dropna().astype(str).tolist())
ordered_labels = sorted(ordered_labels, key=_label_num)
cluster_id_labels = [str(i) for i in sorted(res["cluster_id"].dropna().astype(int).unique().tolist())]
category_orders = {"cluster_label": ordered_labels, "label": ordered_labels, "cluster_legend": cluster_id_labels}

cluster_legend_colors: dict[str, str] = {}
for lbl, color in label_color_map.items():
    num = _label_num(lbl)
    if num:
        cluster_legend_colors[str(num)] = color

c1, c2, c3 = st.columns(3)
c1.metric("Total observasi", f"{len(res):,}")
c2.metric("Jumlah zona", f"{res['cluster'].nunique():,}")
c3.metric("Rata-rata per zona", f"{len(res) / max(res['cluster'].nunique(), 1):.1f}")
if moran_err:
    st.warning(f"Moran's I label (diagnostik): {moran_err}")
elif moran_I is not None:
    st.metric("Moran's I (label, diagnostik)", f"{moran_I:.4f}")

colL, colR = st.columns(2)
with colL:
    fig_bar = px.bar(cluster_counts, x="label", y="n", title="Observations per Zone",
                     color="cluster_legend", color_discrete_map=cluster_legend_colors, category_orders=category_orders)
    fig_bar.update_xaxes(categoryorder="array", categoryarray=ordered_labels)
    fig_bar.update_layout(height=420, legend_title_text="Zona :")
    st.plotly_chart(fig_bar, width="stretch")

with colR:
    fig_map = None
    if "lon" in res.columns and "lat" in res.columns:
        map_df = res.copy()
        map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
        map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
        map_df = map_df.dropna(subset=["lon", "lat"]).copy()
        if len(map_df):
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", color="cluster_legend",
                title=f"Peta Zona Kemiskinan (RQZM Contiguous II, β={float(beta):g})",
                opacity=0.85, color_discrete_map=cluster_legend_colors, category_orders=category_orders,
                hover_data={"cluster_label": True, "lon": ":.5f", "lat": ":.5f"}, zoom=12, height=520,
            )
            ov = load_padukuhan_boundary_overlay(str(BATA_PADUKUHAN_SHP))
            overlay_layers = []
            if ov.get("error"):
                st.warning(f"Overlay batas padukuhan tidak aktif: {ov['error']}")
            else:
                gj = ov.get("geojson")
                if gj is not None:
                    overlay_layers.extend([
                        {"below": "traces", "sourcetype": "geojson", "source": gj, "type": "fill",
                         "color": "rgba(230,230,230,0.40)"},
                        {"below": "traces", "sourcetype": "geojson", "source": gj, "type": "line",
                         "color": "rgba(31,41,55,0.95)", "line": {"width": 1.4}},
                    ])
                lbl = ov.get("labels", pd.DataFrame())
                if len(lbl):
                    label_text = lbl["dusun"].astype(str).str.title()
                    halo, halo2 = 0.00024, 0.00042
                    for dx, dy in [
                        (-halo, 0.0), (halo, 0.0), (0.0, -halo), (0.0, halo),
                        (-halo, -halo), (-halo, halo), (halo, -halo), (halo, halo),
                        (-halo2, 0.0), (halo2, 0.0), (0.0, -halo2), (0.0, halo2),
                        (-halo2, -halo2), (-halo2, halo2), (halo2, -halo2), (halo2, halo2),
                    ]:
                        fig_map.add_trace(go.Scattermapbox(
                            lon=lbl["lng"] + dx, lat=lbl["lat"] + dy, mode="text", text=label_text,
                            textposition="middle center", textfont=dict(size=16, color="rgba(255,255,255,1.0)"),
                            hoverinfo="skip", showlegend=False))
                    fig_map.add_trace(go.Scattermapbox(
                        lon=lbl["lng"], lat=lbl["lat"], mode="text", text=label_text, textposition="middle center",
                        textfont=dict(size=17, color="#000000", family="Arial Black, DejaVu Sans, sans-serif"),
                        hoverinfo="skip", showlegend=False))
            fig_map.update_layout(
                mapbox_style="carto-positron",
                mapbox_layers=[
                    {"below": "traces", "sourcetype": "raster", "sourceattribution": "Esri",
                     "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]},
                    *overlay_layers,
                ],
            )
            fig_map.add_shape(type="rect", xref="paper", yref="paper", x0=0.855, y0=0.71, x1=0.985, y1=0.965,
                              line=dict(color="#111111", width=1), fillcolor="rgba(255,255,255,0.55)", layer="above")
            fig_map.add_annotation(x=0.92, y=0.93, xref="paper", yref="paper", text="N", showarrow=False,
                                   font=dict(size=22, color="#111111", family="Arial Black, DejaVu Sans, sans-serif"),
                                   xanchor="center", yanchor="middle")
            fig_map.add_shape(type="path", xref="paper", yref="paper", path="M 0.92 0.90 L 0.895 0.825 L 0.92 0.853 Z",
                              line=dict(color="#111111", width=1.8), fillcolor="rgba(255,255,255,1)", layer="above")
            fig_map.add_shape(type="path", xref="paper", yref="paper", path="M 0.92 0.90 L 0.92 0.853 L 0.945 0.825 Z",
                              line=dict(color="#111111", width=1.8), fillcolor="rgba(20,20,20,1)", layer="above")
            lon_min, lon_max = float(map_df["lon"].min()), float(map_df["lon"].max())
            lat_mean = float(map_df["lat"].mean())
            width_km = max((lon_max - lon_min) * 111.32 * np.cos(np.deg2rad(lat_mean)), 0.05)
            scale_km = min([0.1, 0.2, 0.5, 1.0, 2.0, 5.0], key=lambda v: abs(v - max(width_km * 0.18, 0.1)))
            half_km = scale_km / 2.0

            def _fmt_km_id(v: float) -> str:
                return str(int(v)) if float(v).is_integer() else str(v).replace(".", ",")

            scale_y, scale_x0, scale_x1 = 0.745, 0.87, 0.97
            fig_map.add_shape(type="line", xref="paper", yref="paper", x0=scale_x0, x1=scale_x1, y0=scale_y, y1=scale_y,
                              line=dict(color="#111111", width=3))
            for i in range(9):
                xi = scale_x0 + (scale_x1 - scale_x0) * (i / 8.0)
                tick_h = 0.011 if i in (0, 2, 4, 8) else 0.006
                fig_map.add_shape(type="line", xref="paper", yref="paper", x0=xi, x1=xi,
                                  y0=scale_y - tick_h, y1=scale_y + tick_h, line=dict(color="#111111", width=2))
            for xi, txt in [(scale_x0, "0"), (scale_x0 + (scale_x1 - scale_x0) * 0.5, _fmt_km_id(half_km)),
                            (scale_x1, f"{_fmt_km_id(scale_km)}km")]:
                fig_map.add_annotation(x=xi, y=scale_y + 0.03, xref="paper", yref="paper", text=txt, showarrow=False,
                                       font=dict(size=11, color="#111111", family="Arial Black, DejaVu Sans, sans-serif"),
                                       xanchor="center")
            fig_map.update_layout(
                legend_title_text="<b>Zona :</b>",
                legend=dict(x=0.01, y=0.02, xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#111111", borderwidth=1, itemsizing="constant",
                            font=dict(size=16, color="#000000", family="Arial Black, Arial, sans-serif"),
                            title_font=dict(size=18, color="#000000", family="Arial Black, Arial, sans-serif")),
                dragmode="pan", margin=dict(l=10, r=10, t=50, b=10),
            )
            fig_map.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                              line=dict(color="#ffffff", width=5), fillcolor="rgba(0,0,0,0)", layer="above")
            fig_map.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                              line=dict(color="#000000", width=2), fillcolor="rgba(0,0,0,0)", layer="above")
        else:
            st.info("Data koordinat lon/lat kosong setelah pembersihan.")
    elif "x_utm" in res.columns and "y_utm" in res.columns:
        fig_map = px.scatter(res, x="x_utm", y="y_utm", color="cluster_legend", title="Zone Map (UTM)",
                             opacity=0.8, color_discrete_map=cluster_legend_colors, category_orders=category_orders)
    else:
        st.info("Kolom koordinat tidak ditemukan (`lon/lat` atau `x_utm/y_utm`).")

    if fig_map is not None:
        is_mapbox_fig = any(getattr(tr, "type", "") == "scattermapbox" for tr in fig_map.data)
        if not is_mapbox_fig:
            fig_map.update_layout(height=420, legend_title_text="Zona :")
        st.plotly_chart(fig_map, width="stretch", config={"scrollZoom": True, "displaylogo": False})

# ---- Export GPKG di bawah peta ---- #
st.markdown("---")
st.markdown("#### Export Hasil Clustering")
st.caption("Download data titik rumah tangga beserta label zona dan skor deprivasi.")

_ex_base = [c for c in ["id_keluarga", "dusun", "cluster_id", "cluster_label", SCORE_COL] if c in res.columns]
_ex_dims = [c for c in DIMENSION_FEATURES if c in res.columns]
_ex_coords = [c for c in ["lon", "lat", "x_utm", "y_utm"] if c in res.columns]
_ex_df = res[list(dict.fromkeys(_ex_base + _ex_dims + _ex_coords))].copy()

if {"x_utm", "y_utm"}.issubset(res.columns):
    _ex_geo_x, _ex_geo_y, _ex_crs = "x_utm", "y_utm", "EPSG:32749"
elif {"lon", "lat"}.issubset(res.columns):
    _ex_geo_x, _ex_geo_y, _ex_crs = "lon", "lat", "EPSG:4326"
else:
    _ex_geo_x = _ex_geo_y = _ex_crs = None

_ex_col_csv, _ex_col_gpkg = st.columns(2)
with _ex_col_csv:
    st.download_button(
        label="Download CSV (.csv)",
        data=_ex_df.to_csv(index=False).encode("utf-8"),
        file_name=f"rqzm_contiguous_zones_b{float(beta):g}.csv",
        mime="text/csv",
        width="stretch",
    )
with _ex_col_gpkg:
    if _ex_geo_x is None:
        st.info("Koordinat tidak tersedia untuk ekspor GeoPackage.")
    else:
        try:
            import tempfile
            import geopandas as gpd
            from shapely.geometry import Point as _ShpPoint
            _ex_valid = _ex_df.dropna(subset=[_ex_geo_x, _ex_geo_y]).copy()
            _ex_gdf = gpd.GeoDataFrame(
                _ex_valid,
                geometry=[_ShpPoint(float(x), float(y)) for x, y in zip(_ex_valid[_ex_geo_x], _ex_valid[_ex_geo_y])],
                crs=_ex_crs,
            )
            with tempfile.TemporaryDirectory() as _ex_td:
                _ex_gpkg_path = Path(_ex_td) / f"rqzm_contiguous_zones_b{float(beta):g}.gpkg"
                _ex_gdf.to_file(_ex_gpkg_path, driver="GPKG", layer="rqzm_contiguous")
                _ex_gpkg_bytes = _ex_gpkg_path.read_bytes()
            st.download_button(
                label="Download GeoPackage (.gpkg) — QGIS",
                data=_ex_gpkg_bytes,
                file_name=f"rqzm_contiguous_zones_b{float(beta):g}.gpkg",
                mime="application/geopackage+sqlite3",
                width="stretch",
            )
        except Exception as _ex_e:
            st.warning(f"GeoPackage tidak tersedia: {_ex_e}")

# ---- Ringkasan & profil ---- #
st.markdown("**Zone Summary**")
show_df(cluster_counts[["cluster_id", "label", "n", "pct"]].rename(columns={"cluster_id": "zona"}), show_no=False)

with st.expander("Auto Label Profile", expanded=True):
    profile_df = profile_df.copy()
    profile_df["cluster_id"] = profile_df["cluster"].map(display_cluster_map).fillna(profile_df["cluster"]).astype(int)
    profile_df = profile_df.sort_values("cluster_id")
    cols_show = ["cluster_id", "cluster_label"] + [
        c for c in ["mean_deprivation_score", "mean_dim_human", "mean_dim_physical",
                    "mean_dim_natural", "mean_dim_social", "mean_dim_financial"] if c in profile_df.columns]
    show_df(profile_df[cols_show].rename(columns={"cluster_id": "zona"}), width="stretch")

st.markdown("**Profil Variabel Antar Zona (Z-Score)**")
profile_long, profile_pivot = _cluster_variable_profile_table(profile_df, display_cluster_map)
if len(profile_long) == 0:
    st.info("Variabel profil tidak tersedia.")
else:
    profile_long = profile_long.copy()
    profile_long["cluster_legend"] = profile_long["cluster_id"].astype(int).astype(str)
    fig_profile = px.line(profile_long, x="variabel", y="zscore", color="cluster_legend", markers=True,
                          color_discrete_map=cluster_legend_colors, category_orders=category_orders,
                          title="Relative Profile per Variable (Z-Score)")
    fig_profile.update_traces(line={"width": 3}, marker={"size": 12})
    fig_profile.update_layout(height=620, xaxis_title="", yaxis_title="Z-Score", legend_title_text="<b>Zona :</b>")
    st.plotly_chart(fig_profile, width="stretch")
    show_df(profile_pivot.rename(columns={"cluster_id": "zona"}))

st.markdown("**Variable Profile (Mean Value per Zona, skala asli)**")
dist_long, dist_pivot = _cluster_mean_profile(res, DIMENSION_FEATURES, label_map, display_cluster_map)
if len(dist_long):
    dist_long = dist_long.copy()
    dist_long["cluster_legend"] = dist_long["cluster_id"].astype(int).astype(str)
    fig_dist = px.line(dist_long, x="variabel", y="mean_value", color="cluster_legend", markers=True,
                       color_discrete_map=cluster_legend_colors, category_orders=category_orders,
                       title="Mean Value per Variabel dan Zona")
    fig_dist.update_traces(line={"width": 3}, marker={"size": 12})
    fig_dist.update_layout(height=430, xaxis_title="", yaxis_title="Mean Value", legend_title_text="<b>Zona :</b>")
    st.plotly_chart(fig_dist, width="stretch")
    show_df(dist_pivot.rename(columns={"cluster_id": "zona"}))

# ---- Diagnostik kontiguitas/kompaksitas (pembanding utama vs NC) ---- #
st.markdown("**Diagnostik Kontiguitas / Kompaksitas per Zona**")
st.caption(
    "n_components rendah & largest_component_share tinggi (mendekati 1) = zona makin kontigu. "
    "Bandingkan angka-angka ini dengan page Non-Contiguous untuk melihat efek β."
)
if w_used is not None:
    frag_tbl = _fragmentation_metrics(res, w_used, cluster_col="cluster")
    if len(frag_tbl):
        frag_tbl = frag_tbl.copy()
        frag_tbl["cluster_id"] = frag_tbl["cluster"].map(display_cluster_map).fillna(frag_tbl["cluster"]).astype(int)
        frag_tbl["cluster_label"] = frag_tbl["cluster"].map(label_map).fillna(frag_tbl["cluster"].astype(str))
        frag_tbl = frag_tbl.sort_values("cluster_id")
        show_df(
            frag_tbl[["cluster_id", "cluster_label", "n_members", "n_components",
                      "largest_component", "largest_component_share", "fragmentation_index"]]
            .rename(columns={"cluster_id": "zona"})
        )
else:
    st.info("Spatial weights tidak tersedia (libpysal belum terpasang).")

st.markdown("**Zone Explanation + Household Characteristics**")
fam_tbl, fam_err = _cluster_household_characteristics(res, label_map)
if fam_err:
    st.warning(fam_err)
else:
    fam_tbl = fam_tbl.copy()
    fam_tbl["cluster_id"] = fam_tbl["cluster"].map(display_cluster_map).fillna(fam_tbl["cluster"]).astype(int)
    fam_tbl = fam_tbl.sort_values("cluster_id")
    show_df(
        fam_tbl[["cluster_id", "cluster_label", "jumlah_kk", "rata2_anggota_keluarga",
                 "persen_mpi_poor", "median_pengeluaran_bulanan", "pekerjaan_dominan",
                 "ijazah_kepala_dominan"]].rename(columns={"cluster_id": "zona"})
    )

st.markdown("**Deprivation CV per Zona (skala asli 0-100)**")
cv_tbl = _cv_by_cluster(res, label_map, display_cluster_map)
if len(cv_tbl):
    show_df(cv_tbl[["cluster_id", "cluster_label", "mean_deprivation_score",
                    "std_deprivation_score", "cv_deprivation_score"]].rename(columns={"cluster_id": "zona"}))

# ---- Cutoff ---- #
if SCORE_COL in res.columns and pd.api.types.is_numeric_dtype(res[SCORE_COL]):
    cutoff_tbl, cutoffs, cutoff_acc = _compute_cutoff_table(res, label_map, score_col=SCORE_COL, cluster_col="cluster")
    st.markdown("**Zone Cutoff Explanation (based on deprivation_score)**")
    st.caption("Cutoff = titik tengah antar mean zona terurut. Aturan interpretasi 1 variabel, bukan pengganti Ward multivariat.")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah cutoff", f"{len(cutoffs)}")
    c2.metric("Akurasi aturan cutoff vs zona", f"{cutoff_acc*100:.2f}%")
    cutoff_tbl = cutoff_tbl.copy()
    cutoff_tbl["cluster_id"] = cutoff_tbl["cluster"].map(display_cluster_map).fillna(cutoff_tbl["cluster"]).astype(int)
    cutoff_tbl = cutoff_tbl.sort_values("cluster_id")
    show_df(cutoff_tbl[["cluster_id", "label", "n", "mean", "median", "min", "max", "cutoff_interval"]]
            .rename(columns={"cluster_id": "zona"}))

    col_cut_l, col_cut_r = st.columns(2)
    with col_cut_l:
        fig_hist = px.histogram(res, x=SCORE_COL, color="cluster_legend", barmode="overlay", opacity=0.45, nbins=50,
                                title="Deprivation Score Distribution + Cutoff Lines",
                                color_discrete_map=cluster_legend_colors, category_orders=category_orders)
        for c in cutoffs:
            fig_hist.add_vline(x=float(c), line_width=2, line_dash="dash", line_color="#c62828")
        fig_hist.update_layout(height=420, legend_title_text="Zona :")
        st.plotly_chart(fig_hist, width="stretch")
    with col_cut_r:
        res["cluster_id_label"] = res["cluster_id"].apply(lambda x: f"{ZONE_PREFIX} {int(x)}")
        fig_box = px.box(res, x="cluster_id_label", y=SCORE_COL, color="cluster_legend", points=False,
                         title="Score Distribution per Zone",
                         color_discrete_map=cluster_legend_colors, category_orders=category_orders)
        for c in cutoffs:
            fig_box.add_hline(y=float(c), line_width=1, line_dash="dot", line_color="#666")
        fig_box.update_layout(height=420, xaxis_title="Zona", legend_title_text="Zona :")
        st.plotly_chart(fig_box, width="stretch")

# ---- Rumah tangga & dusun ---- #
st.markdown("**Households & Hamlets Summary per Zona**")
kk_dusun_summary, kk_dusun_detail, kk_dusun_err = _cluster_kk_dusun_tables(res, label_map)
if kk_dusun_err:
    st.warning(kk_dusun_err)
else:
    kk_dusun_summary = kk_dusun_summary.copy()
    kk_dusun_summary["cluster_id"] = kk_dusun_summary["cluster"].map(display_cluster_map).fillna(kk_dusun_summary["cluster"]).astype(int)
    kk_dusun_summary = kk_dusun_summary.sort_values("cluster_id")
    show_df(kk_dusun_summary[["cluster_id", "cluster_label", "jumlah_kk", "jumlah_dusun", "daftar_dusun"]]
            .rename(columns={"cluster_id": "zona"}))
    with st.expander("Zona x Hamlet Details (Households)", expanded=False):
        kk_dusun_detail = kk_dusun_detail.copy()
        kk_dusun_detail["cluster_id"] = kk_dusun_detail["cluster"].map(display_cluster_map).fillna(kk_dusun_detail["cluster"]).astype(int)
        kk_dusun_detail = kk_dusun_detail.sort_values(["cluster_id", "jumlah_kk", "dusun"], ascending=[True, False, True])
        show_df(kk_dusun_detail[["cluster_id", "cluster_label", "dusun", "jumlah_kk"]].rename(columns={"cluster_id": "zona"}))

# ---- Unduh (CSV + GPKG untuk QGIS) ---- #
st.markdown("---")
st.markdown("#### Download Data Titik Zona")
if {"x_utm", "y_utm"}.issubset(res.columns):
    _geo_x, _geo_y, _crs = "x_utm", "y_utm", "EPSG:32749"
elif {"lon", "lat"}.issubset(res.columns):
    _geo_x, _geo_y, _crs = "lon", "lat", "EPSG:4326"
else:
    _geo_x = _geo_y = _crs = None

_dl_base = [c for c in ["id_keluarga", "dusun", "cluster_id", "cluster_label", SCORE_COL] if c in res.columns]
_dl_dims = [c for c in DIMENSION_FEATURES if c in res.columns]
_dl_coords = [c for c in ["lon", "lat", "x_utm", "y_utm"] if c in res.columns]
_dl_res = res[list(dict.fromkeys(_dl_base + _dl_dims + _dl_coords))].copy()
st.caption(f"{len(_dl_res):,} titik rumah tangga + label zona + skor deprivasi + koordinat. GeoPackage: {_crs or 'N/A'}.")

_c_csv, _c_gpkg = st.columns(2)
with _c_csv:
    st.download_button("CSV (.csv)", data=_dl_res.to_csv(index=False).encode("utf-8"),
                       file_name=f"rqzm_contiguous_zones_b{float(beta):g}.csv", mime="text/csv", width="stretch")
with _c_gpkg:
    if _geo_x is None:
        st.info("Koordinat tidak tersedia untuk ekspor GeoPackage.")
    else:
        try:
            import tempfile
            import geopandas as gpd
            from shapely.geometry import Point
            _valid = _dl_res.dropna(subset=[_geo_x, _geo_y]).copy()
            _gdf = gpd.GeoDataFrame(
                _valid, geometry=[Point(float(x), float(y)) for x, y in zip(_valid[_geo_x], _valid[_geo_y])], crs=_crs)
            with tempfile.TemporaryDirectory() as _td:
                _gpkg_path = Path(_td) / "rqzm_contiguous_zones.gpkg"
                _gdf.to_file(_gpkg_path, driver="GPKG", layer="rqzm_contiguous")
                _gpkg_bytes = _gpkg_path.read_bytes()
            st.download_button("GeoPackage (.gpkg) — QGIS", data=_gpkg_bytes,
                               file_name=f"rqzm_contiguous_zones_b{float(beta):g}.gpkg",
                               mime="application/geopackage+sqlite3", width="stretch")
        except Exception as _e:
            st.warning(f"GeoPackage tidak tersedia: {_e}")

# ---- Dendrogram ---- #
st.markdown("**Dendrogram Hierarki (Ward, [atribut | β×koordinat])**")
st.caption("Dendrogram dari linkage Ward atas ruang fitur ber-β yang sama dengan pembentuk zona (sebelum merge sliver).")
if Z_link is None:
    st.info("Linkage tidak tersedia (mode fallback CSV).")
else:
    rendered, dendro_err = _render_dendrogram_from_Z(Z_link, recommended_k_for_dendro)
    if dendro_err:
        st.warning(dendro_err)
    else:
        fig_dendro, k_target, k_realized, cut_h = rendered
        st.pyplot(fig_dendro, clear_figure=True, width="stretch")
        if k_realized == k_target:
            st.success(f"Dipotong pada tinggi {cut_h:.4f} -> tepat {k_realized} zona (target k={k_target}).")
        else:
            st.warning(f"Dipotong pada tinggi {cut_h:.4f} -> {k_realized} zona (target k={k_target}; bisa beda bila ada ties).")

if RUNINFO_PATH.exists():
    with st.expander("Run Info", expanded=False):
        st.code(RUNINFO_PATH.read_text(encoding="utf-8"), language="text")
   