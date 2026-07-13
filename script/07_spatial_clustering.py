from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "mpi_household.parquet"
DEFAULT_OUT_DIR = ROOT / "outputs"


def normalize_columns(cols: list[str]) -> list[str]:
    out: list[str] = []
    for c in cols:
        c2 = str(c).strip().lower()
        c2 = re.sub(r"\s+", "_", c2)
        c2 = c2.replace("/", "_")
        out.append(c2)
    return out


def read_table(path: Path, sep: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=sep, engine="c", on_bad_lines="skip", low_memory=False)
    df.columns = normalize_columns(list(df.columns))
    return df


def auto_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "deprivation_score",
        "dim_human",
        "dim_physical",
        "dim_natural",
        "dim_social",
        "dim_financial",
    ]
    picked = [c for c in preferred if c in df.columns]
    if picked:
        return picked

    exclude = {
        "lat",
        "lng",
        "lon",
        "x_utm",
        "y_utm",
        "id_keluarga",
        "id_keluargas",
        "cluster",
        "mpi_poor",
    }
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c not in exclude]
    return num_cols[:8]


def build_point_weights(coords: np.ndarray, weight_type: str, k_neighbors: int, distance: float):
    if weight_type == "none":
        return None

    try:
        import libpysal
    except Exception as exc:
        raise SystemExit(
            "Spatial weights membutuhkan libpysal. Install dulu:\n"
            "  pip install libpysal\n"
            f"Detail error: {exc}"
        )

    n = coords.shape[0]
    if n < 3:
        raise SystemExit("Observasi terlalu sedikit untuk spatial weights.")

    if weight_type == "knn":
        k_eff = min(max(1, int(k_neighbors)), n - 1)
        w = libpysal.weights.KNN.from_array(coords, k=k_eff)
    elif weight_type == "distance":
        w = libpysal.weights.DistanceBand.from_array(
            coords, threshold=float(distance), binary=False, silence_warnings=True
        )
    else:
        raise SystemExit(f"weight_type tidak valid untuk point data: {weight_type}")

    w.transform = "R"
    return w.sparse.tocsr()


def build_queen_dataset(
    df: pd.DataFrame,
    features: list[str],
    shp_path: Path,
    shp_key: str,
    data_key: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    try:
        import geopandas as gpd
        import libpysal
    except Exception as exc:
        raise SystemExit(
            "Queen contiguity membutuhkan geopandas dan libpysal.\n"
            "Install dulu:\n"
            "  pip install geopandas libpysal\n"
            f"Detail error: {exc}"
        )

    if not shp_path.exists():
        raise SystemExit(f"Shapefile tidak ditemukan: {shp_path}")

    gdf = gpd.read_file(shp_path)
    gdf.columns = normalize_columns(list(gdf.columns))
    shp_key = shp_key.strip().lower()
    data_key = data_key.strip().lower()

    if shp_key not in gdf.columns:
        raise SystemExit(f"Kolom --shp-key '{shp_key}' tidak ditemukan di shapefile.")
    if data_key not in df.columns:
        raise SystemExit(f"Kolom --data-key '{data_key}' tidak ditemukan di data.")

    agg_input = df[[data_key] + features].copy()
    agg_input[data_key] = agg_input[data_key].astype(str).str.strip().str.lower()
    agg_input = agg_input.dropna(subset=features, how="any")
    if len(agg_input) == 0:
        raise SystemExit("Tidak ada data valid untuk agregasi queen.")

    agg = agg_input.groupby(data_key, dropna=False)[features].mean().reset_index()

    gdf["_join_key"] = gdf[shp_key].astype(str).str.strip().str.lower()
    gdf = gdf.merge(agg, left_on="_join_key", right_on=data_key, how="inner")
    gdf = gdf.dropna(subset=features, how="any")
    if len(gdf) < 3:
        raise SystemExit("Hasil join queen terlalu sedikit (n < 3).")

    w = libpysal.weights.Queen.from_dataframe(gdf)
    w.transform = "R"
    W = w.sparse.tocsr()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf_metric = gdf.to_crs("EPSG:3857")
    centroids_metric = gdf_metric.geometry.centroid
    centroids_ll = gpd.GeoSeries(centroids_metric, crs="EPSG:3857").to_crs("EPSG:4326")

    work = pd.DataFrame(
        {
            "unit_key": gdf["_join_key"].astype(str).values,
            "lon": centroids_ll.x.to_numpy(),
            "lat": centroids_ll.y.to_numpy(),
        }
    )
    for c in features:
        work[c] = pd.to_numeric(gdf[c], errors="coerce").to_numpy()

    X = work[features].to_numpy(dtype=float)
    return work, X, W


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Spatial clustering (KMeans) dengan opsi spatial weights "
            "(none/knn/distance/queen) dan fitur spatial sederhana."
        )
    )
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Path input data (parquet/csv/xlsx)")
    p.add_argument("--sep", default=";", help="Separator untuk CSV (default ;) ")
    p.add_argument("--features", default="", help="Kolom fitur numerik, pisah koma. Jika kosong pakai auto-select.")
    p.add_argument("--k", type=int, default=5, help="Jumlah cluster KMeans")
    p.add_argument(
        "--weights",
        choices=["none", "knn", "distance", "queen"],
        default="knn",
        help="Jenis spatial weights",
    )
    p.add_argument("--k-neighbors", type=int, default=8, help="K tetangga untuk knn")
    p.add_argument(
        "--distance",
        type=float,
        default=150.0,
        help="Threshold jarak untuk distance weights (meter jika UTM, derajat jika lon/lat)",
    )
    p.add_argument("--coord-system", choices=["auto", "geo", "proj"], default="auto", help="Sistem koordinat untuk point weights")
    p.add_argument("--lat-col", default="lat", help="Kolom latitude")
    p.add_argument("--lon-col", default="lng", help="Kolom longitude")
    p.add_argument("--xcoord-col", default="x_utm", help="Kolom X projected")
    p.add_argument("--ycoord-col", default="y_utm", help="Kolom Y projected")
    p.add_argument("--shp", default="", help="Path SHP untuk queen contiguity")
    p.add_argument("--shp-key", default="Dusun", help="Kolom key di SHP untuk queen")
    p.add_argument("--data-key", default="dusun", help="Kolom key di data untuk queen")
    p.add_argument("--scale", action="store_true", help="Standardize fitur sebelum clustering")
    p.add_argument("--add-neighbor-count", action="store_true", help="Tambahkan fitur jumlah tetangga dari weights")
    p.add_argument("--seed", type=int, default=42, help="Random seed KMeans")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Folder output")
    p.add_argument("--out-prefix", default="spatial_clustering", help="Prefix nama file output")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        raise SystemExit(f"Input tidak ditemukan: {input_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path, args.sep)
    if len(df) == 0:
        raise SystemExit("Data kosong.")

    if args.features.strip():
        features = [x.strip().lower() for x in args.features.split(",") if x.strip()]
    else:
        features = auto_feature_columns(df)

    if not features:
        raise SystemExit("Tidak ada fitur numerik yang bisa dipakai untuk clustering.")

    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise SystemExit(f"Kolom fitur tidak ditemukan: {missing_features}")

    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mode = "point"
    W = None
    work: pd.DataFrame
    X_base: np.ndarray

    if args.weights == "queen":
        mode = "polygon_aggregated"
        if not args.shp:
            raise SystemExit("weights=queen membutuhkan --shp.")
        work, X_base, W = build_queen_dataset(
            df=df,
            features=features,
            shp_path=Path(args.shp),
            shp_key=args.shp_key,
            data_key=args.data_key,
        )
    else:
        lat_col = args.lat_col.strip().lower()
        lon_col = args.lon_col.strip().lower()
        xcoord_col = args.xcoord_col.strip().lower()
        ycoord_col = args.ycoord_col.strip().lower()

        use_geo = False
        if args.coord_system == "geo":
            use_geo = True
        elif args.coord_system == "proj":
            use_geo = False
        else:
            use_geo = not (xcoord_col in df.columns and ycoord_col in df.columns)

        coord_cols = [lon_col, lat_col] if use_geo else [xcoord_col, ycoord_col]
        missing_coords = [c for c in coord_cols if c not in df.columns]
        if missing_coords:
            raise SystemExit(f"Kolom koordinat tidak ditemukan: {missing_coords}")

        for c in coord_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        needed = features + coord_cols
        work = df[needed].copy()
        work = work.dropna(subset=needed).reset_index(drop=True)
        if len(work) < 3:
            raise SystemExit("Data valid terlalu sedikit (n < 3).")

        X_base = work[features].to_numpy(dtype=float)
        coords = work[coord_cols].to_numpy(dtype=float)
        W = build_point_weights(coords, args.weights, args.k_neighbors, args.distance)

        if use_geo:
            work = work.rename(columns={lon_col: "lon", lat_col: "lat"})
        else:
            work = work.rename(columns={xcoord_col: "x_utm", ycoord_col: "y_utm"})

    X_model = X_base.copy()
    feature_model = features.copy()

    if W is not None and args.add_neighbor_count:
        neighbor_count = np.diff(W.indptr).astype(float)
        X_model = np.column_stack([X_model, neighbor_count])
        feature_model.append("neighbor_count")
        work["neighbor_count"] = neighbor_count

    if args.scale:
        scaler = StandardScaler()
        X_model = scaler.fit_transform(X_model)

    n = X_model.shape[0]
    if n < max(3, args.k):
        raise SystemExit(f"Jumlah observasi valid ({n}) lebih kecil dari k ({args.k}).")

    model = KMeans(n_clusters=int(args.k), random_state=int(args.seed), n_init="auto")
    labels = model.fit_predict(X_model)
    work["cluster"] = labels.astype(int)

    result_path = out_dir / f"{args.out_prefix}_results.csv"
    summary_path = out_dir / f"{args.out_prefix}_summary.csv"
    meta_path = out_dir / f"{args.out_prefix}_runinfo.txt"

    work.to_csv(result_path, index=False)

    summary = (
        work.groupby("cluster", dropna=False)
        .agg(
            n=("cluster", "size"),
            **{f"mean_{c}": (c, "mean") for c in features if c in work.columns},
        )
        .reset_index()
        .sort_values("cluster")
    )
    summary.to_csv(summary_path, index=False)

    with meta_path.open("w", encoding="utf-8") as f:
        f.write("Spatial Clustering Run Info\n")
        f.write(f"input={input_path}\n")
        f.write(f"mode={mode}\n")
        f.write(f"weights={args.weights}\n")
        f.write(f"k={args.k}\n")
        f.write(f"n={n}\n")
        f.write(f"features={','.join(features)}\n")
        f.write(f"model_features={','.join(feature_model)}\n")
        f.write(f"scale={args.scale}\n")
        f.write(f"add_neighbor_count={args.add_neighbor_count}\n")
        f.write(f"results={result_path}\n")
        f.write(f"summary={summary_path}\n")

    print("Selesai.")
    print(f"- Hasil cluster: {result_path}")
    print(f"- Ringkasan cluster: {summary_path}")
    print(f"- Run info: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
