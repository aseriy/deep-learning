import os, sys, argparse, json
import array
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, select, func, cast, String, Column, Integer
from sqlalchemy.sql.sqltypes import NullType
from sklearn.decomposition import PCA

from dash import Dash, html, dcc, Output, Input, State, no_update
import plotly.express as px

# ---------- CLI ----------
def parse_cli():
    # Parse CLI args
    p = argparse.ArgumentParser()
    p.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    p.add_argument("-t", "--table", required=True, help="Table with centroids")
    p.add_argument("--poll", type=int, default=10, help="Poll seconds")
    p.add_argument("--dims", type=int, choices=[2,3], default=2, help="Output dims")
    p.add_argument("--port", type=int, default=None, help="HTTP port (fallback to $PORT or 9000)")
    return p.parse_args()

ARGS = parse_cli()
ENGINE = create_engine(ARGS.url, pool_pre_ping=True)

# ---------- SQLAlchemy Core table definition (no reflection) ----------

def _split_schema_table(qualified: str):
    # Accept "schema.table" or just "table"
    if "." in qualified and not qualified.startswith('"'):
        schema, name = qualified.split(".", 1)
        return schema, name
    return None, qualified

_SCHEMA, _TABLE_NAME = _split_schema_table(ARGS.table)
_MD = MetaData()
_T = Table(
    _TABLE_NAME,
    _MD,
    Column("id", Integer, primary_key=True),
    Column("centroid", NullType()),  # CockroachDB VECTOR; parsed manually
    Column("epoch", Integer, primary_key=True),
    schema=_SCHEMA,
)
# Derive the clusters table name from the centroid table ("*_centroid" -> "*_clusters")
_CLUSTERS_TNAME = (
    _TABLE_NAME.replace("_centroid", "_clusters")
    if _TABLE_NAME.endswith("_centroid")
    else "passage_passage_vector_clusters"
)

_TC = Table(
    _CLUSTERS_TNAME,
    _MD,
    Column("pid", String, primary_key=True),
    Column("epoch", Integer, primary_key=True),
    Column("cluster_id", Integer),
    schema=_SCHEMA,
)

_ID_COL = _T.c.id
_VEC_COL = _T.c.centroid
_EPOCH_COL = _T.c.epoch

# ---------- helpers ----------
def parse_vec(x):
    # Robust vector parser for various driver return types
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)
    if isinstance(x, array.array) and x.typecode in ("f", "d"):
        return np.frombuffer(x, dtype=float)
    if isinstance(x, memoryview):
        try:
            return np.frombuffer(x, dtype=float)
        except Exception:
            pass  # fall through to string parsing
    # fallback to string parsing like "[0.1,-0.2,...]" or "{...}"
    s = str(x).strip().replace("{", "[").replace("}", "]")
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return np.array([], dtype=float)
    try:
        return np.fromstring(s, sep=",", dtype=float)
    except Exception:
        return np.array([], dtype=float)

def _fetch_cluster_counts():
    latest_epoch = select(func.max(_EPOCH_COL)).scalar_subquery()
    stmt = (
        select(
            _TC.c.cluster_id.label("cid"),
            func.count().label("n_members"),
        )
        .where(_TC.c.epoch == latest_epoch)
        .group_by(_TC.c.cluster_id)
        .order_by(_TC.c.cluster_id)
    )
    with ENGINE.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    # dict like {cid: n_members, ...}
    return {int(cid): int(n) for cid, n in rows}

def _fetch_rows(cast_to_string: bool):
    # Build a SQLAlchemy Core statement that selects latest epoch only
    vec_expr = cast(_VEC_COL, String) if cast_to_string else _VEC_COL
    latest_epoch = select(func.max(_EPOCH_COL)).scalar_subquery()

    stmt = (
        select(
            _ID_COL.label("cid"),
            vec_expr.label("vec"),
        )
        .where(_EPOCH_COL == latest_epoch)
        .order_by(_ID_COL)
    )

    with ENGINE.connect() as conn:
        return conn.execute(stmt).fetchall()


def radii_from_counts_2d(meta_df, counts_by_cid, max_frac=0.10):
    """
    Compute a 2D circle radius per centroid so that circle AREA ~ member count.
    The largest circle's radius is max_frac of the plot's dominant span.
    """
    import numpy as np

    cids = meta_df["cid"].astype(int).to_numpy()
    counts = np.array([counts_by_cid.get(int(cid), 0) for cid in cids], dtype=float)
    if counts.size == 0 or counts.max() <= 0:
        return np.zeros_like(counts)

    # Area proportionality => radius ∝ sqrt(count)
    # Normalize by max count so the biggest circle has radius = max_frac * plot span
    span_x = float(meta_df["x"].max() - meta_df["x"].min())
    span_y = float(meta_df["y"].max() - meta_df["y"].min())
    span = max(span_x, span_y) or 1.0
    r_max = max_frac * span
    return r_max * np.sqrt(counts / counts.max())


def load_centroids():
    # Try without cast first
    rows = _fetch_rows(cast_to_string=False)
    if not rows:
        return pd.DataFrame(), None

    def build_df(rows_):
        df_ = pd.DataFrame(rows_, columns=["cid", "vec"])
        df_["vec"] = df_["vec"].map(parse_vec)
        lens_ = df_["vec"].map(len)
        return df_, lens_

    df, lens = build_df(rows)

    # If parsing failed (all zeros), retry with ::STRING automatically
    if lens.max() == 0:
        rows2 = _fetch_rows(cast_to_string=True)
        if rows2:
            df, lens = build_df(rows2)

    if lens.empty or lens.max() == 0:
        return pd.DataFrame(), None

    # Keep only the dominant vector length, report drops
    vc = lens.value_counts()
    dim = int(vc.idxmax())
    dropped = int((lens != dim).sum())

    df = df[lens == dim].sort_values("cid").reset_index(drop=True)
    X = np.vstack(df["vec"].to_list())    # Build 'meta' and attach attrs after column selection so they persist
    meta = df[["cid"]].copy()
    meta.attrs["dropped"] = dropped
    meta.attrs["dim"] = dim
    return meta, X

def pca_fit(X, n_out):
    p = PCA(n_components=n_out, random_state=42).fit(X)
    return {
        "components": p.components_.tolist(),
        "mean": p.mean_.tolist(),
        "n_out": n_out,
    }

def pca_transform(X, basis):
    comps = np.asarray(basis["components"])
    mean  = np.asarray(basis["mean"])
    Xc = X - mean
    Y = Xc @ comps.T
    return Y

# ---------- Dash app ----------
app = Dash(__name__)
server = app.server  # for gunicorn

app.layout = html.Div([
    html.H2("Centroid Tracker (Dash • PCA)"),
    html.Div([
        html.Label("Dims"),
        dcc.Dropdown(
            id="dims",
            options=[{"label":"2D","value":2},{"label":"3D","value":3}],
            value=2,  # force 2D on initial load
            clearable=False,
            style={"width":"120px"}
        ),
        html.Button("Reset PCA basis", id="reset", n_clicks=0, style={"marginLeft":"12px"}),
        html.Span(id="status", style={"marginLeft":"12px", "opacity":0.7}),
    ], style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"8px"}),

    dcc.Graph(id="graph", style={"height":"70vh"}),

    # Stores
    dcc.Store(id="pca_basis"),         # holds components/mean
    dcc.Store(id="last_basis_dims"),   # tracks dims used for basis
    dcc.Store(id="last_reset_clicks", data=0),   # last seen Reset n_clicks

    # Polling
    dcc.Interval(id="poll", interval=ARGS.poll*1000, n_intervals=0),
])

@app.callback(
    Output("graph","figure"),
    Output("pca_basis","data"),
    Output("last_basis_dims","data"),
    Output("last_reset_clicks","data"),
    Output("status","children"),
    Input("poll","n_intervals"),
    Input("reset","n_clicks"),
    Input("dims","value"),          # <-- moved here (was State)
    State("pca_basis","data"),
    State("last_basis_dims","data"),
    State("last_reset_clicks","data"),
    prevent_initial_call=False,
)
def refresh(_n, reset_clicks, n_out, basis, last_dims, last_reset):
    # Load latest centroids
    meta, X = load_centroids()
    if X is None or meta.empty:
        fig = px.scatter(x=[], y=[])
        return fig, basis, last_dims, last_reset, "No data"

    # Ensure basis exists & matches dims, or reset requested
    # Detect a *new* reset click and decide if we need to refit
    reset_clicks = reset_clicks or 0
    last_reset = last_reset or 0
    clicked_now = reset_clicks > last_reset
    need_fit = (basis is None) or (last_dims != n_out) or clicked_now
    if need_fit:
        basis = pca_fit(X, n_out)
        last_dims = n_out

    # Transform with (cached) basis
    Y = pca_transform(X, basis)

    # Build figure
    meta = meta.copy()
    # Cluster sizes for latest epoch (used for circle/marker sizing)
    counts = _fetch_cluster_counts()
    if n_out == 2:
        meta["x"], meta["y"] = Y[:,0], Y[:,1]
        fig = px.scatter(meta, x="x", y="y", text=meta["cid"].astype(str),
                         title="Centroids (fixed PCA basis)")
        fig.update_traces(marker=dict(size=16, symbol="x"), textposition="top center")

        # Circles sized by cluster member counts (area ∝ count)
        radii = radii_from_counts_2d(meta, counts, max_frac=0.10)
        for (x, y, r) in zip(meta["x"], meta["y"], radii):
            if r <= 0:
                continue
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x - r, x1=x + r,
                y0=y - r, y1=y + r,
                line=dict(width=1, color="rgba(0,0,0,0.25)"),
                layer="below",
            )
        # Keep units equal so circles render as true circles
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Ensure the plot range includes the circles (not just the points)
        if len(radii) and float(np.max(radii)) > 0:
            rmax = float(np.max(radii))
            x_min, x_max = float(meta["x"].min()) - rmax, float(meta["x"].max()) + rmax
            y_min, y_max = float(meta["y"].min()) - rmax, float(meta["y"].max()) + rmax
            fig.update_xaxes(range=[x_min, x_max])
            fig.update_yaxes(range=[y_min, y_max])
    else:
        meta["x"], meta["y"], meta["z"] = Y[:,0], Y[:,1], Y[:,2]
        fig = px.scatter_3d(meta, x="x", y="y", z="z", text=meta["cid"].astype(str),
                            title="Centroids (fixed PCA basis)")

        # Scale marker sizes by √count (proxy for circle area)
        s_counts = np.array([counts.get(int(cid), 0) for cid in meta["cid"]], dtype=float)
        if s_counts.max() > 0:
            sizes = 6.0 + 24.0 * (np.sqrt(s_counts) / np.sqrt(s_counts.max()))
            fig.update_traces(marker=dict(size=sizes, symbol="x"))
        else:
            fig.update_traces(marker=dict(size=6, symbol="x"))

    fig.update_layout(showlegend=False)
    dropped = getattr(meta, "attrs", {}).get("dropped", 0)
    dim = getattr(meta, "attrs", {}).get("dim", X.shape[1])
    status = f"Centroids: {len(meta)} • Dim: {dim} • Dims out: {n_out}" + (f" • Dropped: {dropped}" if dropped else "")
    new_last_reset = reset_clicks if clicked_now else last_reset
    return fig, basis, last_dims, new_last_reset, status

if __name__ == "__main__":
    # Dev mode (useful for quick testing). For production, run via gunicorn below.
    port = ARGS.port or int(os.getenv("PORT", "9000"))
    app.run(host="0.0.0.0", port=port, debug=False)
