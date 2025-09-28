import os, sys, argparse, json
import array
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, select, func, cast, String, Column, Integer, DateTime
from sqlalchemy.sql.sqltypes import NullType
from sklearn.decomposition import PCA

from dash import Dash, html, dcc, Output, Input, State, no_update
import plotly.express as px
from datetime import datetime, timezone

# ---------- CLI ----------
def parse_cli():
    # Parse CLI args
    p = argparse.ArgumentParser()
    p.add_argument("-u", "--url", required=True, help="CockroachDB connection URL")
    p.add_argument("-t", "--table", required=True, help="Target table name")
    p.add_argument("-c", "--column", required=True, help="Column vector column")
    p.add_argument("--poll", type=int, default=10, help="Poll seconds")
    p.add_argument("--dims", type=int, choices=[2,3], default=2, help="Output dims")
    p.add_argument("--port", type=int, default=None, help="HTTP port (fallback to $PORT or 9000)")
    return p.parse_args()

ARGS = parse_cli()
ENGINE = create_engine(ARGS.url, pool_pre_ping=True)

# ---------- SQLAlchemy Core table definition (no reflection) ----------

_MD = MetaData()

_T = Table(
    f"{ARGS.table}_{ARGS.column}_centroid",
    _MD,
    Column("id", Integer, primary_key=True),
    Column("centroid", NullType()),  # CockroachDB VECTOR; parsed manually
    Column("epoch", Integer, primary_key=True)
)

_TC = Table(
    f"{ARGS.table}_{ARGS.column}_clusters",
    _MD,
    Column("pid", String, primary_key=True),
    Column("epoch", Integer, primary_key=True),
    Column("cluster_id", Integer)
)

_TE = Table(
    f"{ARGS.table}_{ARGS.column}_epoch",
    _MD,
    Column("epoch", Integer, primary_key=True),
    Column("current_at", DateTime(timezone=True))
)

_ID_COL = _T.c.id
_VEC_COL = _T.c.centroid
_EPOCH_COL = _TE.c.epoch

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



def _get_current_epoch():
    stmt = select(_TE.c.epoch, _TE.c.current_at).order_by(_TE.c.epoch.desc()).limit(1)
    with ENGINE.connect() as conn:
        row = conn.execute(stmt).first()
    if row is None:
        return None, None
    return row.epoch, row.current_at

def _format_age(dt):
    if not dt:
        return "—"
    now = datetime.now(timezone.utc) if dt.tzinfo else datetime.utcnow()
    delta = now - dt
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def _fetch_cluster_counts(current_epoch):
    stmt = (
        select(
            _TC.c.cluster_id.label("cid"),
            func.count().label("n_members"),
        )
        .where(_TC.c.epoch == current_epoch)
        .group_by(_TC.c.cluster_id)
        .order_by(_TC.c.cluster_id)
    )
    with ENGINE.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    # dict like {cid: n_members, ...}
    return {int(cid): int(n) for cid, n in rows}


def _fetch_rows(current_epoch, cast_to_string: bool):
    # Build a SQLAlchemy Core statement that selects latest epoch only
    vec_expr = cast(_VEC_COL, String) if cast_to_string else _VEC_COL
    stmt = (
        select(
            _ID_COL.label("cid"),
            vec_expr.label("vec"),
        )
        .where(_T.c.epoch == current_epoch)
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


def load_centroids(current_epoch):
    # Try without cast first
    rows = _fetch_rows(current_epoch, cast_to_string=False)
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
        rows2 = _fetch_rows(current_epoch, cast_to_string=True)
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
        ),        html.Span(id="status", style={"marginLeft":"12px", "opacity":0.7}),
    ], style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"8px"}),

    dcc.Graph(id="graph", style={"height":"70vh"}),

    # Stores
    dcc.Store(id="pca_basis"),         # holds components/mean
    dcc.Store(id="last_basis_dims"),   # tracks dims used for basis
    # Polling
    dcc.Interval(id="poll", interval=ARGS.poll*1000, n_intervals=0),
])

@app.callback(
    Output("graph","figure"),
    Output("pca_basis","data"),
    Output("last_basis_dims","data"),
    Output("status","children"),
    Input("poll","n_intervals"),
    Input("dims","value"),
    State("pca_basis","data"),
    State("last_basis_dims","data"),
    prevent_initial_call=False,
)


def refresh(_n, n_out, basis, last_dims):
    current_epoch, current_at = _get_current_epoch()
    
    # Load latest centroids
    meta, X = load_centroids(current_epoch)
    if X is None or meta.empty:
        fig = px.scatter(x=[], y=[], template="plotly_white")
        return fig, basis, last_dims, "No data"

    # Refit PCA on every refresh so axes follow the latest data
    basis = pca_fit(X, n_out)
    last_dims = n_out

    # Transform with (cached) basis
    Y = pca_transform(X, basis)

    # Build figure
    meta = meta.copy()
    # Cluster sizes for latest epoch (used for circle/marker sizing)
    counts = _fetch_cluster_counts(current_epoch)
    if n_out == 2:
        meta["x"], meta["y"] = Y[:,0], Y[:,1]
        fig = px.scatter(meta, x="x", y="y", text=meta["cid"].astype(str),
                         title="Centroids (PCA refit each refresh)", template="plotly_white")
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
                layer="below"
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
                            title="Centroids (PCA refit each refresh)", template="plotly_white")

        # Scale marker sizes by √count (proxy for circle area)
        s_counts = np.array([counts.get(int(cid), 0) for cid in meta["cid"]], dtype=float)
        if s_counts.max() > 0:
            sizes = 6.0 + 24.0 * (np.sqrt(s_counts) / np.sqrt(s_counts.max()))
            fig.update_traces(marker=dict(size=sizes))
        else:
            fig.update_traces(marker=dict(size=6))

    fig.update_layout(showlegend=False)
    dropped = getattr(meta, "attrs", {}).get("dropped", 0)
    dim = getattr(meta, "attrs", {}).get("dim", X.shape[1])
    status = f"Centroids: {len(meta)} • Dim: {dim} • Dims out: {n_out}" + (f" • Dropped: {dropped}" if dropped else "") + f" • Epoch: {current_epoch} • Age: {_format_age(current_at)}"
    return fig, basis, last_dims, status

if __name__ == "__main__":
    # Dev mode (useful for quick testing). For production, run via gunicorn below.
    port = ARGS.port or int(os.getenv("PORT", "9000"))
    app.run(host="0.0.0.0", port=port, debug=False)
