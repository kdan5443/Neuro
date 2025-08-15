#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Causal Discovery Dashboard (Tableau-style) — with AgGrid, URL params, PNG export
# --------------------------------------------------------------------------------
# pip install -r requirements.txt
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io, warnings, sys, math, json, datetime as dt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

from pgmpy.estimators import PC
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

# AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

warnings.filterwarnings("ignore")
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# ---------- Page / Theme ----------
st.set_page_config(layout="wide", page_title="Causal Discovery Dashboard")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 0.8rem;}
.card{padding:14px;border-radius:16px;border:1px solid rgba(255,255,255,0.08);
      background:rgba(255,255,255,0.04)}
.kpi-label{font-size:12px;opacity:.8}
.kpi-value{font-size:28px;font-weight:700;margin-top:2px}
.kpi-delta{font-size:12px;opacity:.9;margin-top:4px}
</style>""", unsafe_allow_html=True)

st.title("Causal Discovery — Explore • Correlate • Infer")
st.caption("Interactive, Tableau-style analytics with correlation, Granger causality, and causal Bayesian networks (PC & DBN).")
st.divider()

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def _coerce_numeric(df: pd.DataFrame):
    df_num = pd.DataFrame(index=df.index)
    dropped = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df_num[c] = df[c]
        else:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() >= max(3, int(0.5 * len(x))):
                df_num[c] = x
            else:
                dropped.append(c)
    return df_num, dropped

@st.cache_data(show_spinner=False)
def _standardize(df: pd.DataFrame):
    scaler = StandardScaler()
    z = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
    return z

def _validate_time_index(df: pd.DataFrame):
    msgs, ok = [], True
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        msgs.append("Index is not datetime/period; Granger/DBN will assume evenly-spaced rows.")
        ok = False
    else:
        if not df.index.is_monotonic_increasing:
            msgs.append("Datetime index not sorted; sorting by time.")
        if df.index.inferred_freq is None:
            msgs.append("Datetime frequency not inferred; ensure roughly even spacing for Granger/DBN.")
    return ok, msgs

@st.cache_data(show_spinner=False)
def compute_corr(df: pd.DataFrame):
    return df.corr(numeric_only=True)

@st.cache_data(show_spinner=False)
def compute_adf_stationary(df: pd.DataFrame, alpha: float):
    transformed, info = {}, {}
    for c in df.columns:
        s = df[c].dropna()
        if len(s) < 3: continue
        n_diffs = 0
        while True:
            try:
                p = adfuller(s)[1]
            except Exception:
                break
            if p <= alpha or n_diffs >= 2 or len(s) < 3:
                transformed[c] = s
                info[c] = n_diffs
                break
            s = s.diff().dropna()
            n_diffs += 1
    if not transformed:
        return pd.DataFrame(index=df.index), {}
    out = pd.concat(transformed, axis=1).dropna()
    return out, info

@st.cache_data(show_spinner=False)
def auto_select_lag(df: pd.DataFrame, max_lag_cap: int = 12, criterion: str = "aic"):
    try:
        model = VAR(df.dropna())
        res = model.select_order(maxlags=min(max_lag_cap, max(1, int(len(df) / 5))))
        k = getattr(res, criterion)
        if k is None or (isinstance(k, float) and math.isnan(k)): return 3
        return int(k)
    except Exception:
        return 3

@st.cache_data(show_spinner=False)
def granger_matrix(df: pd.DataFrame, max_lag: int, alpha: float, fdr: bool = True):
    cols = df.select_dtypes("number").columns.tolist()
    results = []
    for i, cause in enumerate(cols):
        for j, effect in enumerate(cols):
            if i == j: continue
            data = df[[effect, cause]].dropna()
            if len(data) <= max_lag: continue
            try:
                gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                p = gc[max_lag][0]['ssr_ftest'][1]
                results.append((cause, effect, p))
            except Exception:
                continue
    if not results:
        return pd.DataFrame(columns=["cause", "effect", "p_raw", "p_adj", "significant"])
    pvals = [r[2] for r in results]
    if fdr and len(pvals) > 1:
        rej, padj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    else:
        rej, padj = [p < alpha for p in pvals], pvals
    out = pd.DataFrame({
        "cause": [r[0] for r in results],
        "effect": [r[1] for r in results],
        "p_raw": pvals,
        "p_adj": padj,
        "significant": rej
    }).sort_values("p_adj")
    return out

@st.cache_data(show_spinner=False)
def pc_estimate(df: pd.DataFrame, alpha: float, ci_test: str):
    est = PC(data=df)
    model = est.estimate(ci_test=ci_test, significance_level=alpha)
    dag = nx.DiGraph(model.edges())
    return dag, {'num_nodes': dag.number_of_nodes(),
                 'num_edges': dag.number_of_edges(),
                 'edges': list(dag.edges())}

@st.cache_data(show_spinner=False)
def dbn_estimate(df: pd.DataFrame, max_lag: int, alpha: float, ci_test: str):
    lagged = df.copy()
    for lag in range(1, max_lag + 1):
        for c in df.columns:
            lagged[f"{c}_t-{lag}"] = df[c].shift(lag)
    lagged = lagged.add_suffix("_t").dropna()
    est = PC(data=lagged)
    model = est.estimate(ci_test=ci_test, significance_level=alpha)
    dag = nx.DiGraph(model.edges())
    return dag, {'num_nodes': dag.number_of_nodes(),
                 'num_edges': dag.number_of_edges(),
                 'max_lag': max_lag,
                 'edges': list(dag.edges())}

# ---- Plotly graph builders ----
def kpi_card(label, value, delta=None):
    delta_html = f'<div class="kpi-delta">Δ {delta:+.2%}</div>' if delta is not None else ''
    st.markdown(f"""
    <div class="card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)

def fig_corr_heatmap(corr: pd.DataFrame, title="Correlation Matrix"):
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", origin="lower",
        title=title
    )
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    return fig

def fig_scatter(df: pd.DataFrame, x: str, y: str, color=None, title=None, hover=None):
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover, title=title)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    return fig

def fig_timeseries(df: pd.DataFrame, y_cols, title="Time Series"):
    tdf = df.copy()
    tdf["__t__"] = tdf.index
    fig = px.line(tdf, x="__t__", y=y_cols, title=title)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), xaxis_title="Time")
    return fig

def fig_dag_plotly(dag: nx.DiGraph, title="Graph"):
    if dag is None or dag.number_of_nodes() == 0:
        return go.Figure()
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(dag, prog="dot")
    except Exception:
        pos = nx.spring_layout(dag, k=0.8, iterations=50, seed=GLOBAL_SEED)

    # Edges
    edge_x, edge_y = [], []
    for u, v in dag.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=1.5), opacity=0.7, hoverinfo='skip', name="edges")

    # Nodes
    node_x, node_y, text = [], [], []
    deg = dict(dag.degree())
    for n in dag.nodes():
        x, y = pos[n]; node_x.append(x); node_y.append(y)
        text.append(f"{n} (deg={deg[n]})")
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=list(dag.nodes()),
        textposition="top center", hovertext=text, hoverinfo="text",
        marker=dict(size=[10 + 4*deg[n] for n in dag.nodes()], line=dict(width=1))
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=title, showlegend=False,
                      margin=dict(l=10,r=10,t=50,b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def df_download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def download_plotly_png(fig, label: str, filename: str):
    try:
        png_bytes = fig.to_image(format="png", scale=2)  # kaleido backend
        st.download_button(label, png_bytes, file_name=filename, mime="image/png")
    except Exception as e:
        st.caption(f"PNG export requires kaleido. If you installed it, refresh. Error: {e}")

# ---------- URL query param helpers ----------
def _get_qs_list(key: str):
    q = st.query_params.get(key, [])
    if isinstance(q, str):  # single value
        return [q]
    return list(q)

def _set_qs_list(key: str, values: list[str]):
    # Streamlit supports assignment-like API for query params
    if values:
        st.query_params[key] = values
    elif key in st.query_params:
        # remove key when empty to keep URL tidy
        qp = dict(st.query_params)
        qp.pop(key, None)
        st.query_params.clear()
        st.query_params.update(qp)

def _date_to_str(d):
    if isinstance(d, (dt.date, dt.datetime)):
        return d.isoformat()
    return str(d)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload a CSV", type="csv")

    if uploaded_file:
        st.subheader("Analysis")
        alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05, 0.01)
        lag_mode = st.radio("Granger Lag", ["Auto (AIC)", "Manual"], horizontal=True)
        granger_max_lag = st.slider("Manual Granger Max Lag", 1, 15, 3, 1, disabled=(lag_mode=="Auto (AIC)"))
        dbn_max_lag = st.slider("DBN Max Lag", 1, 5, 2, 1)
        corr_threshold = st.slider("Correlation Threshold (abs)", 0.0, 1.0, 0.1, 0.05)
        ci_test = st.selectbox("PC/DBN CI test", ["pearsonr", "partial_correlation"], index=0)
        zscore = st.toggle("Standardize (z-score)", value=True)
        sample_n = st.number_input("Row sample (0 = all)", min_value=0, value=0, step=1000,
                                   help="For large datasets, sample for speed.")
    run_button = st.button("Run", use_container_width=True)

# ---------- Main ----------
if uploaded_file and run_button:
    try:
        # Robust CSV read
        content = uploaded_file.read()
        for enc in ("utf-8", "utf-16", "latin1"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                break
            except Exception:
                continue
        else:
            st.error("Could not read CSV with tried encodings."); st.stop()

        # Use first datetime-like column as index if present
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) \
                   or pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.9]
        if dt_cols:
            c0 = dt_cols[0]
            df[c0] = pd.to_datetime(df[c0], errors="coerce")
            df = df.set_index(c0).sort_index()

        df_num, dropped = _coerce_numeric(df)
        if dropped: st.warning(f"Dropping non-numeric/unusable columns: {', '.join(dropped)}")
        if df_num.shape[1] < 2:
            st.error("Need at least 2 numeric columns after preprocessing."); st.stop()

        # Optional sampling
        if sample_n and sample_n > 0 and len(df_num) > sample_n:
            df_num = df_num.sample(sample_n, random_state=GLOBAL_SEED).sort_index()

        # Standardize
        df_work = _standardize(df_num.fillna(method="ffill").fillna(method="bfill")) if zscore \
                  else df_num.fillna(method="ffill").fillna(method="bfill")

        ok, msgs = _validate_time_index(df_work)
        for m in msgs: st.info(m)

        # ---------- GLOBAL FILTERS (Quick filters with URL param sync) ----------
        st.subheader("Filters & KPIs")
        # Categorical quick filters (low-cardinality non-numeric columns from original df)
        cat_candidates = [c for c in df.columns
                          if (c not in df_work.columns) or (not pd.api.types.is_numeric_dtype(df.get(c, pd.Series(dtype=object))))]
        cat_candidates = [c for c in cat_candidates
                          if c in df.columns and df[c].nunique(dropna=True) <= 50 and not pd.api.types.is_numeric_dtype(df[c])]

        cols_filters = st.columns(4 if len(cat_candidates) >= 3 else max(1, len(cat_candidates)))
        active_cats = {}
        for i, c in enumerate(cat_candidates[:4]):
            with cols_filters[i]:
                opts = sorted(df[c].dropna().astype(str).unique().tolist())
                default = _get_qs_list(c)
                sel = st.multiselect(c, options=opts, default=default)
                if sel: active_cats[c] = sel
                _set_qs_list(c, sel)

        # Date range filter if time index exists
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            min_d, max_d = df_work.index.min().date(), df_work.index.max().date()
            qs_start = st.query_params.get("start", [min_d.isoformat()])
            qs_end = st.query_params.get("end", [max_d.isoformat()])
            try:
                qs_start_date = dt.date.fromisoformat(qs_start[0] if isinstance(qs_start, list) else str(qs_start))
                qs_end_date = dt.date.fromisoformat(qs_end[0] if isinstance(qs_end, list) else str(qs_end))
                default_rng = [qs_start_date, qs_end_date]
            except Exception:
                default_rng = [min_d, max_d]
            date_rng = st.date_input("Date range", default_rng)
            # sync to URL
            if isinstance(date_rng, list) and len(date_rng) == 2:
                st.query_params["start"] = _date_to_str(date_rng[0])
                st.query_params["end"] = _date_to_str(date_rng[1])
        else:
            date_rng = None

        # Apply filters
        mask = pd.Series(True, index=df_work.index)
        for col, sel in active_cats.items():
            aligned = df[col].astype(str)
            aligned = aligned.reindex(df_work.index, method='nearest') if not aligned.index.equals(df_work.index) else aligned
            mask &= aligned.isin(sel)
        if date_rng and len(date_rng) == 2 and isinstance(df_work.index, pd.DatetimeIndex):
            mask &= (df_work.index.date >= date_rng[0]) & (df_work.index.date <= date_rng[1])
        df_filt = df_work.loc[mask]

        # ---------- KPI row ----------
        k1, k2, k3 = st.columns(3)
        with k1: kpi_card("Rows (filtered)", f"{len(df_filt):,}")
        with k2:
            m = df_filt.mean(numeric_only=True).mean()
            kpi_card("Overall mean (cols avg)", f"{m:.3f}")
        with k3:
            if len(df_filt) >= 60:
                a = df_filt.tail(30).mean(numeric_only=True).mean()
                b = df_filt.tail(60).head(30).mean(numeric_only=True).mean()
                delta = (a - b) / (b if abs(b) > 1e-12 else 1.0)
                kpi_card("Last 30d vs prev 30d", f"{a:.3f}", delta=delta)
            else:
                kpi_card("Last 30d vs prev 30d", "—")

        st.divider()

        # ---------- Tabs ----------
        tab_overview, tab_corr, tab_granger, tab_pc, tab_dbn, tab_data = st.tabs([
            "Overview", "Correlation", "Granger Causality", "PC DAG", "DBN", "Data"
        ])

        # ===== Overview =====
        with tab_overview:
            st.subheader("Overview")
            cols_num = df_filt.columns.tolist()
            if len(cols_num) >= 2:
                xcol = st.selectbox("X", options=cols_num, index=0, key="ovx")
                ycol = st.selectbox("Y", options=[c for c in cols_num if c != xcol], index=0, key="ovy")
                fig_sc = fig_scatter(df_filt, xcol, ycol, title=f"{ycol} vs {xcol}", hover=df_filt.columns)
                st.plotly_chart(fig_sc, use_container_width=True)
                download_plotly_png(fig_sc, "Download scatter (PNG)", "overview_scatter.png")
            show_ts = st.toggle("Show time series of selected columns", value=True)
            if show_ts:
                pick = st.multiselect("Columns for time series", options=df_filt.columns, default=df_filt.columns[:3])
                if pick:
                    fig_ts = fig_timeseries(df_filt, pick, title="Time Series")
                    st.plotly_chart(fig_ts, use_container_width=True)
                    download_plotly_png(fig_ts, "Download time series (PNG)", "timeseries.png")

        # ===== Correlation =====
        with tab_corr:
            st.subheader("Correlation Analysis")
            corr = compute_corr(df_filt)
            fig_corr = fig_corr_heatmap(corr)
            st.plotly_chart(fig_corr, use_container_width=True)
            download_plotly_png(fig_corr, "Download correlation heatmap (PNG)", "correlation_heatmap.png")

            # strong pairs table
            strong = (corr.where(~np.eye(len(corr), dtype=bool)).stack().reset_index())
            strong.columns = ["var1", "var2", "corr"]
            strong = strong[strong["corr"].abs() >= corr_threshold].sort_values("corr", ascending=False)
            if strong.empty:
                st.info("No correlations above threshold.")
            else:
                st.dataframe(strong, use_container_width=True, height=360)
                df_download_button(strong, "Download correlations CSV", "correlations.csv")

        # ===== Granger =====
        stationary_df, diff_info = compute_adf_stationary(df_filt, alpha=alpha)
        with tab_granger:
            st.subheader("Granger Causality")
            if stationary_df.empty:
                st.error("Could not form stationary series for Granger.")
            else:
                k = auto_select_lag(stationary_df) if lag_mode == "Auto (AIC)" else granger_max_lag
                st.caption(f"Using lag = **{k}**")
                gmat = granger_matrix(stationary_df, max_lag=k, alpha=alpha, fdr=True)
                if gmat.empty or not gmat["significant"].any():
                    st.info("No significant Granger causal relationships (FDR controlled).")
                else:
                    st.success("Significant relationships:")
                    st.dataframe(gmat[gmat["significant"]], use_container_width=True, height=320)
                    df_download_button(gmat, "Download Granger results CSV", "granger_results.csv")
                if diff_info:
                    st.caption("Differencing (ADF): " + ", ".join([f"{k}: d={v}" for k, v in diff_info.items()]))

        # ===== PC DAG =====
        with tab_pc:
            st.subheader("PC Algorithm — Static Causal Graph")
            if stationary_df.empty:
                st.error("PC requires stationary numeric data.")
            else:
                dag, stats = pc_estimate(stationary_df, alpha=alpha, ci_test=ci_test)
                fig_pc = fig_dag_plotly(dag, "Estimated DAG (PC)")
                st.plotly_chart(fig_pc, use_container_width=True)
                download_plotly_png(fig_pc, "Download PC DAG (PNG)", "pc_dag.png")
                st.caption(f"Nodes: **{stats['num_nodes']}** | Edges: **{stats['num_edges']}**")
                edges_df = pd.DataFrame(stats["edges"], columns=["u", "v"])
                with st.expander("Edges"):
                    st.dataframe(edges_df, use_container_width=True, height=240)
                df_download_button(edges_df, "Download PC edges CSV", "pc_edges.csv")

        # ===== DBN =====
        with tab_dbn:
            st.subheader("Dynamic Bayesian Network (via lagged features + PC)")
            try:
                dbn, dbn_stats = dbn_estimate(df_filt, max_lag=dbn_max_lag, alpha=alpha, ci_test=ci_test)
                fig_dbn = fig_dag_plotly(dbn, f"DBN (max lag {dbn_max_lag})")
                st.plotly_chart(fig_dbn, use_container_width=True)
                download_plotly_png(fig_dbn, "Download DBN (PNG)", "dbn.png")
                st.caption(f"Nodes: **{dbn_stats['num_nodes']}** | Edges: **{dbn_stats['num_edges']}**")
                dbn_edges = pd.DataFrame(dbn_stats["edges"], columns=["u","v"])
                with st.expander("Edges"):
                    st.dataframe(dbn_edges, use_container_width=True, height=240)
                df_download_button(dbn_edges, "Download DBN edges CSV", "dbn_edges.csv")
            except Exception as e:
                st.error(f"DBN error: {e}")

        # ===== Data (AgGrid) =====
        with tab_data:
            st.subheader("Data (filtered) — AgGrid")
            data_view = df_filt.reset_index()
            gob = GridOptionsBuilder.from_dataframe(data_view)
            # Enable pivoting, grouping, sidebar, and aggregates
            gob.configure_side_bar()
            gob.configure_default_column(
                resizable=True, sortable=True, filter=True, enablePivot=True,
                enableRowGroup=True, enableValue=True, editable=False
            )
            gob.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
            # Some sensible aggregations
            for c in data_view.columns:
                if pd.api.types.is_numeric_dtype(data_view[c]):
                    gob.configure_column(c, aggFunc=["sum","avg","min","max"])
            grid_options = gob.build()

            AgGrid(
                data_view,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.NO_UPDATE,
                allow_unsafe_jscode=False,
                theme="alpine",  # alpine, balham, material
                height=480,
                fit_columns_on_grid_load=True
            )

            # Also offer CSV download of current filtered frame
            df_download_button(data_view, "Download filtered data CSV", "data_filtered.csv")

    except Exception as e:
        st.error(f"Error: {e}")

elif uploaded_file and not run_button:
    st.info("Click **Run** in the sidebar to start.")
else:
    st.info("Upload a CSV to begin.")

