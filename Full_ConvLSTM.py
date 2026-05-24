"""
Full_ConvLSTM.py
West Philippine Sea Fishing Ground Prediction System
Streamlit multi-tab dashboard for the ConvLSTM spatiotemporal deep learning pipeline.

Tabs
────
  Dashboard    – overview, large map, metric cards
  Predictions  – browse individual monthly forecasts
  Analytics    – full evaluation, error maps, training curves, variable explorer
  About        – pipeline diagram, architecture, data sources

Run in Colab
────────────
  !pip install streamlit -q
  !npm install -g localtunnel
  # upload or %%writefile this file, then:
  !streamlit run Full_ConvLSTM.py &>/content/logs.txt &
  !lt --port 8501
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WPS Fishing Ground Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#111827!important;color:#94a3b8!important}
[data-testid="stSidebar"]{background:#0f172a!important;border-right:0.5px solid #1e293b!important}
[data-testid="stSidebar"] *{color:#94a3b8!important}
h1{color:#cbd5e1!important;font-size:1.25rem!important;font-weight:500!important}
h2{color:#94a3b8!important;font-size:1rem!important;font-weight:500!important;margin-top:1.2rem!important}
h3{color:#64748b!important;font-size:.88rem!important;font-weight:500!important}
[data-testid="stMetric"]{background:#1e293b!important;border:0.5px solid #334155!important;
  border-radius:6px!important;padding:10px 14px!important}
[data-testid="stMetricLabel"]{color:#475569!important;font-size:.72rem!important}
[data-testid="stMetricValue"]{color:#94a3b8!important;font-size:1.4rem!important}
[data-testid="stMetricDelta"]{font-size:.72rem!important}
[data-testid="stTabs"] button{color:#475569!important;font-size:.82rem!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#94a3b8!important;
  border-bottom:2px solid #475569!important}
.stInfo{background:rgba(71,85,105,.12)!important;border-left:3px solid #475569!important}
.stWarning{background:rgba(180,130,60,.08)!important;border-left:3px solid #92622a!important}
.stSuccess{background:rgba(60,120,80,.08)!important;border-left:3px solid #2e6644!important}
hr{border-color:#1e293b!important}
.stCaption{color:#475569!important;font-size:.72rem!important}
[data-testid="stSelectbox"]>div>div{background:#1e293b!important;
  border:0.5px solid #334155!important;color:#94a3b8!important}
[data-testid="stDataFrame"]{border:0.5px solid #1e293b!important;border-radius:6px!important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 7,  20
LON_MIN, LON_MAX = 114, 120

CMAP_FISHING = LinearSegmentedColormap.from_list(
    "fishing", ["#000033", "#0000FF", "#FF00FF", "#FF0000"]
)

TEST_LABELS = [
    "Oct 2023","Nov 2023","Dec 2023","Jan 2024","Feb 2024","Mar 2024",
    "Apr 2024","May 2024","Jun 2024","Jul 2024","Aug 2024","Sep 2024",
    "Oct 2024","Nov 2024","Dec 2024","Jan 2025","Feb 2025","Mar 2025","Apr 2025",
]

VARIABLES = {
    "SST – sea surface temperature":         "sst",
    "SSH – sea surface height":              "ssh",
    "UO – eastward velocity":                "uo",
    "VO – northward velocity":               "vo",
    "Chlorophyll-a":                         "chl",
    "NPPV – net primary production":         "nppv",
}


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MPL_RC = {
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#0f172a",
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   "#64748b",
    "axes.titlecolor":   "#94a3b8",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "grid.color":        "#1e293b",
    "grid.linewidth":    0.4,
    "text.color":        "#94a3b8",
    "font.family":       "monospace",
    "font.size":         8,
}

def apply_rc():
    plt.rcParams.update(MPL_RC)

def styled_fig(w=10, h=4, nrows=1, ncols=1, **kw):
    apply_rc()
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    fig.patch.set_facecolor("#111827")
    axes = np.array(ax).ravel() if not isinstance(ax, plt.Axes) else [ax]
    for a in axes:
        a.set_facecolor("#0f172a")
        for sp in a.spines.values():
            sp.set_color("#1e293b")
            sp.set_linewidth(0.5)
        a.tick_params(colors="#475569", labelsize=7)
        a.xaxis.label.set_color("#64748b")
        a.yaxis.label.set_color("#64748b")
        a.title.set_color("#94a3b8")
    return fig, ax

def add_colorbar(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="#475569", labelsize=6)
    cb.ax.set_ylabel(label, color="#64748b", fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#475569")
    cb.outline.set_edgecolor("#1e293b")
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_arrays(data_dir):
    out = {}
    for name in ["predictions.npy","y_test.npy","X_test.npy"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            out[name.replace(".npy","")] = np.load(p)
    p = os.path.join(data_dir, "evaluation_results.csv")
    if os.path.exists(p):
        out["eval_df"] = pd.read_csv(p)
    p = os.path.join(data_dir, "training_history.json")
    if os.path.exists(p):
        with open(p) as f:
            out["history"] = json.load(f)
    p = os.path.join(data_dir, "preprocessed_features.nc")
    if os.path.exists(p):
        out["features_path"] = p
    return out

@st.cache_resource(show_spinner=False)
def load_model(data_dir):
    try:
        import tensorflow as tf
        for name in ["convlstm_model.h5","best_model.h5","convlstm_model.keras"]:
            p = os.path.join(data_dir, name)
            if os.path.exists(p):
                return tf.keras.models.load_model(p), name
    except Exception as e:
        return None, str(e)
    return None, "not found"


# ─────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_arr(d, key):
    a = d.get(key)
    if a is None:
        return None
    return a[:, :, :, 0] if a.ndim == 4 else a

def metric_val(eval_df, name, fmt="{:.4f}"):
    if eval_df is None:
        return "--"
    row = eval_df[eval_df["Metric"].str.upper() == name.upper()]
    if len(row) == 0:
        return "--"
    try:
        return fmt.format(float(row["Value"].values[0]))
    except (ValueError, TypeError):
        return str(row["Value"].values[0])

def test_label(i, n):
    return TEST_LABELS[i] if i < len(TEST_LABELS) else f"Sample {i+1}"

def ema(series, alpha=0.55):
    out = []
    for i, v in enumerate(series):
        out.append(v if i == 0 else alpha*v + (1-alpha)*out[-1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 WPS FishAI")
    st.caption("West Philippine Sea · 2015–2024")
    st.divider()

    data_dir = st.text_input(
        "Drive path",
        value="/content/drive/MyDrive/fishing_project/",
        help="Google Drive folder containing .npy / .json / .csv outputs",
    )

    st.divider()
    st.markdown("#### Visualization")
    threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.05)
    show_contour = st.toggle("Threshold contour", value=True)
    show_grid    = st.toggle("Grid overlay",       value=True)

    st.divider()
    st.markdown("#### Model")
    for k, v in [("Architecture","ConvLSTM2D"),("Seq. length","3 months"),
                 ("Horizon","1 month"),("Grid","41 × 25"),
                 ("Resolution","0.25°"),("Parameters","~272 K")]:
        c1, c2 = st.columns([1.1, 1])
        c1.caption(k); c2.caption(f"**{v}**")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading outputs…"):
    data     = load_arrays(data_dir)

preds   = get_arr(data, "predictions")
y_test  = get_arr(data, "y_test")
X_test  = data.get("X_test")
eval_df = data.get("eval_df")
history = data.get("history")
n_test  = len(preds) if preds is not None else 0

has_preds   = preds is not None and y_test is not None
has_history = history is not None


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## West Philippine Sea Fishing Ground Prediction System")
st.caption("ConvLSTM2D spatiotemporal deep learning · CMEMS + AIS · 0.25° · 41×25 grid")
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_dash, tab_pred, tab_anal, tab_about = st.tabs([
    "Dashboard", "Predictions", "Analytics", "About"
])


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # ── metric row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE",        metric_val(eval_df, "RMSE",        "{:.4f}"))
    m2.metric("MAE",         metric_val(eval_df, "MAE",         "{:.4f}"))
    m3.metric("F1 score",    metric_val(eval_df, "F1",          "{:.4f}"))
    m4.metric("SSI",         metric_val(eval_df, "SSI",         "{:.4f}"))
    m5.metric("Wasserstein", metric_val(eval_df, "Wasserstein", "{:.4f}"))

    st.divider()

    # ── map + stat cards ──
    map_col, stat_col = st.columns([3, 1], gap="large")

    with map_col:
        st.markdown("#### Fishing probability map")
        if not has_preds:
            st.info("Load `predictions.npy` and `y_test.npy` to render the map.")
            fig, ax = styled_fig(9, 4.5)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_xlim(LON_MIN, LON_MAX)
            ax.set_ylim(LAT_MIN, LAT_MAX)
            ax.set_title("Fishing probability — no data loaded")
            ax.grid(True, linewidth=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            latest = preds[-1]
            fig, ax = styled_fig(9, 4.5)
            im = ax.imshow(
                latest * 1.0,
                extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                cmap=CMAP_FISHING, vmin=0, vmax=1,
                aspect="auto", origin="lower",
            )
            if show_contour:
                ax.contour(latest, levels=[threshold],
                           colors=["#94a3b8"], linewidths=0.6,
                           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
            if show_grid:
                ax.grid(True, linewidth=0.3)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(f"Predicted fishing probability · {test_label(n_test-1, n_test)}")
            add_colorbar(fig, im, ax, "probability")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with stat_col:
        st.markdown("#### Key statistics")
        if has_preds:
            latest_pred = preds[-1]
            latest_obs  = y_test[-1]
            high_cells  = int((latest_pred > threshold).sum())
            peak        = float(np.nanmax(latest_pred))
            avg_pred    = float(np.nanmean(latest_pred))
        else:
            high_cells = peak = avg_pred = None

        st.metric("Latest month", test_label(n_test-1, n_test) if n_test else "--")
        st.metric("Peak probability",
                  f"{peak:.3f}" if peak is not None else "--")
        st.metric(f"High cells (≥ {threshold})",
                  str(high_cells) if high_cells is not None else "--")
        st.metric("Mean probability",
                  f"{avg_pred:.3f}" if avg_pred is not None else "--")
        st.metric("Test samples", str(n_test) if n_test else "--")

    st.divider()

    # ── RMSE over time (full width) ──
    st.markdown("#### RMSE over test months")
    if not has_preds:
        st.caption("Awaiting predictions.")
    else:
        from sklearn.metrics import mean_squared_error as _mse
        monthly_rmse = [
            float(np.sqrt(_mse(
                np.nan_to_num(y_test[i].flatten(), nan=0.0),
                np.nan_to_num(preds[i].flatten(), nan=0.0),
            )))
            for i in range(n_test)
        ]
        labels = [test_label(i, n_test) for i in range(n_test)]
        mean_r = float(np.mean(monthly_rmse))

        fig, ax = styled_fig(12, 2.8)
        bar_colors = [
            "#92622a" if v > mean_r else "#2e4a6a"
            for v in monthly_rmse
        ]
        ax.bar(range(n_test), monthly_rmse, color=bar_colors, alpha=0.8, width=0.65)
        ax.plot(range(n_test), monthly_rmse,
                color="#64748b", linewidth=1.2, marker="o", markersize=3, zorder=5)
        ax.axhline(mean_r, color="#5a4a2a", linewidth=0.8, linestyle="--", alpha=0.7,
                   label=f"mean {mean_r:.4f}")
        ax.set_xticks(range(n_test))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", linewidth=0.3)
        ax.legend(fontsize=7, facecolor="#0f172a", edgecolor="#1e293b",
                  labelcolor="#64748b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTIONS TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("### Monthly forecast browser")
    st.caption("Select any test month to inspect the observed vs. predicted fishing probability map.")

    if not has_preds:
        st.warning("No prediction data found. Run Phase 3 & 4 notebooks first, "
                   "then check the Drive path in the sidebar.")
        st.stop()

    # ── controls ──
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        month_i = st.slider("Test sample", 0, n_test - 1, n_test - 1,
                            format=f"sample %d of {n_test}")
    with ctrl2:
        vmax = st.slider("Color scale max", 0.1, 1.0, 1.0, 0.05)
    with ctrl3:
        show_diff = st.toggle("Show difference map", value=False)

    month_lbl = test_label(month_i, n_test)
    st.markdown(f"#### {month_lbl}")

    # ── side-by-side or diff ──
    obs_map  = y_test[month_i]
    pred_map = preds[month_i]

    if show_diff:
        fig, axes = styled_fig(10, 4, ncols=3)
        maps_data = [
            (obs_map,              CMAP_FISHING, 0, vmax, "Observed AIS"),
            (pred_map,             CMAP_FISHING, 0, vmax, "Predicted probability"),
            (pred_map - obs_map,   "RdBu_r",    -0.4, 0.4, "Difference (pred − obs)"),
        ]
    else:
        fig, axes = styled_fig(9, 4, ncols=2)
        maps_data = [
            (obs_map,  CMAP_FISHING, 0, vmax, "Observed AIS"),
            (pred_map, CMAP_FISHING, 0, vmax, "Predicted probability"),
        ]

    for ax, (arr, cmap, vmin, vmax_c, title) in zip(np.array(axes).ravel(), maps_data):
        im = ax.imshow(arr, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                       cmap=cmap, vmin=vmin, vmax=vmax_c,
                       aspect="auto", origin="lower")
        if show_contour and cmap is CMAP_FISHING:
            ax.contour(arr, levels=[threshold], colors=["#94a3b8"],
                       linewidths=0.6,
                       extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
        if show_grid:
            ax.grid(True, linewidth=0.3)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(title)
        add_colorbar(fig, im, ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── cell statistics ──
    st.divider()
    st.markdown("#### Cell-level statistics — " + month_lbl)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Peak predicted probability", f"{float(np.nanmax(pred_map)):.3f}")
    s2.metric(f"High cells (≥ {threshold})",
              str(int((pred_map > threshold).sum())))
    s3.metric("Mean predicted probability", f"{float(np.nanmean(pred_map)):.3f}")
    s4.metric("Mean observed AIS",          f"{float(np.nanmean(obs_map)):.3f}")

    # ── row profile charts ──
    st.divider()
    st.markdown("#### Spatial profile — latitudinal mean")
    fig, (ax1, ax2) = styled_fig(10, 2.5, ncols=2)
    lat_vals = np.linspace(LAT_MIN, LAT_MAX, pred_map.shape[0])
    ax1.plot(lat_vals, np.nanmean(obs_map, axis=1),
             color="#475569", linewidth=1.2, label="Observed")
    ax1.plot(lat_vals, np.nanmean(pred_map, axis=1),
             color="#64748b", linewidth=1.2, linestyle="--", label="Predicted")
    ax1.set_xlabel("Latitude (°N)")
    ax1.set_ylabel("Mean probability")
    ax1.set_title("Latitudinal mean")
    ax1.legend(fontsize=7, facecolor="#0f172a", edgecolor="#1e293b",
               labelcolor="#64748b")
    ax1.grid(True, linewidth=0.3)

    lon_vals = np.linspace(LON_MIN, LON_MAX, pred_map.shape[1])
    ax2.plot(lon_vals, np.nanmean(obs_map, axis=0),
             color="#475569", linewidth=1.2, label="Observed")
    ax2.plot(lon_vals, np.nanmean(pred_map, axis=0),
             color="#64748b", linewidth=1.2, linestyle="--", label="Predicted")
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Mean probability")
    ax2.set_title("Longitudinal mean")
    ax2.legend(fontsize=7, facecolor="#0f172a", edgecolor="#1e293b",
               labelcolor="#64748b")
    ax2.grid(True, linewidth=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── download ──
    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        dl1.download_button(
            "Download prediction array (.npy)",
            data=pred_map.astype(np.float32).tobytes(),
            file_name=f"pred_{month_lbl.replace(' ','_')}.npy",
            mime="application/octet-stream",
        )
    with dl2:
        if eval_df is not None:
            dl2.download_button(
                "Download evaluation results (.csv)",
                data=eval_df.to_csv(index=False).encode(),
                file_name="evaluation_results.csv",
                mime="text/csv",
            )


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_anal:
    st.markdown("### Model evaluation & analytics")

    # ── evaluation table ──
    st.markdown("#### Evaluation results")
    if eval_df is not None:
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
    else:
        st.info("No `evaluation_results.csv` found. Run Phase 4 notebook.")

    st.divider()

    # ── observed vs predicted grid ──
    st.markdown("#### Observed vs. predicted — 6 test samples")
    st.caption("Top row: observed AIS fishing effort. Bottom row: ConvLSTM2D predicted probability.")

    if not has_preds:
        st.warning("Prediction data not loaded.")
    else:
        n_show = min(6, n_test)
        start_i = st.slider("Starting sample", 0, max(0, n_test - n_show), 0,
                            key="ovp_start")
        indices = list(range(start_i, start_i + n_show))

        fig, axes = styled_fig(14, 5, nrows=2, ncols=n_show)
        for col, i in enumerate(indices):
            lbl = test_label(i, n_test)
            for row, (arr, row_lbl) in enumerate([
                (y_test[i],  "Observed"),
                (preds[i],   "Predicted"),
            ]):
                ax = axes[row, col]
                im = ax.imshow(arr,
                               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                               cmap=CMAP_FISHING, vmin=0, vmax=1,
                               aspect="auto", origin="lower")
                ax.set_title(lbl if row == 0 else "", fontsize=7)
                if col == 0:
                    ax.set_ylabel(row_lbl, fontsize=7)
                ax.set_xlabel("" if row == 0 else "Lon", fontsize=6)
                ax.tick_params(labelsize=6)
                if show_grid:
                    ax.grid(True, linewidth=0.25)

        fig.subplots_adjust(right=0.87, hspace=0.3, wspace=0.25)
        cax = fig.add_axes([0.89, 0.1, 0.012, 0.8])
        cb  = fig.colorbar(im, cax=cax)
        cb.ax.yaxis.set_tick_params(color="#475569", labelsize=6)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#475569")
        cb.outline.set_edgecolor("#1e293b")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    # ── error maps ──
    st.markdown("#### Spatial error analysis")
    st.caption("Errors averaged across all test samples.")

    if not has_preds:
        st.warning("Prediction data not loaded.")
    else:
        y_np = np.nan_to_num(y_test.astype(float), nan=0.0)
        p_np = np.nan_to_num(preds.astype(float),  nan=0.0)
        err  = p_np - y_np

        mean_err  = err.mean(axis=0)
        mae_map   = np.abs(err).mean(axis=0)
        rmse_map  = np.sqrt((err ** 2).mean(axis=0))

        fig, axes = styled_fig(13, 4, ncols=3)
        panels = [
            (mean_err,  "RdBu_r",  -0.3,  0.3, "Mean error (pred − obs)"),
            (mae_map,   "YlOrRd",   0.0,  0.3, "MAE per cell"),
            (rmse_map,  "plasma",   0.0,  0.3, "RMSE per cell"),
        ]
        for ax, (arr, cmap, vmin, vmax, title) in zip(axes, panels):
            im = ax.imshow(arr,
                           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="auto", origin="lower")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            if show_grid:
                ax.grid(True, linewidth=0.25)
            add_colorbar(fig, im, ax)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # error summary
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Max MAE cell",        f"{mae_map.max():.4f}")
        ec2.metric("Mean MAE (all cells)", f"{mae_map.mean():.4f}")
        ec3.metric("Max RMSE cell",       f"{rmse_map.max():.4f}")
        ec4.metric("Mean bias",           f"{mean_err.mean():.4f}")

    st.divider()

    # ── training history ──
    st.markdown("#### Training history")
    if not has_history:
        st.info("No `training_history.json` found. Run Phase 3 notebook.")
    else:
        smooth = st.toggle("Smooth curves (EMA)", value=True, key="smooth_train")

        loss_tr = history.get("loss", [])
        loss_vl = history.get("val_loss", [])
        mae_tr  = history.get("mae",      history.get("mean_absolute_error", []))
        mae_vl  = history.get("val_mae",  history.get("val_mean_absolute_error", []))
        epochs  = list(range(1, len(loss_tr) + 1))

        fig, (ax1, ax2) = styled_fig(12, 3.5, ncols=2)

        def plot_curve(ax, train, val, ylabel, title):
            tr = ema(train) if smooth else train
            vl = ema(val)   if smooth else val
            ep = list(range(1, len(tr)+1))
            ax.plot(ep, tr, color="#2e4a6a", linewidth=1.4, label="train")
            ax.plot(ep[:len(vl)], vl,
                    color="#64748b", linewidth=1.2, linestyle="--", label="val")
            ax.fill_between(ep, tr, alpha=0.06, color="#2e4a6a")
            if val:
                best = int(np.argmin(val))
                ax.axvline(best+1, color="#5a4a2a", linewidth=0.7,
                           linestyle=":", alpha=0.7,
                           label=f"best val ep {best+1}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, linewidth=0.3)
            ax.legend(fontsize=7, facecolor="#0f172a", edgecolor="#1e293b",
                      labelcolor="#64748b")

        plot_curve(ax1, loss_tr, loss_vl, "Binary cross-entropy", "Loss")
        plot_curve(ax2, mae_tr,  mae_vl,  "MAE",                  "MAE")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # training summary
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Total epochs",     str(len(epochs)))
        tc2.metric("Best val loss",    f"{min(loss_vl):.4f}" if loss_vl else "--")
        tc3.metric("Final train MAE",  f"{mae_tr[-1]:.4f}"   if mae_tr  else "--")
        tc4.metric("Final val MAE",    f"{mae_vl[-1]:.4f}"   if mae_vl  else "--")

    st.divider()

    # ── variable explorer ──
    st.markdown("#### Variable explorer")
    st.caption("Inspect any preprocessed oceanographic variable from `preprocessed_features.nc`.")

    features_path = data.get("features_path")
    if features_path is None:
        st.info("No `preprocessed_features.nc` found. Run Phase 2 notebook.")
    else:
        try:
            import xarray as xr
            feats = xr.open_dataset(features_path)
            var_names = list(feats.data_vars)

            vc1, vc2 = st.columns(2)
            with vc1:
                sel_var = st.selectbox("Variable", var_names, key="varexp_var")
            with vc2:
                n_times = len(feats.time)
                time_i  = st.slider("Time step (month index)",
                                    0, n_times - 1, 0, key="varexp_t")

            arr = feats[sel_var].isel(time=time_i).values
            fig, ax = styled_fig(8, 4)
            im = ax.imshow(arr,
                           extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                           cmap="viridis", aspect="auto", origin="lower")
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(f"{sel_var} · time index {time_i}")
            add_colorbar(fig, im, ax, sel_var)
            if show_grid:
                ax.grid(True, linewidth=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            feats.close()

        except ImportError:
            st.warning("Install xarray and netCDF4: `!pip install xarray netCDF4`")
        except Exception as e:
            st.error(f"Could not load features: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# ABOUT TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### About this system")

    # ── pipeline ──
    st.markdown("#### Pipeline")
    steps = [
        ("CMEMS ocean data",     "Physics (0.083°) + BGC (0.25°)\nSST, SSH, UO, VO, Chl-a, NPPV\n2015–2024"),
        ("AIS fishing effort",   "Global Fishing Watch v3.0\nMonthly 0.1° · 2015–2024\nglobalfishingwatch.org"),
        ("Preprocessing",        "Regrid to 0.25° · time align\nLinear gap fill · Phase 2"),
        ("Normalization",        "Min-max per variable\nOutput range [0, 1]\nPhase 2"),
        ("ConvLSTM2D",           "2-layer · 64+32 filters\nSeq=3 → Pred=1 month\n~272 K parameters"),
        ("Probability map",      "41 × 25 grid output\nSigmoid activation\nPhase 5 · visualization"),
    ]
    cols = st.columns(len(steps))
    for col, (title, body) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(body.replace("\n", "  \n"))

    st.divider()

    # ── architecture ──
    acol, ocol = st.columns(2)
    with acol:
        st.markdown("#### Model architecture")
        st.code("""Input  (batch, 3, 41, 25, 6)
  ConvLSTM2D  filters=64  kernel=3×3
  BatchNorm + Dropout(0.2)
  ConvLSTM2D  filters=32  kernel=3×3
  BatchNorm + Dropout(0.2)
  Conv2D      filters=1   kernel=1×1
              activation=sigmoid
Output (batch, 41, 25, 1)
       fishing probability ∈ [0, 1]

Optimizer : Adam
Loss      : binary_crossentropy
Epochs    : 50 (early stopping p=10)
Batch     : 8
Split     : 70 / 15 / 15 %""", language="text")

    with ocol:
        st.markdown("#### Output files")
        outputs = {
            "preprocessed_features.nc": "Normalised 6-channel feature cube",
            "ais_gridded.csv":           "AIS effort on 0.25° monthly grid",
            "best_model.h5":             "Best validation checkpoint",
            "convlstm_model.h5":         "Final trained model",
            "training_history.json":     "Epoch-wise loss & MAE",
            "X_test.npy":                "Test input (N, 3, 41, 25, 6)",
            "y_test.npy":                "Test labels (N, 41, 25, 1)",
            "predictions.npy":           "Model predictions (N, 41, 25, 1)",
            "evaluation_results.csv":    "RMSE, MAE, F1, SSI, Wasserstein",
        }
        rows = []
        for fname, desc in outputs.items():
            fpath  = os.path.join(data_dir, fname)
            status = "✅" if os.path.exists(fpath) else "⬜"
            rows.append({"Status": status, "File": fname, "Description": desc})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── data sources ──
    st.markdown("#### Data sources")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.info("**CMEMS Physics** `GLOBAL_MULTIYEAR_PHY_001_030`  \n"
                "thetao · uo · vo · zos  \n0.083° monthly · 5 depths")
    with d2:
        st.info("**CMEMS BGC** `GLOBAL_MULTIYEAR_BGC_001_029`  \n"
                "chl · nppv  \n0.25° monthly · 5 depths")
    with d3:
        st.info("**AIS Fishing Effort** Global Fishing Watch v3.0  \n"
                "Monthly 0.1° fleet CSV · 2015–2024")

    st.warning("Known issue: AIS loading uses `range(2020, 2025)`. "
               "Update to `range(2015, 2025)` in Phase 1 for the full dataset.")

    st.divider()
    st.caption("West Philippine Sea Fishing Ground Prediction System · "
               "ConvLSTM2D · CMEMS + AIS · 2015–2024 · Built with Streamlit")
