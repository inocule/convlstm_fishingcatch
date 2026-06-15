"""
strmlt_ConvLSTM.py  –  West Philippine Sea Fishing Ground Prediction System
Streamlit dashboard for the ConvLSTM spatiotemporal deep learning pipeline.

Aligned to:
  01_data_loading.ipynb   – AIS 2019-2024, flat CSV folder, 72 months
  02_preprocessing.ipynb  – 7 channels (sst,ssh,vo,uo,chl,nppv,fishing_effort)
                            LAT 10-20°N · LON 114-120°E · 0.25° grid 41×25
  03_buildntrain.ipynb – input_shape (3,41,25,7), 70/15/15 split
                            saves: convlstm_weights.npz, X_test.npy, y_test.npy,
                                   training_history.json, data_summary.json
  04_evaluation.ipynb     – RMSE,MAE,F1_best,F1_config,SSI,Wasserstein; values stored as strings
                            saves: predictions.npy, evaluation_results.csv
  05_visualization.ipynb  – extent LAT 7-20, fishing cmap, Reds for MAE/RMSE
                            saves: obs_vs_pred.png, error_maps.png,
                                   rmse_over_time.png, training_history.png

Run in Colab
────────────
!pip install streamlit -q
!npm install -g localtunnel
# upload or %%writefile this file, then:
!streamlit run Full_ConvLSTM.py --server.enableCORS=false --server.enableXsrfProtection=false &>/content/logs.txt &
!lt --port 8501
# paste the IP printed below as the tunnel password
import urllib; print(urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode().strip())
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# THEME  (matches the mockup: #111827 bg, slate text)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#111827!important;color:#94a3b8!important}
[data-testid="stSidebar"]{background:#0f172a!important;border-right:0.5px solid #1e293b!important}
[data-testid="stSidebar"] *{color:#94a3b8!important}
h1{color:#cbd5e1!important;font-size:1.2rem!important;font-weight:500!important}
h2{color:#94a3b8!important;font-size:1rem!important;font-weight:500!important;margin-top:1.2rem!important}
h3{color:#64748b!important;font-size:.88rem!important;font-weight:500!important}
[data-testid="stMetric"]{background:#1e293b!important;border:0.5px solid #334155!important;
  border-radius:6px!important;padding:10px 14px!important}
[data-testid="stMetricLabel"]{color:#475569!important;font-size:.72rem!important}
[data-testid="stMetricValue"]{color:#94a3b8!important;font-size:1.35rem!important}
[data-testid="stTabs"] button{color:#475569!important;font-size:.82rem!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#94a3b8!important;
  border-bottom:2px solid #475569!important}
.stInfo{background:rgba(71,85,105,.1)!important;border-left:3px solid #334155!important}
.stWarning{background:rgba(120,90,30,.08)!important;border-left:3px solid #7a5a20!important}
.stSuccess{background:rgba(30,80,50,.08)!important;border-left:3px solid #1e5032!important}
hr{border-color:#1e293b!important}
.stCaption{color:#475569!important;font-size:.72rem!important}
[data-testid="stSelectbox"]>div>div{background:#1e293b!important;
  border:0.5px solid #334155!important;color:#94a3b8!important}
/* Upload tab dropzone styling */
[data-testid="stFileUploader"]{background:#0f172a!important;
  border:1.5px dashed #334155!important;border-radius:10px!important;
  padding:10px!important;transition:border-color .2s!important}
[data-testid="stFileUploader"]:hover{border-color:#475569!important}
[data-testid="stFileUploader"] label{color:#64748b!important;font-size:.82rem!important}
[data-testid="stFileUploader"] small{color:#334155!important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  – taken directly from the notebooks
# ─────────────────────────────────────────────────────────────────────────────
# 02_preprocessing: bounding box used for data selection
LAT_MIN_DATA, LAT_MAX_DATA = 10, 20
LON_MIN,      LON_MAX      = 114, 120

# 05_visualization: extent used in imshow (slightly wider latitude)
LAT_MIN_VIZ, LAT_MAX_VIZ = 7, 20

DATE_START = "2019-01-01"
DATE_END   = "2024-12-31"

# 7 channels: sst ssh vo uo chl nppv fishing_effort  (03_model_training cell[4])
CHANNELS = ["sst","ssh","vo","uo","chl","nppv","fishing_effort"]
CHANNEL_LABELS = {
    "sst":            "SST – sea surface temperature",
    "ssh":            "SSH – sea surface height",
    "vo":             "VO – northward velocity",
    "uo":             "UO – eastward velocity",
    "chl":            "Chlorophyll-a",
    "nppv":           "NPPV – net primary production",
    "fishing_effort": "Fishing effort (AIS, log-norm)",
}

SEQ_LEN = 3   # 03_model_training
N_LAT   = 41  # 02_preprocessing target grid (BGC native)
N_LON   = 25

# 05_visualization custom colormap
CMAP_FISHING = LinearSegmentedColormap.from_list(
    "fishing", ["#000033","#0000FF","#FF00FF","#FF0000"]
)

# Global test labels (will be resolved dynamically once data is loaded)
TEST_LABELS = [str(p) for p in pd.period_range("2023-11-01", "2024-12-31", freq="M")]



# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE  – dark, minimal, research-grade
# ─────────────────────────────────────────────────────────────────────────────
MPL_RC = {
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#0f172a",
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   "#64748b",
    "axes.titlecolor":   "#94a3b8",
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "grid.color":        "#1e293b",
    "grid.linewidth":    0.4,
    "text.color":        "#94a3b8",
    "font.family":       "monospace",
    "font.size":         8,
    "legend.facecolor":  "#0f172a",
    "legend.edgecolor":  "#1e293b",
    "legend.fontsize":   7,
}

def sfig(w=10, h=4, nrows=1, ncols=1, **kw):
    """Create a styled dark figure."""
    plt.rcParams.update(MPL_RC)
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    fig.patch.set_facecolor("#111827")
    axes = np.array(ax).ravel() if not isinstance(ax, plt.Axes) else [ax]
    for a in axes:
        a.set_facecolor("#0f172a")
        for sp in a.spines.values():
            sp.set_color("#1e293b")
            sp.set_linewidth(0.5)
        a.tick_params(colors="#475569", labelsize=7)
    return fig, ax

def add_cb(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="#475569", labelsize=6)
    cb.ax.set_ylabel(label, color="#64748b", fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#475569")
    cb.outline.set_edgecolor("#1e293b")
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_outputs(data_dir):
    out = {}
    for name in ["predictions.npy", "y_test.npy", "X_test.npy"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            out[name.replace(".npy", "")] = np.load(p)

    p = os.path.join(data_dir, "evaluation_results.csv")
    if os.path.exists(p):
        out["eval_df"] = pd.read_csv(p)

    p = os.path.join(data_dir, "training_history.json")
    if os.path.exists(p):
        with open(p) as fh:
            out["history"] = json.load(fh)

    p = os.path.join(data_dir, "data_summary.json")
    if os.path.exists(p):
        with open(p) as fh:
            out["data_summary"] = json.load(fh)

    for fname in ["preprocessed_features.nc", "ais_fishing_effort_gridded.nc"]:
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            out[fname] = p

    return out

@st.cache_resource(show_spinner=False)
def load_model(data_dir):
    """Load NumPy weights (.npz) saved by 03_buildntrain_numpy.ipynb.
    Returns (weights_dict, filename) or (None, error_message)."""
    for name in ["convlstm_weights.npz"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            try:
                return np.load(p), name
            except Exception as e:
                return None, str(e)
    return None, "convlstm_weights.npz not found"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def squeeze2d(arr):
    """(N,41,25,1) or (N,41,25) → (N,41,25)."""
    if arr is None:
        return None
    return arr[:,:,:,0] if arr.ndim == 4 else arr

def get_metric(eval_df, name):
    """Return float from evaluation_results.csv (values stored as strings)."""
    if eval_df is None:
        return None
    row = eval_df[eval_df["Metric"].str.upper() == name.upper()]
    if len(row) == 0:
        return None
    try:
        return float(row["Value"].values[0])
    except (ValueError, TypeError):
        return None

def fmt(v, spec=".4f"):
    return f"{v:{spec}}" if v is not None else "--"

def tlabel(i):
    return TEST_LABELS[i] if i < len(TEST_LABELS) else f"Sample {i+1}"

def ema(series, alpha=0.55):
    out = []
    for i, v in enumerate(series):
        out.append(v if i == 0 else alpha*v + (1-alpha)*out[-1])
    return out

def monthly_rmse_list(y2d, p2d):
    return [
        float(np.sqrt(np.mean(
            (np.nan_to_num(y2d[i].flatten(), nan=0.0) - np.nan_to_num(p2d[i].flatten(), nan=0.0)) ** 2
        )))
        for i in range(len(y2d))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - PART 1 (Drive Path input)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 WPS FishAI")
    
    # Default to the folder containing this script so local runs work out-of-the-box.
    # In Colab you can still paste your Drive path manually.
    _default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_output")
    data_dir = st.text_input(
        "Data path",
        value=_default_dir,
        help="Folder containing .npy / .json / .csv pipeline outputs",
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & DYNAMIC DATES RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading pipeline outputs…"):
    data = load_outputs(data_dir)

preds_raw  = data.get("predictions")
y_test_raw = data.get("y_test")
X_test     = data.get("X_test")
eval_df    = data.get("eval_df")
history    = data.get("history")

preds  = squeeze2d(preds_raw)
y_test = squeeze2d(y_test_raw)

has_preds   = preds is not None and y_test is not None
has_history = history is not None
n_test      = len(preds) if has_preds else 0

# Dynamic Date Resolution
fallback_start = "2019-01-01"
fallback_end   = "2024-12-31"

# 1. Try to load date range from data_summary.json
summary_path = os.path.join(data_dir, "data_summary.json")
if os.path.exists(summary_path):
    try:
        with open(summary_path) as f:
            summary = json.load(f)
            if "date_range" in summary:
                parts = summary["date_range"].split(" to ")
                if len(parts) == 2:
                    fallback_start = parts[0]
                    fallback_end = parts[1]
    except Exception:
        pass

# 2. Try to get exact times from preprocessed NetCDF if available
all_months = []
feat_path = data.get("preprocessed_features.nc")
if feat_path and os.path.exists(feat_path):
    try:
        import xarray as xr
        with xr.open_dataset(feat_path) as ds:
            all_months = [str(pd.Timestamp(t).to_period("M")) for t in ds.time.values]
            fallback_start = str(pd.Timestamp(ds.time.values[0]).strftime("%Y-%m-%d"))
            fallback_end = str(pd.Timestamp(ds.time.values[-1]).strftime("%Y-%m-%d"))
    except Exception:
        pass

# Generate monthly periods between fallback start/end if NetCDF wasn't read
if not all_months:
    try:
        periods = pd.period_range(fallback_start, fallback_end, freq="M")
        all_months = [str(p) for p in periods]
    except Exception:
        all_months = [f"Month {i+1}" for i in range(72)]

date_start_dt = pd.to_datetime(fallback_start)
date_end_dt   = pd.to_datetime(fallback_end)
start_year    = date_start_dt.year
end_year      = date_end_dt.year
total_months  = len(all_months)

seq_months = all_months[SEQ_LEN:]
if n_test > 0:
    TEST_LABELS = seq_months[-n_test:]
else:
    _n_seq = len(seq_months)
    _train_n = int(0.70 * _n_seq)
    _val_n = int(0.15 * _n_seq)
    TEST_LABELS = [str(p) for p in seq_months[_train_n + _val_n:]]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - PART 2 (Dependent on resolved dates)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"West Philippine Sea · {start_year}–{end_year}")
    st.divider()
    st.markdown("#### Visualization")
    threshold    = st.slider("Prediction threshold", 0.0, 1.0, 0.10, 0.05)
    show_contour = st.toggle("Threshold contour", value=True)
    show_grid    = st.toggle("Grid overlay",       value=True)

    st.divider()
    st.markdown("#### Model")
    specs = [
        ("Architecture",  "ConvLSTM2D"),
        ("Channels",      "7"),
        ("Seq. length",   "3 months"),
        ("Horizon",       "1 month"),
        ("Grid",          "41 × 25"),
        ("Resolution",    "0.25°"),
        ("Input shape",   "(3,41,25,7)"),
        ("Parameters",    "20,313"),
        ("Train/Val/Test","70/15/15"),
    ]
    for k, v in specs:
        c1, c2 = st.columns([1.2, 1])
        c1.caption(k); c2.caption(f"**{v}**")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## West Philippine Sea Fishing Ground Prediction System")
st.caption(
    "ConvLSTM2D · CMEMS (physics 0.083° + BGC 0.25°) + AIS (GFW v3.0) · "
    f"7 channels · {N_LAT}×{N_LON} grid · 0.25° · {start_year}–{end_year}"
)
st.divider()


tab_upload, tab_dash, tab_pred, tab_anal, tab_pipeline,  tab_about = st.tabs([
    "Data Upload","Dashboard", "Predictions", "Analytics", "Pipeline",  "About"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab_dash:

    # ── metrics (values come from evaluation_results.csv, stored as strings) ──
    m_rmse = get_metric(eval_df, "RMSE")
    m_mae  = get_metric(eval_df, "MAE")
    m_f1_best = get_metric(eval_df, "F1_best")
    m_f1_cfg  = get_metric(eval_df, "F1_config")
    m_ssi  = get_metric(eval_df, "SSI")
    m_wd   = get_metric(eval_df, "Wasserstein")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("RMSE",        fmt(m_rmse, ".6f"))
    c2.metric("MAE",         fmt(m_mae,  ".6f"))
    c3.metric("F1 (Best)",   fmt(m_f1_best, ".4f"))
    c4.metric("F1 (Config)", fmt(m_f1_cfg,  ".4f"))
    c5.metric("SSI",         fmt(m_ssi,  ".4f"))
    c6.metric("Wasserstein", fmt(m_wd,   ".4f"))

    st.divider()

    # ── large map + stat cards ──
    map_col, stat_col = st.columns([3, 1], gap="large")

    with map_col:
        st.markdown("#### Fishing probability map")
        st.caption(
            f"Latest test sample · {tlabel(n_test-1) if n_test else '--'} · "
            "target: AIS fishing effort (log-normalised)"
        )

        fig, ax = sfig(9, 4.5)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN_VIZ, LAT_MAX_VIZ)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

        if not has_preds:
            ax.set_title("No data loaded — run Phase 3 & 4 notebooks first")
            ax.text(
                (LON_MIN + LON_MAX) / 2,
                (LAT_MIN_VIZ + LAT_MAX_VIZ) / 2,
                "predictions.npy not found",
                ha="center", va="center", color="#334155", fontsize=10,
            )
            if show_grid:
                ax.grid(True, linewidth=0.3)
        else:
            latest = preds[-1]
            im = ax.imshow(
                latest,
                extent=[LON_MIN, LON_MAX, LAT_MIN_VIZ, LAT_MAX_VIZ],
                cmap=CMAP_FISHING, vmin=0, vmax=1,
                aspect="auto", origin="lower",
            )
            if show_contour:
                ax.contour(
                    latest, levels=[threshold],
                    colors=["#64748b"], linewidths=0.7,
                    extent=[LON_MIN, LON_MAX, LAT_MIN_VIZ, LAT_MAX_VIZ],
                )
            if show_grid:
                ax.grid(True, linewidth=0.3)
            ax.set_title(
                f"Predicted fishing effort · {tlabel(n_test-1)} · "
                f"threshold = {threshold}"
            )
            add_cb(fig, im, ax, "probability")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with stat_col:
        st.markdown("#### Statistics")

        if has_preds:
            latest_pred = preds[-1]
            latest_obs  = y_test[-1]
            peak        = float(np.nanmax(latest_pred))
            high_cells  = int((latest_pred > threshold).sum())
            mean_pred   = float(np.nanmean(latest_pred))
            mean_obs    = float(np.nanmean(latest_obs))
        else:
            peak = high_cells = mean_pred = mean_obs = None

        st.metric("Latest sample",   tlabel(n_test-1) if n_test else "--")
        st.metric("Test samples",    str(n_test) if n_test else "--")
        st.metric("Peak probability",
                  f"{peak:.3f}" if peak is not None else "--")
        st.metric(f"High cells (≥{threshold})",
                  str(high_cells) if high_cells is not None else "--")
        st.metric("Mean predicted",
                  f"{mean_pred:.3f}" if mean_pred is not None else "--")
        st.metric("Mean observed",
                  f"{mean_obs:.3f}" if mean_obs is not None else "--")

    st.divider()

    # ── RMSE over test months  (05_visualization cell[8]) ──
    st.markdown("#### RMSE over test months")
    st.caption("Per-sample RMSE on the held-out test set — matches 05_visualization.ipynb §3")

    if not has_preds:
        st.info("Load prediction data to render this chart.")
    else:
        rmse_list = monthly_rmse_list(y_test, preds)
        labels    = [tlabel(i) for i in range(n_test)]
        mean_r    = float(np.mean(rmse_list))

        fig, ax = sfig(12, 3)
        bar_c = ["#7a3a20" if v > mean_r else "#2a4a6a" for v in rmse_list]
        ax.bar(range(n_test), rmse_list, color=bar_c, alpha=0.8, width=0.65)
        ax.plot(range(n_test), rmse_list,
                color="#64748b", linewidth=1.2,
                marker="o", markersize=3.5, zorder=5)
        ax.axhline(mean_r, color="#6a5020", linewidth=0.8, linestyle="--",
                   alpha=0.7, label=f"mean = {mean_r:.4f}")
        ax.set_xticks(range(n_test))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Test sample (month)")
        ax.set_title("Prediction accuracy over time")
        ax.grid(axis="y", linewidth=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("### Monthly forecast browser")
    st.caption(
        "Browse any individual test-set sample. Observed = normalised AIS fishing "
        "effort (ground truth). Predicted = ConvLSTM2D sigmoid output."
    )

    if not has_preds:
        st.warning(
            "No prediction data found.  "
            "Run `04_evaluation.ipynb` (which calls `model.predict()` and saves "
            "`predictions.npy`, `y_test.npy`) then check the Drive path in the sidebar."
        )

    if has_preds:
      # ── controls ──
      ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
      with ctrl1:
        month_i = st.slider(
            "Test sample index", 0, n_test - 1, n_test - 1,
            format=f"sample %d  ({'{}'})".format(
                tlabel(st.session_state.get("_mi", n_test-1))
                if "_mi" in st.session_state else tlabel(n_test-1)
            ),
            key="_mi",
        )
      with ctrl2:
        vmax_pred = st.slider("Color scale max", 0.1, 1.0, 1.0, 0.05,
                              key="vmax_pred")
      with ctrl3:
        show_diff = st.toggle("Difference map", value=False)

      lbl = tlabel(month_i)
      st.markdown(f"#### {lbl}  —  sample {month_i + 1} of {n_test}")

      obs_map  = y_test[month_i]
      pred_map = preds[month_i]

      # ── maps (05_visualization layout) ──
      if show_diff:
          fig, axes = sfig(13, 4.5, ncols=3)
          panels = [
              (obs_map,              CMAP_FISHING, 0,     vmax_pred, "Observed AIS effort"),
              (pred_map,             CMAP_FISHING, 0,     vmax_pred, "Predicted probability"),
              (pred_map - obs_map,   "RdBu_r",    -0.3,   0.3,      "Difference (pred − obs)"),
          ]
      else:
          fig, axes = sfig(10, 4.5, ncols=2)
          panels = [
              (obs_map,  CMAP_FISHING, 0, vmax_pred, "Observed AIS effort"),
              (pred_map, CMAP_FISHING, 0, vmax_pred, "Predicted probability"),
          ]

      for ax, (arr, cmap, vmin, vmax_c, title) in zip(np.array(axes).ravel(), panels):
          im = ax.imshow(
              arr,
              extent=[LON_MIN, LON_MAX, LAT_MIN_VIZ, LAT_MAX_VIZ],
              cmap=cmap, vmin=vmin, vmax=vmax_c,
              aspect="auto", origin="lower",
          )
          if show_contour and cmap is CMAP_FISHING:
              ax.contour(arr, levels=[threshold], colors=["#64748b"],
                         linewidths=0.6,
                         extent=[LON_MIN, LON_MAX, LAT_MIN_VIZ, LAT_MAX_VIZ])
          if show_grid:
              ax.grid(True, linewidth=0.3)
          ax.set_xlabel("Longitude (°E)")
          ax.set_ylabel("Latitude (°N)")
          ax.set_title(f"{title} · {lbl}")
          add_cb(fig, im, ax)

      plt.tight_layout()
      st.pyplot(fig, use_container_width=True)
      plt.close(fig)

      st.caption(
          "Colormap: `fishing` — #000033 → #0000FF → #FF00FF → #FF0000  "
          "(matches 05_visualization.ipynb)"
      )

      # ── cell statistics ──
      st.divider()
      st.markdown("#### Cell statistics — " + lbl)
      s1, s2, s3, s4 = st.columns(4)
      s1.metric("Peak predicted",       f"{float(np.nanmax(pred_map)):.4f}")
      s2.metric(f"High cells (≥{threshold})", str(int((pred_map > threshold).sum())))
      s3.metric("Mean predicted",       f"{float(np.nanmean(pred_map)):.4f}")
      s4.metric("Mean observed",        f"{float(np.nanmean(obs_map)):.4f}")

      # ── spatial profiles ──
      st.divider()
      st.markdown("#### Spatial profiles")
      st.caption("Latitudinal and longitudinal mean of observed vs. predicted.")

      fig, (ax1, ax2) = sfig(11, 2.8, ncols=2)
      lat_vals = np.linspace(LAT_MIN_DATA, LAT_MAX_DATA, pred_map.shape[0])
      lon_vals = np.linspace(LON_MIN,      LON_MAX,      pred_map.shape[1])

      for ax, xvals, obs_agg, pred_agg, xlabel in [
          (ax1, lat_vals,
           np.nanmean(obs_map, axis=1), np.nanmean(pred_map, axis=1),
           "Latitude (°N)"),
          (ax2, lon_vals,
           np.nanmean(obs_map, axis=0), np.nanmean(pred_map, axis=0),
           "Longitude (°E)"),
      ]:
          ax.plot(xvals, obs_agg,  color="#475569", linewidth=1.2, label="Observed")
          ax.plot(xvals, pred_agg, color="#64748b", linewidth=1.2,
                  linestyle="--", label="Predicted")
          ax.set_xlabel(xlabel)
          ax.set_ylabel("Mean probability")
          ax.grid(True, linewidth=0.3)
          ax.legend()

      ax1.set_title("Latitudinal mean")
      ax2.set_title("Longitudinal mean")
      plt.tight_layout()
      st.pyplot(fig, use_container_width=True)
      plt.close(fig)

      # ── download ──
      st.divider()
      dl1, dl2 = st.columns(2)
      with dl1:
          st.download_button(
              "Download prediction array (.npy)",
              data=pred_map.astype(np.float32).tobytes(),
              file_name=f"pred_{lbl.replace(' ','_')}.npy",
              mime="application/octet-stream",
          )
      with dl2:
          if eval_df is not None:
              st.download_button(
                  "Download evaluation results (.csv)",
                  data=eval_df.to_csv(index=False).encode(),
                  file_name="evaluation_results.csv",
                  mime="text/csv",
              )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_anal:
    st.markdown("### Model evaluation & analytics")

    # ── evaluation table ──
    st.markdown("#### Evaluation results")
    st.caption(
        "From `evaluation_results.csv` (04_evaluation.ipynb). "
        "Metrics: RMSE, MAE computed globally; F1_best (optimal threshold sweep) and F1_config; "
        "SSI and Wasserstein per-sample then averaged."
    )
    if eval_df is not None:
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
    else:
        st.info("No `evaluation_results.csv` found. Run `04_evaluation.ipynb`.")

    st.divider()

    # ── observed vs predicted grid  (05_visualization cell[5]) ──
    st.markdown("#### Observed vs. predicted heatmaps")
    st.caption(
        "2-row grid: observed AIS (top) vs. predicted probability (bottom). "
        "Shared colorbar. Matches `obs_vs_pred.png` from 05_visualization.ipynb."
    )

    if not has_preds:
        st.warning("Prediction data not loaded.")
    else:
        n_show  = min(6, n_test)
        start_i = st.slider("Starting sample", 0, max(0, n_test - n_show), 0,
                            key="ovp_start")
        indices = list(range(start_i, start_i + n_show))

        fig, axes = sfig(16, 6, nrows=2, ncols=n_show)

        last_im = None
        for col, i in enumerate(indices):
            for row, (arr, row_lbl) in enumerate([
                (y_test[i],  "Observed"),
                (preds[i],   "Predicted"),
            ]):
                ax = axes[row, col]
                im = ax.imshow(
                    arr,
                    extent=[LON_MIN, LON_MAX, LAT_MIN_VIZ, LAT_MAX_VIZ],
                    cmap=CMAP_FISHING, vmin=0, vmax=1,
                    aspect="auto", origin="lower",
                )
                last_im = im
                ax.set_title(tlabel(i) if row == 0 else "", fontsize=7.5)
                if col == 0:
                    ax.set_ylabel(row_lbl + "\nLatitude", fontsize=7)
                ax.set_xlabel("Longitude" if row == 1 else "", fontsize=7)
                ax.tick_params(labelsize=6)
                if show_grid:
                    ax.grid(True, linewidth=0.25)

        # Horizontal colorbar below — matches 05_visualization layout
        fig.subplots_adjust(bottom=0.14, hspace=0.25, wspace=0.25)
        cbar_ax = fig.add_axes([0.15, 0.04, 0.70, 0.025])
        cb = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal",
                          label="Fishing probability")
        cb.ax.xaxis.set_tick_params(color="#475569", labelsize=6)
        cb.ax.set_xlabel("Fishing probability", color="#64748b", fontsize=7)
        cb.outline.set_edgecolor("#1e293b")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    # ── error maps  (05_visualization cell[6]) ──
    st.markdown("#### Spatial error maps")
    st.caption(
        "Errors averaged across all test samples. "
        "Colormaps: `RdBu_r` (mean error), `Reds` (MAE), `Reds` (RMSE). "
        "Matches `error_maps.png` from 05_visualization.ipynb."
    )

    if not has_preds:
        st.warning("Prediction data not loaded.")
    else:
        # error = preds - y_test  (05_visualization cell[6])
        # Both are (N,41,25); add channel dim back to match notebook
        err_arr = preds_raw - y_test_raw   # (N,41,25,1)

        mean_err  = err_arr.mean(axis=0)[:,:,0]
        mae_map   = np.abs(err_arr).mean(axis=0)[:,:,0]
        rmse_map  = np.sqrt((err_arr**2).mean(axis=0)[:,:,0])

        fig, axes = sfig(14, 4.2, ncols=3)
        panels = [
            (mean_err,  "RdBu_r", -0.3, 0.3, "Mean Error",   "Pred − Obs"),
            (mae_map,   "Reds",    0.0, 0.3, "MAE",          "|Pred − Obs|"),
            (rmse_map,  "Reds",    0.0, 0.3, "RMSE per cell","RMSE"),
        ]
        for ax, (arr, cmap, vmin, vmax, title, cblbl) in zip(axes, panels):
            im = ax.imshow(
                arr,
                extent=[LON_MIN, LON_MAX, LAT_MIN_VIZ, LAT_MAX_VIZ],
                cmap=cmap, vmin=vmin, vmax=vmax,
                aspect="auto", origin="lower",
            )
            ax.set_title(title)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            if show_grid:
                ax.grid(True, linewidth=0.25)
            add_cb(fig, im, ax, cblbl)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # summary row
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Max MAE cell",          f"{mae_map.max():.4f}")
        e2.metric("Mean MAE (all cells)",  f"{mae_map.mean():.4f}")
        e3.metric("Max RMSE cell",         f"{rmse_map.max():.4f}")
        e4.metric("Mean bias",             f"{mean_err.mean():.4f}")

    st.divider()

    # ── training history  (05_visualization cell[9]) ──
    st.markdown("#### Training history")
    st.caption(
        "Loss = binary cross-entropy · MAE tracked per epoch. "
        "Keys: `history['loss']`, `history['val_loss']`, `history['mae']`, "
        "`history['val_mae']`."
    )

    if not has_history:
        st.info("No `training_history.json` found. Run `03_model_training.ipynb`.")
    else:
        smooth_tog = st.toggle("Smooth curves (EMA α=0.55)", value=True,
                               key="smooth_train")

        loss_tr = history.get("loss", [])
        loss_vl = history.get("val_loss", [])
        mae_tr  = history.get("mae",     history.get("mean_absolute_error", []))
        mae_vl  = history.get("val_mae", history.get("val_mean_absolute_error", []))
        n_ep    = len(loss_tr)

        fig, (ax1, ax2) = sfig(12, 3.5, ncols=2)

        def plot_curve(ax, train, val, ylabel, title):
            tr = ema(train) if smooth_tog else train
            vl = ema(val)   if smooth_tog else val
            ep = list(range(1, len(tr)+1))
            ax.plot(ep,          tr, color="#2a4a6a", linewidth=1.4, label="Train")
            ax.plot(ep[:len(vl)],vl, color="#64748b", linewidth=1.2,
                    linestyle="--", label="Val")
            ax.fill_between(ep, tr, alpha=0.06, color="#2a4a6a")
            if val:
                best = int(np.argmin(val))
                ax.axvline(best+1, color="#6a5020", linewidth=0.7,
                           linestyle=":", alpha=0.7,
                           label=f"best val ep {best+1}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, linewidth=0.3)
            ax.legend()

        plot_curve(ax1, loss_tr, loss_vl, "Binary cross-entropy", "Training Loss")
        plot_curve(ax2, mae_tr,  mae_vl,  "MAE",                  "Training MAE")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total epochs",    str(n_ep))
        t2.metric("Best val loss",   f"{min(loss_vl):.6f}" if loss_vl else "--")
        t3.metric("Final train MAE", f"{mae_tr[-1]:.6f}"   if mae_tr  else "--")
        t4.metric("Final val MAE",   f"{mae_vl[-1]:.6f}"   if mae_vl  else "--")

    st.divider()

    # ── variable explorer  (preprocessed_features.nc) ──
    st.markdown("#### Variable explorer")
    st.caption(
        "Inspect any of the 7 preprocessed channels from `preprocessed_features.nc` "
        "(chl, nppv, ssh, sst, uo, vo, fishing_effort). "
        "All min-max normalised to [0, 1]."
    )

    feat_path = data.get("preprocessed_features.nc")
    if feat_path is None:
        st.info("No `preprocessed_features.nc` found. Run `02_preprocessing.ipynb`.")
    else:
        try:
            import xarray as xr
            feats    = xr.open_dataset(feat_path)
            var_list = list(feats.data_vars)   # 7 vars
            n_times  = len(feats.time)

            vc1, vc2 = st.columns(2)
            with vc1:
                sel_var = st.selectbox(
                    "Variable", var_list,
                    format_func=lambda v: CHANNEL_LABELS.get(v, v),
                    key="varexp_var",
                )
            with vc2:
                time_i = st.slider(f"Month index (0 = {all_months[0]})",
                                   0, n_times - 1, 0, key="varexp_t")

            arr_var = feats[sel_var].isel(time=time_i).values
            month_str = str(pd.Timestamp(feats.time.values[time_i]).to_period("M"))

            fig, ax = sfig(9, 4)
            im = ax.imshow(
                arr_var,
                extent=[LON_MIN, LON_MAX, LAT_MIN_DATA, LAT_MAX_DATA],
                cmap="viridis", vmin=0, vmax=1,
                aspect="auto", origin="lower",
            )
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(
                f"{CHANNEL_LABELS.get(sel_var, sel_var)} · {month_str} "
                f"(normalised [0,1])"
            )
            if show_grid:
                ax.grid(True, linewidth=0.3)
            add_cb(fig, im, ax, "normalised value")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            feats.close()

        except ImportError:
            st.warning("`xarray` or `netCDF4` not installed.  "
                       "Run `!pip install xarray netCDF4` in Colab.")
        except Exception as e:
            st.error(f"Could not load features: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PIPELINE STATUS
# ═════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("### Pipeline Status")
    st.caption("Step-by-step checklist of all notebook output files. Green = present in Drive path.")

    # ── data_summary.json contents ──
    data_summary = data.get("data_summary")
    if data_summary:
        st.markdown("#### Dataset metadata (`data_summary.json`)")
        ds_col1, ds_col2, ds_col3, ds_col4 = st.columns(4)
        ds_col1.metric("Months",    str(data_summary.get("n_months", "--")))
        ds_col2.metric("Grid",
                       f"{data_summary.get('n_lat','?')}×{data_summary.get('n_lon','?')}")
        ds_col3.metric("Channels",  str(data_summary.get("n_channels", "--")))
        ds_col4.metric("Seq length",str(data_summary.get("seq_len", "--")))
        ds_col5, ds_col6, ds_col7, ds_col8 = st.columns(4)
        ds_col5.metric("Train sequences", str(data_summary.get("n_train", "--")))
        ds_col6.metric("Val sequences",   str(data_summary.get("n_val",   "--")))
        ds_col7.metric("Test sequences",  str(data_summary.get("n_test",  "--")))
        ds_col8.metric("Best epoch",      str(data_summary.get("best_epoch", "--")))
        st.divider()
    else:
        st.info("No `data_summary.json` found. Run `03_model_training.ipynb` to generate it.")
        st.divider()

    # ── per-phase output checklist ──
    phases = [
        ("Phase 1 — Data Loading (`01_data_loading.ipynb`)", [
            ("physics_raw_region.nc",   "CMEMS physics regional cache (0.083° NetCDF)"),
            ("bgc_raw_region.nc",       "CMEMS BGC regional cache (0.25° NetCDF)"),
            ("ais_raw_region.parquet",  "AIS fishing effort regional cache (Parquet)"),
        ]),
        ("Phase 2 — Preprocessing (`02_preprocessing.ipynb`)", [
            ("preprocessed_features.nc",      "7-channel normalised feature cube"),
            ("ais_fishing_effort_gridded.nc",  "AIS aggregated to 0.25° model grid"),
        ]),
        ("Phase 3 — Model Training (`03_buildntrain_numpy.ipynb`)", [
            ("convlstm_weights.npz",  "Trained NumPy ConvLSTM weights (best val_loss checkpoint)"),
            ("X_test.npy",            "Test input sequences (N, 3, 41, 25, 7) channel-last"),
            ("y_test.npy",            "Test target maps (N, 41, 25, 1)"),
            ("training_history.json", "Loss & MAE per epoch"),
            ("data_summary.json",     "Dataset metadata for dashboard"),
        ]),
        ("Phase 4 — Evaluation (`04_evaluation.ipynb`)", [
            ("predictions.npy",        "Model predictions on test set (N, 41, 25, 1)"),
            ("evaluation_results.csv", "RMSE, MAE, F1_best, F1_config, SSI, Wasserstein metrics"),
        ]),
        ("Phase 5 — Visualization (`05_visualization.ipynb`)", [
            ("obs_vs_pred.png",     "Observed vs. predicted heatmap grid"),
            ("error_maps.png",      "Spatial mean error / MAE / RMSE maps"),
            ("rmse_over_time.png",  "Per-sample RMSE bar chart"),
            ("training_history.png","Training loss & MAE curves"),
        ]),
    ]

    for phase_title, file_list in phases:
        st.markdown(f"#### {phase_title}")
        rows = []
        for fname, desc in file_list:
            fpath  = os.path.join(data_dir, fname)
            exists = os.path.exists(fpath)
            size_str = ""
            if exists:
                try:
                    size_kb = os.path.getsize(fpath) / 1024
                    size_str = (f"{size_kb/1024:.1f} MB" if size_kb > 1024
                                else f"{size_kb:.0f} KB")
                except Exception:
                    pass
            rows.append({
                "": "✅" if exists else "⬜",
                "File": fname,
                "Description": desc,
                "Size": size_str,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    # ── overall readiness ──
    all_critical = [
        "preprocessed_features.nc",
        "convlstm_weights.npz",
        "X_test.npy", "y_test.npy",
        "predictions.npy", "evaluation_results.csv",
    ]
    missing = [fn for fn in all_critical
               if not os.path.exists(os.path.join(data_dir, fn))]
    if not missing:
        st.success("All critical pipeline files present — dashboard is fully operational.")
    else:
        st.warning(
            f"Missing {len(missing)} critical file(s): {', '.join(missing)}  \n"
            "Run the corresponding notebook(s) to generate them."
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Data Upload")
    st.caption(
        "Drag-and-drop your raw data files here. AIS CSVs will be saved to the "
        "`ais_raw/` sub-folder; CMEMS NetCDF files to `cmems_raw/`. "
        "You can upload an entire folder's worth of files at once using the multi-file selector."
    )

    # ── destination config ──
    st.markdown("#### Destination folders")
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        ais_dest = st.text_input(
            "AIS output folder",
            value=os.path.join(data_dir, "ais_raw"),
            key="ais_dest",
            help="AIS CSV files will be saved here (created if absent)",
        )
    with up_col2:
        cmems_dest = st.text_input(
            "CMEMS output folder",
            value=os.path.join(data_dir, "cmems_raw"),
            key="cmems_dest",
            help="CMEMS NetCDF files will be saved here (created if absent)",
        )

    st.divider()

    # ══════════════════════════════════════════
    # AIS CSV UPLOADER
    # ══════════════════════════════════════════
    st.markdown("#### AIS Fishing Effort CSVs")
    st.caption(
        "Select all CSV files from your AIS folder "
        "(`fleet-monthly-csvs-10-v3-YYYY-MM-DD.csv`). "
        "You can select multiple files at once (Ctrl/Cmd+A to select all in the picker)."
    )

    ais_files = st.file_uploader(
        "Drop AIS CSV files here",
        type=["csv"],
        accept_multiple_files=True,
        key="ais_uploader",
        help="Select all monthly AIS CSV files — Ctrl+A selects all in the file dialog",
        label_visibility="collapsed",
    )

    if ais_files:
        # ── preview table ──
        ais_rows = []
        total_ais_bytes = 0
        for f in ais_files:
            sz = len(f.getvalue())
            total_ais_bytes += sz
            sz_str = f"{sz/1024:.1f} KB" if sz < 1024*1024 else f"{sz/1024/1024:.2f} MB"
            # Light validation: check header for expected AIS columns
            try:
                sample = pd.read_csv(f, nrows=1)
                f.seek(0)
                has_lat = any(c.lower() in ("lat", "latitude", "cell_ll_lat") for c in sample.columns)
                has_lon = any(c.lower() in ("lon", "longitude", "cell_ll_lon") for c in sample.columns)
                valid = "✅" if (has_lat and has_lon) else "⚠️"
            except Exception:
                f.seek(0)
                valid = "❓"
            ais_rows.append({"File": f.name, "Size": sz_str, "Valid": valid})

        total_ais_mb = total_ais_bytes / 1024 / 1024
        st.success(
            f"**{len(ais_files)} AIS file(s) staged** — "
            f"Total: {total_ais_mb:.2f} MB"
        )
        with st.expander(f"Preview {len(ais_files)} staged AIS files", expanded=len(ais_files) <= 20):
            st.dataframe(
                pd.DataFrame(ais_rows),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("💾 Save AIS files to disk", key="save_ais", type="primary"):
            os.makedirs(ais_dest, exist_ok=True)
            prog = st.progress(0, text="Saving AIS files…")
            saved, skipped, errors = [], [], []
            for i, f in enumerate(ais_files):
                out_path = os.path.join(ais_dest, f.name)
                try:
                    with open(out_path, "wb") as fh:
                        fh.write(f.getvalue())
                    saved.append(f.name)
                except Exception as e:
                    errors.append((f.name, str(e)))
                prog.progress((i + 1) / len(ais_files),
                              text=f"Saving {f.name} ({i+1}/{len(ais_files)})…")
            prog.empty()
            if saved:
                st.success(
                    f"✅ Saved **{len(saved)}** AIS file(s) to `{ais_dest}`"
                )
            if errors:
                for fname_e, err in errors:
                    st.error(f"❌ {fname_e}: {err}")
    else:
        st.info(
            "No AIS files staged yet. Click above to open the file picker, "
            "then navigate to your AIS folder and select all CSV files "
            "(use **Ctrl+A** or **Cmd+A** to select all)."
        )

    st.divider()

    # ══════════════════════════════════════════
    # CMEMS NetCDF UPLOADER
    # ══════════════════════════════════════════
    st.markdown("#### CMEMS NetCDF Files")
    st.caption(
        "Upload your CMEMS physics and/or BGC NetCDF files "
        "(`physics_raw_region.nc`, `bgc_raw_region.nc`, or any `.nc` / `.nc4` file). "
        "Large files may take a moment to transfer through the browser."
    )

    cmems_files = st.file_uploader(
        "Drop CMEMS NetCDF files here",
        type=["nc", "nc4", "netcdf"],
        accept_multiple_files=True,
        key="cmems_uploader",
        help="Physics and BGC NetCDF files from CMEMS downloads",
        label_visibility="collapsed",
    )

    if cmems_files:
        cmems_rows = []
        total_cmems_bytes = 0
        for f in cmems_files:
            sz = len(f.getvalue())
            total_cmems_bytes += sz
            sz_str = f"{sz/1024:.1f} KB" if sz < 1024*1024 else f"{sz/1024/1024:.2f} MB"
            # Light validation: try to open with xarray
            valid = "❓"
            dims_str = ""
            try:
                import xarray as xr, tempfile, shutil
                tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
                tmp.write(f.getvalue())
                tmp.close()
                f.seek(0)
                with xr.open_dataset(tmp.name) as ds_chk:
                    dims_str = " × ".join(f"{k}:{v}" for k, v in ds_chk.dims.items())
                    valid = "✅"
                os.unlink(tmp.name)
            except ImportError:
                valid = "⚠️ xarray missing"
            except Exception:
                f.seek(0)
                valid = "⚠️"
            cmems_rows.append({"File": f.name, "Size": sz_str, "Dimensions": dims_str, "Valid": valid})

        total_cmems_mb = total_cmems_bytes / 1024 / 1024
        st.success(
            f"**{len(cmems_files)} CMEMS file(s) staged** — "
            f"Total: {total_cmems_mb:.2f} MB"
        )
        with st.expander(f"Preview {len(cmems_files)} staged CMEMS files", expanded=True):
            st.dataframe(
                pd.DataFrame(cmems_rows),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("💾 Save CMEMS files to disk", key="save_cmems", type="primary"):
            os.makedirs(cmems_dest, exist_ok=True)
            prog = st.progress(0, text="Saving CMEMS files…")
            saved_c, errors_c = [], []
            for i, f in enumerate(cmems_files):
                out_path = os.path.join(cmems_dest, f.name)
                try:
                    with open(out_path, "wb") as fh:
                        fh.write(f.getvalue())
                    saved_c.append(f.name)
                except Exception as e:
                    errors_c.append((f.name, str(e)))
                prog.progress((i + 1) / len(cmems_files),
                              text=f"Saving {f.name} ({i+1}/{len(cmems_files)})…")
            prog.empty()
            if saved_c:
                st.success(
                    f"✅ Saved **{len(saved_c)}** CMEMS file(s) to `{cmems_dest}`"
                )
            if errors_c:
                for fname_e, err in errors_c:
                    st.error(f"❌ {fname_e}: {err}")
    else:
        st.info(
            "No CMEMS files staged yet. Click above to open the file picker "
            "and select your `.nc` / `.nc4` files."
        )

    st.divider()

    # ── folder scan (show what's already on disk) ──
    st.markdown("#### Current raw data on disk")
    st.caption("Files already present in the AIS and CMEMS destination folders.")

    scan_col1, scan_col2 = st.columns(2)

    def scan_folder(folder, exts):
        """Return a DataFrame of files in folder matching extensions."""
        rows = []
        if not os.path.isdir(folder):
            return None
        for fname in sorted(os.listdir(folder)):
            if any(fname.lower().endswith(e) for e in exts):
                fpath = os.path.join(folder, fname)
                try:
                    sz = os.path.getsize(fpath)
                    sz_str = (f"{sz/1024/1024:.2f} MB" if sz >= 1024*1024
                              else f"{sz/1024:.1f} KB")
                except Exception:
                    sz_str = "?"
                rows.append({"File": fname, "Size": sz_str})
        return pd.DataFrame(rows) if rows else None

    with scan_col1:
        st.markdown(f"**AIS raw folder** — `{ais_dest}`")
        ais_scan = scan_folder(ais_dest, [".csv"])
        if ais_scan is not None:
            st.caption(f"{len(ais_scan)} CSV file(s) found")
            st.dataframe(ais_scan, use_container_width=True, hide_index=True)
        elif os.path.isdir(ais_dest):
            st.info("Folder exists but contains no CSV files yet.")
        else:
            st.info("Folder does not exist yet — it will be created on first save.")

    with scan_col2:
        st.markdown(f"**CMEMS raw folder** — `{cmems_dest}`")
        cmems_scan = scan_folder(cmems_dest, [".nc", ".nc4", ".netcdf"])
        if cmems_scan is not None:
            st.caption(f"{len(cmems_scan)} NetCDF file(s) found")
            st.dataframe(cmems_scan, use_container_width=True, hide_index=True)
        elif os.path.isdir(cmems_dest):
            st.info("Folder exists but contains no NetCDF files yet.")
        else:
            st.info("Folder does not exist yet — it will be created on first save.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### About this system")

    # ── pipeline ──
    st.markdown("#### Pipeline")
    steps = [
        ("01 Data loading",
         "Mount Drive · load CMEMS physics (0.083°) and BGC (0.25°) NetCDF · "
         f"load AIS CSVs (flat folder, bbox-filtered) · {total_months} months {start_year}-{end_year}"),
        ("02 Preprocessing",
         "Select depth 0.49 m (physics) · depth-average 0–5 m (BGC) · "
         "resample monthly · regrid physics → 0.25° · linear gap fill · "
         "min-max normalise 6 ocean vars · log1p-normalise AIS → 7 channels"),
        ("03 Model training",
         "Stack 7 channels (SST,SSH,VO,UO,Chl,NPPV,fishing_effort) · "
         "create seq=3 → pred=1 sequences · 70/15/15 split · "
         "2-layer ConvLSTM (16→8) + 1×1 matmul sigmoid · focal loss · pure NumPy"),
        ("04 Evaluation",
         "Load `convlstm_weights.npz` · reconstruct NumPy model · predict on X_test (NaN→0) · "
         "compute RMSE, MAE, F1 threshold sweep, SSI, Wasserstein · save CSV"),
        ("05 Visualization",
         "Load predictions.npy + y_test.npy · obs-vs-pred grid · "
         "error maps (RdBu_r / Reds) · RMSE over time · training curves"),
    ]
    for title, body in steps:
        with st.expander(title):
            st.caption(body)

    st.divider()

    # ── architecture + output files side by side ──
    a_col, o_col = st.columns(2)

    with a_col:
        st.markdown("#### Model architecture")
        st.code("""# 03_buildntrain_numpy.ipynb  (pure NumPy — no torch/tf)
Input  (batch, SEQ_LEN=3, 41, 25, 7)  channel-last
  ConvLSTMLayer(7  → 16, return_sequences=True)   im2col + matmul
  BatchNorm(16)
  Dropout(0.2)
  ConvLSTMLayer(16 → 8,  return_sequences=False)
  BatchNorm(8)
  Dropout(0.2)
  1×1 matmul (8 → 1) + bias
  sigmoid
Output (batch, 41, 25, 1)   ∈ [0, 1]

optimizer=ManualAdam(lr=1e-3)
loss=focal_loss(alpha=1-pos_ratio, gamma=2.0)
grad_clip=1.0
epochs=50  batch_size=8  patience=10
weights saved as: convlstm_weights.npz""", language="text")

    with o_col:
        st.markdown("#### Output file status")
        st.caption("See the **Pipeline** tab for a detailed per-phase checklist with file sizes.")
        outputs = {
            "data_summary.json":            "Phase 1 — dataset metadata",
            "preprocessed_features.nc":     "Phase 2 — 7-channel normalised cube",
            "ais_fishing_effort_gridded.nc": "Phase 2 — AIS on 0.25° grid",
            "convlstm_weights.npz":         "Phase 3 — NumPy ConvLSTM weights (best val checkpoint)",
            "training_history.json":        "Phase 3 — loss & MAE per epoch",
            "X_test.npy":                   "Phase 3 — test inputs (N,3,41,25,7)",
            "y_test.npy":                   "Phase 3 — test labels (N,41,25,1)",
            "predictions.npy":              "Phase 4 — model predictions (N,41,25,1)",
            "evaluation_results.csv":       "Phase 4 — RMSE,MAE,F1_best,F1_config,SSI,Wasserstein",
        }
        rows = []
        for fname, desc in outputs.items():
            fpath  = os.path.join(data_dir, fname)
            status = "✅" if os.path.exists(fpath) else "⬜"
            rows.append({"": status, "File": fname, "Description": desc})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── data sources ──
    st.markdown("#### Data sources")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.info(
            "**CMEMS Physics** `GLOBAL_MULTIYEAR_PHY_001_030`  \n"
            "thetao (SST) · uo · vo · zos (SSH)  \n"
            "0.083° monthly · depth sel. 0.49 m  \n"
            f"2015–{end_year} (sliced to {start_year}–{end_year})"
        )
    with d2:
        st.info(
            "**CMEMS BGC** `GLOBAL_MULTIYEAR_BGC_001_029`  \n"
            "chl · nppv  \n"
            "0.25° monthly · depth avg 0–5 m  \n"
            f"2015–{end_year} (sliced to {start_year}–{end_year})"
        )
    with d3:
        st.info(
            "**AIS** Global Fishing Watch v3.0  \n"
            "fleet-monthly-csvs-10-v3-YYYY-MM-DD.csv  \n"
            "flat folder · bbox-filtered · log1p-normalised  \n"
            f"{total_months} months · {fallback_start[:7]} to {fallback_end[:7]}"
        )

    st.divider()
    st.caption(
        "West Philippine Sea Fishing Ground Prediction System · "
        f"ConvLSTM2D · CMEMS + AIS GFW v3.0 · {start_year}–{end_year} · Built with Streamlit"
    )
