# ConvLSTM Fishing Ground Prediction
### West Philippine Sea · 2019–2024

A spatiotemporal deep learning pipeline that predicts monthly fishing ground probability maps for the **West Philippine Sea (10°–20°N, 114°–120°E)** using Convolutional LSTM (ConvLSTM2D). The model learns from **7 channels** (6 oceanographic variables + AIS fishing effort history) derived from two CMEMS reanalysis products and the Global Fishing Watch AIS dataset, producing a 41×25 grid (at 0.25° resolution) of fishing probability for the following month.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Required Datasets](#required-datasets)
5. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
   - [Phase 1: Data Loading](#phase-1-data-loading)
   - [Phase 2: Preprocessing](#phase-2-preprocessing)
   - [Phase 3: Model Training](#phase-3-model-training)
   - [Phase 4: Model Evaluation](#phase-4-model-evaluation)
   - [Phase 5: Visualization](#phase-5-visualization)
6. [Streamlit Dashboard](#streamlit-dashboard)
7. [Model Architecture](#model-architecture)
8. [Shared CONFIG Dictionary](#shared-config-dictionary)
9. [Output Files](#output-files)
10. [Known Issues & Notes](#known-issues--notes)

---

## Project Overview

```
  Oceanographic features (6 channels)    AIS Fishing Effort (label + channel)
  ┌────────────────────────────┐          ┌──────────────────────┐
  │  SST, SSH, UO, VO          │          │  fishing_hours       │
  │  Chlorophyll-a, NPP        │          │  log1p-normalised    │
  └──────────────┬─────────────┘          └──────────┬───────────┘
                 │   7 channels · 3 months of history │ target month
                 └──────────────┬────────────────────-┘
                           ConvLSTM2D
                                │
                    Predicted fishing probability
                       map for the next month
                        (41 × 25 grid, 0.25°)
```

**Key design choices:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Channels | **7** (sst, ssh, vo, uo, chl, nppv, fishing_effort) | AIS effort as an auto-regressive feature improves F1 |
| Sequence length | **3 months → predict 1 month ahead** | Captures seasonal lagging effects |
| Spatial grid | **0.25° · 41 lat × 25 lon** | BGC native resolution; physics bilinearly regridded |
| Model bbox | **10°–20°N, 114°–120°E** | WPS core fishing zone; narrow domain improves class balance |
| Staging bbox | **0°–30°N, 110°–140°E** | Wider cache in Phase 1; narrowed in Phase 2 onward |
| Training date range | **2019-01 – 2024-12 (72 months)** | Post-2019 AIS has denser coverage → better label quality |
| Train/Val/Test split | **70% / 15% / 15%** | Chronological (no shuffling to prevent data leakage) |
| Normalization | **min-max [0, 1]** for ocean vars; **log1p then min-max** for AIS | Compresses heavy-tailed fishing distribution |

---

## Repository Structure

```
convlstm_fishingcatch/
├── 01_data_loading.ipynb       # Phase 1 — raw CMEMS + AIS ingest & regional cache
├── 02_preprocessing.ipynb      # Phase 2 — subset, regrid, normalize, save feature cube
├── 03_model_training.ipynb     # Phase 3 — build & train ConvLSTM2D, save outputs
├── 04_evaluation.ipynb         # Phase 4 — load model, predict, compute 5 metrics
├── 05_visualization.ipynb      # Phase 5 — maps, error plots, training curves
├── Full_ConvLSTM.py            # Streamlit dashboard (all 5 phases in one app)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

All data files are read from and written to **Google Drive** at:
```
My Drive/fishing_project/
```

> **Modularity:** Every notebook shares an identical `CONFIG` dictionary at the top. To change the model bbox, date range, normalization method, or any file name — edit one cell and copy it to the next notebook. See [Shared CONFIG Dictionary](#shared-config-dictionary).

---

## Prerequisites

The notebooks run on **Google Colab** with a GPU runtime (T4 recommended).

All dependencies are installed inline with `!pip install -q` at the top of each notebook:

```bash
# Phase 1 & 2
pip install xarray netCDF4 pandas numpy pyarrow scipy

# Phase 3
pip install tensorflow

# Phase 4
pip install scikit-learn scipy tensorflow

# Phase 5
pip install matplotlib scikit-learn

# Dashboard (Full_ConvLSTM.py)
pip install streamlit xarray netCDF4 tensorflow scikit-learn
```

> **GPU required:** ConvLSTM2D training is very slow on CPU. Set Colab runtime to **GPU** (Runtime → Change runtime type → T4 GPU) before running Phase 3.

---

## Required Datasets

Place all files in `My Drive/fishing_project/` before running the notebooks.

### 1. CMEMS Global Ocean Physics Reanalysis — Wind Velocities (0.083°)
- **Product:** `GLOBAL_MULTIYEAR_PHY_001_030`
- **Variables:** `uo` (eastward velocity), `vo` (northward velocity)
- **Resolution:** 0.083° monthly
- **Time range:** 2014-01 – 2024-12 *(wider than model window; Phase 1 caches and Phase 2 subsets)*
- **Region:** 0°–30°N, 110°–140°E *(staging bbox; narrowed in Phase 2)*
- **Filename:** `cmems_mod_glo_phy_my_0.083deg_P1M-m_1779636039565.nc`
- **Portal:** [CMEMS Data Store](https://data.marine.copernicus.eu)

### 2. CMEMS Global Ocean Physics Reanalysis — Height & Temperature (0.083°)
- **Product:** `GLOBAL_MULTIYEAR_PHY_001_030`
- **Variables:** `thetao` (SST), `zos` (SSH)
- **Resolution:** 0.083° monthly
- **Filename:** `cmems_mod_glo_phy_my_0.083deg_P1M-m_1779635380319.nc`
- **Portal:** [CMEMS Data Store](https://data.marine.copernicus.eu)

### 3. CMEMS Global Ocean Biogeochemistry Hindcast (0.25°)
- **Product:** `GLOBAL_MULTIYEAR_BGC_001_029`
- **Variables:** `chl` (Chlorophyll-a), `nppv` (Net Primary Production)
- **Resolution:** 0.25° monthly
- **Filename:** `cmems_mod_glo_bgc_my_0.25deg_P1M-m_1779635372583.nc`
- **Portal:** [CMEMS Data Store](https://data.marine.copernicus.eu)

### 4. AIS Fishing Effort — Global Fishing Watch v3.0
- **Product:** Monthly apparent fishing hours by flag state and gear type
- **Format:** `fleet-monthly-csvs-10-v3-YYYY-MM-DD.csv` (0.1° resolution)
- **Years required:** 2014–2024
- **Folder layout:** All CSVs in a **flat folder** — no year subfolders:
  ```
  My Drive/fishing_project/ais_fishing/
      fleet-monthly-csvs-10-v3-2019-01-01.csv
      fleet-monthly-csvs-10-v3-2019-02-01.csv
      ...
      fleet-monthly-csvs-10-v3-2024-12-01.csv
  ```
- **Portal:** [Global Fishing Watch Data Portal](https://globalfishingwatch.org/datasets-and-code/)

> **Why monthly 0.1° files?** The daily 0.01° files are 1–2 GB each. The monthly 0.1° files are ~1,000× smaller, load instantly, and bin cleanly onto the 0.25° target grid.

---

## Step-by-Step Walkthrough

### Phase 1: Data Loading
**Notebook:** `01_data_loading.ipynb` | **Outputs:** `physics_raw_region.nc`, `bgc_raw_region.nc`, `ais_raw_region.parquet`

**Goal:** Mount Drive, load raw CMEMS NetCDF and AIS CSVs, verify time alignment, and cache regional subsets for Phase 2.

1. **Mount Google Drive** and set `DATA_DIR` from `CONFIG['data_dir']`.
2. **Load two CMEMS physics NetCDF files** and merge them into one `xr.Dataset` with variables `uo`, `vo`, `thetao`, `zos`.
3. **Load CMEMS BGC NetCDF** → variables `chl`, `nppv`.
4. **Load AIS CSVs** from the flat `ais_fishing/` folder using `load_ais_filtered()`:
   - Reads only the 4 required columns (`date`, `cell_ll_lat`, `cell_ll_lon`, `fishing_hours`)
   - Filters to `CONFIG['bbox_regional']` immediately (drops ~98% of rows)
   - Skips files outside `CONFIG['date_full']` by filename date
5. **Time range verification** — asserts all three sources span the configured date range.
6. **Save regional caches** to Drive:
   - `physics_raw_region.nc` — merged physics dataset
   - `bgc_raw_region.nc` — BGC dataset
   - `ais_raw_region.parquet` — filtered AIS rows (CSV.gz fallback if pyarrow unavailable)

> **Run Phase 1 once.** The cached files are the only inputs to Phase 2 — you never need to re-load the multi-GB source files again.

---

### Phase 2: Preprocessing
**Notebook:** `02_preprocessing.ipynb` | **Inputs:** Phase 1 outputs | **Output:** `preprocessed_features.nc`, `ais_fishing_effort_gridded.nc`

**Goal:** Subset to the WPS model bbox and date window, regrid, normalize all 7 channels, and save a clean feature cube.

**Step 1 — Load and subset to WPS model bbox:**
- Opens `physics_raw_region.nc` and `bgc_raw_region.nc`.
- Physics: selects nearest surface depth (`CONFIG['physics_surface_depth']` = 0.49 m), then subsets to `CONFIG['bbox_model']` (10°–20°N, 114°–120°E) and `CONFIG['date_model']` (2019–2024).
- BGC: depth-averages across `CONFIG['bgc_depth_range']` (0.51–5.14 m), then applies the same spatial/temporal subset.

**Step 2 — Monthly resampling:**
- Resamples both datasets to `CONFIG['resample_freq']` = `'1ME'` (modern replacement for deprecated `'1M'`).

**Step 3 — Regrid physics to 0.25°:**
- The BGC native grid (41×25 at 0.25°) is the **canonical target grid**.
- Physics (0.083°) is bilinearly interpolated onto it with `xr.DataArray.interp()`.

**Step 4 — Alignment verification:**
- `np.allclose` assertions confirm lat/lon/time are identical between datasets.

**Step 5 — Gap filling:**
- NaN values (land/masked cells) are linearly interpolated along the time axis using `interpolate_na(..., fill_value='extrapolate')`.

**Step 6 — Normalize 6 ocean channels:**

| Variable | Source dataset | Physical meaning |
|----------|---------------|-----------------|
| `sst` | Physics `thetao` | Sea surface temperature |
| `ssh` | Physics `zos` | Sea surface height |
| `uo` | Physics `uo` | Eastward sea water velocity |
| `vo` | Physics `vo` | Northward sea water velocity |
| `chl` | BGC `chl` | Chlorophyll-a |
| `nppv` | BGC `nppv` | Net primary production |

All 6 are min-max normalized to **[0, 1]** globally over the full time series. Method is controlled by `CONFIG['norm_method']` (`'minmax'` or `'zscore'`).

**Step 7 — Process AIS data (7th channel + label):**
- Loads `ais_raw_region.parquet` from Phase 1.
- Narrows to `CONFIG['bbox_model']` and `CONFIG['date_model']`.
- Bins 0.1° GFW rows onto the 0.25° target grid with `aggregate_ais_to_grid()`.
- Aligns to the CMEMS time axis (missing months filled with 0).
- Applies **log1p then min-max normalization** to compress the heavy-tailed distribution.
- Saves raw gridded AIS to `ais_fishing_effort_gridded.nc`.

**Step 8 — Save 7-channel feature cube:**
- All 7 normalized arrays are merged into `preprocessed_features.nc`.
- Final shape: `(72, 41, 25)` per variable, 7 variables.

---

### Phase 3: Model Training
**Notebook:** `03_model_training.ipynb` | **Input:** `preprocessed_features.nc` | **Outputs:** `convlstm_model.keras`, `best_model.keras`, `X_test.npy`, `y_test.npy`, `training_history.json`, `data_summary.json`

**Goal:** Build sliding-window sequences and train a two-layer ConvLSTM2D model.

**Sequence creation:**
- Sliding window over `(72, 41, 25, 7)` feature cube.
- `SEQ_LEN = 3` → each input sample is 3 consecutive monthly maps.
- `PRED_LEN = 1` → each label is the next month's AIS fishing effort grid.
- Result: `X_seq` shape `(69, 3, 41, 25, 7)` and `y_seq` shape `(69, 1, 41, 25)`.

**Train/Val/Test split (chronological):**

| Split | Samples | Approx. dates |
|-------|---------|---------------|
| Train (70%) | 48 | Jan 2019 – Dec 2022 |
| Val (15%)   | 10 | Jan 2023 – Oct 2023 |
| Test (15%)  | 11 | Nov 2023 – Dec 2024 |

**Label reshaping:**
- `y_seq` → `(N, 41, 25, 1)` to match the Conv2D output shape.

**Callbacks:**
- `EarlyStopping(patience=10)` — stops training if validation loss stagnates.
- `ModelCheckpoint` → saves best weights to `best_model.keras` (native Keras format).

**Training:**
- Up to **50 epochs**, batch size **8**, optimizer **Adam**, loss **binary cross-entropy**.

**Saved outputs:**

| File | Contents |
|------|----------|
| `convlstm_model.keras` | Final trained model (native Keras, no legacy HDF5 warning) |
| `best_model.keras` | Best validation loss checkpoint |
| `X_test.npy` | Test input sequences `(11, 3, 41, 25, 7)` |
| `y_test.npy` | Test labels `(11, 41, 25, 1)` |
| `training_history.json` | Epoch-wise loss/MAE for train and val (read by dashboard) |
| `data_summary.json` | Dataset metadata: shapes, split sizes, bbox, best epoch (read by dashboard Pipeline tab) |

---

### Phase 4: Model Evaluation
**Notebook:** `04_evaluation.ipynb` | **Inputs:** Phase 3 outputs | **Outputs:** `predictions.npy`, `evaluation_results.csv`

**Goal:** Quantify model performance on the held-out test set using five metrics.

The model is loaded from `convlstm_model.keras` (`.h5` legacy fallback if not found). Land/coastal NaN values in `X_test` are replaced with `0.0` before prediction.

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — overall magnitude of prediction errors |
| **MAE** | Mean Absolute Error — average absolute deviation per pixel |
| **F1 Score** | Binary classification score at threshold `CONFIG['f1_threshold']` = 0.5 |
| **SSI** | Structural Similarity Index — compares local luminance, contrast, structure |
| **Wasserstein Distance** | Earth Mover's Distance between predicted and observed distributions |

Results are printed as a formatted table and saved to `evaluation_results.csv`.

---

### Phase 5: Visualization
**Notebook:** `05_visualization.ipynb` | **Inputs:** Phase 3/4 outputs | **Outputs:** 4 PNG files

**Goal:** Produce publication-quality plots comparing predictions to observations.

Four plots are generated and saved at 300 DPI to the Drive folder:

1. **`obs_vs_pred.png`** — 2×N grid: observed AIS effort (top) vs. ConvLSTM predictions (bottom) for up to 6 test months. Custom `fishing` colormap: `#000033 → #0000FF → #FF00FF → #FF0000`.

2. **`error_maps.png`** — Spatial error maps averaged across all test months:
   - *Mean Error* (Pred − Obs): signed bias, `RdBu_r` colormap
   - *MAE map*: absolute average error per cell, `Reds`
   - *RMSE map*: per-cell RMSE across test period, `Reds`

3. **`rmse_over_time.png`** — Bar + line chart of per-sample RMSE across the 11 test months.

4. **`training_history.png`** — Dual-panel training curves (Loss and MAE for train/val), loaded from `training_history.json`.

> **Note on visualization latitude extent:** Plots use `lat 7°–20°N` for geographic framing context (South China Sea coastline), while the model bounding box is `10°–20°N`. The 7° padding is intentional and does not affect model inputs.

---

## Streamlit Dashboard

**File:** `Full_ConvLSTM.py`

A dark-themed Streamlit dashboard that reads all notebook outputs from Google Drive and presents them interactively. Run it in Colab alongside the notebooks:

```bash
!pip install streamlit -q
!npm install -g localtunnel
!streamlit run Full_ConvLSTM.py --server.enableCORS=false --server.enableXsrfProtection=false &>/content/logs.txt &
!lt --port 8501
# Paste the IP below as the localtunnel password:
import urllib; print(urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode().strip())
```

**Five tabs:**

| Tab | Contents |
|-----|----------|
| **Dashboard** | 5 metric cards (RMSE/MAE/F1/SSI/Wasserstein) · Latest prediction map · RMSE bar chart over test months |
| **Predictions** | Month browser slider · Observed vs. predicted maps · Optional difference map · Spatial lat/lon profiles · Download buttons |
| **Analytics** | Evaluation results table · 2×N obs-vs-pred grid (matches `obs_vs_pred.png`) · Spatial error maps · Training history curves · Variable explorer (all 7 channels from `preprocessed_features.nc`) |
| **Pipeline** | Per-phase file checklist with sizes · `data_summary.json` metric cards (split counts, best epoch) · Overall readiness banner |
| **About** | Architecture code block · Output file status table · Data source cards |

**Data loaded by the dashboard:**

| File | Loaded by |
|------|-----------|
| `predictions.npy` | Dashboard (all tabs) |
| `y_test.npy` | Dashboard (all tabs) |
| `X_test.npy` | Dashboard |
| `evaluation_results.csv` | Dashboard metrics & Analytics |
| `training_history.json` | Analytics → Training history |
| `data_summary.json` | Pipeline tab metadata cards |
| `preprocessed_features.nc` | Analytics → Variable explorer |

---

## Model Architecture

```
Input: (batch, 3 timesteps, 41 lat, 25 lon, 7 channels)
         │
         ▼
  ConvLSTM2D(filters=64, kernel=3×3, padding='same', return_sequences=True)
         │   ~163,840 params — extracts spatiotemporal patterns across all 3 timesteps
         ▼
  BatchNormalization + Dropout(0.2)
         │
         ▼
  ConvLSTM2D(filters=32, kernel=3×3, padding='same', return_sequences=False)
         │   ~110,720 params — collapses temporal dimension, outputs final spatial state
         ▼
  BatchNormalization + Dropout(0.2)
         │
         ▼
  Conv2D(filters=1, kernel=1×1, activation='sigmoid')
         │   33 params — projects to single-channel fishing probability
         ▼
Output: (batch, 41, 25, 1)   ← probability map in [0, 1]

Total trainable parameters: ~274,785
optimizer='adam'  loss='binary_crossentropy'  metrics=['mae']
epochs=50  batch_size=8  patience=10  (EarlyStopping on val_loss)
```

---

## Shared CONFIG Dictionary

Every notebook (01–05) and the dashboard share a single `CONFIG` dictionary declared at the top. It is the **sole source of truth** for all spatial, temporal, path, and model parameters.

```python
CONFIG = {
    # Spatial: Phase 1 caches the regional bbox; Phase 2+ uses the model bbox
    'bbox_regional': {'lat_min': 0,  'lat_max': 30,  'lon_min': 110, 'lon_max': 140},
    'bbox_model':    {'lat_min': 10, 'lat_max': 20,  'lon_min': 114, 'lon_max': 120},

    # Temporal
    'date_full':  {'start': '2014-01-01', 'end': '2024-12-31'},  # Phase 1 CMEMS/AIS load
    'date_model': {'start': '2019-01-01', 'end': '2024-12-31'},  # Phase 2+ model window

    # All filenames
    'files': {
        'preprocessed_nc': 'preprocessed_features.nc',
        'model_keras':     'convlstm_model.keras',
        'best_model':      'best_model.keras',
        'X_test_npy':      'X_test.npy',
        'y_test_npy':      'y_test.npy',
        'history_json':    'training_history.json',
        'summary_json':    'data_summary.json',
        'predictions_npy': 'predictions.npy',
        'eval_csv':        'evaluation_results.csv',
        # ...and all other intermediate files
    },

    # Model
    'seq_len': 3, 'pred_len': 1, 'n_channels': 7,
    'epochs': 50, 'batch_size': 8, 'patience': 10,
    'train_frac': 0.70, 'val_frac': 0.15,

    # Preprocessing
    'norm_method': 'minmax', 'resample_freq': '1ME',
    'physics_surface_depth': 0.49,
    'bgc_depth_range': (0.51, 5.14),
    'f1_threshold': 0.5,
}
```

> To adapt this pipeline for a different region or date range, update `CONFIG` in **one cell** and copy it to the other notebooks.

---

## Output Files

All outputs are written to `My Drive/fishing_project/`:

| File | Phase | Description |
|------|-------|-------------|
| `physics_raw_region.nc` | 1 | CMEMS physics regional cache |
| `bgc_raw_region.nc` | 1 | CMEMS BGC regional cache |
| `ais_raw_region.parquet` | 1 | AIS regional cache (CSV.gz fallback) |
| `preprocessed_features.nc` | 2 | 7-channel normalised feature cube `(72, 41, 25)` |
| `ais_fishing_effort_gridded.nc` | 2 | AIS gridded to 0.25° monthly `(72, 41, 25)` |
| `convlstm_model.keras` | 3 | Final trained model (native Keras format) |
| `best_model.keras` | 3 | Best validation loss checkpoint |
| `X_test.npy` | 3 | Test input sequences `(11, 3, 41, 25, 7)` |
| `y_test.npy` | 3 | Test labels `(11, 41, 25, 1)` |
| `training_history.json` | 3 | Epoch-wise loss/MAE for train and val |
| `data_summary.json` | 3 | Pipeline metadata for dashboard Pipeline tab |
| `predictions.npy` | 4 | Model predictions on test set `(11, 41, 25, 1)` |
| `evaluation_results.csv` | 4 | RMSE, MAE, F1, SSI, Wasserstein scores |
| `obs_vs_pred.png` | 5 | Observed vs. predicted heatmap grid |
| `error_maps.png` | 5 | Spatial mean error, MAE, and RMSE maps |
| `rmse_over_time.png` | 5 | Per-sample RMSE bar chart |
| `training_history.png` | 5 | Training and validation loss/MAE curves |

---

## Known Issues & Notes

> **Most issues listed in the original README have been resolved.** The table below documents the remaining operational considerations.

| Issue | Status | Detail |
|-------|--------|--------|
| `resample(time='1M')` FutureWarning | ✅ Fixed | All notebooks use `CONFIG['resample_freq']` = `'1ME'` |
| `Dataset.dims` FutureWarning | ✅ Fixed | Notebook 03 uses `.sizes` instead of `.dims` |
| HDF5 `.h5` legacy format warning | ✅ Fixed | Notebooks 03/04 save/load `.keras` format; `.h5` fallback retained in 04 |
| Duplicate `load_ais_filtered` definition | ✅ Fixed | Defined once in Notebook 01; Notebook 02 loads from Parquet |
| F1 score = 0.0000 | ⚠️ Monitor | The model's sigmoid output rarely exceeds 0.5 on sparse fishing pixels. Consider lowering `CONFIG['f1_threshold']` or using a weighted binary cross-entropy loss |
| AIS data sparsity | ℹ️ By design | The narrow WPS bbox (10°–20°N, 114°–120°E) was chosen specifically to improve the positive-pixel ratio and thus F1 score vs. the wider regional bbox |
| Training speed | ℹ️ Expected | ConvLSTM2D is compute-heavy. T4 GPU completes 50 epochs in ~10 min. CPU will be ~50× slower |
| Colab session reset | ℹ️ By design | After a session reset, re-run the `CONFIG` cell in each notebook before continuing. All heavy data is cached in Drive |
