# ConvLSTM Fishing Ground Prediction
### West Philippine Sea - 2017-2024

A spatiotemporal deep learning pipeline that predicts monthly fishing ground probability maps for the West Philippine Sea (10-20 N, 114-120 E) using a custom Convolutional LSTM (ConvLSTM) model implemented in PyTorch with manual low-level operations (pure tensor math). The model learns from 7 channels (6 oceanographic variables + AIS fishing effort history) derived from two CMEMS reanalysis products and the Global Fishing Watch AIS dataset, producing a 41x25 grid (at 0.25 deg resolution) of fishing probability for the following month.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Required Datasets](#required-datasets)
5. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
   - [Phase 1: Data Loading](#phase-1-data-loading)
   - [Phase 2: Preprocessing](#phase-2-preprocessing)
   - [Phase 3: Model Training](#phase-3-build-and-train)
   - [Phase 4: Model Evaluation](#phase-4-model-evaluation)
   - [Phase 5: Visualization](#phase-5-visualization)
6. [Streamlit Dashboard](#streamlit-dashboard)
7. [Model Architecture](#model-architecture)
8. [Shared CONFIG Dictionary](#shared-config-dictionary)
9. [Output Files](#output-files)
10. [Known Issues & Resolution Notes](#known-issues--resolution-notes)

---

## Project Overview

```
  Oceanographic features (6 channels)    AIS Fishing Effort (label + channel)
  +----------------------------+          +----------------------+
  |  SST, SSH, UO, VO          |          |  fishing_hours       |
  |  Chlorophyll-a, NPP        |          |  log1p-normalised    |
  +--------------+-------------+          +----------+-----------+
                 |   7 channels * 3 months of history | target month
                 +--------------+---------------------+
                           ConvLSTM2D
                                |
                    Predicted fishing probability
                       map for the next month
                       (41 x 25 grid, 0.25 deg)
```

**Key design choices:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Channels | **7** (sst, ssh, vo, uo, chl, nppv, fishing_effort) | AIS effort as an auto-regressive feature improves F1 |
| Sequence length | **3 months -> predict 1 month ahead** | Captures seasonal lagging effects |
| Spatial grid | **0.25 deg * 41 lat x 25 lon** | BGC native resolution; physics bilinearly regridded |
| Model bbox | **10-20 N, 114-120 E** | WPS core fishing zone; narrow domain improves class balance |
| Staging bbox | **0-30 N, 110-140 E** | Wider cache in Phase 1; narrowed in Phase 2 onward |
| Training date range | **2017-01 - 2024-12 (96 months)** | Expanded window adds training context and addresses data sparsity |
| Train/Val/Test split | **70% / 15% / 15%** | Chronological (no shuffling to prevent temporal data leakage) |
| Normalization | **min-max [0, 1]** for ocean vars; **log1p then min-max** for AIS | Compresses heavy-tailed fishing distribution; parameters saved to JSON |

---

## Repository Structure

```
convlstm_fishingcatch/
├── 01_data_loading.ipynb          # Phase 1 - raw CMEMS + AIS ingest & regional cache
├── 02_preprocessing.ipynb         # Phase 2 - subset, regrid, normalize, save features & stats
├── 03_buildntraintorch.ipynb      # Phase 3 - train custom PyTorch ConvLSTM (manual ops & focal loss)
├── 04_evaluation.ipynb            # Phase 4 - pure NumPy inference using exported weights
├── 05_visualization.ipynb         # Phase 5 - publication-grade Cartopy mapping & error curves
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

All data files are read from and written to **Google Drive** at:
```
My Drive/fishing_project/
```

> [!NOTE]
> **Modularity:** Every notebook shares an identical `CONFIG` dictionary at the top. To change the model bbox, date range, normalization method, or any file name - edit one cell and copy it to the next notebook. See [Shared CONFIG Dictionary](#shared-config-dictionary).

---

## Prerequisites

The notebooks are designed to run on **Google Colab** with a CPU runtime or GPU runtime (recommended for acceleration in Notebook 03).

All dependencies are installed inline with `!pip install -q` at the top of each notebook:

```bash
# Phase 1 & 2
pip install xarray netCDF4 pandas numpy pyarrow scipy

# Phase 3 (PyTorch-Manual)
pip install torch xarray netCDF4 numpy

# Phase 4 (NumPy Evaluation)
pip install scikit-learn scipy numpy pandas

# Phase 5 (Geographic Visualisation)
pip install matplotlib scikit-learn cartopy numpy

# Dashboard (frontend/strmlt_ConvLSTM.py)
pip install streamlit xarray netCDF4 scikit-learn matplotlib pandas numpy
```

> [!TIP]
> **Computation:** Training is done in PyTorch but bypasses high-level model modules in favor of custom-written forward/backward tensor passes. While CPU execution is supported, a GPU runtime dramatically reduces the training time for the 100-epoch limit.

---

## Required Datasets

Place all files in `My Drive/fishing_project/` before running the notebooks.

### 1. CMEMS Global Ocean Physics Reanalysis - Wind Velocities (0.083 deg)
- **Product:** `GLOBAL_MULTIYEAR_PHY_001_030`
- **Variables:** `uo` (eastward velocity), `vo` (northward velocity)
- **Resolution:** 0.083 deg monthly
- **Time range:** 2014-01 - 2024-12 *(wider than model window; Phase 1 caches and Phase 2 subsets)*
- **Region:** 0-30 N, 110-140 E *(staging bbox; narrowed in Phase 2)*
- **Filename:** `cmems_mod_glo_phy_my_0.083deg_P1M-m_1779636039565.nc`
- **Portal:** [CMEMS Data Store](https://data.marine.copernicus.eu)

### 2. CMEMS Global Ocean Physics Reanalysis - Height & Temperature (0.083 deg)
- **Product:** `GLOBAL_MULTIYEAR_PHY_001_030`
- **Variables:** `thetao` (SST), `zos` (SSH)
- **Resolution:** 0.083 deg monthly
- **Filename:** `cmems_mod_glo_phy_my_0.083deg_P1M-m_1779635380319.nc`
- **Portal:** [CMEMS Data Store](https://data.marine.copernicus.eu)

### 3. CMEMS Global Ocean Biogeochemistry Hindcast (0.25 deg)
- **Product:** `GLOBAL_MULTIYEAR_BGC_001_029`
- **Variables:** `chl` (Chlorophyll-a), `nppv` (Net Primary Production)
- **Resolution:** 0.25 deg monthly
- **Filename:** `cmems_mod_glo_bgc_my_0.25deg_P1M-m_1779635372583.nc`
- **Portal:** [CMEMS Data Store](https://data.marine.copernicus.eu)

### 4. AIS Fishing Effort - Global Fishing Watch v3.0
- **Product:** Monthly apparent fishing hours by flag state and gear type
- **Format:** `fleet-monthly-csvs-10-v3-YYYY-MM-DD.csv` (0.1 deg resolution)
- **Years required:** 2014-2024
- **Folder layout:** All CSVs in a **flat folder** - no year subfolders:
  ```
  My Drive/fishing_project/ais_fishing/
      fleet-monthly-csvs-10-v3-2019-01-01.csv
      fleet-monthly-csvs-10-v3-2019-02-01.csv
      ...
      fleet-monthly-csvs-10-v3-2024-12-01.csv
  ```
- **Portal:** [Global Fishing Watch Data Portal](https://globalfishingwatch.org/datasets-and-code/)

> **Why monthly 0.1 deg files?** The daily 0.01 deg files are extremely large. The monthly 0.1 deg files load quickly, filter efficiently in Pandas, and bin cleanly onto the 0.25 deg target grid.

---

## Step-by-Step Walkthrough

### Phase 1: Data Loading
**Notebook:** [01_data_loading.ipynb](file:///d:/Programming/convlstm/convlstm_fishingcatch/01_data_loading.ipynb) | **Outputs:** `physics_raw_region.nc`, `bgc_raw_region.nc`, `ais_raw_region.parquet`

**Goal:** Mount Drive, load raw CMEMS NetCDF and AIS CSVs, verify time alignment, and cache regional subsets for Phase 2.

1. **Mount Google Drive** and set `DATA_DIR` from `CONFIG['data_dir']`.
2. **Load two CMEMS physics NetCDF files** and merge them into one `xr.Dataset` with variables `uo`, `vo`, `thetao`, `zos`.
3. **Load CMEMS BGC NetCDF** -> variables `chl`, `nppv`.
4. **Load AIS CSVs** from the flat `ais_fishing/` folder using `load_ais_filtered()`:
   - Reads only the 4 required columns (`date`, `cell_ll_lat`, `cell_ll_lon`, `fishing_hours`)
   - Filters to `CONFIG['bbox_regional']` immediately (drops ~98% of rows)
   - Skips files outside `CONFIG['date_full']` by filename date
5. **Time range verification** - asserts all three sources span the configured date range.
6. **Save regional caches** to Drive:
   - `physics_raw_region.nc` - merged physics dataset
   - `bgc_raw_region.nc` - BGC dataset
   - `ais_raw_region.parquet` - filtered AIS rows (CSV.gz fallback if pyarrow unavailable)

> **Run Phase 1 once.** The cached files are the only inputs to Phase 2 - you never need to re-load the multi-GB source files again.

---

### Phase 2: Preprocessing
**Notebook:** [02_preprocessing.ipynb](file:///d:/Programming/convlstm/convlstm_fishingcatch/02_preprocessing.ipynb) | **Inputs:** Phase 1 outputs | **Outputs:** `preprocessed_features.nc`, `norm_stats.json`, `ais_fishing_effort_gridded.nc`

**Goal:** Subset to the WPS model bbox and date window, regrid, normalize all 7 channels, save normalization parameters, and output a clean feature cube.

**Step 1 - Load and subset to WPS model bbox:**
- Opens `physics_raw_region.nc` and `bgc_raw_region.nc`.
- Physics: selects nearest surface depth (`CONFIG['physics_surface_depth']` = 0.49 m), then subsets to `CONFIG['bbox_model']` (10-20 N, 114-120 E) and `CONFIG['date_model']` (2017-2024).
- BGC: depth-averages across `CONFIG['bgc_depth_range']` (0.51-5.14 m), then applies the same spatial/temporal subset.

**Step 2 - Monthly resampling:**
- Resamples both datasets to `CONFIG['resample_freq']` = `'1ME'` (modern replacement for deprecated `'1M'`).

**Step 3 - Regrid physics to 0.25 deg:**
- The BGC native grid (41x25 at 0.25 deg) is the **canonical target grid**.
- Physics (0.083 deg) is bilinearly interpolated onto it with `xr.DataArray.interp()`.

**Step 4 - Alignment verification:**
- `np.allclose` assertions confirm lat/lon/time are identical between datasets.

**Step 5 - Gap filling:**
- NaN values (land/masked cells) are linearly interpolated along the time axis using `interpolate_na(..., fill_value='extrapolate')`.

**Step 6 - Normalize 6 ocean channels:**

| Variable | Source dataset | Physical meaning |
|----------|---------------|-----------------|
| `sst` | Physics `thetao` | Sea surface temperature |
| `ssh` | Physics `zos` | Sea surface height |
| `uo` | Physics `uo` | Eastward sea water velocity |
| `vo` | Physics `vo` | Northward sea water velocity |
| `chl` | BGC `chl` | Chlorophyll-a |
| `nppv` | BGC `nppv` | Net primary production |

All 6 are min-max normalized to **[0, 1]** over the training months only to prevent data leakage. Normalization bounds are exported to `norm_stats.json` for inverse scaling during inference.

**Step 7 - Process AIS data (7th channel + label):**
- Loads `ais_raw_region.parquet` from Phase 1.
- Narrows to `CONFIG['bbox_model']` and `CONFIG['date_model']` (2017-2024).
- Bins 0.1 deg GFW rows onto the 0.25 deg target grid with `aggregate_ais_to_grid()`.
- Aligns to the CMEMS time axis (missing months filled with 0).
- Applies **log1p then min-max normalization** (based on training bounds) to compress the heavy-tailed distribution.
- Saves raw gridded AIS to `ais_fishing_effort_gridded.nc`.

**Step 8 - Save 7-channel feature cube:**
- All 7 normalized arrays are merged into `preprocessed_features.nc`.
- Final shape: `(96, 41, 25)` per variable, 7 variables.

---

### Phase 3: Model Training
**Notebook:** [03_buildntraintorch.ipynb](file:///d:/Programming/convlstm/convlstm_fishingcatch/03_buildntraintorch.ipynb) | **Input:** `preprocessed_features.nc` | **Outputs:** `convlstm_weights.npz`, `best_model.pt`, `convlstm_model.pt`, `X_test.npy`, `y_test.npy`, `training_history.json`, `data_summary.json`

**Goal:** Build sliding-window sequences and train a custom two-layer ConvLSTM model in PyTorch using manual low-level operations.

**Sequence creation:**
- Sliding window over `(96, 41, 25, 7)` feature cube.
- `SEQ_LEN = 3` -> each input sample is 3 consecutive monthly maps.
- `PRED_LEN = 1` -> each label is the next month's AIS fishing effort grid.
- Result: `X_seq` shape `(93, 3, 41, 25, 7)` and `y_seq` shape `(93, 1, 41, 25)`.

**Train/Val/Test split (chronological 70/15/15):**

| Split | Samples | Approx. dates |
|-------|---------|---------------|
| Train (70%) | **65** | Nov 2017 - May 2022 |
| Val (15%)   | **13** | Jun 2022 - Jun 2023 |
| Test (15%)  | **15** | Jul 2023 - Dec 2024 |

**Label reshaping:**
- `y_seq` -> `(N, 41, 25, 1)` to match the final output shape.

**Custom PyTorch Manual Model Operations:**
- **im2col convolution:** Slides a kernel over the 2D maps using PyTorch's `F.unfold` (im2col) followed by `torch.matmul` and spatial reshaping, bypassing standard `nn.Conv2d`.
- **Manual BatchNorm:** Custom running-stat normalizer. Computes batch mean/variance, tracks them as non-gradient parameters, and scales inputs (using `eps=1e-4` to handle small batch sizes).
- **Manual Dropout:** Custom inverted dropout using a Bernoulli mask (`torch.bernoulli`) scaled by `1 / (1 - rate)` during training.
- **Focal Loss:** Replaces standard BCE to address the extreme class imbalance (empty ocean cells). Modulates training using a focal factor to focus on hard boundaries:
  $$\text{Focal Loss} = - \alpha (1 - p_t)^\gamma \log(p_t)$$
  Where $\alpha = 0.75$, $\gamma = 2.0$, and $p_t$ represents the continuous probability distance.
- **Manual Adam:** Custom optimization loop updating parameters based on first and second moments manually, compiled via tensor operations.
- **Validation F1 Monitoring:** Early stopping is tracked on the validation F1-score rather than raw loss. The training runs up to 100 epochs with a patience of 15 and automated learning rate reduction (factor 0.5, patience 7).

**Saved outputs:**
- `best_model.pt` & `convlstm_model.pt`: Saved model state dictionaries.
- `convlstm_weights.npz`: Saved model weights in a NumPy-compatible archive format for Notebook 04, mapping PyTorch filter formats `(C_out, C_in, kH, kW)` to the expected layout `(kH*kW*C_in, C_out)`.
- `X_test.npy` / `y_test.npy`: Saved chronological test splits.
- `training_history.json`: Log of training metrics.
- `data_summary.json`: Run metadata containing split sizes, best epochs, and configurations.

---

### Phase 4: Model Evaluation
**Notebook:** [04_evaluation.ipynb](file:///d:/Programming/convlstm/convlstm_fishingcatch/04_evaluation.ipynb) | **Inputs:** Phase 3 outputs | **Outputs:** `predictions.npy`, `evaluation_results.csv`

**Goal:** Reconstruct the custom model structure in pure NumPy, load the saved `.npz` weights, run inference on the test set, and evaluate performance.

The model is instantiated in pure NumPy. Weight arrays are loaded directly from `convlstm_weights.npz` and slot into the `im2col` matrix multiplication routines without further transposition. 

The pipeline evaluates predictions using six metrics:

| Metric | Test Set Value | Description |
|--------|----------------|-------------|
| **RMSE** | **0.0954** | Root Mean Squared Error - overall magnitude of prediction errors |
| **MAE** | **0.0412** | Mean Absolute Error - average absolute deviation per pixel (~4 pp) |
| **F1 (Best)** | **0.3852** | Binary classification score at optimal threshold = **0.08** |
| **F1 (Config)** | **0.2451** | Binary classification score at standard configuration threshold = **0.15** |
| **SSI** | **0.1850 ± 0.065** | Structural Similarity Index - compares local luminance, contrast, structure |
| **Wasserstein Distance** | **8.4520 ± 3.120** | Earth Mover's Distance between predicted and observed spatial distributions |

Results are printed as a formatted table and saved to `evaluation_results.csv`.

---

### Phase 5: Visualization
**Notebook:** [05_visualization.ipynb](file:///d:/Programming/convlstm/convlstm_fishingcatch/05_visualization.ipynb) | **Inputs:** Phase 3/4 outputs | **Outputs:** 4 PNG files

**Goal:** Produce publication-quality plots comparing predictions to observations, overlaying geographic maps using Cartopy.

Four plots are generated and saved at 300 DPI to the Drive folder:

1. **`obs_vs_pred.png`** - 2xN grid: observed AIS effort (top) vs. ConvLSTM predictions (bottom) for test months. Uses custom `fishing` colormap (`#000033 -> #0000FF -> #FF00FF -> #FF0000`) and overlays land masses.
2. **`error_maps.png`** - Spatial error maps (Mean Error via `RdBu_r`, MAE/RMSE via `Reds`) indicating where the model under- or over-predicts.
3. **`rmse_over_time.png`** - Bar + line chart of per-sample RMSE across the 15 test months.
4. **`training_history.png`** - Dual-panel training curves (Loss and MAE for train/val), loaded from `training_history.json`.

> **Note on visualization latitude extent:** Spatial plots expand the latitude range to `7-20 N` (using Cartopy for coastline drawing), while the model bounding box is `10-20 N`. The 3-degree southern padding is intentional and allows framing the Philippine coastline (Luzon/Mindoro) as a geographic reference.

---

## Streamlit Dashboard

**File:** [strmlt_ConvLSTM.py](file:///d:/Programming/convlstm/frontend/strmlt_ConvLSTM.py)

A dark-themed Streamlit dashboard that reads notebook outputs from Google Drive and presents them interactively. Run it in Colab alongside the notebooks:

```bash
# Install Streamlit and localtunnel
pip install streamlit -q
npm install -g localtunnel

# Run the app in the background
streamlit run frontend/strmlt_ConvLSTM.py --server.enableCORS=false --server.enableXsrfProtection=false &>/content/logs.txt &
```

**Six tabs:**

| Tab | Contents |
|-----|----------|
| **Data Upload** | Drag-and-drop or select folder path containing the pipeline outputs (`.npy`, `.csv`, `.nc`) |
| **Dashboard** | 6 metric cards (RMSE/MAE/F1_best/F1_config/SSI/Wasserstein) - Latest prediction map - RMSE bar chart over test months |
| **Predictions** | Month browser slider - Observed vs. predicted maps - Optional difference map - Spatial lat/lon profiles - Download buttons |
| **Analytics** | Evaluation results table - 2xN obs-vs-pred grid - Spatial error maps - Training history curves - Variable explorer (all 7 channels) |
| **Pipeline** | Per-phase file checklist with sizes - `data_summary.json` metric cards (split counts, best epoch) - Overall readiness banner |
| **About** | Architecture details - Output file status table - Data source cards |

---

## Model Architecture

```
Input: (batch, 3 timesteps, 7 channels, 41 lat, 25 lon) [PyTorch Tensor Layout]
         |
         v
  ConvLSTM Layer 1 (filters=32, kernel=3x3, padding='same', return_sequences=True)
         |   Extracts spatiotemporal patterns across all 3 timesteps
         v
  Manual BatchNorm + Dropout(0.2)
         |
         v
  ConvLSTM Layer 2 (filters=16, kernel=3x3, padding='same', return_sequences=False)
         |   Collapses temporal dimension, outputs final spatial state
         v
  Manual BatchNorm + Dropout(0.2)
         |
         v
  1x1 Output Conv (filters=1, activation='sigmoid')  [Manual channel-wise matmul]
         |   Projects 16 hidden channels to a single-channel fishing probability
         v
Output: (batch, 1, 41, 25)   <- probability map in [0, 1]

Total trainable parameters: 72,881
Loss: Focal Loss (gamma=2.0, alpha=0.75)
Optimizer: Manual Adam implementation
Early Stopping: 15 epochs patience based on Validation F1-score
```

---

## Shared CONFIG Dictionary

Every notebook (01-05) and the dashboard share a single `CONFIG` dictionary declared at the top. It is the **sole source of truth** for all spatial, temporal, path, and model parameters.

```python
CONFIG = {
    # Spatial: Phase 1 caches the regional bbox; Phase 2+ uses the model bbox
    'bbox_regional': {'lat_min': 0,  'lat_max': 30,  'lon_min': 110, 'lon_max': 140},
    'bbox_model':    {'lat_min': 10, 'lat_max': 20,  'lon_min': 114, 'lon_max': 120},

    # Temporal
    'date_full':  {'start': '2014-01-01', 'end': '2024-12-31'},  # Phase 1 CMEMS/AIS load
    'date_model': {'start': '2017-01-01', 'end': '2024-12-31'},  # Phase 2+ model window

    # All filenames
    'files': {
        'physics_w_nc':    'cmems_mod_glo_phy_my_0.083deg_P1M-m_1779636039565.nc',
        'physics_ht_nc':   'cmems_mod_glo_phy_my_0.083deg_P1M-m_1779635380319.nc',
        'bgc_src_nc':      'cmems_mod_glo_bgc_my_0.25deg_P1M-m_1779635372583.nc',
        'physics_nc':      'physics_raw_region.nc',
        'bgc_nc':          'bgc_raw_region.nc',
        'ais_parquet':     'ais_raw_region.parquet',
        'ais_csv_gz':      'ais_raw_region.csv.gz',
        'ais_gridded_nc':  'ais_fishing_effort_gridded.nc',
        'preprocessed_nc': 'preprocessed_features.nc',
        'norm_stats_json': 'norm_stats.json',
        'model_pt':        'convlstm_model.pt',
        'best_model_pt':   'best_model.pt',
        'weights_npz':     'convlstm_weights.npz',
        'X_test_npy':      'X_test.npy',
        'y_test_npy':      'y_test.npy',
        'history_json':    'training_history.json',
        'summary_json':    'data_summary.json',
        'predictions_npy': 'predictions.npy',
        'eval_csv':        'evaluation_results.csv',
    },

    # AIS columns
    'ais_use_cols': ['date', 'cell_ll_lat', 'cell_ll_lon', 'fishing_hours'],

    # Model
    'seq_len': 3, 'pred_len': 1, 'n_channels': 7,
    'epochs': 100, 'batch_size': 4, 'patience': 15,
    'train_frac': 0.70, 'val_frac': 0.15,

    # Preprocessing & Evaluation
    'norm_method': 'minmax', 'resample_freq': '1ME',
    'physics_surface_depth': 0.49,
    'bgc_depth_range': (0.51, 5.14),
    'f1_threshold': 0.15,
}
```

---

## Output Files

All outputs are written to `My Drive/fishing_project/`:

| File | Phase | Description |
|------|-------|-------------|
| `physics_raw_region.nc` | 1 | CMEMS physics regional cache |
| `bgc_raw_region.nc` | 1 | CMEMS BGC regional cache |
| `ais_raw_region.parquet` | 1 | AIS regional cache (CSV.gz fallback) |
| `norm_stats.json` | 2 | Min-max normalization parameters for inverse scaling |
| `preprocessed_features.nc` | 2 | 7-channel normalised feature cube `(96, 41, 25)` |
| `ais_fishing_effort_gridded.nc` | 2 | AIS gridded to 0.25° monthly `(96, 41, 25)` |
| `best_model.pt` / `convlstm_model.pt` | 3 | Trained PyTorch model checkpoints |
| `convlstm_weights.npz` | 3 | Exported model weights in NumPy-compatible structure |
| `X_test.npy` | 3 | Test input sequences `(15, 3, 41, 25, 7)` |
| `y_test.npy` | 3 | Test labels `(15, 41, 25, 1)` |
| `training_history.json` | 3 | Epoch-wise loss/MAE for train and val |
| `data_summary.json` | 3 | Pipeline metadata and training run statistics |
| `predictions.npy` | 4 | NumPy model predictions on the test set `(15, 41, 25, 1)` |
| `evaluation_results.csv` | 4 | Final RMSE, MAE, F1, SSI, and Wasserstein metrics |
| `obs_vs_pred.png` | 5 | Observed vs. predicted heatmap grid (Cartopy mapping) |
| `error_maps.png` | 5 | Spatial mean error, MAE, and RMSE maps |
| `rmse_over_time.png` | 5 | Per-sample RMSE bar chart |
| `training_history.png` | 5 | Training and validation loss/MAE curves |

---

## Known Issues & Resolution Notes

### 1. BCE Loss Class Imbalance (Resolved)
- **Problem:** The WPS area is ~95% empty ocean (no fishing). Standard BCE loss led to a "prediction collapse" where the model minimized loss by predicting near-zero everywhere, resulting in an F1-score of **0.0065** at a threshold of 0.15.
- **Resolution:** Implemented a custom **Focal Loss** layer with $\alpha = 0.75$ and $\gamma = 2.0$ to focus updates on active fishing boundaries. Under the new training configuration, the test F1-score at a 0.15 threshold improved to **0.2451**, and the optimal threshold sweep achieved an F1 of **0.3852** (at threshold 0.08).

### 2. Data Sparsity & Bbox Selection (Resolved)
- **Problem:** Apparent fishing effort in Global Fishing Watch is highly sparse prior to 2021.
- **Resolution:** The training model window was expanded from `2019-2024` to `2017-2024` to maximize chronological training samples. The narrow model bounding box (`10-20 N, 114-120 E`) helps concentrate positive labels, improving density and training stability.

### 3. Keras & HDF5 Format Warnings (Resolved)
- **Problem:** TensorFlow/Keras serialization formats frequently throw environment-specific compatibility warnings on Colab.
- **Resolution:** Replaced the Keras library entirely with a PyTorch model configured with manual parameter dictionaries. Weights are exported to `.npz` files, allowing pure NumPy inference during evaluation and dashboard rendering.
