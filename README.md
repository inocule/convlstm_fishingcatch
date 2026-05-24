# ConvLSTM Fishing Ground Prediction
### West Philippine Sea · 2015–2024

A spatiotemporal deep learning pipeline that predicts monthly fishing ground probability maps for the **West Philippine Sea (10°–20°N, 114°–120°E)** using Convolutional LSTM (ConvLSTM2D). The model learns from 6 oceanographic variables derived from two CMEMS reanalysis products and produces a 41×25 grid (at 0.25° resolution) of fishing probability for the following month.

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
6. [Model Architecture](#model-architecture)
7. [Output Files](#output-files)
8. [Known Issues & Notes](#known-issues--notes)

---

## Project Overview

```
Oceanographic features (6 channels)    AIS Fishing Effort (label)
  ┌────────────────────────────┐          ┌──────────────────┐
  │  SST, SSH, UO, VO          │          │  fishing_hours   │
  │  Chlorophyll-a, NPP        │          │  gridded 0.25°   │
  └──────────────┬─────────────┘          └────────┬─────────┘
                 │   3 months of history            │ target month
                 └──────────────┬───────────────────┘
                           ConvLSTM2D
                                │
                    Predicted fishing probability
                       map for the next month
                        (41 × 25 grid, 0.25°)
```

**Key design choices:**
- **Sequence length:** 3 months of lagged features → predict 1 month ahead
- **Spatial grid:** 0.25° resolution (41 lat × 25 lon), covering the West Philippine Sea
- **Time range:** January 2015 – December 2024 (120 months total)
- **Train/Val/Test split:** 70% / 15% / 15% (chronological, no shuffling)

---

## Repository Structure

```
convlstm/
├── Full_ConvLSTM.ipynb   # All-in-one notebook (run this in Colab)
├── 01_data_loading.ipynb
├── 02_preprocessing.ipynb
├── 03_model_training.ipynb
├── 04_evaluation.ipynb
├── 05_visualization.ipynb
└── README.md             # This file
```

The **`Full_ConvLSTM.ipynb`** consolidates all five phases into a single notebook for convenience. The numbered notebooks are the individual modular versions.

All data files are expected in **Google Drive** at:
```
My Drive/fishing_project/
```

---

## Prerequisites

The notebook runs on **Google Colab** with a GPU runtime (T4 recommended).

All dependencies are installed inline at the start of each phase:

```bash
pip install xarray netCDF4 pandas numpy matplotlib
pip install tensorflow
pip install scikit-learn scipy
pip install matplotlib cartopy
```

---

## Required Datasets

You must download the following files and place them in your Google Drive folder (`My Drive/fishing_project/`):

### 1. CMEMS Global Ocean Physics Reanalysis (0.083°)
- **Product:** `GLOBAL_MULTIYEAR_PHY_001_030`
- **Variables:** `thetao` (SST), `uo` (eastward velocity), `vo` (northward velocity), `zos` (SSH)
- **Resolution:** 0.083° monthly, 5 depth levels
- **Time range:** 2015-01–2024-12
- **Region:** 10°–20°N, 114°–120°E (download with margin)
- **Filename:** `cmems_mod_glo_phy_my_0.083deg_P1M-m_*.nc`
- **Portal:** [CMEMS Data Store](https://data.marine.copernicus.eu)

### 2. CMEMS Global Ocean Biogeochemistry Hindcast (0.25°)
- **Product:** `GLOBAL_MULTIYEAR_BGC_001_029`
- **Variables:** `chl` (Chlorophyll-a), `nppv` (Net Primary Production)
- **Resolution:** 0.25° monthly, 5 depth levels
- **Time range:** 2015-01–2024-12
- **Region:** 10°–20°N, 114°–120°E
- **Filename:** `cmems_mod_glo_bgc_my_0.25deg_P1M-m_*.nc`
- **Portal:** [CMEMS Data Store](https://data.marine.copernicus.eu)

### 3. AIS Fishing Effort (Global Fishing Watch v3.0)
- **Product:** Monthly apparent fishing hours by flag state and gear type
- **Format:** `fleet-monthly-csvs-10-v3-{year}.zip` (0.1° resolution, **not** the daily 0.01° files)
- **Years required:** 2015–2024 (one zip per year)
- **Portal:** [Global Fishing Watch Data Portal](https://globalfishingwatch.org/datasets-and-code/) or [Zenodo](https://zenodo.org)
- **After downloading:** Extract each zip and rename/save the CSVs as `ais_effort_{year}.csv` inside your Drive folder

> **Why the monthly 0.1° files?** The daily 100th-degree (0.01°) files are 1–2 GB each per year and will exceed Colab's RAM. The monthly 0.1° files are ~1,000× smaller, load instantly, and still bin cleanly onto the 0.25° target grid.

---

## Step-by-Step Walkthrough

### Phase 1: Data Loading

**Goal:** Mount Google Drive, load raw NetCDF and CSV files, verify time alignment.

1. **Mount Google Drive** and point `DATA_DIR` to `My Drive/fishing_project/`.
2. **Load the Physics NetCDF** (`xr.open_dataset`) → confirms 4 variables, 120 time steps (2015–2024), shape `(120, 5, 121, 72)`.
3. **Load the BGC NetCDF** → confirms 2 variables, shape `(120, 5, 41, 25)`.
4. **Load AIS CSVs** using a list comprehension and `pd.concat` for all years 2015–2024.
5. **Time range verification** — asserts all three datasets span Jan 2015 → Dec 2024.
6. Saves a `data_summary.json` to Drive as a lightweight checkpoint.

---

### Phase 2: Preprocessing

**Goal:** Align all data onto a single 0.25° grid, fill gaps, normalize, and save a clean feature cube.

**Step 1 — Load and subset spatially:**
- Physics is loaded and sliced to the **nearest surface depth (~0.49 m)**, then cut to `lat 10°–20°N, lon 114°–120°E`.
- BGC is depth-averaged across the upper 5 levels (~0–5 m) and cut to the same bounding box.

**Step 2 — Monthly aggregation:**
- Both datasets are already monthly. The `.resample(time='1M').mean()` call is a safety step to ensure uniform monthly timestamps.
- Result: `(120, 121, 72)` for physics and `(120, 41, 25)` for BGC.

**Step 3 — Regrid physics to 0.25°:**
- The BGC grid (`41 × 25` at 0.25°) is the **target grid** for everything.
- Physics (originally 0.083°, higher resolution) is bilinearly interpolated down to the BGC grid using `xr.DataArray.interp()`.
- Result: both datasets now share shape `(120, 41, 25)`.

**Step 4 — Alignment verification:**
- `np.allclose` assertions confirm latitude, longitude, and time coordinates are identical between the two regridded datasets.

**Step 5 — Gap filling:**
- Any remaining `NaN` values (e.g., land cells or data gaps) are filled along the time dimension using linear interpolation with `interpolate_na(..., fill_value='extrapolate')`.

**Step 6 — Extract and normalize:**
- Six variables are extracted and individually **min-max normalized** to `[0, 1]`:

  | Variable | Source | Physical meaning |
  |----------|--------|-----------------|
  | `sst`  | Physics (`thetao`) | Sea surface temperature |
  | `ssh`  | Physics (`zos`)    | Sea surface height (anomaly) |
  | `uo`   | Physics (`uo`)     | Eastward sea water velocity |
  | `vo`   | Physics (`vo`)     | Northward sea water velocity |
  | `chl`  | BGC (`chl`)        | Chlorophyll-a concentration |
  | `nppv` | BGC (`nppv`)       | Net primary production |

**Step 7 — Process AIS data (label generation):**
- AIS CSVs are loaded, then each position's `lat`/`lon` is binned onto the 0.25° grid using `pd.cut`.
- Fishing effort is grouped by `(month, lat_bin, lon_bin)` and summed into `fishing_hours`.
- The resulting gridded table is saved as `ais_gridded.csv`.

**Step 8 — Save preprocessed features:**
- All 6 normalized arrays are merged into a single `xr.Dataset` and saved as `preprocessed_features.nc`.
- Final shape: `(120, 41, 25)` per variable, 6 variables.

---

### Phase 3: Model Training

**Goal:** Build sliding-window sequences and train a two-layer ConvLSTM2D model.

**Sequence creation:**
- A sliding window converts the `(120, 41, 25, 6)` feature cube into overlapping sequences.
- `SEQ_LEN = 3` → each input sample is 3 consecutive monthly maps.
- `PRED_LEN = 1` → each label is the immediately following month's fishing grid.
- Result: `X_seq` shape `(117, 3, 41, 25, 6)` and `y_seq` shape `(117, 1, 41, 25)`.

**Train/Val/Test split (chronological):**

| Split | Samples | Approx. dates |
|-------|---------|---------------|
| Train (70%) | 81 | ~Jan 2015 – ~Apr 2022 |
| Val (15%)   | 17 | ~May 2022 – ~Sep 2023 |
| Test (15%)  | 19 | ~Oct 2023 – ~Dec 2024 |

**Label reshaping:**
- `y_seq` is squeezed from `(N, 1, 41, 25)` → `(N, 41, 25, 1)` to match the Conv2D output shape.

**Model architecture:** *(See next section for diagram)*

**Callbacks:**
- `EarlyStopping(patience=10)` — stops training if validation loss stops improving.
- `ModelCheckpoint` — saves the best weights to `best_model.h5`.

**Training:**
- Up to **50 epochs**, batch size **8**, optimizer **Adam**, loss **binary cross-entropy**.

**Saved outputs:**
- `convlstm_model.h5` — final trained model
- `training_history.json` — epoch-by-epoch loss/MAE
- `X_test.npy`, `y_test.npy` — test set arrays for evaluation

---

### Phase 4: Model Evaluation

**Goal:** Quantify model performance on the held-out test set using five metrics.

The model is loaded from `convlstm_model.h5` and run on the 19 test samples. Land/coastal `NaN` values in inputs are replaced with `0.0` before prediction.

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — overall magnitude of prediction errors |
| **MAE** | Mean Absolute Error — average absolute deviation per cell |
| **F1 Score** | Binary classification score at threshold 0.5 (fishing / not fishing) |
| **SSI** | Structural Similarity Index — compares local luminance, contrast, and structure |
| **Wasserstein Distance** | Earth Mover's Distance between predicted and observed spatial distributions |

Results are printed as a table and saved to `evaluation_results.csv`.

---

### Phase 5: Visualization

**Goal:** Produce publication-quality plots comparing predictions to observations.

Four plots are generated and saved at 300 DPI:

1. **`obs_vs_pred.png`** — Side-by-side grid showing 6 test months: observed fishing effort (top row) vs. model predictions (bottom row), using a custom deep-blue-to-red colormap.

2. **`error_maps.png`** — Three spatial error panels averaged across all test months:
   - *Mean Error* (bias) — signed, `RdBu` colormap
   - *MAE map* — magnitude of average error per cell
   - *RMSE map* — per-cell RMSE across the test period

3. **`temporal_performance.png`** — Line plot of RMSE per test sample, showing how prediction accuracy evolves over the 19 test months.

4. **`training_curves.png`** — Dual-panel training history (Loss and MAE) for both train and validation sets, loaded from `training_history.json`.

---

## Model Architecture

```
Input: (batch, 3 timesteps, 41 lat, 25 lon, 6 channels)
         │
         ▼
  ConvLSTM2D(filters=64, kernel=3×3, padding='same', return_sequences=True)
         │   161,536 params — extracts spatiotemporal patterns across all 3 timesteps
         ▼
  BatchNormalization + Dropout(0.2)
         │
         ▼
  ConvLSTM2D(filters=32, kernel=3×3, padding='same', return_sequences=False)
         │   110,720 params — collapses temporal dimension, outputs final spatial state
         ▼
  BatchNormalization + Dropout(0.2)
         │
         ▼
  Conv2D(filters=1, kernel=1×1, activation='sigmoid')
         │   33 params — projects to single-channel fishing probability
         ▼
Output: (batch, 41, 25, 1)   ← probability map in [0, 1]

Total trainable parameters: ~272,481
```

---

## Output Files

All outputs are written to `My Drive/fishing_project/`:

| File | Description |
|------|-------------|
| `data_summary.json` | Dataset shape/date-range summary from Phase 1 |
| `preprocessed_features.nc` | Normalized 6-channel feature cube (120 × 41 × 25) |
| `ais_gridded.csv` | AIS fishing hours binned to 0.25° monthly grid |
| `best_model.h5` | Checkpoint of best validation loss during training |
| `convlstm_model.h5` | Final saved model (all epochs) |
| `training_history.json` | Epoch-wise loss and MAE for train/val |
| `X_test.npy` | Test input sequences `(19, 3, 41, 25, 6)` |
| `y_test.npy` | Test labels `(19, 41, 25, 1)` |
| `predictions.npy` | Model predictions on test set `(19, 41, 25, 1)` |
| `evaluation_results.csv` | RMSE, MAE, F1, SSI, Wasserstein scores |
| `obs_vs_pred.png` | Side-by-side observed vs. predicted maps |
| `error_maps.png` | Spatial mean error, MAE, and RMSE maps |
| `temporal_performance.png` | RMSE over test months |
| `training_curves.png` | Training and validation loss/MAE curves |

---

## Known Issues & Notes

- **AIS year range in code:** The current code loads `ais_effort_{year}.csv` for `range(2020, 2025)` (only 5 years). To use the full 2015–2024 range matching the oceanographic data, update this to `range(2015, 2025)`.
- **NaN training label:** The `y_array` (AIS target) is initialized as all zeros. Until real AIS data is loaded and binned, the model trains on a zero-valued label, which explains the `NaN` loss and zero metrics seen in earlier runs. The AIS processing step in Phase 2 must be completed first.
- **FutureWarning (`Dataset.dims`):** Using `dict(ds.dims)` triggers a deprecation warning in newer xarray versions. Replace with `dict(ds.sizes)` to suppress it.
- **HDF5 save format:** Keras will warn that `.h5` is a legacy format. You can safely switch to `.keras` format by changing `model.save(...)` to use a `.keras` extension.
- **GPU required:** ConvLSTM2D training is very slow on CPU. Make sure Colab runtime is set to **GPU** (Runtime → Change runtime type → T4 GPU).
