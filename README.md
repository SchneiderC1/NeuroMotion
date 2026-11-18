# NeuroMotion

## EEG motor imagery → virtual movement
- Decode **Rest / Left-hand / Right-hand imagery** from non-invasive EEG and use the prediction to move a virtual object left, right or not at all.
- Data: [PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, 64 channels, 160 Hz).
- Runs used: **R04, R08, R12** (imagined left/right fist; `T0` = rest, `T1` = left, `T2` = right).

## Files

| File | Purpose |
|---|---|
| `NeuroMotion_Training_Notebook.ipynb` | Training + testing notebook (every step explained in the notebook) |
| `app.py` | Streamlit frontend |
| `requirements.txt` | Python dependencies |

## Folder layout
```
project/
├── data/                     <- PhysioNet files: data/S001/S001R04.edf, S001R08.edf, S001R12.edf, S002/...
├── outputs/                  <- written by the notebook (CV results, plots, cache)
└── artifacts/all_subject_deployment/   <- written by the notebook (weights + config for the app)
```
`app.py` looks for `data`, `outputs`, `artifacts/...` relative to itself, so keep the notebook next to `app.py`.

## Recommended Windows setup

Using Miniconda/Anaconda Prompt:

```powershell
conda create -n neuromotion python=3.11 -y
conda activate neuromotion
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name neuromotion --display-name "Python (neuromotion)"
jupyter notebook
```

Open `NeuroMotion_Training_Notebook.ipynb` and select the `Python (neuromotion)` kernel. If the installed PyTorch build cannot use the GPU, the notebook will run on CPU; for CUDA, install the PyTorch build that matches your NVIDIA driver using the official PyTorch installation selector before running the rest of the requirements.

## Data path and split

Download from PhysioNet

Point `DATA_ROOT` in the notebook to the folder containing the subject directories, for example:

```text
data/
  S001/
    S001R04.edf
    S001R08.edf
    S001R12.edf
```
## Train and test
The notebook:
1. loads all subjects exactly the way the app will (filter 4–38 Hz, window 0.5–3.5 s, µV) and auto-excludes anomalous recordings;
2. **Protocol A — regular, subject-independent training:** 5-fold cross-validation grouped by subject, a full evaluation report, then a deployment model trained with 6 held-out **demo subjects**, calibrated confidence, a validation-chosen gate threshold, and export of weights + JSON config;
3. verifies the exported files behave identically under an independent re-implementation of the app's loading code;
4. **Protocol B — all-subject training (Step 15):** trains on all subjects, holding out only a run (default `R12`, configurable via `HOLDOUT_RUNS`) of each of the `N_BEST_SUBJECTS` (default 10) best-decoding subjects. Reported next to the subject-independent score on the same epochs and exported to `artifacts/all_subjects_best_runs_holdout/`. This is a *personalised* result on good decoders, so it is optimistic by design — never quote it as general accuracy.

**Every result** (CV, demo subjects, all-subject model) is reported by one function with: accuracy, balanced accuracy, per-class and macro/weighted **precision, recall, F1**, Cohen's κ, MCC, ROC-AUC, average precision, log-loss, ECE, majority baseline; and graphs — confusion matrices (counts and normalised), per-class P/R/F1 bars, ROC and precision–recall curves, reliability diagram, confidence histograms, confidence-gate trade-off, per-fold and per-subject accuracy. Figures go to `outputs/figures/`, tables to `outputs/reports/`.

## Use the result in the app

```bash
streamlit run app.py
```
Sidebar defaults already point at `artifacts\all_subjects_best_runs_holdout\neuromotion_all_subjects_best_runs_holdout_eegnet_state_dict.pt` and `..._config.json`.
Choose one of the **held-out demo subjects** (printed at the end of the notebook and stored in the config as `held_out_demo_subjects`), click **Load model**, **Load recording**, **Play**.
The **Results** page reads `outputs/all_subject_cross_validation/` (previous results are backed up to `..._backup_<timestamp>` automatically).

## Why Braindecode EEGNet?

Braindecode provides EEG-oriented neural architectures and an `EEGClassifier` interface compatible with the familiar `.fit()` / `.predict_proba()` workflow. EEGNet is compact and explicitly designed for EEG decoding. 

## Dataset citation and references

- Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset (version 1.0.0).* PhysioNet. https://doi.org/10.13026/C28G6P
- Dataset page: https://physionet.org/content/eegmmidb/1.0.0/
- Braindecode stable docs: https://braindecode.org/stable/index.html
- Braindecode EEGNet: https://braindecode.org/stable/generated/braindecode.models.EEGNet.html
- Braindecode training tutorial: https://braindecode.org/stable/auto_examples/model_building/plot_basic_training_epochs.html
