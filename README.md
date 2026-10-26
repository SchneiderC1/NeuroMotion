# NeuroMotion Milestone 4 — First Model Exploration

This folder contains a reproducible first model experiment using **Braindecode EEGNet** for the NeuroMotion project.

## Contents

- `training_experiments.ipynb` — full workflow: data loading, run-level split, preprocessing, training, validation metrics, failure inspection, confidence-threshold comparison, latency check, and saved-model handoff.
- `requirements.txt` — Python dependencies.
- `SUBMISSION_NOTE.md` — assignment summary and the important local-run limitation.
- `.gitignore` — excludes raw EEG data and local outputs/checkpoints until you choose what to commit.

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

Open `training_experiments.ipynb` and select the `Python (neuromotion)` kernel. If the installed PyTorch build cannot use the GPU, the notebook will run on CPU; for CUDA, install the PyTorch build that matches your NVIDIA driver using the official PyTorch installation selector before running the rest of the requirements.

## Data path and split

Point `DATA_ROOT` in the notebook to the folder containing the subject directories, for example:

```text
data/
  S001/
    S001R04.edf
    S001R08.edf
    S001R12.edf
```

This warm-up uses only `S001R04.edf` for training and `S001R08.edf` for validation. `S001R12.edf` is explicitly reserved and not loaded. Each recording is split as a whole, so windows from one run are not scattered across train/validation. The notebook applies the project mapping valid for imagined unilateral fist runs: T0=Rest, T1=Left Hand imagery, T2=Right Hand imagery.

## Why Braindecode EEGNet?

Braindecode provides EEG-oriented neural architectures and an `EEGClassifier` interface compatible with the familiar `.fit()` / `.predict_proba()` workflow. EEGNet is compact and explicitly designed for EEG decoding. The alternative candidates (ShallowFBCSPNet and CSP+LDA) remain candidates for a later fair comparison, not claimed measured winners.

## Dataset citation and references

- Schalk, G. (2009). *EEG Motor Movement/Imagery Dataset (version 1.0.0).* PhysioNet. https://doi.org/10.13026/C28G6P
- Dataset page: https://physionet.org/content/eegmmidb/1.0.0/
- Braindecode stable docs: https://braindecode.org/stable/index.html
- Braindecode EEGNet: https://braindecode.org/stable/generated/braindecode.models.EEGNet.html
- Braindecode training tutorial: https://braindecode.org/stable/auto_examples/model_building/plot_basic_training_epochs.html
