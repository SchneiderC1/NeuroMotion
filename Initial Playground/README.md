# NeuroMotion

## Dataset
PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB), version 1.0.0.

Source: https://physionet.org/content/eegmmidb/1.0.0/

DOI: 10.13026/C28G6P

License: Open Data Commons Attribution License v1.0

Retrieved/obtained: 2026-10-05

The dataset contains 64-channel EEG recorded at 160 Hz in EDF+ format. For the NeuroMotion prototype, the initial classification task uses imagined unilateral fist runs R04, R08, and R12. In those runs, T0 = rest, T1 = left-fist imagery, and T2 = right-fist imagery.

## Files
- `playground.ipynb` — data retrieval, inspection, visualization, and quality checks.
- `requirements.txt` — Python dependencies.
- `.gitignore` — keeps the large raw dataset and generated files out of Git.

## Local dataset layout
Place the downloaded PhysioNet dataset under `data/` or change `DATA_ROOT` in the notebook.

```text
data/
├── S001/
│   ├── S001R01.edf
│   ├── ...
│   └── S001R14.edf
├── S002/
└── ...
```

## Reproduction
1. Install dependencies with `pip install -r requirements.txt`.
2. Place the downloaded dataset in `data/` or set `DATA_ROOT` to its location.
3. Open `playground.ipynb`.
4. Run the notebook from top to bottom.

No credentials or external API keys are required.

## Citation
Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P
