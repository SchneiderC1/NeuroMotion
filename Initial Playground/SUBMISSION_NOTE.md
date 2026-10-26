# Submission Note

**Dataset:** PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB), v1.0.0.

**Storage:** The dataset was downloaded locally and is referenced by the notebook through the configurable `DATA_ROOT` path. The raw EDF files are excluded from Git because of their size; the notebook documents the source and expected directory structure.

**What I noticed:** The dataset contains 64-channel EEG recorded at 160 Hz. A key labeling issue is that `T1` and `T2` change meaning depending on the run. For the NeuroMotion prototype, I therefore restrict the initial task to imagined unilateral fist runs R04, R08, and R12, where T0/T1/T2 map consistently to rest/left-hand imagery/right-hand imagery.

**Core prototype:** EEG-only classification of three classes, followed later by a confidence threshold that can suppress uncertain commands.
