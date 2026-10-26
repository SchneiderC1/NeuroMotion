# Submission note — NeuroMotion first model experiment

## Selected role and candidate

NeuroMotion's AI component classifies a short recorded EEG window into Rest, Left Hand imagery, or Right Hand imagery. The first candidate is Braindecode's EEGNet architecture, trained from scratch through Braindecode's EEGClassifier wrapper. EEGNet was selected because it is compact and EEG-oriented and the documented API supports a straightforward fit/predict workflow. ShallowFBCSPNet and CSP+LDA remain alternative candidates for later experiments.

## Data and split

Dataset: PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0, subject S001. The notebook uses R04 as training and R08 as validation, with R12 held out for the later final test. This whole-run split prevents windows from the same recording from being split across train/validation. R04/R08/R12 are the unilateral motor-imagery runs, so T0=Rest, T1=Left Hand imagery, and T2=Right Hand imagery. Preprocessing is a 4–38 Hz filter and 0.5–3.5 second epochs following event onset. Channel-wise normalization is fitted on the training windows only.

## Experiment configuration and planned comparison

| Run | Configuration | Change | Evaluation |
|---|---|---|---|
| Initial model | EEGNet, 10 epochs, batch size 8, Adam, learning rate 0.001, seed 42, confidence threshold 0.60 | First end-to-end run | Validation accuracy, balanced accuracy, macro-F1, confusion matrix, confidence-gate output rates, wrong movement command rates, and CPU/GPU latency |
| Revised gate | Same trained EEGNet weights, same validation probabilities, threshold 0.70 | One targeted change: threshold only | Same validation examples and metrics; compare false movement against accepted-output rate |
| Majority baseline | Always predict the most common class in the training run | Non-AI reference baseline | Same validation set and raw classification metrics |

