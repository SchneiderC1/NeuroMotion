from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    import mne
except Exception as exc:  # Keep the UI able to explain a missing EEG dependency.
    mne = None
    MNE_IMPORT_ERROR = exc
else:
    MNE_IMPORT_ERROR = None

try:
    import torch
    from braindecode.models import EEGNet
except Exception as exc:  # The app can still open in an unconfigured/preview state.
    torch = None
    EEGNet = None
    MODEL_IMPORT_ERROR = exc
else:
    MODEL_IMPORT_ERROR = None


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

st.set_page_config(
    page_title="NeuroMotion | Mind → Movement",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root { --nm-bg:#071629; --nm-panel:#0c2038; --nm-panel2:#102944; --nm-border:#1b3a59;
        --nm-muted:#91a9c3; --nm-text:#e8f1fb; --nm-blue:#4e9cf4; --nm-green:#25d39a;
        --nm-purple:#a477f5; --nm-orange:#f2a45a; }
html, body, [data-testid="stAppViewContainer"] { background:var(--nm-bg); color:var(--nm-text); }
[data-testid="stHeader"] { background:rgba(7,22,41,.92); }
[data-testid="stSidebar"] { background:#081a30; border-right:1px solid var(--nm-border); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#dce9f8; }
.block-container { padding-top:1.15rem; padding-bottom:2rem; max-width:1600px; }
h1,h2,h3,h4,p,span,label { color:var(--nm-text); }
.nm-brand { display:flex; align-items:center; gap:13px; padding:8px 0 17px; }
.nm-brand-icon { width:48px;height:48px;border-radius:15px;background:linear-gradient(145deg,#173d68,#0d2746);
 display:flex;align-items:center;justify-content:center;font-size:27px;box-shadow:0 0 28px rgba(54,142,244,.12); }
.nm-brand-title { font-size:25px;font-weight:750;letter-spacing:-.6px;line-height:1.15; }
.nm-brand-sub { color:#8db9ed;font-size:13px;letter-spacing:.3px;margin-top:3px; }
.nm-topline {display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;color:#9bb3ce;font-size:13px;}
.nm-card { background:linear-gradient(145deg,rgba(15,40,67,.98),rgba(9,29,51,.98));border:1px solid var(--nm-border);
 border-radius:14px;padding:16px 18px;box-shadow:0 8px 24px rgba(0,0,0,.13); }
.nm-card-title {font-size:16px;font-weight:650;margin:0 0 13px 0;color:#edf5ff;}
.nm-small {font-size:12px;color:var(--nm-muted);}
.nm-muted {color:var(--nm-muted);}
.nm-kpi-label {font-size:13px;color:#aac2dc;margin-bottom:6px;}
.nm-kpi-value {font-size:20px;font-weight:700;line-height:1.2;}
.nm-green {color:var(--nm-green)!important;}
.nm-blue {color:#70b5ff!important;}
.nm-purple {color:#b69aff!important;}
.nm-orange {color:#ffc17b!important;}
.nm-pill {display:inline-block;border:1px solid #2b5276;border-radius:99px;padding:4px 10px;font-size:11px;color:#bcd2e9;background:#102944;}
.nm-divider {height:1px;background:#1a3957;margin:10px 0 13px;}
.nm-pipeline {background:#0b2038;border:1px solid var(--nm-border);border-radius:13px;padding:15px 16px;margin-top:14px;}
.nm-pipeline-title {font-weight:650;margin-bottom:12px;}
.nm-stage {background:#102d4c;border:1px solid #2a5276;border-radius:10px;padding:12px 7px;text-align:center;min-height:76px;}
.nm-stage-icon {font-size:21px;margin-bottom:3px;}
.nm-stage-name {font-weight:650;font-size:12px;}
.nm-stage-sub {font-size:10px;color:#a4bad2;margin-top:3px;}
.nm-arrow {color:#6e91b7;text-align:center;font-size:22px;padding-top:18px;}
.nm-state {font-size:11px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;color:#a8c5e2;}
.nm-callout {border:1px solid #315576;background:#0b2541;border-radius:10px;padding:11px 13px;color:#d9e8f7;font-size:13px;}
.nm-warn {border-color:#8c6535;background:rgba(125,83,30,.18);}
.nm-good {border-color:#247c65;background:rgba(19,112,82,.14);}
.nm-danger {border-color:#86494f;background:rgba(120,40,47,.16);}
.stButton > button {border-radius:9px;border:1px solid #2b5276;background:#123252;color:#edf6ff;font-weight:600;}
.stButton > button:hover {border-color:#4c9ce9;background:#173d63;color:#fff;}
.stButton > button[kind="primary"] {background:linear-gradient(90deg,#158d78,#1dbd92);border:0;color:#041b20;}
[data-testid="stMetric"] {background:transparent;}
[data-testid="stMetricValue"] {color:#f0f6ff;}
[data-testid="stMetricLabel"] {color:#a8bfd8;}
[data-testid="stDataFrame"] {border:1px solid var(--nm-border);border-radius:10px;overflow:hidden;}
[data-testid="stProgressBar"] > div > div {background:linear-gradient(90deg,#198e80,#27d2a0);}
hr {border-color:#1a3957;}
footer {visibility:hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Session state ----------
def init_state() -> None:
    defaults = {
        "page": "Home",
        "model_bundle": None,
        "model_error": None,
        "recording": None,
        "recording_error": None,
        "is_playing": False,
        "next_index": 0,
        "current_result": None,
        "history": [],
        "object_x": 0,
        "threshold": 0.60,
        "last_dataset_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ---------- Model and data loading ----------
@st.cache_resource(show_spinner=False)
def load_model_bundle(weights_path_str: str, config_path_str: str, device_mode: str):
    if torch is None or EEGNet is None:
        raise RuntimeError(f"PyTorch/Braindecode import failed: {MODEL_IMPORT_ERROR}")
    weights_path = Path(weights_path_str).expanduser().resolve()
    config_path = Path(config_path_str).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Model configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    if device_mode == "CPU" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Match the EEGNet constructor used in the Braindecode training notebook.
    model = EEGNet(
        n_chans=int(config["n_channels"]),
        n_outputs=int(config.get("n_outputs", 3)),
        n_times=int(config["n_times"]),
        sfreq=float(config.get("sampling_rate_hz", 160.0)),
    )
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, config, device, str(weights_path), str(config_path)


def locate_edf(data_root: str, subject: str, run: str) -> Path:
    root = Path(data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    filename = f"{subject}{run}.edf"
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} anywhere under {root.resolve()}")
    if len(matches) > 1:
        # Deterministic choice; make duplicates visible in the UI.
        st.warning(f"Found {len(matches)} copies of {filename}; using {matches[0]}.")
    return matches[0]


def load_recording(data_root: str, subject: str, run: str, config: dict[str, Any]) -> dict[str, Any]:
    if mne is None:
        raise RuntimeError(f"MNE import failed: {MNE_IMPORT_ERROR}")
    edf_path = locate_edf(data_root, subject, run)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    expected_sfreq = float(config.get("sampling_rate_hz", 160.0))
    if not np.isclose(sfreq, expected_sfreq):
        raise ValueError(f"Recording is {sfreq:g} Hz; model expects {expected_sfreq:g} Hz.")

    raw.pick(picks="eeg")
    expected_channels = list(config.get("channel_names_in_order", []))
    if not expected_channels:
        raise ValueError("Model config is missing channel_names_in_order. Retrain/export the model using the updated notebook cell.")
    missing = sorted(set(expected_channels) - set(raw.ch_names))
    extra = sorted(set(raw.ch_names) - set(expected_channels))
    if missing or extra:
        raise ValueError(
            "EEG channel set does not match the model. "
            f"Missing: {missing[:8]}; extra: {extra[:8]}. "
            "The app will not silently drop or invent channels."
        )
    raw.pick(picks=expected_channels)  # Enforce training channel order.

    preprocessing = config.get("preprocessing", {})
    bandpass = preprocessing.get("bandpass_hz", config.get("bandpass_hz", [4.0, 38.0]))
    low_cut, high_cut = float(bandpass[0]), float(bandpass[1])
    raw.filter(l_freq=low_cut, h_freq=high_cut, picks="eeg", verbose="ERROR")

    event_to_label = config.get("annotation_to_label_id", {"T0": 0, "T1": 1, "T2": 2})
    class_names = config.get("class_names", ["Rest", "Left Hand", "Right Hand"])
    events, found_event_id = mne.events_from_annotations(
        raw, regexp=r"^(T0|T1|T2)$", verbose="ERROR"
    )
    missing_events = sorted(set(event_to_label) - set(found_event_id))
    if missing_events:
        raise ValueError(f"Recording is missing required annotations: {missing_events}")
    selected_event_id = {name: found_event_id[name] for name in event_to_label}

    epoch_window = preprocessing.get("epoch_seconds_after_event", config.get("epoch_seconds_after_event", [0.5, 3.5]))
    tmin, tmax = float(epoch_window[0]), float(epoch_window[1])
    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    X_uv = epochs.get_data().astype(np.float32) * 1e6
    if len(X_uv) == 0:
        raise ValueError("No valid T0/T1/T2 epochs remained after annotation rejection.")
    if not np.isfinite(X_uv).all():
        raise ValueError("EEG epochs contain NaN or infinite values; replay is blocked.")

    code_to_label = {selected_event_id[name]: int(event_to_label[name]) for name in event_to_label}
    code_to_annotation = {selected_event_id[name]: name for name in event_to_label}
    labels = np.asarray([code_to_label[int(code)] for code in epochs.events[:, 2]], dtype=np.int64)
    onsets = (epochs.events[:, 0] - raw.first_samp) / sfreq
    meta = pd.DataFrame({
        "window_index": np.arange(len(labels)),
        "event_onset_sec": np.round(onsets, 3),
        "annotation": [code_to_annotation[int(code)] for code in epochs.events[:, 2]],
        "label_id": labels,
        "reference_class": [class_names[int(v)] for v in labels],
    })

    norm_cfg = config.get("normalization", {})
    if norm_cfg.get("method") == "per_recording_channel_zscore":
        # Statistics come from the recording being replayed (no labels used) - identical to training.
        mean = X_uv.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = X_uv.std(axis=(0, 2), keepdims=True).astype(np.float32)
    else:
        mean_values = norm_cfg.get("channel_mean_microvolts")
        std_values = norm_cfg.get("channel_std_microvolts")
        if mean_values is None or std_values is None:
            raise ValueError("Model config is missing its training normalization statistics.")
        mean = np.asarray(mean_values, dtype=np.float32).reshape(1, -1, 1)
        std = np.asarray(std_values, dtype=np.float32).reshape(1, -1, 1)
        if mean.shape[1] != X_uv.shape[1] or std.shape[1] != X_uv.shape[1]:
            raise ValueError("Normalization channel count does not match the loaded EEG recording.")
    std = np.where(std < 1e-6, 1.0, std)
    X_norm = ((X_uv - mean) / std).astype(np.float32)
    if not np.isfinite(X_norm).all():
        raise ValueError("Normalized EEG contains non-finite values; replay is blocked.")

    return {
        "path": str(edf_path.resolve()),
        "subject": subject,
        "run": run,
        "sfreq": sfreq,
        "channel_names": list(raw.ch_names),
        "X_uv": X_uv,
        "X_norm": X_norm,
        "labels": labels,
        "metadata": meta,
        "n_requested_events": int(len(events)),
        "n_retained_epochs": int(len(epochs)),
        "n_rejected_epochs": int(len(events) - len(epochs)),
        "tmin": tmin,
        "tmax": tmax,
        "class_names": class_names,
    }


def infer_window(index: int) -> dict[str, Any]:
    bundle = st.session_state.model_bundle
    recording = st.session_state.recording
    if bundle is None or recording is None:
        raise RuntimeError("Model or recording is unavailable.")
    model, config, device, *_ = bundle
    if index < 0 or index >= len(recording["X_norm"]):
        raise IndexError("Replay window index is outside the loaded recording.")
    x = torch.from_numpy(recording["X_norm"][index:index + 1]).to(device)
    with torch.no_grad():
        output = model(x)
        logits = output.reshape(output.shape[0], -1)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)
    if len(probabilities) != 3 or not np.isfinite(probabilities).all():
        raise RuntimeError(f"Model returned invalid class probabilities: {probabilities}")
    probabilities = probabilities / probabilities.sum()
    class_names = recording["class_names"]
    predicted_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_id])
    predicted_class = class_names[predicted_id]
    threshold = float(st.session_state.threshold)

    # Safety order: invalid/low-confidence predictions are rejected before class actions.
    # A valid REST class is not a movement command. Rejected predictions never move the object.
    if confidence < threshold:
        final_state = "NO COMMAND"
        reason = f"Low confidence ({confidence:.1%}) is below the {threshold:.0%} gate."
        action = "Hold position"
        command = None
    elif predicted_id == 0:
        final_state = "REST"
        reason = "REST is a valid class; no movement is requested."
        action = "No movement"
        command = None
    elif predicted_id == 1:
        final_state = "LEFT COMMAND"
        reason = "Accepted left-hand imagery prediction."
        action = "Move left"
        command = "LEFT"
    else:
        final_state = "RIGHT COMMAND"
        reason = "Accepted right-hand imagery prediction."
        action = "Move right"
        command = "RIGHT"

    row = recording["metadata"].iloc[index]
    result = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "window_index": int(index),
        "event_onset_sec": float(row["event_onset_sec"]),
        "reference_class": str(row["reference_class"]),
        "predicted_class": predicted_class,
        "predicted_id": predicted_id,
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
        "final_state": final_state,
        "reason": reason,
        "action": action,
        "command": command,
        "threshold": threshold,
    }
    if command == "LEFT":
        st.session_state.object_x = max(-3, int(st.session_state.object_x) - 1)
    elif command == "RIGHT":
        st.session_state.object_x = min(3, int(st.session_state.object_x) + 1)
    return result


def process_next_window() -> None:
    recording = st.session_state.recording
    if recording is None:
        st.session_state.is_playing = False
        return
    index = int(st.session_state.next_index)
    if index >= len(recording["X_norm"]):
        st.session_state.is_playing = False
        return
    result = infer_window(index)
    st.session_state.current_result = result
    st.session_state.history.insert(0, result.copy())
    st.session_state.history = st.session_state.history[:30]
    st.session_state.next_index = index + 1
    if st.session_state.next_index >= len(recording["X_norm"]):
        st.session_state.is_playing = False


# ---------- Visual components ----------
def render_header(page: str) -> None:
    st.markdown(
        f"""
        <div class="nm-topline">
          <div class="nm-brand">
            <div class="nm-brand-icon">🧠</div>
            <div><div class="nm-brand-title">NeuroMotion</div>
                 <div class="nm-brand-sub">Mind &nbsp;→&nbsp; Movement</div></div>
          </div>
          <div><span class="nm-pill">EEG MOTOR-IMAGERY CLASSIFICATION PROTOTYPE</span>&nbsp;&nbsp; ⚙</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{page}  /  Recorded EEG replay · Local processing · Research prototype only")


def render_top_status() -> None:
    bundle = st.session_state.model_bundle
    recording = st.session_state.recording
    result = st.session_state.current_result
    if recording is None:
        system_status, status_class = "Data unavailable", "nm-orange"
    elif bundle is None:
        system_status, status_class = "Model unavailable", "nm-orange"
    elif st.session_state.is_playing:
        system_status, status_class = "Playing replay", "nm-green"
    elif result:
        system_status, status_class = "Paused / ready", "nm-blue"
    else:
        system_status, status_class = "Ready · waiting", "nm-green"

    if result:
        pred = result["predicted_class"]
        conf = f"{result['confidence']:.1%}"
        action = result["action"]
        gate = result["final_state"]
        actual = result["reference_class"]
        actual_class = "nm-green" if actual == pred else "nm-orange"
        actual_text = f"{actual} {'✓' if actual == pred else '✗'}"
    else:
        pred, conf, action, gate = "Waiting for window", "—", "No action", "NOT RUN"
        actual_text, actual_class = "—", ""

    values = [
        ("System Status", system_status, status_class),
        ("Current Prediction", pred, "nm-green" if result and result["predicted_id"] in (1, 2) else "nm-blue"),
        ("Actual Command", actual_text, actual_class),
        ("Confidence", conf, ""),
        ("Action", action, "nm-green" if "Move" in action else ""),
        ("Gate Decision", gate, "nm-orange" if gate == "NO COMMAND" else "nm-green" if "COMMAND" in gate else "nm-blue"),
    ]
    with st.container(border=True):
        cols = st.columns([1.0, 1.2, 1.2, .75, 1.0, 1.0])
        for col, (label, value, color) in zip(cols, values):
            with col:
                st.markdown(f'<div class="nm-kpi-label">{label}</div><div class="nm-kpi-value {color}">{value}</div>', unsafe_allow_html=True)


def make_signal_figure(recording: dict[str, Any] | None, result: dict[str, Any] | None) -> go.Figure | None:
    if recording is None or result is None:
        return None
    idx = int(result["window_index"])
    signal = recording["X_uv"][idx]
    names = recording["channel_names"]
    sfreq = recording["sfreq"]
    tmin = recording["tmin"]
    times = np.arange(signal.shape[1]) / sfreq + tmin

    # Prefer central motor channels when available; otherwise use the first six ordered channels.
    preferred = []
    for target in ("C3", "Cz", "C4", "FC3", "FCz", "FC4"):
        match = next((i for i, name in enumerate(names) if name.rstrip(".").upper() == target), None)
        if match is not None and match not in preferred:
            preferred.append(match)
    indices = preferred[:6]
    if len(indices) < 6:
        indices.extend(i for i in range(len(names)) if i not in indices)
        indices = indices[:6]

    fig = go.Figure()
    palette = ["#55b5ff", "#f4a35d", "#2ed6a0", "#9e82ee", "#dc72ca", "#7b91ff"]
    spacing = max(float(np.nanpercentile(np.abs(signal[indices]), 90)) * 2.5, 8.0)
    for display_idx, channel_idx in enumerate(indices):
        y = signal[channel_idx] + display_idx * spacing
        fig.add_trace(go.Scatter(
            x=times, y=y, mode="lines", name=names[channel_idx].rstrip("."),
            line=dict(width=1.15, color=palette[display_idx % len(palette)]),
            hovertemplate=f"{names[channel_idx]}<br>%{{x:.2f}} s<br>%{{y:.2f}} µV<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", height=270, margin=dict(l=8, r=8, t=5, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(4,17,31,.25)",
        font=dict(color="#cbdcf0", size=11), legend=dict(orientation="v", x=1.01, y=1),
        xaxis=dict(title="Time (s)", gridcolor="#1a3855", zeroline=False),
        yaxis=dict(title="Amplitude (µV, offset for display)", gridcolor="#1a3855", zeroline=False, showticklabels=False),
    )
    return fig


def render_virtual_environment(result: dict[str, Any] | None) -> None:
    x = int(st.session_state.object_x)
    final_state = result["final_state"] if result else "WAITING"
    action = result["action"] if result else "Load a recording and step through a window."
    if result and result["command"] == "LEFT":
        direction = "←"
        accent = "#2ed6a0"
    elif result and result["command"] == "RIGHT":
        direction = "→"
        accent = "#55b5ff"
    elif result and final_state == "REST":
        direction = "—"
        accent = "#a78af6"
    else:
        direction = "⏸"
        accent = "#f1ad63"
    offset = x * 64
    html = f"""
    <div class="scene-wrap">
      <div class="scene-sky"></div><div class="scene-horizon"></div><div class="scene-floor"></div>
      <div class="scene-grid"></div>
      <div class="scene-arrow scene-left" style="color:{accent}">←</div>
      <div class="scene-arrow scene-right" style="color:{accent}">→</div>
      <div class="scene-cube" style="transform:translateX(calc(-50% + {offset}px))"></div>
      <div class="scene-bottom"><b>{final_state}</b><span>{action}</span><small>Recorded EEG replay · virtual object only</small></div>
    </div>
    <style>
      .scene-wrap{{position:relative;height:356px;border-radius:12px;overflow:hidden;background:#9db4c7;border:1px solid #254767;}}
      .scene-sky{{position:absolute;inset:0 0 52% 0;background:linear-gradient(180deg,#48a0da,#b7d7e9);}}
      .scene-horizon{{position:absolute;left:0;right:0;top:48%;height:12%;background:linear-gradient(180deg,#c5d0d7,#8c9ea9);}}
      .scene-floor{{position:absolute;inset:52% 0 0;background:linear-gradient(180deg,#a9b8c1,#7e909e);}}
      .scene-grid{{position:absolute;inset:52% 0 0;opacity:.38;background-image:linear-gradient(rgba(35,64,88,.5) 1px,transparent 1px),linear-gradient(90deg,rgba(35,64,88,.45) 1px,transparent 1px);background-size:48px 34px;transform:perspective(160px) rotateX(20deg);transform-origin:top;}}
      .scene-cube{{position:absolute;left:50%;top:61%;width:82px;height:82px;background:linear-gradient(135deg,#3b94ff,#1456b8);border:1px solid #79b8ff;box-shadow:0 14px 28px rgba(4,22,43,.32);border-radius:3px;transition:transform .35s ease;}}
      .scene-cube:before{{content:'';position:absolute;left:0;right:0;top:-15px;height:15px;background:#6cb3ff;transform:skewX(-42deg);transform-origin:bottom;}}
      .scene-cube:after{{content:'';position:absolute;right:-15px;top:-8px;width:15px;height:90px;background:#17488c;transform:skewY(-42deg);transform-origin:left;}}
      .scene-arrow{{position:absolute;top:68%;font-size:42px;font-weight:800;text-shadow:0 3px 8px rgba(0,0,0,.2);}}
      .scene-left{{left:16%;}} .scene-right{{right:16%;}}
      .scene-bottom{{position:absolute;left:12px;right:12px;bottom:12px;background:rgba(4,22,42,.88);padding:11px 14px;border-radius:9px;color:#f1f7ff;display:grid;grid-template-columns:1fr auto;gap:3px 8px;font-size:13px;}}
      .scene-bottom span{{text-align:right;color:#bcd2e8;}} .scene-bottom small{{grid-column:1 / -1;color:#91abc5;font-size:10px;}}
    </style>
    """
    components.html(html, height=365, scrolling=False)


def actual_command_info(reference_class: str) -> tuple[str, str]:
    """Map the recorded label (dataset cue T0/T1/T2) to a command name and the action it requests."""
    table = {"Rest": ("REST", "No movement"), "Left Hand": ("LEFT", "Move left"), "Right Hand": ("RIGHT", "Move right")}
    return table.get(str(reference_class), (str(reference_class).upper(), "—"))


def render_actual_vs_predicted(result: dict[str, Any] | None) -> None:
    if not result:
        return
    actual_cmd, actual_action = actual_command_info(result["reference_class"])
    ok = result["reference_class"] == result["predicted_class"]
    css = "nm-good" if ok else "nm-danger"
    verdict = "✓ Prediction matches the actual command" if ok else "✗ Prediction differs from the actual command"
    st.markdown(
        f'<div class="nm-callout {css}"><b>Actual command</b> (recorded cue): {result["reference_class"]} → {actual_action}<br>'
        f'<b>Predicted:</b> {result["predicted_class"]} ({result["confidence"]:.0%}) · gate: {result["final_state"]}<br>{verdict}</div>',
        unsafe_allow_html=True,
    )


def render_probabilities(result: dict[str, Any] | None) -> None:
    st.markdown('<div class="nm-card-title">Classification Probabilities</div>', unsafe_allow_html=True)
    class_names = ["Left Hand Imagery", "Right Hand Imagery", "Rest"]
    colors = ["#29c99a", "#4b98ee", "#9b75ed"]
    if result is None:
        st.info("No prediction yet. Load the model and a recording, then select Step Window or Play.")
        return
    probs = result["probabilities"]
    # Class order in the model is Rest, Left, Right; present the two movements first like the mock-up.
    order = [(1, class_names[0], colors[0]), (2, class_names[1], colors[1]), (0, class_names[2], colors[2])]
    for class_id, label, color in order:
        value = float(probs[class_id])
        st.markdown(f"<div style='display:flex;justify-content:space-between;margin:7px 0 3px'><span>{label}</span><b>{value:.0%}</b></div>", unsafe_allow_html=True)
        st.progress(value)


def render_recent_predictions() -> None:
    st.markdown('<div class="nm-card-title">Recent Predictions</div>', unsafe_allow_html=True)
    history = st.session_state.history
    if not history:
        st.caption("No replay windows processed in this session.")
        return
    frame = pd.DataFrame(history[:8])
    frame["match"] = np.where(frame["reference_class"] == frame["predicted_class"], "✓", "✗")
    frame = frame[["timestamp", "window_index", "reference_class", "predicted_class", "match", "confidence", "final_state"]].copy()
    frame["confidence"] = frame["confidence"].map(lambda v: f"{v:.0%}")
    frame.columns = ["Time", "Window", "Actual", "Prediction", "Match", "Confidence", "Gate decision"]
    st.dataframe(frame, hide_index=True, use_container_width=True, height=235)


def render_pipeline() -> None:
    st.markdown(
        """
        <div class="nm-pipeline">
          <div class="nm-pipeline-title">System Pipeline</div>
          <div style="display:grid;grid-template-columns:1fr 22px 1fr 22px 1fr 22px 1fr 22px 1fr 22px 1fr;gap:5px;align-items:stretch">
            <div class="nm-stage"><div class="nm-stage-icon">〰</div><div class="nm-stage-name">EEG Data</div><div class="nm-stage-sub">(recorded EDF)</div></div><div class="nm-arrow">›</div>
            <div class="nm-stage"><div class="nm-stage-icon">⚙</div><div class="nm-stage-name">Preprocessing</div><div class="nm-stage-sub">(filter, epoch, normalize)</div></div><div class="nm-arrow">›</div>
            <div class="nm-stage"><div class="nm-stage-icon">◉</div><div class="nm-stage-name">Feature Learning</div><div class="nm-stage-sub">(EEGNet filters)</div></div><div class="nm-arrow">›</div>
            <div class="nm-stage"><div class="nm-stage-icon">🧠</div><div class="nm-stage-name">AI Classifier</div><div class="nm-stage-sub">(Braindecode EEGNet)</div></div><div class="nm-arrow">›</div>
            <div class="nm-stage"><div class="nm-stage-icon">⇄</div><div class="nm-stage-name">Confidence Gate</div><div class="nm-stage-sub">(REST / reject / move)</div></div><div class="nm-arrow">›</div>
            <div class="nm-stage"><div class="nm-stage-icon">▣</div><div class="nm-stage-name">Virtual Action</div><div class="nm-stage-sub">(move object)</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_control_bar() -> None:
    recording = st.session_state.recording
    if recording is None:
        st.markdown('<div class="nm-callout nm-warn">Load a compatible model and one EDF recording from the sidebar to enable recorded replay.</div>', unsafe_allow_html=True)
        return
    total = len(recording["X_norm"])
    next_idx = min(int(st.session_state.next_index), max(total - 1, 0))
    st.markdown('<div class="nm-card-title">Recorded Replay Controls</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.4])
    with c1:
        if st.button("▶ Play", key="play_btn", type="primary", disabled=st.session_state.is_playing or total == 0, use_container_width=True):
            if st.session_state.next_index >= total:
                st.session_state.next_index = 0
                st.session_state.object_x = 0
                st.session_state.current_result = None
                st.session_state.history = []
            st.session_state.is_playing = True
            st.rerun()
    with c2:
        if st.button("Ⅱ Pause", key="pause_btn", disabled=not st.session_state.is_playing, use_container_width=True):
            st.session_state.is_playing = False
            st.rerun()
    with c3:
        if st.button("Step Window", key="step_btn", disabled=st.session_state.is_playing or total == 0, use_container_width=True):
            if st.session_state.next_index < total:
                try:
                    process_next_window()
                except Exception as exc:
                    st.session_state.is_playing = False
                    st.session_state.recording_error = f"Inference stopped safely: {exc}"
                    st.error(st.session_state.recording_error)
            st.rerun()
    with c4:
        if st.button("↺ Restart", key="restart_btn", disabled=total == 0, use_container_width=True):
            st.session_state.is_playing = False
            st.session_state.next_index = 0
            st.session_state.current_result = None
            st.session_state.history = []
            st.session_state.object_x = 0
            st.rerun()
    with c5:
        speed = st.selectbox("Replay speed", [0.5, 1.0, 2.0], index=1, format_func=lambda x: f"{x:g}×", key="replay_speed")
    progress = (int(st.session_state.next_index) / max(total, 1))
    st.progress(min(progress, 1.0), text=f"Window {min(st.session_state.next_index, total)} / {total} · {speed:g}× replay")


def render_dashboard(page: str) -> None:
    render_header(page)
    render_control_bar()
    interval = (1.0 / float(st.session_state.get("replay_speed", 1.0))) if st.session_state.is_playing else None

    @st.fragment(run_every=interval)
    def dashboard_fragment():
        if st.session_state.is_playing:
            try:
                process_next_window()
            except Exception as exc:
                st.session_state.is_playing = False
                st.session_state.recording_error = f"Inference stopped safely: {exc}"
                st.error(st.session_state.recording_error)
                st.rerun()
            if not st.session_state.is_playing and st.session_state.recording is not None:
                # Rebuild the page to remove the fragment timer when replay reaches its final window.
                st.rerun()
        render_top_status()
        result = st.session_state.current_result
        recording = st.session_state.recording
        left, right = st.columns([1.06, .84], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown('<div class="nm-card-title">EEG Signal (Processed)</div>', unsafe_allow_html=True)
                fig = make_signal_figure(recording, result)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    if result:
                        st.caption(f"{recording['subject']}{recording['run']} · window {result['window_index']} · event onset {result['event_onset_sec']:.3f}s · {recording['sfreq']:.0f} Hz")
                else:
                    st.info("The processed EEG waveform will appear here after you load a recording and process a window.")
            lower_left, lower_right = st.columns([1.08, .92], gap="small")
            with lower_left:
                with st.container(border=True):
                    render_probabilities(result)
            with lower_right:
                with st.container(border=True):
                    render_recent_predictions()
        with right:
            with st.container(border=True):
                st.markdown('<div class="nm-card-title">Virtual Environment</div>', unsafe_allow_html=True)
                render_virtual_environment(result)
                render_actual_vs_predicted(result)
                if result:
                    if result["final_state"] == "NO COMMAND":
                        st.markdown(f'<div class="nm-callout nm-warn"><b>NO COMMAND — LOW CONFIDENCE</b><br>{result["reason"]}<br>The virtual object remains stationary.</div>', unsafe_allow_html=True)
                    elif result["final_state"] == "REST":
                        st.markdown('<div class="nm-callout">REST is a valid class. No movement is executed.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="nm-callout nm-good">{result["reason"]}</div>', unsafe_allow_html=True)
                elif recording is None:
                    st.markdown('<div class="nm-callout nm-warn">DATA UNAVAILABLE — no recording is loaded.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="nm-callout">READY — waiting for the first replay window.</div>', unsafe_allow_html=True)
        render_pipeline()

    dashboard_fragment()


def list_subjects(root_text: str) -> list[str]:
    root = Path(root_text).expanduser()
    if not root.exists():
        return []
    found = set()
    for path in root.rglob("S[0-9][0-9][0-9]R04.edf"):
        found.add(path.name[:4])
    return sorted(found)


def render_results_page() -> None:
    render_header("Results")
    st.markdown("## Experiment results")
    st.caption("The dashboard reads locally saved outputs. It does not generate or infer metrics when result files are missing.")
    output_root_text = st.text_input("Results folder", value="../outputs")
    output_root = Path(output_root_text).expanduser()
    cv_dir = output_root / "all_subject_cross_validation"
    summary_path = cv_dir / "cross_validation_summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        cols = st.columns(4)
        cols[0].metric("Eligible subjects", summary.get("eligible_subjects", "—"))
        cols[1].metric("Held-out epochs", summary.get("total_tested_epochs_out_of_fold", "—"))
        cols[2].metric("OOF accuracy", f"{summary.get('accuracy', float('nan')):.1%}" if "accuracy" in summary else "—")
        cols[3].metric("OOF macro-F1", f"{summary.get('macro_f1', float('nan')):.3f}" if "macro_f1" in summary else "—")
        fold_path = cv_dir / "fold_metrics.csv"
        if fold_path.exists():
            st.markdown("### Fold-by-fold evaluation")
            st.dataframe(pd.read_csv(fold_path), hide_index=True, use_container_width=True)
        gate_path = cv_dir / "oof_gate_metrics.csv"
        if gate_path.exists():
            st.markdown("### Confidence-gate diagnostics")
            st.dataframe(pd.read_csv(gate_path), hide_index=True, use_container_width=True)
        subject_path = cv_dir / "oof_subject_metrics.csv"
        if subject_path.exists():
            st.markdown("### Subject-level performance")
            subject_df = pd.read_csv(subject_path)
            st.dataframe(subject_df, hide_index=True, use_container_width=True, height=300)
        with st.expander("Full cross-validation summary"):
            st.json(summary)
    else:
        st.info(f"No all-subject cross-validation summary found at {summary_path}. Run the all-subject notebook cells first.")
    if st.session_state.history:
        st.markdown("### This replay session")
        session_df = pd.DataFrame(st.session_state.history)
        st.download_button(
            "Download replay log (CSV)",
            data=session_df.to_csv(index=False).encode("utf-8"),
            file_name="neuromotion_replay_log.csv",
            mime="text/csv",
        )


def render_settings_page() -> None:
    render_header("Settings")
    st.markdown("## Runtime settings")
    st.markdown(
        "The confidence threshold is an operational gate, not a calibrated probability or guarantee of correctness. "
        "Use validation data for threshold selection; do not tune against the held-out test folds."
    )
    current = float(st.session_state.threshold)
    new_threshold = st.slider("Minimum confidence to accept a movement class", 0.50, 0.95, current, 0.01, key="settings_threshold")
    st.session_state.threshold = float(new_threshold)
    st.caption("Below this threshold the system produces NO COMMAND and the virtual object does not move.")

    bundle = st.session_state.model_bundle
    if bundle:
        model, config, device, weights_path, config_path = bundle
        st.markdown("### Loaded model")
        st.write({
            "Model": config.get("model", config.get("model_class", "Braindecode EEGNet")),
            "Device": str(device),
            "Training scope": config.get("training_scope", config.get("subject", "All-subject model")),
            "Training subjects": config.get("n_training_subjects", "not recorded"),
            "Channels": config.get("n_channels"),
            "Window samples": config.get("n_times"),
            "Weights": weights_path,
            "Config": config_path,
        })
    else:
        st.info("No model is loaded. Use the sidebar to select the model weights and JSON handoff configuration.")

    recording = st.session_state.recording
    if recording:
        st.markdown("### Loaded recording")
        st.write({
            "Recording": f"{recording['subject']}{recording['run']}",
            "EDF": recording["path"],
            "Retained epochs": recording["n_retained_epochs"],
            "Rejected epochs": recording["n_rejected_epochs"],
            "Channels": len(recording["channel_names"]),
            "Sampling rate": recording["sfreq"],
        })

    st.markdown("### State semantics")
    st.markdown(
        "- **REST:** valid predicted class; no movement.\n"
        "- **NO COMMAND — LOW CONFIDENCE:** prediction rejected by the gate; no movement and the user may resume replay.\n"
        "- **PAUSED:** replay control state; it is not an EEG class.\n"
        "- **DATA UNAVAILABLE:** no compatible model/recording or invalid signal; no prediction is made."
    )


# ---------- Sidebar and app routing ----------
with st.sidebar:
    st.markdown(
        '<div class="nm-brand"><div class="nm-brand-icon">🧠</div><div><div class="nm-brand-title">NeuroMotion</div><div class="nm-brand-sub">Mind → Movement</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", ["Home", "Live Demo", "Results", "Settings"], label_visibility="collapsed", key="page_nav")
    st.markdown("---")
    st.markdown("#### Local data and model")
    data_root = st.text_input("EEG dataset folder", value="data", help="Folder containing S001/S001R04.edf, S002/..., etc.")
    subjects = list_subjects(data_root)
    if subjects:
        default_subject_index = subjects.index("S001") if "S001" in subjects else 0
        subject = st.selectbox("Subject", subjects, index=default_subject_index)
    else:
        subject = st.text_input("Subject ID", value="S001", help="Example: S001")
    run = st.selectbox("Run", ["R04", "R08", "R12"], help="Unilateral imagined-fist runs: T0 Rest, T1 Left, T2 Right.")
    st.markdown('<div class="nm-small">Model weights and config exported by the notebook</div>', unsafe_allow_html=True)
    weights_path = st.text_input("Weights (.pt)", value=r"artifacts\all_subjects_best_runs_holdout\neuromotion_all_subjects_best_runs_holdout_eegnet_state_dict.pt")
    config_path = st.text_input("Model config (.json)", value=r"artifacts\all_subjects_best_runs_holdout\neuromotion_all_subjects_best_runs_holdout_eegnet_config.json")
    device_mode = st.selectbox("Inference device", ["Auto", "CPU"], index=0)
    if st.button("Load model", use_container_width=True, type="primary"):
        try:
            load_model_bundle.clear()
            bundle = load_model_bundle(weights_path, config_path, device_mode)
            config = bundle[1]
            if "subject" in config and str(config["subject"]) != str(subject):
                raise ValueError(
                    f"This artifact is calibrated for {config['subject']}, but {subject} is selected. "
                    "Use the matching subject or an all-subject deployment artifact."
                )
            st.session_state.model_bundle = bundle
            st.session_state.model_error = None
            st.session_state.threshold = float(config.get("confidence_threshold", config.get("active_threshold_candidate", 0.60)))
            st.session_state.recording = None
            st.session_state.current_result = None
            st.session_state.next_index = 0
            st.session_state.history = []
            st.session_state.object_x = 0
            st.success(f"Model loaded on {bundle[2]}.")
        except Exception as exc:
            st.session_state.model_error = str(exc)
            st.session_state.model_bundle = None
            st.error(f"Model could not be loaded: {exc}")

    if st.button("Load recording", use_container_width=True, disabled=st.session_state.model_bundle is None):
        try:
            cfg = st.session_state.model_bundle[1]
            recording = load_recording(data_root, subject, run, cfg)
            st.session_state.recording = recording
            st.session_state.recording_error = None
            st.session_state.is_playing = False
            st.session_state.next_index = 0
            st.session_state.current_result = None
            st.session_state.history = []
            st.session_state.object_x = 0
            st.session_state.last_dataset_key = f"{subject}-{run}"
            st.success(f"Loaded {subject}{run}: {recording['n_retained_epochs']} windows retained.")
        except Exception as exc:
            st.session_state.recording_error = str(exc)
            st.session_state.recording = None
            st.error(f"Recording could not be loaded: {exc}")

    if st.session_state.model_error:
        st.caption(f"Model issue: {st.session_state.model_error}")
    if st.session_state.recording_error:
        st.caption(f"Recording issue: {st.session_state.recording_error}")
    st.markdown("---")
    st.markdown('<div class="nm-small">Local recorded EEG only · No live headset · Not for medical use</div>', unsafe_allow_html=True)


if page in ("Home", "Live Demo"):
    render_dashboard("Home" if page == "Home" else "Live Demo")
elif page == "Results":
    render_results_page()
else:
    render_settings_page()
