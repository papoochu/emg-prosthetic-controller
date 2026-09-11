from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import joblib
import numpy as np
from scipy.io import loadmat

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from features import extract_features


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "prosthetic_command_rf.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "S1_A1_E2.mat"
)


# Colors

BACKGROUND = "#0F172A"
CARD = "#111827"
CARD_LIGHT = "#1E293B"
TEXT = "#F8FAFC"
MUTED_TEXT = "#94A3B8"
ACCENT = "#38BDF8"
SUCCESS = "#22C55E"
ERROR = "#EF4444"
BORDER = "#334155"


# Load trained model

model = joblib.load(MODEL_PATH)


# Prosthetic commands

command_map = {
    0: "REST",
    6: "FIST",
    7: "POINT",
    8: "OPEN HAND",
    13: "WRIST FLEXION",
    14: "WRIST EXTENSION",
}

command_emoji = {
    0: "🖐️",
    6: "✊",
    7: "☝️",
    8: "🖐️",
    13: "👇",
    14: "👆",
}


# Load EMG data

data = loadmat(DATA_PATH)

emg = data["emg"]
labels = data["restimulus"].flatten()
repetitions = data["rerepetition"].flatten()


# Controller settings

window_size = 20
step_size = 10
sampling_rate = 100


# Temporal smoothing settings

history_length = 5

probability_history = deque(
    maxlen=history_length
)


# Build demonstration stream

demo_commands = [
    6,
    7,
    8,
    13,
    14,
]

demo_emg = []
demo_labels = []

for command in demo_commands:

    indices = np.where(
        (labels == command)
        & (repetitions == 1)
    )[0]

    start = max(
        indices[0] - 100,
        0
    )

    end = min(
        indices[-1] + 100,
        len(emg)
    )

    demo_emg.append(
        emg[start:end]
    )

    demo_labels.append(
        labels[start:end]
    )

demo_emg = np.concatenate(
    demo_emg
)

demo_labels = np.concatenate(
    demo_labels
)


# Use recent model probabilities to stabilize predictions

def get_smoothed_prediction(features):

    current_probabilities = model.predict_proba(
        features
    )[0]

    probability_history.append(
        current_probabilities
    )

    history = np.array(
        probability_history
    )

    # Newer predictions receive more weight
    weights = np.arange(
        1,
        len(history) + 1,
        dtype=float
    )

    weights /= weights.sum()

    smoothed_probabilities = np.average(
        history,
        axis=0,
        weights=weights
    )

    best_index = np.argmax(
        smoothed_probabilities
    )

    prediction = model.classes_[
        best_index
    ]

    confidence = smoothed_probabilities[
        best_index
    ]

    raw_confidence = np.max(
        current_probabilities
    )

    return (
        prediction,
        confidence,
        raw_confidence
    )


# Create application window

root = tk.Tk()

root.title(
    "EMG Prosthetic Controller"
)

root.geometry(
    "1100x820"
)

root.minsize(
    950,
    720
)

root.configure(
    bg=BACKGROUND
)


# Configure ttk styling

style = ttk.Style()

style.theme_use(
    "clam"
)

style.configure(
    "Confidence.Horizontal.TProgressbar",
    troughcolor=CARD_LIGHT,
    background=ACCENT,
    bordercolor=CARD_LIGHT,
    lightcolor=ACCENT,
    darkcolor=ACCENT,
)


# Header

header = tk.Frame(
    root,
    bg=BACKGROUND
)

header.pack(
    fill="x",
    padx=35,
    pady=(25, 18)
)


title_label = tk.Label(
    header,
    text="EMG Prosthetic Controller",
    font=("Segoe UI", 24, "bold"),
    fg=TEXT,
    bg=BACKGROUND,
)

title_label.pack(
    anchor="w"
)


subtitle_label = tk.Label(
    header,
    text=(
        "Real-time prosthetic command prediction "
        "from surface EMG signals"
    ),
    font=("Segoe UI", 11),
    fg=MUTED_TEXT,
    bg=BACKGROUND,
)

subtitle_label.pack(
    anchor="w",
    pady=(3, 0)
)


# Main layout

main_frame = tk.Frame(
    root,
    bg=BACKGROUND
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=(0, 30)
)

main_frame.grid_columnconfigure(
    0,
    weight=2
)

main_frame.grid_columnconfigure(
    1,
    weight=3
)

main_frame.grid_rowconfigure(
    0,
    weight=1
)


# Prediction card

prediction_card = tk.Frame(
    main_frame,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
)

prediction_card.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 12)
)


card_heading = tk.Label(
    prediction_card,
    text="CURRENT PREDICTION",
    font=("Segoe UI", 10, "bold"),
    fg=MUTED_TEXT,
    bg=CARD,
)

card_heading.pack(
    pady=(28, 10)
)


# Large command emoji

emoji_label = tk.Label(
    prediction_card,
    text=command_emoji[0],
    font=("Segoe UI Emoji", 86),
    fg=TEXT,
    bg=CARD,
)

emoji_label.pack(
    pady=(5, 0)
)


# Predicted command

prediction_label = tk.Label(
    prediction_card,
    text="REST",
    font=("Segoe UI", 27, "bold"),
    fg=TEXT,
    bg=CARD,
)

prediction_label.pack(
    pady=(4, 4)
)


# Prediction status

status_label = tk.Label(
    prediction_card,
    text="● WAITING",
    font=("Segoe UI", 11, "bold"),
    fg=MUTED_TEXT,
    bg=CARD,
)

status_label.pack(
    pady=(4, 20)
)


# Separator

separator = tk.Frame(
    prediction_card,
    bg=BORDER,
    height=1,
)

separator.pack(
    fill="x",
    padx=30,
    pady=(0, 20)
)


# Actual movement

actual_caption = tk.Label(
    prediction_card,
    text="RECORDED MOVEMENT",
    font=("Segoe UI", 9, "bold"),
    fg=MUTED_TEXT,
    bg=CARD,
)

actual_caption.pack()


actual_label = tk.Label(
    prediction_card,
    text="REST",
    font=("Segoe UI", 16, "bold"),
    fg=TEXT,
    bg=CARD,
)

actual_label.pack(
    pady=(4, 22)
)


# Smoothed confidence

confidence_header = tk.Frame(
    prediction_card,
    bg=CARD
)

confidence_header.pack(
    fill="x",
    padx=30
)


confidence_title = tk.Label(
    confidence_header,
    text="Smoothed confidence",
    font=("Segoe UI", 10),
    fg=MUTED_TEXT,
    bg=CARD,
)

confidence_title.pack(
    side="left"
)


confidence_value = tk.Label(
    confidence_header,
    text="0.0%",
    font=("Segoe UI", 11, "bold"),
    fg=TEXT,
    bg=CARD,
)

confidence_value.pack(
    side="right"
)


confidence_bar = ttk.Progressbar(
    prediction_card,
    orient="horizontal",
    mode="determinate",
    maximum=100,
    style="Confidence.Horizontal.TProgressbar",
)

confidence_bar.pack(
    fill="x",
    padx=30,
    pady=(8, 12)
)


# Raw model confidence

raw_confidence_label = tk.Label(
    prediction_card,
    text="Current-window confidence: 0.0%",
    font=("Segoe UI", 9),
    fg=MUTED_TEXT,
    bg=CARD,
)

raw_confidence_label.pack(
    pady=(0, 20)
)


# Technical information

info_frame = tk.Frame(
    prediction_card,
    bg=CARD_LIGHT
)

info_frame.pack(
    fill="x",
    padx=30,
    pady=(0, 30)
)


info_text = tk.Label(
    info_frame,
    text=(
        "200 ms window   •   "
        "100 ms update   •   "
        "10 channels\n"
        "5-window temporal smoothing"
    ),
    font=("Segoe UI", 9),
    fg=MUTED_TEXT,
    bg=CARD_LIGHT,
)

info_text.pack(
    pady=10
)


# Right-side panel

right_panel = tk.Frame(
    main_frame,
    bg=BACKGROUND
)

right_panel.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(12, 0)
)

right_panel.grid_columnconfigure(
    0,
    weight=1
)

right_panel.grid_rowconfigure(
    0,
    weight=3
)

right_panel.grid_rowconfigure(
    1,
    weight=2
)


# EMG signal card

signal_card = tk.Frame(
    right_panel,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
)

signal_card.grid(
    row=0,
    column=0,
    sticky="nsew",
    pady=(0, 12)
)


signal_header = tk.Frame(
    signal_card,
    bg=CARD
)

signal_header.pack(
    fill="x",
    padx=22,
    pady=(20, 5)
)


signal_title = tk.Label(
    signal_header,
    text="Live EMG Signal",
    font=("Segoe UI", 14, "bold"),
    fg=TEXT,
    bg=CARD,
)

signal_title.pack(
    side="left"
)


channel_label = tk.Label(
    signal_header,
    text="CHANNEL 1",
    font=("Segoe UI", 9, "bold"),
    fg=ACCENT,
    bg=CARD,
)

channel_label.pack(
    side="right"
)


# Create EMG graph

figure = Figure(
    figsize=(6.5, 3.1),
    dpi=100,
    facecolor=CARD,
)

axis = figure.add_subplot(
    111
)

axis.set_facecolor(
    CARD
)

axis.tick_params(
    colors=MUTED_TEXT,
    labelsize=8
)

axis.spines["top"].set_visible(
    False
)

axis.spines["right"].set_visible(
    False
)

axis.spines["left"].set_color(
    BORDER
)

axis.spines["bottom"].set_color(
    BORDER
)

axis.set_xlabel(
    "200 ms window",
    color=MUTED_TEXT,
    fontsize=9
)

axis.set_ylabel(
    "Amplitude",
    color=MUTED_TEXT,
    fontsize=9
)

x_values = np.arange(
    window_size
)

initial_signal = np.zeros(
    window_size
)

emg_line, = axis.plot(
    x_values,
    initial_signal,
    linewidth=2,
)

axis.set_xlim(
    0,
    window_size - 1
)


graph_canvas = FigureCanvasTkAgg(
    figure,
    master=signal_card
)

graph_canvas.draw()

graph_canvas.get_tk_widget().configure(
    bg=CARD,
    highlightthickness=0
)

graph_canvas.get_tk_widget().pack(
    fill="both",
    expand=True,
    padx=15,
    pady=(0, 15)
)


# Prediction history card

history_card = tk.Frame(
    right_panel,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
)

history_card.grid(
    row=1,
    column=0,
    sticky="nsew",
    pady=(12, 0)
)


history_header = tk.Frame(
    history_card,
    bg=CARD
)

history_header.pack(
    fill="x",
    padx=22,
    pady=(18, 10)
)


history_title = tk.Label(
    history_header,
    text="Recent Controller Outputs",
    font=("Segoe UI", 14, "bold"),
    fg=TEXT,
    bg=CARD,
)

history_title.pack(
    side="left"
)


memory_label = tk.Label(
    history_header,
    text="TEMPORALLY SMOOTHED",
    font=("Segoe UI", 8, "bold"),
    fg=ACCENT,
    bg=CARD,
)

memory_label.pack(
    side="right"
)


history_container = tk.Frame(
    history_card,
    bg=CARD
)

history_container.pack(
    fill="both",
    expand=True,
    padx=22,
    pady=(0, 18)
)


# Current position in simulated stream

index = 0

prediction_history = []


# Update prediction history display

def update_history(
    emoji,
    predicted_name,
    confidence_percent,
    correct
):

    prediction_history.insert(
        0,
        (
            emoji,
            predicted_name,
            confidence_percent,
            correct,
        )
    )

    prediction_history[:] = (
        prediction_history[:5]
    )

    for widget in history_container.winfo_children():
        widget.destroy()

    for emoji, name, confidence, correct in prediction_history:

        row = tk.Frame(
            history_container,
            bg=CARD_LIGHT
        )

        row.pack(
            fill="x",
            pady=3
        )

        emoji_display = tk.Label(
            row,
            text=emoji,
            font=("Segoe UI Emoji", 17),
            fg=TEXT,
            bg=CARD_LIGHT,
            width=3,
        )

        emoji_display.pack(
            side="left",
            padx=(8, 2),
            pady=5
        )

        name_display = tk.Label(
            row,
            text=name,
            font=("Segoe UI", 10, "bold"),
            fg=TEXT,
            bg=CARD_LIGHT,
            anchor="w",
        )

        name_display.pack(
            side="left",
            fill="x",
            expand=True
        )

        confidence_display = tk.Label(
            row,
            text=f"{confidence:.0f}%",
            font=("Segoe UI", 10),
            fg=MUTED_TEXT,
            bg=CARD_LIGHT,
        )

        confidence_display.pack(
            side="right",
            padx=(5, 6)
        )

        result_display = tk.Label(
            row,
            text="✓" if correct else "✕",
            font=("Segoe UI", 12, "bold"),
            fg=SUCCESS if correct else ERROR,
            bg=CARD_LIGHT,
        )

        result_display.pack(
            side="right",
            padx=5
        )


# Process one EMG window and update the interface

def update_prediction():

    global index

    # Stop when the demonstration reaches the end
    if index + window_size > len(demo_emg):

        emoji_label.config(
            text="✅"
        )

        prediction_label.config(
            text="DEMO COMPLETE"
        )

        status_label.config(
            text="● COMPLETE",
            fg=SUCCESS,
        )

        actual_label.config(
            text="—"
        )

        confidence_value.config(
            text="—"
        )

        raw_confidence_label.config(
            text=""
        )

        confidence_bar["value"] = 0

        return

    # Get current 200 ms EMG window
    window = demo_emg[
        index:index + window_size
    ]

    # Extract MAV, RMS, and waveform length features
    features = extract_features(
        window
    ).reshape(
        1,
        -1
    )

    # Predict using the current window and recent probability history
    (
        prediction,
        confidence,
        raw_confidence
    ) = get_smoothed_prediction(
        features
    )

    confidence_percent = (
        confidence * 100
    )

    raw_confidence_percent = (
        raw_confidence * 100
    )

    # Determine the recorded movement at the center of the window
    true_label = demo_labels[
        index
        + window_size // 2
    ]

    predicted_name = command_map.get(
        prediction,
        "OTHER"
    )

    true_name = command_map.get(
        true_label,
        "OTHER"
    )

    predicted_emoji = command_emoji.get(
        prediction,
        "❓"
    )

    correct = (
        prediction == true_label
    )

    # Update main prediction display
    emoji_label.config(
        text=predicted_emoji
    )

    prediction_label.config(
        text=predicted_name
    )

    actual_label.config(
        text=true_name
    )

    confidence_value.config(
        text=f"{confidence_percent:.1f}%"
    )

    raw_confidence_label.config(
        text=(
            "Current-window confidence: "
            f"{raw_confidence_percent:.1f}%"
        )
    )

    confidence_bar["value"] = (
        confidence_percent
    )

    # Update prediction status
    if correct:

        status_label.config(
            text="● CORRECT",
            fg=SUCCESS,
        )

    else:

        status_label.config(
            text="● MISMATCH",
            fg=ERROR,
        )

    # Update live EMG graph
    channel_signal = window[
        :,
        0
    ]

    emg_line.set_ydata(
        channel_signal
    )

    signal_min = np.min(
        channel_signal
    )

    signal_max = np.max(
        channel_signal
    )

    padding = max(
        (signal_max - signal_min) * 0.2,
        0.01
    )

    axis.set_ylim(
        signal_min - padding,
        signal_max + padding
    )

    graph_canvas.draw_idle()

    # Add temporally smoothed output to prediction history
    update_history(
        predicted_emoji,
        predicted_name,
        confidence_percent,
        correct,
    )

    # Advance stream by 100 ms
    index += step_size

    # Schedule next prediction
    root.after(
        100,
        update_prediction
    )


# Start simulator

update_prediction()

root.mainloop()