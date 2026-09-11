from pathlib import Path
import tkinter as tk

import joblib
import numpy as np
from scipy.io import loadmat

from features import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "prosthetic_command_rf.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "S1_A1_E2.mat"

model = joblib.load(MODEL_PATH)

command_map = {
    0: "REST",
    6: "FIST",
    7: "POINT",
    8: "OPEN HAND",
    13: "WRIST FLEXION",
    14: "WRIST EXTENSION"
}

data = loadmat(DATA_PATH)

emg = data["emg"]
labels = data["restimulus"].flatten()
repetitions = data["rerepetition"].flatten()

window_size = 20
step_size = 10

target_command = 6
target_repetition = 1

indices = np.where(
    (labels == target_command) &
    (repetitions == target_repetition)
)[0]

start = max(indices[0] - 200, 0)
end = min(indices[-1] + 200, len(emg))

segment = emg[start:end]
segment_labels = labels[start:end]

root = tk.Tk()
root.title("EMG Prosthetic Controller")

title_label = tk.Label(
    root,
    text="Predicted Prosthetic Command",
    font=("Arial", 18)
)
title_label.pack(pady=20)

prediction_label = tk.Label(
    root,
    text="REST",
    font=("Arial", 32, "bold")
)
prediction_label.pack(pady=20)

actual_label = tk.Label(
    root,
    text="Actual: REST",
    font=("Arial", 16)
)
actual_label.pack(pady=10)

index = 0


def update_prediction():
    global index

    if index + window_size > len(segment):
        prediction_label.config(text="DONE")
        return

    window = segment[index:index + window_size]

    features = extract_features(window)

    prediction = model.predict(
        features.reshape(1, -1)
    )[0]

    true_label = segment_labels[
        index + window_size // 2
    ]

    predicted_name = command_map[prediction]
    true_name = command_map.get(true_label, "OTHER")

    prediction_label.config(
        text=predicted_name
    )

    actual_label.config(
        text=f"Actual: {true_name}"
    )

    index += step_size

    root.after(
        100,
        update_prediction
    )


update_prediction()

root.mainloop()