from pathlib import Path
import time

import joblib
from scipy.io import loadmat

from features import extract_features

import numpy as np


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

print("Starting simulated EMG stream...\n")

for i in range(
    0,
    len(segment) - window_size + 1,
    step_size
):
    window = segment[i:i + window_size]

    features = extract_features(window)

    prediction = model.predict(
        features.reshape(1, -1)
    )[0]

    true_label = segment_labels[
        i + window_size // 2
    ]

    predicted_name = command_map[prediction]

    true_name = command_map.get(
        true_label,
        "OTHER"
    )

    print(
        f"Actual: {true_name:<18} | "
        f"Predicted: {predicted_name}"
    )

    time.sleep(0.1)