import numpy as np


def extract_features(window):
    mav = np.mean(np.abs(window), axis=0)

    rms = np.sqrt(
        np.mean(window ** 2, axis=0)
    )

    waveform_length = np.sum(
        np.abs(np.diff(window, axis=0)),
        axis=0
    )

    return np.concatenate([
        mav,
        rms,
        waveform_length
    ])