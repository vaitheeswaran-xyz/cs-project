"""
preprocessing.py

Functions for preprocessing ECG signals.

Pipeline:
----------
1. Remove baseline wander
2. Bandpass filtering
3. Remove powerline interference
4. Normalize signal
"""

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, iirnotch


# ---------------------------------------------------------
# Bandpass Filter
# ---------------------------------------------------------

def bandpass_filter(signal,
                    fs,
                    lowcut=0.5,
                    highcut=40):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        N=4,
        Wn=[low, high],
        btype='band'
    )

    filtered = filtfilt(b, a, signal)

    return filtered


# ---------------------------------------------------------
# Notch Filter
# ---------------------------------------------------------

def notch_filter(signal,
                 fs,
                 notch_freq=50,
                 quality_factor=30):

    nyquist = 0.5 * fs

    w0 = notch_freq / nyquist

    b, a = iirnotch(w0, quality_factor)

    filtered = filtfilt(b, a, signal)

    return filtered


# ---------------------------------------------------------
# Normalize
# ---------------------------------------------------------

def normalize(signal):

    signal = np.asarray(signal)

    signal = (signal - np.mean(signal)) / np.std(signal)

    return signal


# ---------------------------------------------------------
# Complete Preprocessing Pipeline
# ---------------------------------------------------------

def preprocess_signal(signal,
                      fs):

    # NeuroKit removes baseline wander
    clean = nk.ecg_clean(
        signal,
        sampling_rate=fs,
        method="neurokit"
    )

    # Bandpass
    clean = bandpass_filter(
        clean,
        fs
    )

    # Remove 50Hz powerline noise
    clean = notch_filter(
        clean,
        fs
    )

    # Normalize
    clean = normalize(
        clean
    )

    return clean