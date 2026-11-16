"""
hrv_features.py

Extract Heart Rate Variability (HRV) features
from a single ECG lead (preferably Lead II).

Author: ECG CHD Project
"""

import numpy as np
import neurokit2 as nk


def extract_hrv(signal, fs):
    """
    Parameters
    ----------
    signal : ndarray
        Clean ECG signal (Lead II)

    fs : int
        Sampling frequency

    Returns
    -------
    dict
        HRV features
    """

    features = {}

    try:

        # Detect R-peaks
        _, rpeaks = nk.ecg_peaks(
            signal,
            sampling_rate=fs
        )

        # Compute HRV
        hrv = nk.hrv_time(
            rpeaks,
            sampling_rate=fs,
            show=False
        )

    except Exception:

        return features

    # -----------------------------
    # Selected HRV Features
    # -----------------------------

    selected = [

        "HRV_MeanNN",
        "HRV_SDNN",
        "HRV_RMSSD",
        "HRV_SDSD",
        "HRV_CVNN",
        "HRV_CVSD",
        "HRV_MedianNN",
        "HRV_MadNN",
        "HRV_IQRNN",
        "HRV_pNN50",
        "HRV_pNN20",
        "HRV_HTI",
        "HRV_TINN"

    ]

    for feature in selected:

        if feature in hrv.columns:

            features[feature] = float(hrv[feature].iloc[0])

        else:

            features[feature] = np.nan

    # -----------------------------
    # Heart Rate
    # -----------------------------

    rr = np.diff(rpeaks["ECG_R_Peaks"]) / fs

    if len(rr):

        features["heart_rate"] = 60 / np.mean(rr)

        features["mean_rr"] = np.mean(rr)

        features["std_rr"] = np.std(rr)

        features["min_rr"] = np.min(rr)

        features["max_rr"] = np.max(rr)

        features["rr_range"] = np.max(rr) - np.min(rr)

    else:

        features["heart_rate"] = np.nan
        features["mean_rr"] = np.nan
        features["std_rr"] = np.nan
        features["min_rr"] = np.nan
        features["max_rr"] = np.nan
        features["rr_range"] = np.nan

    return features