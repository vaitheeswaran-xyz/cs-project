"""
morphology.py

Extract morphological ECG features from a SINGLE lead.

Features:
---------
P amplitude
Q amplitude
R amplitude
S amplitude
T amplitude

P duration
PR interval
QRS duration
QT interval
QTc

R/S ratio

Returns
-------
Dictionary
"""

import numpy as np
import neurokit2 as nk


# ----------------------------------------------------------
# Helper
# ----------------------------------------------------------

def safe_mean(values):

    values = np.array(values)

    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan

    return np.mean(values)


# ----------------------------------------------------------
# Main Function
# ----------------------------------------------------------

def extract_morphology(signal, fs):

    features = {}

    try:

        # Detect R peaks

        _, rpeaks = nk.ecg_peaks(
            signal,
            sampling_rate=fs
        )

        # Delineate PQRST

        _, waves = nk.ecg_delineate(
            signal,
            rpeaks,
            sampling_rate=fs,
            method="dwt"
        )

    except:

        return features

    # ------------------------------------------------------
    # R Peaks
    # ------------------------------------------------------

    r_locations = rpeaks["ECG_R_Peaks"]

    features["num_beats"] = len(r_locations)

    if len(r_locations):

        features["mean_r_amp"] = safe_mean(
            signal[r_locations]
        )

    # ------------------------------------------------------
    # Q Peaks
    # ------------------------------------------------------

    q_locations = waves["ECG_Q_Peaks"]

    q_amp = []

    for q in q_locations:

        if not np.isnan(q):

            q_amp.append(signal[int(q)])

    features["mean_q_amp"] = safe_mean(q_amp)

    # ------------------------------------------------------
    # S Peaks
    # ------------------------------------------------------

    s_locations = waves["ECG_S_Peaks"]

    s_amp = []

    for s in s_locations:

        if not np.isnan(s):

            s_amp.append(signal[int(s)])

    features["mean_s_amp"] = safe_mean(s_amp)

    # ------------------------------------------------------
    # P Peaks
    # ------------------------------------------------------

    p_locations = waves["ECG_P_Peaks"]

    p_amp = []

    for p in p_locations:

        if not np.isnan(p):

            p_amp.append(signal[int(p)])

    features["mean_p_amp"] = safe_mean(p_amp)

    # ------------------------------------------------------
    # T Peaks
    # ------------------------------------------------------

    t_locations = waves["ECG_T_Peaks"]

    t_amp = []

    for t in t_locations:

        if not np.isnan(t):

            t_amp.append(signal[int(t)])

    features["mean_t_amp"] = safe_mean(t_amp)

    # ------------------------------------------------------
    # P Duration
    # ------------------------------------------------------

    p_on = waves["ECG_P_Onsets"]
    p_off = waves["ECG_P_Offsets"]

    duration = []

    for a, b in zip(p_on, p_off):

        if np.isnan(a) or np.isnan(b):
            continue

        duration.append((b-a)/fs)

    features["mean_p_duration"] = safe_mean(duration)

    # ------------------------------------------------------
    # PR Interval
    # ------------------------------------------------------

    pr = []

    for p, q in zip(p_on, q_locations):

        if np.isnan(p) or np.isnan(q):
            continue

        pr.append((q-p)/fs)

    features["mean_pr_interval"] = safe_mean(pr)

    # ------------------------------------------------------
    # QRS Duration
    # ------------------------------------------------------

    qrs = []

    for q, s in zip(q_locations, s_locations):

        if np.isnan(q) or np.isnan(s):
            continue

        qrs.append((s-q)/fs)

    features["mean_qrs_duration"] = safe_mean(qrs)

    # ------------------------------------------------------
    # QT Interval
    # ------------------------------------------------------

    t_off = waves["ECG_T_Offsets"]

    qt = []

    for q, t in zip(q_locations, t_off):

        if np.isnan(q) or np.isnan(t):
            continue

        qt.append((t-q)/fs)

    features["mean_qt_interval"] = safe_mean(qt)

    # ------------------------------------------------------
    # QTc (Bazett)
    # ------------------------------------------------------

    rr = np.diff(r_locations) / fs

    if len(rr) > 0 and len(qt) > 0:

        qtc = []

        for qti, rri in zip(qt[:-1], rr):

            qtc.append(qti / np.sqrt(rri))

        features["mean_qtc"] = safe_mean(qtc)

    else:

        features["mean_qtc"] = np.nan

    # ------------------------------------------------------
    # R/S Ratio
    # ------------------------------------------------------

    if abs(features["mean_s_amp"]) > 1e-6:

        features["rs_ratio"] = (
            features["mean_r_amp"] /
            abs(features["mean_s_amp"])
        )

    else:

        features["rs_ratio"] = np.nan

    return features