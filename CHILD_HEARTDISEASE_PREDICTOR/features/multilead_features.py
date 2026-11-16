"""
multilead_features.py

Extract ECG features that require information from
multiple leads.

Author:
ECG CHD Project
"""

import numpy as np


def safe(feature_dict, lead, feature):
    """
    Safely retrieve a feature.
    """

    try:
        return feature_dict[lead][feature]
    except Exception:
        return np.nan


def extract_multilead_features(ecg_record, morphology):

    features = {}

    # =====================================================
    # Limb Leads
    # =====================================================

    limb = [
        safe(morphology, "I", "mean_r_amp"),
        safe(morphology, "II", "mean_r_amp"),
        safe(morphology, "III", "mean_r_amp"),
        safe(morphology, "aVR", "mean_r_amp"),
        safe(morphology, "aVL", "mean_r_amp"),
        safe(morphology, "aVF", "mean_r_amp")
    ]

    features["limb_mean_r"] = np.nanmean(limb)
    features["limb_std_r"] = np.nanstd(limb)

    # =====================================================
    # Chest Leads
    # =====================================================

    chest = [
        safe(morphology, "V1", "mean_r_amp"),
        safe(morphology, "V2", "mean_r_amp"),
        safe(morphology, "V3", "mean_r_amp"),
        safe(morphology, "V4", "mean_r_amp"),
        safe(morphology, "V5", "mean_r_amp"),
        safe(morphology, "V6", "mean_r_amp")
    ]

    features["precordial_mean_r"] = np.nanmean(chest)
    features["precordial_std_r"] = np.nanstd(chest)

    # =====================================================
    # R Wave Progression
    # =====================================================

    progression = np.diff(chest)

    features["r_progression_mean"] = np.nanmean(progression)
    features["r_progression_std"] = np.nanstd(progression)

    # =====================================================
    # Sokolow-Lyon
    # =====================================================

    s_v1 = abs(safe(morphology, "V1", "mean_s_amp"))

    r_v5 = safe(morphology, "V5", "mean_r_amp")

    r_v6 = safe(morphology, "V6", "mean_r_amp")

    features["sokolow_lyon"] = s_v1 + max(r_v5, r_v6)

    # =====================================================
    # Cornell Voltage
    # =====================================================

    r_avl = safe(morphology, "aVL", "mean_r_amp")

    s_v3 = abs(safe(morphology, "V3", "mean_s_amp"))

    features["cornell_voltage"] = r_avl + s_v3

    # =====================================================
    # R/S Ratios
    # =====================================================

    for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:

        r = safe(morphology, lead, "mean_r_amp")

        s = abs(safe(morphology, lead, "mean_s_amp"))

        if s > 1e-6:

            features[f"{lead}_rs_ratio"] = r / s

        else:

            features[f"{lead}_rs_ratio"] = np.nan

    # =====================================================
    # Average ECG Intervals
    # =====================================================

    pr = []
    qrs = []
    qt = []
    qtc = []

    for lead in morphology:

        pr.append(
            safe(morphology, lead, "mean_pr_interval")
        )

        qrs.append(
            safe(morphology, lead, "mean_qrs_duration")
        )

        qt.append(
            safe(morphology, lead, "mean_qt_interval")
        )

        qtc.append(
            safe(morphology, lead, "mean_qtc")
        )

    features["global_pr"] = np.nanmean(pr)
    features["global_qrs"] = np.nanmean(qrs)
    features["global_qt"] = np.nanmean(qt)
    features["global_qtc"] = np.nanmean(qtc)

    return features