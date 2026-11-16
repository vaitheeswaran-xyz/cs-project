"""
feature_extractor.py

Master feature extraction pipeline.

This module combines all feature extraction modules into
a single feature vector for one ECG record.

Author:
ECG CHD Project
"""

from preprocessing import preprocess_signal

from morphology import extract_morphology
from wavelet_features import extract_wavelet
from frequency_features import extract_frequency
from entropy_features import extract_entropy
from hrv_features import extract_hrv
from multilead_features import extract_multilead_features


def extract_features(ecg_record):
    """
    Parameters
    ----------
    ecg_record : ECGRecord

    Returns
    -------
    dict
        Complete feature dictionary
    """

    features = {}

    morphology_dict = {}

    fs = ecg_record.fs

    # ======================================================
    # Process Every Lead
    # ======================================================

    for lead_name, signal in ecg_record.leads.items():

        # -------------------------------
        # Preprocess
        # -------------------------------

        clean = preprocess_signal(
            signal,
            fs
        )

        # -------------------------------
        # Morphology
        # -------------------------------

        morph = extract_morphology(
            clean,
            fs
        )

        morphology_dict[lead_name] = morph

        for key, value in morph.items():

            features[f"{lead_name}_{key}"] = value

        # -------------------------------
        # Wavelet
        # -------------------------------

        wave = extract_wavelet(
            clean,
            fs
        )

        for key, value in wave.items():

            features[f"{lead_name}_{key}"] = value

        # -------------------------------
        # Frequency
        # -------------------------------

        freq = extract_frequency(
            clean,
            fs
        )

        for key, value in freq.items():

            features[f"{lead_name}_{key}"] = value

        # -------------------------------
        # Entropy
        # -------------------------------

        entropy = extract_entropy(
            clean,
            fs
        )

        for key, value in entropy.items():

            features[f"{lead_name}_{key}"] = value

    # ======================================================
    # HRV (Lead II Only)
    # ======================================================

    if "II" in ecg_record.leads:

        clean_lead2 = preprocess_signal(
            ecg_record.leads["II"],
            fs
        )

        hrv = extract_hrv(
            clean_lead2,
            fs
        )

        for key, value in hrv.items():

            features[f"LeadII_{key}"] = value

    # ======================================================
    # Multi-lead Features
    # ======================================================

    multi = extract_multilead_features(
        ecg_record,
        morphology_dict
    )

    features.update(multi)

    return features