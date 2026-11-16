"""
entropy_features.py

Extract nonlinear complexity features from a single ECG lead.

Author:
ECG CHD Project
"""

import numpy as np
import antropy as ant


def extract_entropy(signal, fs):
    """
    Parameters
    ----------
    signal : ndarray
        Clean ECG signal

    fs : int
        Sampling frequency (currently unused, kept for API consistency)

    Returns
    -------
    dict
        Entropy / nonlinear features
    """

    features = {}

    signal = np.asarray(signal)

    try:

        # ---------------------------------------
        # Sample Entropy
        # ---------------------------------------

        features["sample_entropy"] = ant.sample_entropy(signal)

        # ---------------------------------------
        # Approximate Entropy
        # ---------------------------------------

        features["approx_entropy"] = ant.app_entropy(signal)

        # ---------------------------------------
        # Permutation Entropy
        # ---------------------------------------

        features["permutation_entropy"] = ant.perm_entropy(signal)

        # ---------------------------------------
        # SVD Entropy
        # ---------------------------------------

        features["svd_entropy"] = ant.svd_entropy(signal)

        # ---------------------------------------
        # Spectral Entropy
        # ---------------------------------------

        features["spectral_entropy"] = ant.spectral_entropy(
            signal,
            sf=fs,
            method="welch",
            normalize=True
        )

        # ---------------------------------------
        # Hjorth Parameters
        # ---------------------------------------

        mobility, complexity = ant.hjorth_params(signal)

        features["hjorth_mobility"] = mobility
        features["hjorth_complexity"] = complexity

        # ---------------------------------------
        # Zero Crossing Count
        # ---------------------------------------

        features["zero_crossings"] = ant.num_zerocross(signal)

    except Exception:

        feature_names = [
            "sample_entropy",
            "approx_entropy",
            "permutation_entropy",
            "svd_entropy",
            "spectral_entropy",
            "hjorth_mobility",
            "hjorth_complexity",
            "zero_crossings"
        ]

        for name in feature_names:
            features[name] = np.nan

    return features