"""
wavelet_features.py

Extract Discrete Wavelet Transform (DWT) features
from a single ECG lead.

Wavelet:
---------
db4

Decomposition Level:
--------------------
4

Author:
ECG CHD Project
"""

import numpy as np
import pywt


# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

def shannon_entropy(coeff):
    """
    Compute Shannon entropy of wavelet coefficients.
    """

    coeff = np.asarray(coeff)

    energy = coeff ** 2

    total_energy = np.sum(energy)

    if total_energy == 0:
        return 0

    p = energy / total_energy

    p = p[p > 0]

    return -np.sum(p * np.log2(p))


# ----------------------------------------------------
# Main Feature Extraction
# ----------------------------------------------------

def extract_wavelet(signal,
                     wavelet="db4",
                     level=4):

    features = {}

    coeffs = pywt.wavedec(
        signal,
        wavelet=wavelet,
        level=level
    )

    # coeffs:
    #
    # cA4
    # cD4
    # cD3
    # cD2
    # cD1

    names = [
        "A4",
        "D4",
        "D3",
        "D2",
        "D1"
    ]

    for name, coeff in zip(names, coeffs):

        coeff = np.asarray(coeff)

        # -------------------------
        # Energy
        # -------------------------

        features[f"{name}_energy"] = np.sum(coeff ** 2)

        # -------------------------
        # Mean
        # -------------------------

        features[f"{name}_mean"] = np.mean(coeff)

        # -------------------------
        # Standard Deviation
        # -------------------------

        features[f"{name}_std"] = np.std(coeff)

        # -------------------------
        # Variance
        # -------------------------

        features[f"{name}_variance"] = np.var(coeff)

        # -------------------------
        # RMS
        # -------------------------

        features[f"{name}_rms"] = np.sqrt(
            np.mean(coeff ** 2)
        )

        # -------------------------
        # Maximum
        # -------------------------

        features[f"{name}_max"] = np.max(coeff)

        # -------------------------
        # Minimum
        # -------------------------

        features[f"{name}_min"] = np.min(coeff)

        # -------------------------
        # Entropy
        # -------------------------

        features[f"{name}_entropy"] = shannon_entropy(coeff)

    return features