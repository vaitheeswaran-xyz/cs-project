"""
frequency_features.py

Extract FFT-based features from a single ECG lead.

Author:
ECG CHD Project
"""

import numpy as np


def spectral_entropy(power):

    power = np.asarray(power)

    power = power / np.sum(power)

    power = power[power > 0]

    return -np.sum(power * np.log2(power))


def extract_frequency(signal, fs):

    features = {}

    N = len(signal)

    fft = np.fft.rfft(signal)

    freq = np.fft.rfftfreq(N, d=1/fs)

    magnitude = np.abs(fft)

    power = magnitude ** 2

    # ---------------------------------------
    # Basic FFT Statistics
    # ---------------------------------------

    features["fft_mean"] = np.mean(magnitude)

    features["fft_std"] = np.std(magnitude)

    features["fft_max"] = np.max(magnitude)

    features["fft_energy"] = np.sum(power)

    # ---------------------------------------
    # Dominant Frequency
    # ---------------------------------------

    dominant = np.argmax(power)

    features["dominant_frequency"] = freq[dominant]

    # ---------------------------------------
    # Spectral Centroid
    # ---------------------------------------

    features["spectral_centroid"] = np.sum(
        freq * power
    ) / np.sum(power)

    # ---------------------------------------
    # Spectral Bandwidth
    # ---------------------------------------

    centroid = features["spectral_centroid"]

    bandwidth = np.sqrt(

        np.sum(

            ((freq-centroid)**2) * power

        ) / np.sum(power)

    )

    features["spectral_bandwidth"] = bandwidth

    # ---------------------------------------
    # Spectral Entropy
    # ---------------------------------------

    features["spectral_entropy"] = spectral_entropy(power)

    # ---------------------------------------
    # Frequency Bands
    # ---------------------------------------

    bands = {

        "0_5": (0,5),

        "5_15": (5,15),

        "15_30": (15,30),

        "30_45": (30,45)

    }

    for name,(low,high) in bands.items():

        idx = np.logical_and(

            freq>=low,

            freq<high

        )

        features[f"bandpower_{name}"] = np.sum(

            power[idx]

        )

    return features