"""
load_record.py

Loads a WFDB ECG record (.hea + .dat)
and converts it into a convenient format.

Author: Vaithee's ECG Feature Extraction Pipeline
"""

import wfdb
import numpy as np


class ECGRecord:
    """
    Stores one ECG record and provides
    easy access to each lead.
    """

    def __init__(self, signal, fs, lead_names, comments, record_name):

        self.signal = signal
        self.fs = fs
        self.lead_names = lead_names
        self.comments = comments
        self.record_name = record_name

        # Dictionary:
        # ecg["II"]
        # ecg["V1"]
        self.leads = {}
        for i, lead in enumerate(lead_names):

            if i < signal.shape[1]:
                self.leads[lead] = signal[:, i]

    def get_lead(self, lead_name):
        """
        Return one lead.
        """

        if lead_name not in self.leads:
            raise ValueError(f"Lead '{lead_name}' not found.")

        return self.leads[lead_name]

    def get_all_leads(self):
        """
        Return dictionary containing all leads.
        """

        return self.leads

    def num_samples(self):

        return self.signal.shape[0]

    def num_leads(self):

        return self.signal.shape[1]


def load_record(record_path):
    """
    Parameters
    ----------
    record_path : str

    Example:
    Dataset/P08999_E01

    Returns
    -------
    ECGRecord object
    """

    record = wfdb.rdrecord(record_path)

    signal = record.p_signal.astype(np.float32)

    fs = record.fs

    lead_names = record.sig_name

    comments = record.comments

    return ECGRecord(signal, fs, lead_names, comments)