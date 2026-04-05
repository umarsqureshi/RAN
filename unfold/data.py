"""Data loading and normalization for jet substructure observables."""

import numpy as np
from .observables import OBS_NAMES


def load_data(data_file, max_events=150000):
    """Load and normalize jet substructure data.

    Returns dict with normalized arrays and normalization params.
    """
    data = np.load(data_file)
    streams = ['_true', '_true_alt', '_reco', '_reco_alt']

    arrays = {}
    norm_params = {}

    for var in OBS_NAMES:
        # Truncate to max_events
        for stream in streams:
            key = var + stream
            arrays[key] = data[key][:max_events].astype(np.float32)

        # Normalize using Pythia gen (true) stats
        mu = np.mean(arrays[var + '_true'])
        sig = np.std(arrays[var + '_true'])
        norm_params[var] = (mu, sig)

        for stream in streams:
            key = var + stream
            arrays[key] = (arrays[key] - mu) / sig

    return arrays, norm_params
