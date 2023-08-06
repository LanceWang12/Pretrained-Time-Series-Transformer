import stumpy as sp
import numpy as np


def matrix_profile(query, whole_series, top_k: int = 100):
    distance_profile = sp.mass(query, whole_series)
    idxs = np.argsort(distance_profile)
    return idxs[:top_k]
