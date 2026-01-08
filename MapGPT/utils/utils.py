import numpy as np


def compute_entropy(vec: np.ndarray):
    # as the vec is the log probability, we need to convert the orginal formula
    vec += 1e-10
    entropy = -np.sum(np.exp(vec) * vec)
    return entropy