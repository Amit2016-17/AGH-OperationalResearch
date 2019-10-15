import numpy as np

def calculate_cost(a: np.array, b: np.array, c: np.array, d: np.array, f: float, **kwargs) -> float:
    """ Calculates cost of given solution"""
    return np.sum(b * c[a, :]) + np.sum(f * d[a])
