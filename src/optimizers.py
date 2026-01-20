import numpy as np

def soft_threshold(z: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)

def compute_lipschitz_lr(X: np.ndarray) -> float:
    n = X.shape[0]
    fro2 = np.sum(X * X)
    L = (2.0 / n) * fro2
    if L <= 0:
        return 0.01
    return min(0.01, 1.0 / L)
