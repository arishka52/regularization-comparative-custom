from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from src.optimizers import soft_threshold


@dataclass
class FitResult:
    n_iter: int
    final_loss: float


class LinearRegressorBase:
    def __init__(self):
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.w + self.b


class OLSRegressor(LinearRegressorBase):
    def fit(self, X: np.ndarray, y: np.ndarray, lr=0.01, max_iter=5000, tol=1e-7) -> FitResult:
        n, p = X.shape
        self.w = np.zeros(p, dtype=float)
        self.b = 0.0

        prev = np.inf
        for it in range(1, max_iter + 1):
            y_pred = X @ self.w + self.b
            err = y_pred - y

            grad_w = (2.0 / n) * (X.T @ err)
            grad_b = (2.0 / n) * np.sum(err)

            self.w -= lr * grad_w
            self.b -= lr * grad_b

            loss = float(np.mean(err ** 2))
            if abs(prev - loss) < tol:
                return FitResult(it, loss)
            prev = loss

        return FitResult(max_iter, prev)


class RidgeRegressor(LinearRegressorBase):
    def __init__(self, lam: float):
        super().__init__()
        self.lam = float(lam)

    def fit(self, X: np.ndarray, y: np.ndarray, lr=0.01, max_iter=5000, tol=1e-7) -> FitResult:
        n, p = X.shape
        self.w = np.zeros(p, dtype=float)
        self.b = 0.0

        prev = np.inf
        for it in range(1, max_iter + 1):
            y_pred = X @ self.w + self.b
            err = y_pred - y

            grad_w = (2.0 / n) * (X.T @ err) + 2.0 * self.lam * self.w
            grad_b = (2.0 / n) * np.sum(err)

            self.w -= lr * grad_w
            self.b -= lr * grad_b

            loss = float(np.mean(err ** 2) + self.lam * np.sum(self.w ** 2))
            if abs(prev - loss) < tol:
                return FitResult(it, loss)
            prev = loss

        return FitResult(max_iter, prev)


class LassoRegressor(LinearRegressorBase):
    def __init__(self, lam: float):
        super().__init__()
        self.lam = float(lam)

    def fit(self, X: np.ndarray, y: np.ndarray, lr=0.01, max_iter=5000, tol=1e-7) -> FitResult:
        n, p = X.shape
        self.w = np.zeros(p, dtype=float)
        self.b = 0.0

        prev = np.inf
        for it in range(1, max_iter + 1):
            y_pred = X @ self.w + self.b
            err = y_pred - y

            grad_w = (2.0 / n) * (X.T @ err)
            grad_b = (2.0 / n) * np.sum(err)

            w_tmp = self.w - lr * grad_w
            self.w = soft_threshold(w_tmp, lr * self.lam)
            self.b -= lr * grad_b

            loss = float(np.mean(err ** 2) + self.lam * np.sum(np.abs(self.w)))
            if abs(prev - loss) < tol:
                return FitResult(it, loss)
            prev = loss

        return FitResult(max_iter, prev)


class ElasticNetRegressor(LinearRegressorBase):
    def __init__(self, lam: float, alpha: float):
        super().__init__()
        self.lam = float(lam)
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray, lr=0.01, max_iter=5000, tol=1e-7) -> FitResult:
        n, p = X.shape
        self.w = np.zeros(p, dtype=float)
        self.b = 0.0

        prev = np.inf
        for it in range(1, max_iter + 1):
            y_pred = X @ self.w + self.b
            err = y_pred - y

            grad_w = (2.0 / n) * (X.T @ err) + 2.0 * self.lam * (1.0 - self.alpha) * self.w
            grad_b = (2.0 / n) * np.sum(err)

            w_tmp = self.w - lr * grad_w
            self.w = soft_threshold(w_tmp, lr * self.lam * self.alpha)
            self.b -= lr * grad_b

            loss = float(
                np.mean(err ** 2)
                + self.lam * self.alpha * np.sum(np.abs(self.w))
                + self.lam * (1.0 - self.alpha) * np.sum(self.w ** 2)
            )
            if abs(prev - loss) < tol:
                return FitResult(it, loss)
            prev = loss

        return FitResult(max_iter, prev)
