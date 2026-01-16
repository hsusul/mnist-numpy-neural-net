import numpy as np


class Dense:
    """
    Fully-connected layer: Z = XW + b
    X: (batch, in_dim)
    W: (in_dim, out_dim)
    b: (out_dim,)
    Z: (batch, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # He init (good default with ReLU)
        self.W = (rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)

        self.X = None  # cache for backward
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        dZ = dZ.astype(np.float32, copy=False)
        # grads
        self.dW = self.X.T @ dZ
        self.db = dZ.sum(axis=0)
        # pass gradient down
        return dZ @ self.W.T


class ReLU:
    """
    ReLU: A = max(0, Z)
    """

    def __init__(self):
        self.Z = None  # cache

    def forward(self, Z: np.ndarray) -> np.ndarray:
        Z = Z.astype(np.float32, copy=False)
        self.Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        dA = dA.astype(np.float32, copy=False)
        return dA * (self.Z > 0)
