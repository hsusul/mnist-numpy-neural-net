import numpy as np

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = y_pred.astype(np.int64, copy=False)
    y_true = y_true.astype(np.int64, copy=False)
    return float((y_pred == y_true).mean())

def batch_iterator(X, y, batch_size: int, shuffle: bool = True, seed: int = 42):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, N, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
