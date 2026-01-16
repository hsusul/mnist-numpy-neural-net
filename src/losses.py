import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Stable softmax over last dimension.
    logits: (batch, num_classes)
    returns probs: (batch, num_classes)
    """
    logits = logits.astype(np.float32, copy=False)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """
    probs: (batch, num_classes)
    y: (batch,) integer class labels
    returns average loss (float)
    """
    probs = probs.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)

    eps = 1e-12
    p_correct = probs[np.arange(y.shape[0]), y]
    return float(-np.mean(np.log(p_correct + eps)))


def softmax_cross_entropy_with_logits(logits: np.ndarray, y: np.ndarray):
    """
    Combined softmax + cross-entropy.
    Returns:
      loss: float
      dlogits: (batch, num_classes) gradient w.r.t logits
    """
    y = y.astype(np.int64, copy=False)
    probs = softmax(logits)
    loss = cross_entropy(probs, y)

    # gradient: probs - one_hot(y), averaged over batch
    dlogits = probs.copy()
    dlogits[np.arange(y.shape[0]), y] -= 1.0
    dlogits /= y.shape[0]

    return loss, dlogits
