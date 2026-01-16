import numpy as np

from data import get_data
from model import TwoLayerNet


MODEL_PATH = "artifacts/model_mnist.npz"


def topk(probs_row, k=3):
    idx = np.argsort(probs_row)[::-1][:k]
    return [(int(i), float(probs_row[i])) for i in idx]


def main():
    # load data + model
    X_train, y_train, X_test, y_test = get_data()
    model = TwoLayerNet.load(MODEL_PATH)

    # pick 10 random test examples
    rng = np.random.default_rng(0)
    idx = rng.choice(X_test.shape[0], size=10, replace=False)

    X = X_test[idx]
    y_true = y_test[idx]

    # get probabilities
    logits = model.forward(X)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    y_pred = np.argmax(probs, axis=1)

    for i in range(len(idx)):
        print(f"Example {i+1}")
        print(f"  true: {int(y_true[i])}  pred: {int(y_pred[i])}")
        print(f"  top-3: {topk(probs[i], k=3)}")
        print()


if __name__ == "__main__":
    main()
