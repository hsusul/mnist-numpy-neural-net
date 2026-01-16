import numpy as np

from layers import Dense, ReLU
from losses import softmax, softmax_cross_entropy_with_logits

class TwoLayerNet:
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, seed=0):
        self.dense1 = Dense(input_dim, hidden_dim, seed=seed)
        self.relu = ReLU()
        self.dense2 = Dense(hidden_dim, num_classes, seed=seed + 1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        z1 = self.dense1.forward(X)
        a1 = self.relu.forward(z1)
        logits = self.dense2.forward(a1)
        return logits

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X)
        probs = softmax(logits)
        return np.argmax(probs, axis=1)

    def train_step(self, Xb: np.ndarray, yb: np.ndarray, lr: float) -> float:
        # forward
        logits = self.forward(Xb)

        # loss + gradient wrt logits
        loss, dlogits = softmax_cross_entropy_with_logits(logits, yb)

        # backward
        da1 = self.dense2.backward(dlogits)
        dz1 = self.relu.backward(da1)
        _ = self.dense1.backward(dz1)

        # SGD update
        self.dense2.W -= lr * self.dense2.dW
        self.dense2.b -= lr * self.dense2.db
        self.dense1.W -= lr * self.dense1.dW
        self.dense1.b -= lr * self.dense1.db

        return loss

    def save(self, path: str):
        # Make sure the folder exists manually (e.g., artifacts/)
        np.savez_compressed(
            path,
            W1=self.dense1.W, b1=self.dense1.b,
            W2=self.dense2.W, b2=self.dense2.b,
        )

    @staticmethod
    def load(path: str):
        d = np.load(path)
        in_dim = d["W1"].shape[0]
        hidden_dim = d["W1"].shape[1]
        num_classes = d["W2"].shape[1]

        model = TwoLayerNet(in_dim, hidden_dim, num_classes, seed=0)
        model.dense1.W = d["W1"].astype(np.float32)
        model.dense1.b = d["b1"].astype(np.float32)
        model.dense2.W = d["W2"].astype(np.float32)
        model.dense2.b = d["b2"].astype(np.float32)
        return model
