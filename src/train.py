import numpy as np

from data import get_data
from model import TwoLayerNet
from utils import batch_iterator, accuracy

MODEL_PATH = "artifacts/model_mnist.npz"

def main():
    # Load data (uses your cache from data.py)
    X_train, y_train, X_test, y_test = get_data()

    # Hyperparams
    epochs = 5
    batch_size = 128
    lr = 0.1
    hidden_dim = 128

    model = TwoLayerNet(input_dim=784, hidden_dim=hidden_dim, num_classes=10, seed=0)

    for epoch in range(1, epochs + 1):
        losses = []
        for Xb, yb in batch_iterator(X_train, y_train, batch_size=batch_size, shuffle=True, seed=epoch):
            loss = model.train_step(Xb, yb, lr=lr)
            losses.append(loss)

        # Train metrics (quick)
        train_pred = model.predict(X_train[:5000])
        train_acc = accuracy(train_pred, y_train[:5000])

        # Test metrics
        test_pred = model.predict(X_test)
        test_acc = accuracy(test_pred, y_test)

        print(f"Epoch {epoch}/{epochs} | loss={np.mean(losses):.4f} | train_acc~={train_acc:.4f} | test_acc={test_acc:.4f}")

    # Save model (make sure artifacts/ exists)
    model.save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
