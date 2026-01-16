import numpy as np

from data import get_data
from model import TwoLayerNet

MODEL_PATH = "artifacts/model_mnist.npz"


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    # accuracy for each true class = correct / total in that row
    row_sums = cm.sum(axis=1)
    acc = np.zeros((cm.shape[0],), dtype=np.float64)
    for i in range(cm.shape[0]):
        acc[i] = (cm[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0
    return acc


def main():
    # load data + model
    _, _, X_test, y_test = get_data()
    model = TwoLayerNet.load(MODEL_PATH)

    # predict on test
    y_pred = model.predict(X_test)

    # confusion matrix + metrics
    cm = confusion_matrix(y_test, y_pred, num_classes=10)
    overall_acc = float((y_pred == y_test).mean())
    class_acc = per_class_accuracy(cm)

    print("Overall test accuracy:", overall_acc)
    print("\nConfusion matrix (rows=true, cols=pred):\n")
    print(cm)

    print("\nPer-class accuracy:")
    for i, a in enumerate(class_acc):
        print(f"  {i}: {a:.4f}  (n={cm.sum(axis=1)[i]})")

    # top confusions (optional, helpful)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat_idx = np.argsort(cm_off.ravel())[::-1]

    print("\nTop confusions (true -> pred : count):")
    shown = 0
    for k in flat_idx:
        count = int(cm_off.ravel()[k])
        if count == 0:
            break
        i = k // 10
        j = k % 10
        print(f"  {i} -> {j} : {count}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
