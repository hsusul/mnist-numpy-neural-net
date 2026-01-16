from data import get_data
from model import TwoLayerNet
from utils import accuracy

MODEL_PATH = "artifacts/model_mnist.npz"

def main():
    _, _, X_test, y_test = get_data()
    model = TwoLayerNet.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    print("Test accuracy:", accuracy(y_pred, y_test))

if __name__ == "__main__":
    main()
