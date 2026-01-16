import numpy as np

RAW_ARFF_PATH = "data/raw/mnist_784.arff"
PROCESSED_NPZ_PATH = "data/processed/mnist_processed.npz"

def load_arff_mnist(path):
    """
    Reads an ARFF MNIST file (784 pixels + label) 
    Returns: 
    X: (N, 784) float32, normalized to [0,1]
    y: (N,) int64 labels 0-9
    """
    X_rows = []
    y_rows = []
    in_data = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue
            parts = [p.strip() for p in line.split(",")]
            y_rows.append(int(parts[-1]))
            X_rows.append([float(v) for v in parts[:-1]])
    X = np.array(X_rows, dtype = np.float32)
    y = np.array(y_rows, dtype = np.int64)
    if X.max() > 1.0:
        X /= 255.0
    return X, y

def train_test_split(X, y, test_size = 0.2, seed = 42):
    """
    Train/Test splits with shuffling to reduce bias
    returns xtrain ytrain xtest ytest
    """
    N = X.shape[0]
    rng = np.random.default_rng(seed)

    idx = np.arange(N)
    rng.shuffle(idx)

    split = int(N * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test

def save_processed_npz(path: str, X_train, y_train, X_test, y_test):
    """
    Saves processed arrays into one compressed .npz file
    """
    try:
        np.savez_compressed(
            path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Folder not found. Create 'data/processed/' first, then re-run."
        ) from e

def load_processed_npz(path: str):
    """
    Loads processed arrays from the .npz file 
    returns xtrain ytrain xtest ytest
    """
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


def get_data(
    raw_path: str = RAW_ARFF_PATH,
    processed_path: str = PROCESSED_NPZ_PATH,
    test_size: float = 0.2,
    seed: int = 42,
    use_cache: bool = True,
):
    """
    Main entry point
    1. tries to laod cached processed data if use_cache = True
    otherwise parses raw ARFF splits save cache returns array
    """
    X, y = load_arff_mnist(raw_path)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=test_size, seed=seed)

    save_processed_npz(processed_path, X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = get_data()

    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes: ", X_test.shape, y_test.shape)
    print("X min/max:   ", float(X_train.min()), float(X_train.max()))
    print("Labels:      ", np.unique(y_train))
    print("Features:    ", X_train.shape[1], "(should be 784)")


if __name__ == "__main__":
    main()


