# MNIST Neural Network From Scratch (NumPy) + Draw Demo (FastAPI + Docker)

A **2-layer neural network** implemented **from scratch using only NumPy** to classify handwritten digits (0–9) from MNIST.
Includes a **web demo** where you can draw a digit and see the model’s prediction, probabilities, and hidden-layer activations.

* **Architecture:** `784 → 128 → 10` (ReLU)
* **Test accuracy:** ~**95%**

---

## Demo (Draw in Browser)

### Run locally

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Run with Docker

```bash
docker build -t mnist-draw .
docker run -p 8000:8000 mnist-draw
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Training & Evaluation

Create folders (if needed):

```bash
mkdir -p data/processed artifacts
```

Train (saves weights to `artifacts/model_mnist.npz`):

```bash
python src/train.py
```

Evaluate:

```bash
python src/eval.py
python src/confusion.py
python src/predict.py
```

---

## Data

Place the ARFF MNIST file here:

```
data/raw/mnist_784.arff
```

On first run, it is parsed and cached to:

```
data/processed/mnist_processed.npz
```

---

## Project Structure

```
app/
  __init__.py
  api.py
  static/
    index.html
    app.js
    style.css

src/
  data.py       # load ARFF MNIST, normalize, train/test split, cache to npz
  layers.py     # Dense + ReLU layers (forward/backward)
  losses.py     # softmax, cross-entropy, combined loss+gradient
  utils.py      # batching + accuracy helpers
  model.py      # TwoLayerNet wrapper + save/load
  train.py      # training loop
  eval.py       # evaluation script
  predict.py    # CLI demo: random test samples + top-3 probs
  confusion.py  # confusion matrix + per-class accuracy

data/           # raw + processed data (ignored by git)
artifacts/      # saved model weights
Dockerfile
.dockerignore
requirements.txt
```

---

## Notes

* The web demo includes simple MNIST-style preprocessing (crop + pad + resize + center) to better match the training distribution.
* If `artifacts/model_mnist.npz` is missing, run `python src/train.py` first.
