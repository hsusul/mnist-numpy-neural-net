# MNIST Neural Network From Scratch (NumPy)

A **2-layer neural network** implemented **from scratch using only NumPy** to classify handwritten digits (0–9) from the MNIST dataset.
Implements the full training pipeline: **forward pass → softmax + cross-entropy → backpropagation → SGD updates**, plus **save/load** for inference.

## Results

* **Test accuracy:** **95.25%**
* Architecture: `784 → 128 → 10` (ReLU)

## Project Structure

```
src/
  data.py      # load ARFF MNIST, normalize, train/test split, cache to npz
  layers.py    # Dense + ReLU layers (forward/backward)
  losses.py    # softmax, cross-entropy, combined loss+gradient
  utils.py     # batching + accuracy helpers
  model.py     # TwoLayerNet wrapper + save/load
  train.py     # training loop
  eval.py      # evaluation script

data/          # ignored (raw + processed data)
artifacts/     # ignored (saved model)
```

## Setup

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

This project expects the ARFF MNIST file at:

```
data/raw/mnist_784.arff
```

On first run, the file is parsed and cached to:

```
data/processed/mnist_processed.npz
```

> `data/` is in `.gitignore` so the dataset will not be committed.

## Train

Create these folders if they don’t exist:

* `data/processed/`
* `artifacts/`

Run:

```bash
python src/train.py
```

Saves model weights to:

```
artifacts/model_mnist.npz
```

## Evaluate

```bash
python src/eval.py
```

## Model Details

* Hidden activation: **ReLU**
* Output: **Softmax** (10 classes)
* Loss: **Cross-Entropy**
* Optimizer: **Mini-batch SGD**
