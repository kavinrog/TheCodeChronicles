# Deep Neural Network from Scratch

This project implements two versions of a Deep Neural Network (DNN) from scratch using NumPy:

1. `DeepNNScratch` — without ReLU activation.
2. `DeepNNScratchwithReLU` — with ReLU activation and two hidden layers.

It includes comparison of model performance using decision boundaries and loss plots.

## Project Structure

```
.
├── dataset.py          # Data loading and preprocessing
├── deepnn.py           # DNN implementation without ReLU
├── deepnnrelu.py       # DNN implementation with ReLU
├── train.py            # Training and evaluation logic
├── main.py             # Argument-based script to run models
└── images/             # Folder for storing generated plots
```

## Models

- **Without ReLU**: Simple two-layer neural network using sigmoid activation.
- **With ReLU**: Deep network using ReLU in hidden layers, sigmoid in output.

## Dataset

Uses `make_moons` from `sklearn.datasets` with standard scaling and train-test split.

## How to Run

Use the `main.py` file and specify the model type using `--model`:

```bash
python main.py --model dnnworelu
python main.py --model dnnwithrelu
```

## Example Outputs without ReLU

| Loss Plot | Decision Boundary |
|-----------|-------------------|
| ![Loss](./images/Training%20Loss%20Over%20Time%20without%20ReLU.png.png) | ![Decision](./images/Decision%20Boundary%20of%20Data%20without%20ReLU.png.png) |

## Example Outputs with ReLU

| Loss Plot | Decision Boundary |
|-----------|-------------------|
| ![Loss](./images/Training%20Loss%20Over%20Time%20with%20ReLU.png) | ![Decision](./images/Decision%20Boundary%20of%20Dataset%20with%20ReLU.png) |

## Metrics

- Training Accuracy
- Testing Accuracy

Accuracy results are printed after training.

## Requirements

- `numpy`
- `matplotlib`
- `scikit-learn`

Install with:

```bash
pip install -r requirements.txt
```

---

Built with idea to understand the core of neural networks.