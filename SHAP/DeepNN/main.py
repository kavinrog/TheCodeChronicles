"""
main.py – Neural Network Training & Evaluation Script

This script allows you to train and evaluate two variants of a deep neural network:
1. `DeepNNScratch` without ReLU activation (pure sigmoid-based)
2. `DeepNNScratch` with ReLU activation in hidden layers

It uses a synthetic dataset (make_moons) and visualizes:
- Training loss over iterations
- Decision boundaries for classification

Usage:
Run from the command line with:
python main.py --model dnnworelu    # Without ReLU
python main.py --model dnnwithrelu  # With ReLU

Modules:
- argparse: for command-line arguments
- dataset: provides the `load_data()` function
- deepnn / deepnnrelu: custom neural network implementations
- matplotlib, numpy: for plotting and numerical operations
"""
import train