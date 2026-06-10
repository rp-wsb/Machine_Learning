# Pneumonia Classification — CNN vs CapsNet

A comparative study of two deep learning approaches for binary classification of pneumonia from chest X-ray images. The project implements a Convolutional Neural Network (CNN) and a Capsule Network (CapsNet) across several configurations, then evaluates and contrasts their performance.

## Repository Structure

```
Machine_Learning/
├── cnn.py            # CNN model with focal loss
├── capsnet.py        # CapsNet v2 with dynamic routing
├── tuning.py         # CapsNet fine-tuning with focal loss
├── test.py           # Testing script
├── porownanie.py     # Model comparison — plots and summary tables
├── chest_xray/       # Dataset directory
│   ├── train/
│   └── test/
└── wyniki/           # Output: .pkl result files, plots, summaries
```

## Models

### CNN (`cnn.py`)

A four-block convolutional network using Conv2D, Batch Normalization, MaxPooling, and Dropout layers. Focal loss (γ=2.0, α=0.25) replaces standard binary cross-entropy to handle class imbalance. Optimized with Adam at a learning rate of 1e-4, trained for up to 15 epochs.

### CapsNet v2 (`capsnet.py`)

A capsule network built on top of a three-block convolutional backbone. Key design choices:

- PrimaryCaps layer: 8 capsules, dimension 16
- DigitCaps layer: 2 capsules, dimension 32
- Dynamic routing with 3 iterations
- Gradient clipping (`clipnorm=1.0`) for training stability
- Capsule activation diagnostics logged after the first epoch

Optimized with Adam at a learning rate of 1e-3, trained for up to 30 epochs.

### CapsNet + Focal Loss (`tuning.py`)

Fine-tuning of the CapsNet model using focal loss. Results are evaluated at two decision thresholds:

- **Youden** — threshold that maximizes the Youden index (sensitivity + specificity − 1)
- **Sens >= 0.92** — threshold that guarantees sensitivity of at least 92%

## Comparison Script (`porownanie.py`)

Loads serialized results from all three trained models (`.pkl`) and produces:

- Learning curves (accuracy, loss, AUC)
- Confusion matrices for all four model variants
- ROC and Precision-Recall curves
- Bar charts comparing sensitivity, specificity, F1, and accuracy
- Training time and epoch count plots
- A summary table exported as `podsumowanie.csv` and `podsumowanie.txt`

## Dataset

**Chest X-Ray Images (Pneumonia)** from Kaggle.

The dataset should be downloaded separately and placed in the `chest_xray/` directory, with the following structure:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The path can be changed via the `DATA_DIR` variable at the top of each script.

## Configuration

| Parameter     | CNN     | CapsNet |
|---------------|---------|---------|
| Image size    | 150x150 | 150x150 |
| Batch size    | 32      | 32      |
| Max epochs    | 15      | 30      |
| Optimizer     | Adam    | Adam    |
| Learning rate | 1e-4    | 1e-3    |
| Random seed   | 42      | 42      |

## Getting Started

Install dependencies:

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn pandas
```

Run training and evaluation in order:

```bash
# Train CNN
python cnn.py

# Train CapsNet
python capsnet.py

# Fine-tune CapsNet with focal loss (requires capsnet.py output)
python tuning.py

# Generate comparison plots and summary (requires all of the above)
python porownanie.py
```

All outputs are saved to the `./wyniki/` directory.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Pandas
