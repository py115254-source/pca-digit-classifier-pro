# PCA-Based Handwritten Digit Classification

## Project Overview
This project implements a machine learning pipeline to classify handwritten digits (0–9) using the Scikit-learn Digits dataset. Principal Component Analysis (PCA) reduces the dimensionality of the input data while retaining most of its information.

## Objectives
- Classify handwritten digits using Logistic Regression.
- Apply feature scaling and PCA through a machine learning pipeline.
- Evaluate performance using accuracy and a classification report.
- Visualize PCA projections and the confusion matrix.
- Export predictions and evaluation metrics for further analysis.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Methodology
1. Load the handwritten digits dataset.
2. Split the data into training and testing sets using stratified sampling.
3. Standardize input features.
4. Apply PCA to retain the configured proportion of explained variance.
5. Train a Logistic Regression classifier.
6. Evaluate predictions on unseen test data.
7. Save visualizations, predictions, and metrics.

## Project Structure

```text
pca-digit-classifier-pro/
├── main.py
├── requirements.txt
├── README.md
└── outputs/
    ├── sample_digits.png
    ├── pca_scatter.png
    ├── confusion_matrix.png
    ├── predictions.csv
    └── metrics.csv
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

Customize the test split, PCA variance, and output directory:

```bash
python main.py --test-size 0.25 --pca-variance 0.95 --output-dir outputs
```

## Outputs
- `sample_digits.png`: examples from the dataset.
- `pca_scatter.png`: test samples projected onto the first two principal components.
- `confusion_matrix.png`: predicted versus actual digit classes.
- `predictions.csv`: actual labels, predicted labels, and correctness.
- `metrics.csv`: test sample count, retained PCA components, explained variance, and accuracy.

## Reproducibility
The project uses a fixed random seed for data splitting and model configuration. The pipeline helps keep preprocessing consistent between training and prediction.

## Limitations
The dataset contains small, low-resolution digit images. Results may differ on handwritten digits from other sources, so additional external validation would be needed before real-world deployment.

## Author
Machine Learning Project