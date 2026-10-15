
"""Professional PCA-based handwritten digit classification project."""

from pathlib import Path
import argparse
import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
PCA_VARIANCE = 0.95
OUTPUT_DIR = Path("outputs")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """Read configurable settings from the command line."""
    parser = argparse.ArgumentParser(
        description="PCA-based handwritten digit classification"
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--pca-variance", type=float, default=PCA_VARIANCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size must be between 0.1 and 0.4")
    if not 0.80 <= args.pca_variance <= 0.99:
        parser.error("--pca-variance must be between 0.80 and 0.99")

    return args


def load_data():
    """Load the handwritten digits dataset."""
    dataset = load_digits()
    return dataset.data, dataset.target, dataset.images


def build_pipeline(pca_variance):
    """Build a reproducible preprocessing and classification pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_variance, random_state=RANDOM_STATE)),
        ("classifier", LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        )),
    ])


def save_sample_digits(images, labels, output_dir):
    """Save a grid of example handwritten digits."""
    figure, axes = plt.subplots(2, 5, figsize=(10, 5))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index], cmap="gray_r")
        axis.set_title(f"Label: {labels[index]}")
        axis.axis("off")

    figure.suptitle("Sample Handwritten Digits")
    figure.tight_layout()
    figure.savefig(output_dir / "sample_digits.png", dpi=160)
    plt.close(figure)


def save_pca_scatter(model, X_test, y_test, output_dir):
    """Plot test samples using the first two fitted PCA components."""
    scaler = model.named_steps["scaler"]
    pca = model.named_steps["pca"]

    scaled_data = scaler.transform(X_test)
    projected_data = pca.transform(scaled_data)

    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        c=y_test,
        cmap="viridis",
        alpha=0.8,
        s=24,
    )
    axis.set_xlabel("Principal Component 1")
    axis.set_ylabel("Principal Component 2")
    axis.set_title("PCA Projection of Handwritten Digits")
    figure.colorbar(scatter, ax=axis, label="Digit Label")
    figure.tight_layout()
    figure.savefig(output_dir / "pca_scatter.png", dpi=160)
    plt.close(figure)


def save_confusion_matrix(model, X_test, y_test, output_dir):
    """Save the classification confusion matrix."""
    figure, axis = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, cmap="Blues", ax=axis
    )
    axis.set_title("Digit Classification Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def main():
    """Train, evaluate, visualize, and save project outputs."""
    args = parse_arguments()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        X, y, images = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        model = build_pipeline(args.pca_variance)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        pca = model.named_steps["pca"]

        save_sample_digits(images, y, args.output_dir)
        save_pca_scatter(model, X_test, y_test, args.output_dir)
        save_confusion_matrix(model, X_test, y_test, args.output_dir)

        pd.DataFrame({
            "actual_digit": y_test,
            "predicted_digit": predictions,
            "correct": y_test == predictions,
        }).to_csv(args.output_dir / "predictions.csv", index=False)

        metrics = pd.DataFrame([{
            "test_samples": len(y_test),
            "pca_components": pca.n_components_,
            "variance_retained": pca.explained_variance_ratio_.sum(),
            "accuracy": accuracy,
        }])
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)

        LOGGER.info("Total samples: %d", len(X))
        LOGGER.info("Training samples: %d", len(X_train))
        LOGGER.info("Testing samples: %d", len(X_test))
        LOGGER.info("Original features: %d", X.shape[1])
        LOGGER.info("PCA components retained: %d", pca.n_components_)
        LOGGER.info("Variance retained: %.2f%%",
                    pca.explained_variance_ratio_.sum() * 100)
        LOGGER.info("Test accuracy: %.2f%%", accuracy * 100)
        print("\nClassification Report:\n")
        print(classification_report(y_test, predictions, zero_division=0))
        LOGGER.info("Output files saved in: %s", args.output_dir.resolve())

    except (OSError, ValueError) as error:
        LOGGER.error("Project execution failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
