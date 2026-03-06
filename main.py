# main.py

import os
import numpy as np

from src.data.loaders import load_dataset, load_dataset_with_masks, load_test_dataset
from src.train import train_model
from src.infer import load_trained_model, predict_masks, threshold_predictions
from src.evaluate import run_full_evaluation
from src.utils.visualization import (
    plot_all_predictions,
    plot_predictions2,
    plot_overlay_predictions,
)


# =========================
# CONFIG
# =========================

# modos possíveis:
# "train"
# "infer"
# "evaluate"
# "train_and_evaluate"
MODE = "train_and_evaluate"

# paths principais
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATA_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test")
WEIGHTS_PATH = os.path.join(BASE_DIR, "src", "models", "model_augfin5.weights.h5")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# parâmetros
IMG_SIZE = (256, 256)
DOWNSAMPLE_FACTOR = 1.0
INPUT_SHAPE = (256, 256, 1)

TEST_SIZE = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50

THRESHOLD = 0.5

# visualização
SHOW_PREDICTIONS = True
SHOW_OVERLAYS = True
NUM_SAMPLES_TO_SHOW = 10


# =========================
# AUXILIARY
# =========================

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# TRAIN
# =========================

def run_train():
    print("=== TRAIN MODE ===")

    X, Y = load_dataset(
        TRAIN_DATA_DIR,
        img_size=IMG_SIZE,
        downsample_factor=DOWNSAMPLE_FACTOR
    )

    model, history, X_train, X_test, y_train, y_test = train_model(
        X,
        Y,
        input_shape=INPUT_SHAPE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        save_weights_path=WEIGHTS_PATH,
    )

    return model, history, X_train, X_test, y_train, y_test


# =========================
# INFER
# =========================

def run_infer_on_test_images():
    print("=== INFERENCE MODE ===")

    X_test_only = load_test_dataset(
        TEST_DATA_DIR,
        img_size=IMG_SIZE,
        downsample_factor=DOWNSAMPLE_FACTOR
    )

    model = load_trained_model(
        WEIGHTS_PATH,
        input_shape=INPUT_SHAPE
    )

    predictions = predict_masks(model, X_test_only)
    predictions_binary = threshold_predictions(predictions, threshold=THRESHOLD)

    if SHOW_PREDICTIONS:
        n = min(NUM_SAMPLES_TO_SHOW, len(X_test_only))
        plot_predictions2(X_test_only[:n], predictions_binary[:n], num_samples=n)

    if SHOW_OVERLAYS:
        n = min(NUM_SAMPLES_TO_SHOW, len(X_test_only))
        plot_overlay_predictions(
            X_test_only[:n],
            predictions_binary[:n],
            num_samples=n,
            save_dir=None,
            alpha=0.4,
            color='red'
        )

    return X_test_only, predictions, predictions_binary


# =========================
# EVALUATE
# =========================

def run_evaluate_on_test_with_masks():
    print("=== EVALUATION MODE ===")

    X_test, Y_test = load_dataset_with_masks(
        TEST_DATA_DIR,
        img_size=IMG_SIZE,
        downsample_factor=DOWNSAMPLE_FACTOR
    )

    Y_test = (Y_test > 0).astype("float32")

    model = load_trained_model(
        WEIGHTS_PATH,
        input_shape=INPUT_SHAPE
    )

    predictions = predict_masks(model, X_test)
    predictions_binary = threshold_predictions(predictions, threshold=THRESHOLD)

    if SHOW_PREDICTIONS:
        n = min(NUM_SAMPLES_TO_SHOW, len(X_test))
        plot_all_predictions(X_test[:n], Y_test[:n], predictions_binary[:n])

    if SHOW_OVERLAYS:
        n = min(NUM_SAMPLES_TO_SHOW, len(X_test))
        plot_overlay_predictions(
            X_test[:n],
            predictions_binary[:n],
            num_samples=n,
            save_dir=None,
            alpha=0.4,
            color='red'
        )

    results = run_full_evaluation(
        Y_test,
        predictions_binary,
        y_prob=predictions,
        output_dir=RESULTS_DIR,
        save_confusion_path=os.path.join(RESULTS_DIR, "confusion_matrix_global.png")
    )

    return X_test, Y_test, predictions, predictions_binary, results


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    ensure_results_dir()

    if MODE == "train":
        run_train()

    elif MODE == "infer":
        run_infer_on_test_images()

    elif MODE == "evaluate":
        run_evaluate_on_test_with_masks()

    elif MODE == "train_and_evaluate":
        model, history, X_train, X_test, y_train, y_test = run_train()

        predictions = predict_masks(model, X_test)
        predictions_binary = threshold_predictions(predictions, threshold=THRESHOLD)

        if SHOW_PREDICTIONS:
            n = min(NUM_SAMPLES_TO_SHOW, len(X_test))
            plot_all_predictions(X_test[:n], y_test[:n], predictions_binary[:n])

        if SHOW_OVERLAYS:
            n = min(NUM_SAMPLES_TO_SHOW, len(X_test))
            plot_overlay_predictions(
                X_test[:n],
                predictions_binary[:n],
                num_samples=n,
                save_dir=None,
                alpha=0.4,
                color='red'
            )

        run_full_evaluation(
            y_test,
            predictions_binary,
            y_prob=predictions,
            output_dir=RESULTS_DIR,
            save_confusion_path=os.path.join(RESULTS_DIR, "confusion_matrix_global.png")
        )

    else:
        raise ValueError(f"MODE inválido: {MODE}")