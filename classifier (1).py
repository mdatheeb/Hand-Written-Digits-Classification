# -*- coding: utf-8 -*-
"""
MNIST Handwritten Digit Classification
TensorFlow/Keras neural network, with an experiment comparing hidden layer size.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
EPOCHS = 10


def load_data():
    """Load MNIST and split off a validation set using train_test_split."""
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    print("Training images:", x_train.shape)
    print("Validation images:", x_val.shape)
    print("Testing images:", x_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test


def show_sample_images(images, labels, n=10):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def preprocess(*datasets):
    """Scale one or more image arrays from 0-255 to 0-1."""
    return [d.astype("float32") / 255.0 for d in datasets]


def build_model(hidden_neurons=128, activation="relu"):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_neurons, activation=activation),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs=EPOCHS):
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
    )


def plot_history(history, title_suffix=""):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy {title_suffix}".strip())
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss {title_suffix}".strip())
    plt.legend()
    plt.show()


def show_predictions(model, x_test, y_test, n=5, seed=RANDOM_SEED):
    """Pick n random test images and compare actual vs predicted labels."""
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(x_test), n, replace=False)

    sample_images = x_test[sample_indices]
    actual_labels = y_test[sample_indices]

    predictions = model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(sample_images[i], cmap="gray")
        plt.title(f"Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return actual_labels, predicted_labels


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    show_sample_images(x_train, y_train)

    x_train, x_val, x_test = preprocess(x_train, x_val, x_test)

    # --- Baseline model ---
    model = build_model(hidden_neurons=128)
    history = train_model(model, x_train, y_train, x_val, y_val)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    plot_history(history, title_suffix="(128 neurons)")
    show_predictions(model, x_test, y_test)

    # --- Experiment: wider hidden layer ---
    experiment_model = build_model(hidden_neurons=256)
    experiment_history = train_model(experiment_model, x_train, y_train, x_val, y_val)

    exp_loss, exp_accuracy = experiment_model.evaluate(x_test, y_test)
    print("Experiment Test Loss:", exp_loss)
    print("Experiment Test Accuracy:", exp_accuracy)

    plot_history(experiment_history, title_suffix="(256 neurons)")

    print("\nComparison")
    print(f"{'Model':<15}{'Hidden Neurons':<16}{'Test Accuracy':<16}{'Test Loss'}")
    print(f"{'Original':<15}{128:<16}{test_accuracy:<16.4f}{test_loss:.4f}")
    print(f"{'Experimental':<15}{256:<16}{exp_accuracy:<16.4f}{exp_loss:.4f}")


if __name__ == "__main__":
    main()
