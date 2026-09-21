# Deep Learning Assignment
## MNIST Handwritten Digit Classification using TensorFlow/Keras

---

## 1. Aim

To build and train a simple neural network using TensorFlow/Keras to classify handwritten digits from 0 to 9 using the MNIST dataset, evaluate its performance, visualize training results, test the model on five handwritten digit images from the test dataset, and perform one experiment by modifying the number of neurons in the hidden layer.

## 2. Objectives

1. Load and explore the MNIST handwritten digit dataset.
2. Display sample images from the dataset.
3. Split the training data into training and validation sets, and normalize the image data.
4. Design a neural network using TensorFlow/Keras.
5. Compile and train the neural network.
6. Evaluate the model using test accuracy.
7. Visualize training and validation accuracy and loss.
8. Test the trained model on five randomly selected handwritten digit images from the MNIST test dataset.
9. Compare actual and predicted labels.
10. Perform one experiment by changing the number of neurons in the hidden layer.
11. Compare the original and modified model results.

---

## 3. Introduction

Handwritten digit recognition is a common introductory problem in machine learning and deep learning. The objective is to identify which digit, from 0 to 9, is represented by an input image.

For this experiment, the **MNIST dataset** is used. MNIST contains grayscale images of handwritten digits and is widely used for evaluating image classification algorithms.

A simple feed-forward neural network is implemented using **TensorFlow/Keras**. The network is organized into small, reusable functions (data loading, preprocessing, model building, training, evaluation, and visualization) rather than one long script, and `train_test_split` from scikit-learn is used to create a dedicated validation set.

## 4. Dataset Description

| Property | Value |
|---|---|
| Training images (before split) | 60,000 |
| Testing images | 10,000 |
| Number of classes | 10 (digits 0–9) |
| Image size | 28 × 28 pixels |
| Image type | Grayscale |
| Pixels per image | 784 (28 × 28) |
| Original pixel range | 0–255 |

---

## 5. Theory

### 5.1 Neural Network

A neural network consists of interconnected layers of neurons. Each neuron receives input values, applies weights and a bias, and passes the result through an activation function:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

where $x_i$ is the input, $w_i$ the weight, $b$ the bias, and $z$ the weighted sum. The activation function then transforms $z$.

### 5.2 ReLU Activation

The hidden layer uses the ReLU activation function:

$$ReLU(x) = \max(0, x)$$

ReLU introduces non-linearity, allowing the model to learn more complex patterns.

### 5.3 Softmax Activation

The output layer has 10 neurons, one per digit. Softmax converts the raw outputs into probabilities:

$$P(y=i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$$

The class with the highest probability is the predicted digit.

### 5.4 Data Normalization

Pixel values (0–255) are scaled to the range 0–1:

$$X_{normalized} = \frac{X}{255}$$

Normalization helps the network train more effectively.

### 5.5 Train/Validation Split

Instead of letting Keras carve out a validation set automatically (via `validation_split`), the training data is split explicitly using scikit-learn's `train_test_split`, with `stratify=y_train_full` so each digit class is proportionally represented in both the training and validation sets. This gives more visibility and control over exactly how the split is made.

### 5.6 Loss Function

The model uses **Sparse Categorical Cross-Entropy**, since the target labels are integer class values (0–9). It measures the difference between the true class and the predicted probability distribution.

### 5.7 Optimizer

The **Adam optimizer** updates the network's weights during training, combining momentum with adaptive learning rates.

---

## 6. Model Architecture

```text
Input Image (28 × 28)
        |
        v
     Flatten
   (784 values)
        |
        v
  Dense Layer (128 neurons)
        |
        v
   ReLU Activation
        |
        v
  Dense Layer (10 neurons)
        |
        v
  Softmax Activation
        |
        v
  Predicted Digit (0–9)
```

The model is created by a reusable `build_model()` function so the same code can build both the original (128-neuron) and experimental (256-neuron) versions:

```python
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
```

---

## 7. Algorithm

1. Import TensorFlow, NumPy, Matplotlib, and scikit-learn's `train_test_split`.
2. Load the MNIST dataset using Keras.
3. Split the training data into training and validation sets with `train_test_split` (stratified by label).
4. Display sample images from the training dataset.
5. Normalize pixel values from 0–255 to 0–1 for the training, validation, and test sets.
6. Build a Sequential neural network via `build_model()`: Flatten → Dense(128, ReLU) → Dense(10, Softmax).
7. Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.
8. Train the model for 10 epochs using the explicit validation set.
9. Evaluate the trained model on the test dataset.
10. Plot training and validation accuracy, and training and validation loss.
11. Randomly select five images from the test dataset (fixed seed for reproducibility) and predict their digits.
12. Compare actual and predicted labels.
13. Build a second model with 256 neurons in the hidden layer using the same `build_model()` function, train, and evaluate it.
14. Compare the original and experimental models in a summary table.

---

## 8. Implementation

The implementation is organized as a set of functions plus a `main()` pipeline, rather than one continuous script.

### 8.1 Imports

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
```

### 8.2 Loading the Data and Creating the Validation Split

```python
def load_data():
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full,
    )

    print("Training images:", x_train.shape)
    print("Validation images:", x_val.shape)
    print("Testing images:", x_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test
```

Expected shapes:

```text
Training images:   (54000, 28, 28)
Validation images: (6000, 28, 28)
Testing images:    (10000, 28, 28)
```

### 8.3 Dataset Exploration

```python
def show_sample_images(images, labels, n=10):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
```

**[Insert the sample-images output from the notebook here.]**

The displayed images show handwritten digits represented as grayscale images.

### 8.4 Preprocessing

```python
def preprocess(*datasets):
    return [d.astype("float32") / 255.0 for d in datasets]
```

This is applied to the training, validation, and test images together, scaling all pixel values to the 0–1 range.

### 8.5 Building and Compiling the Model

```python
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
```

| Component | Selection |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| Metric | Accuracy |
| Hidden Activation | ReLU (configurable) |
| Output Activation | Softmax |
| Hidden Neurons | 128 (baseline), 256 (experiment) |

### 8.6 Training the Model

```python
def train_model(model, x_train, y_train, x_val, y_val, epochs=10):
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
    )
```

The model trains for 10 epochs, using the validation set created explicitly in §8.2 rather than an automatic split.

### 8.7 Evaluating the Model

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

**Test Accuracy:** `__________`
**Test Loss:** `__________`

> Fill in with the actual values from your notebook run.

The test accuracy indicates how correctly the trained model classified unseen MNIST test images.

### 8.8 Visualizing Accuracy and Loss

```python
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
```

**[Insert the accuracy graph here.]**
**[Insert the loss graph here.]**

**Observation:** Training and validation accuracy generally increase, while loss decreases, as the number of epochs increases — indicating the network is learning useful features. The validation curves show how well the model generalizes to unseen data.

### 8.9 Testing on Five Handwritten Images

Five images are selected at random (with a fixed seed) from the MNIST test dataset, rather than using separately created handwritten images:

```python
def show_predictions(model, x_test, y_test, n=5, seed=42):
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
```

**[Insert the five-image prediction output here.]**

**Prediction Results**

| Image | Actual Label | Predicted Label | Result |
|---|---:|---:|---|
| 1 | ____ | ____ | ____ |
| 2 | ____ | ____ | ____ |
| 3 | ____ | ____ | ____ |
| 4 | ____ | ____ | ____ |
| 5 | ____ | ____ | ____ |

---

## 9. Experiment: Changing the Number of Hidden Neurons

The original model uses 128 neurons in the hidden Dense layer. For the experiment, `build_model()` and `train_model()` are reused with `hidden_neurons=256`, keeping all other parameters unchanged:

```python
experiment_model = build_model(hidden_neurons=256)
experiment_history = train_model(experiment_model, x_train, y_train, x_val, y_val)

exp_loss, exp_accuracy = experiment_model.evaluate(x_test, y_test)

print("Experiment Test Loss:", exp_loss)
print("Experiment Test Accuracy:", exp_accuracy)
```

Because both models are built from the same function, the comparison isolates the effect of hidden-layer size — nothing else in the architecture or training setup differs.

**Comparison**

| Model | Hidden Neurons | Epochs | Test Accuracy | Test Loss |
|---|---:|---:|---:|---:|
| Original | 128 | 10 | ______ | ______ |
| Experimental | 256 | 10 | ______ | ______ |

> Fill in using the actual results printed by `main()`.

**Observation:** Increasing the number of neurons from 128 to 256 increases the hidden layer's capacity, allowing it to learn a larger number of patterns. Whether this meaningfully improves test accuracy should be judged from the actual results — more neurons do not automatically guarantee better performance.

---

## 10. Results

- Training images (after split): ~54,000 | Validation images: ~6,000 | Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Validation split method: `train_test_split` (stratified, 10%, seed 42)
- Hidden layer: 128 neurons, ReLU activation
- Output layer: 10 neurons, Softmax activation
- Epochs: 10 | Optimizer: Adam | Loss: Sparse Categorical Cross-Entropy
- **Test accuracy:** `__________`
- Experimental hidden neurons: 256 | **Experimental test accuracy:** `__________`

## 11. Discussion

This experiment demonstrates the basic workflow of a deep learning classification problem, implemented as a small set of reusable functions rather than a single linear script. The MNIST images were normalized to make input values suitable for training, and an explicit, stratified `train_test_split` was used to create the validation set instead of relying on Keras's automatic split — giving more control and transparency over how the data is partitioned.

The Flatten layer converted each two-dimensional image into a one-dimensional input vector. A Dense hidden layer with ReLU activation learned patterns in the images, and the final Softmax layer produced per-class probabilities, with the highest-probability class taken as the prediction.

Training and validation curves were used to observe performance across epochs, and the model was evaluated on a held-out test set that was untouched during training or validation. Finally, because model construction is factored into `build_model()`, the experiment (256 neurons) reuses the exact same code path as the baseline, isolating hidden-layer size as the only variable being tested.

## 12. Advantages

1. Simple and easy to implement.
2. High classification accuracy on the MNIST dataset.
3. TensorFlow/Keras provides a straightforward model-building interface.
4. The model trains relatively quickly.
5. Reusable functions (`build_model`, `train_model`, `plot_history`) make it easy to run further experiments with minimal code duplication.
6. The explicit, stratified train/validation split gives more control than an automatic split.

## 13. Limitations

1. Uses a fully connected network rather than a Convolutional Neural Network (CNN).
2. Flattening the image discards some spatial structure.
3. Performance on more complex real-world handwriting may be lower than on MNIST.
4. Only one architectural parameter (hidden neuron count) was changed in the experiment.
5. The five "handwritten" test samples come from the MNIST test set, not separately created images.

## 14. Future Improvements

- Use a Convolutional Neural Network (CNN).
- Add dropout layers to reduce overfitting.
- Tune the number and size of hidden layers.
- Apply image augmentation.
- Test different activation functions (the `activation` parameter in `build_model()` already supports this).
- Use batch normalization.
- Perform systematic hyperparameter tuning.
- Test the model on handwritten images created outside the MNIST dataset.

## 15. Conclusion

A simple neural network for handwritten digit classification was implemented using TensorFlow/Keras and the MNIST dataset, structured as reusable functions for data loading, preprocessing, model building, training, evaluation, and visualization. The dataset was explored, split into training/validation/test sets using a stratified `train_test_split`, normalized, and used to train a network consisting of a Flatten layer, a Dense hidden layer with ReLU activation, and a Softmax output layer with ten classes.

The model was evaluated using test accuracy, with training and validation accuracy/loss visualized across epochs. Five randomly selected test images were classified and compared against their actual labels. A follow-up experiment increased the hidden layer from 128 to 256 neurons using the same `build_model()` function, and the two configurations were compared in a summary table.

Overall, the exercise provides a practical understanding of the deep learning workflow: dataset preparation, model design, training, evaluation, prediction, visualization, and experimentation.

---

## 16. Technologies Used

Python · TensorFlow · Keras · NumPy · Matplotlib · scikit-learn · Jupyter Notebook / Google Colab

## 17. Repository Structure

```text
MNIST-Digit-Classification/
├── classifier.py                       (or MNIST_Digit_Classification.ipynb)
├── MNIST_Report.md
├── README.md
├── requirements.txt
└── handwritten_images/   (optional — test images are drawn from MNIST itself)
```

## 18. References

1. TensorFlow Documentation — MNIST Dataset and Keras.
2. Keras Documentation — Sequential Models and Dense Layers.
3. scikit-learn Documentation — `train_test_split`.
4. MNIST Database of Handwritten Digits.
5. Python NumPy Documentation.
6. Matplotlib Documentation.
