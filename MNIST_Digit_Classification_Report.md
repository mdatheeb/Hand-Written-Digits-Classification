## MNIST Handwritten Digit Classification using TensorFlow/Keras

---

## 1. Aim

To build and train a simple neural network using TensorFlow/Keras to classify handwritten digits from 0 to 9 using the MNIST dataset, evaluate its performance, visualize training results, test the model on five handwritten digit images from the test dataset, and perform one experiment by modifying the number of neurons in the hidden layer.

## 2. Objectives

1. Load and explore the MNIST handwritten digit dataset.
2. Display sample images from the dataset.
3. Preprocess and normalize the image data.
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

A simple feed-forward neural network is implemented using **TensorFlow/Keras**. The network learns patterns from the pixel values of the training images and uses those learned patterns to classify unseen test images.

## 4. Dataset Description

| Property | Value |
|---|---|
| Training images | 60,000 |
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

### 5.5 Loss Function

The model uses **Sparse Categorical Cross-Entropy**, since the target labels are integer class values (0–9). It measures the difference between the true class and the predicted probability distribution.

### 5.6 Optimizer

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

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

---

## 7. Algorithm

1. Import TensorFlow, NumPy, and Matplotlib.
2. Load the MNIST dataset using Keras.
3. Separate the dataset into training and testing data.
4. Display sample images from the training dataset.
5. Normalize pixel values from 0–255 to 0–1.
6. Create a Sequential neural network.
7. Flatten each 28 × 28 image into 784 input values.
8. Add a Dense hidden layer with 128 neurons and ReLU activation.
9. Add an output layer with 10 neurons and Softmax activation.
10. Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.
11. Train the model for 10 epochs, using 10% of the training data for validation.
12. Evaluate the trained model on the test dataset.
13. Plot training and validation accuracy, and training and validation loss.
14. Randomly select five images from the test dataset and predict their digits.
15. Compare actual and predicted labels.
16. Create a second model with 256 neurons in the hidden layer, train, and evaluate it.
17. Compare the original and experimental models.

---

## 8. Implementation

### 8.1 Import Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
```

### 8.2 Load the Dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)
```

Expected output:

```text
Training images: (60000, 28, 28)
Training labels: (60000,)
Testing images: (10000, 28, 28)
Testing labels: (10000,)
```

### 8.3 Dataset Exploration

```python
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
```

The displayed images show handwritten digits represented as grayscale images.

### 8.4 Data Preprocessing

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

This normalizes pixel values to the range 0–1 ($X_{normalized} = X / 255$), making the inputs smaller and easier for the network to train on.

### 8.5 Building the Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

- `Flatten` converts each 28 × 28 image into a 1-D vector of 784 values.
- The hidden `Dense` layer has 128 neurons with ReLU activation.
- The output `Dense` layer has 10 neurons (one per digit) with Softmax activation.

### 8.6 Compiling the Model

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

| Component | Selection |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| Metric | Accuracy |
| Hidden Activation | ReLU |
| Output Activation | Softmax |

### 8.7 Training the Model

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.1
)
```

The model trains for 10 epochs, with the training data split into 90% training / 10% validation. The validation set monitors performance on unseen samples during training.

### 8.8 Evaluating the Model

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

**Test Accuracy:** `__________`
**Test Loss:** `__________`


The test accuracy indicates how correctly the trained model classified unseen MNIST test images.

### 8.9 Visualizing Accuracy and Loss

```python
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()
```



```python
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
```

**Observation:** Training and validation accuracy generally increase, while loss decreases, as the number of epochs increases — indicating the network is learning useful features. The validation curves show how well the model generalizes to unseen data.

### 8.10 Testing on Five Handwritten Images

Since separate handwritten image files were not used, five images were randomly selected (with a fixed seed for reproducibility) from the MNIST test dataset.

```python
np.random.seed(42)

sample_indices = np.random.choice(len(x_test), 5, replace=False)
sample_images = x_test[sample_indices]
actual_labels = y_test[sample_indices]

predictions = model.predict(sample_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)
```

```python
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(sample_images[i], cmap="gray")
    plt.title(f"Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
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

The original model uses 128 neurons in the hidden Dense layer. For the experiment, this was increased to 256 neurons, keeping all other parameters unchanged.

```python
experiment_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

experiment_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

experiment_history = experiment_model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.1
)

experiment_loss, experiment_accuracy = experiment_model.evaluate(x_test, y_test)

print("Experiment Test Loss:", experiment_loss)
print("Experiment Test Accuracy:", experiment_accuracy)
```

**Comparison**

| Model | Hidden Neurons | Epochs | Test Accuracy | Test Loss |
|---|---:|---:|---:|---:|
| Original | 128 | 10 | ______ | ______ |
| Experimental | 256 | 10 | ______ | ______ |

> Fill in using the actual results produced by the notebook.

**Observation:** Increasing the number of neurons from 128 to 256 increases the hidden layer's capacity, allowing it to learn a larger number of patterns. Whether this meaningfully improves test accuracy should be judged from the actual results — more neurons do not automatically guarantee better performance.

---

## 10. Results

- Training dataset: 60,000 images | Testing dataset: 10,000 images
- Image dimensions: 28 × 28 pixels
- Hidden layer: 128 neurons, ReLU activation
- Output layer: 10 neurons, Softmax activation
- Epochs: 10 | Optimizer: Adam | Loss: Sparse Categorical Cross-Entropy
- **Test accuracy:** `__________`
- Experimental hidden neurons: 256 | **Experimental test accuracy:** `__________`

## 11. Discussion

This experiment demonstrates the basic workflow of a deep learning classification problem. The MNIST images were normalized to make input values suitable for training, and the Flatten layer converted each two-dimensional image into a one-dimensional input vector. A Dense hidden layer with ReLU activation learned patterns in the images, and the final Softmax layer produced per-class probabilities, with the highest-probability class taken as the prediction.

Training and validation curves were used to observe performance across epochs, and the model was evaluated on a held-out test set. Finally, the number of hidden neurons was changed from 128 to 256 to observe the effect of increasing model capacity.

## 12. Advantages

1. Simple and easy to implement.
2. High classification accuracy on the MNIST dataset.
3. TensorFlow/Keras provides a straightforward model-building interface.
4. The model trains relatively quickly.
5. The experiment demonstrates how architecture choices affect performance.

## 13. Limitations

1. Uses a fully connected network rather than a Convolutional Neural Network (CNN).
2. Flattening the image discards some spatial structure.
3. Performance on more complex real-world handwriting may be lower than on MNIST.
4. Only one architectural parameter was changed in the experiment.
5. The five "handwritten" test samples come from the MNIST test set, not separately created images.

## 14. Future Improvements

- Use a Convolutional Neural Network (CNN).
- Add dropout layers to reduce overfitting.
- Tune the number and size of hidden layers.
- Apply image augmentation.
- Test different activation functions.
- Use batch normalization.
- Perform systematic hyperparameter tuning.
- Test the model on handwritten images created outside the MNIST dataset.

## 15. Conclusion

A simple neural network for handwritten digit classification was implemented using TensorFlow/Keras and the MNIST dataset. The dataset was explored, normalized, and used to train a network consisting of a Flatten layer, a Dense hidden layer with ReLU activation, and a Softmax output layer with ten classes.

The model was evaluated using test accuracy, with training and validation accuracy/loss visualized across epochs. Five randomly selected test images were classified and compared against their actual labels. A follow-up experiment increased the hidden layer from 128 to 256 neurons, and the two configurations were compared.

Overall, the exercise provides a practical understanding of the deep learning workflow: dataset preparation, model design, training, evaluation, prediction, visualization, and experimentation.

---

## 16. Technologies Used

Python · TensorFlow · Keras · NumPy · Matplotlib · Jupyter Notebook / Google Colab

## 17. Repository Structure

```text
MNIST-Digit-Classification/
├── MNIST_Digit_Classification.ipynb
├── MNIST_Report.md
├── README.md
├── requirements.txt
```

## 18. References

1. TensorFlow Documentation — MNIST Dataset and Keras.
2. Keras Documentation — Sequential Models and Dense Layers.
3. MNIST Database of Handwritten Digits.
4. Python NumPy Documentation.
5. Matplotlib Documentation.
