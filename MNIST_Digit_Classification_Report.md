# Deep Learning Assignment
## MNIST Handwritten Digit Classification using TensorFlow/Keras

---

## 1. Aim

To build and train a simple neural network using TensorFlow/Keras to classify handwritten digits from 0 to 9 using the MNIST dataset, evaluate its performance, visualize training results, test the model on five handwritten digit images from the test dataset, and perform one experiment by modifying the number of neurons in the hidden layer.

---

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

---

## 4. Dataset Description

The MNIST dataset contains:

- 60,000 training images
- 10,000 testing images
- 10 different classes
- Classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9
- Image size: 28 × 28 pixels
- Image type: Grayscale

Each image contains 784 pixels because:

\[
28 \times 28 = 784
\]

Each pixel originally has an intensity value between 0 and 255.

---

## 5. Theory

### 5.1 Neural Network

A neural network consists of interconnected layers of neurons. Each neuron receives input values, applies weights and a bias, and passes the result through an activation function.

For a neuron, the weighted sum can be represented as:

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

where:

- \(x_i\) = input
- \(w_i\) = weight
- \(b\) = bias
- \(z\) = weighted sum

The activation function then transforms the value of \(z\).

### 5.2 ReLU Activation

The hidden layer uses the ReLU activation function:

\[
ReLU(x) = max(0,x)
\]

ReLU introduces non-linearity into the network and allows the model to learn more complex patterns.

### 5.3 Softmax Activation

The output layer contains 10 neurons, one for each digit from 0 to 9.

Softmax converts the output values into probabilities:

\[
P(y=i)=\frac{e^{z_i}}{\sum_{j=1}^{10}e^{z_j}}
\]

The class having the highest probability is selected as the predicted digit.

### 5.4 Data Normalization

The original pixel values range from 0 to 255. They are normalized to a range between 0 and 1:

\[
X_{normalized} = \frac{X}{255}
\]

Normalization helps the neural network train more effectively.

### 5.5 Loss Function

The model uses **Sparse Categorical Cross-Entropy** because the target labels are integer class values from 0 to 9.

The loss function measures the difference between the actual class and the predicted probability distribution.

### 5.6 Optimizer

The **Adam optimizer** is used to update the network's weights during training. Adam combines ideas from momentum and adaptive learning rates and is commonly used for neural network training.

---

## 6. Model Architecture

The neural network consists of the following layers:

```text
Input Image
   28 × 28
      |
      v
Flatten
   784 values
      |
      v
Dense Layer
 128 neurons
      |
      v
 ReLU Activation
      |
      v
Dense Layer
 10 neurons
      |
      v
Softmax Activation
      |
      v
Predicted Digit
  0 - 9
```

### Original Model

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
11. Train the model for 10 epochs.
12. Use 10% of the training data for validation.
13. Evaluate the trained model on the test dataset.
14. Plot training and validation accuracy.
15. Plot training and validation loss.
16. Randomly select five images from the test dataset.
17. Predict the digits in the five selected images.
18. Compare actual and predicted labels.
19. Create a second model with 256 neurons in the hidden layer.
20. Train and evaluate the modified model.
21. Compare the original and experimental models.

---

## 8. Implementation

### 8.1 Import Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

### 8.2 Load the MNIST Dataset

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)
```

Expected dataset dimensions:

```text
Training images: (60000, 28, 28)
Training labels: (60000,)
Testing images: (10000, 28, 28)
Testing labels: (10000,)
```

---

## 9. Dataset Exploration

Sample images were displayed to understand the structure of the MNIST dataset.

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

**Insert the output image from the Jupyter Notebook here.**

---

## 10. Data Preprocessing

The pixel values were converted to floating-point values and normalized between 0 and 1.

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

Normalization is performed using:

\[
X_{normalized} = \frac{X}{255}
\]

This makes the input values smaller and helps the neural network train efficiently.

---

## 11. Creating the Neural Network

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

The `Flatten` layer converts each 28 × 28 image into a one-dimensional vector containing 784 values.

The hidden Dense layer contains 128 neurons and uses ReLU activation.

The final Dense layer contains 10 neurons, corresponding to the ten possible digit classes.

---

## 12. Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

The model uses:

| Component | Selection |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| Metric | Accuracy |
| Hidden Activation | ReLU |
| Output Activation | Softmax |

---

## 13. Model Training

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.1
)
```

The model was trained for 10 epochs.

The training dataset was divided into:

- 90% training data
- 10% validation data

The validation data was used to monitor the model's performance on unseen samples during training.

---

## 14. Model Evaluation

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

### Test Result

**Test Accuracy:** `__________`

**Test Loss:** `__________`

> Enter the actual values obtained from your Jupyter Notebook.

The test accuracy indicates how correctly the trained model classified the unseen MNIST test images.

---

## 15. Training and Validation Accuracy

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

**Insert the accuracy graph from the Jupyter Notebook here.**

### Observation

Training and validation accuracy generally increase as the number of epochs increases. This indicates that the neural network is learning useful features from the training images.

---

## 16. Training and Validation Loss

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

**Insert the loss graph from the Jupyter Notebook here.**

### Observation

The training loss generally decreases as the model learns from the training data. The validation loss provides an indication of how well the model generalizes to unseen data.

---

## 17. Testing on Five Handwritten Images

Since separate handwritten image files were not used, five randomly selected handwritten digit images were taken from the MNIST test dataset.

A fixed random seed was used so that the same five images can be selected again.

```python
np.random.seed(42)

sample_indices = np.random.choice(
    len(x_test),
    5,
    replace=False
)

sample_images = x_test[sample_indices]
actual_labels = y_test[sample_indices]

predictions = model.predict(
    sample_images,
    verbose=0
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)
```

The images and predictions were displayed using:

```python
plt.figure(figsize=(12, 4))

for i in range(5):

    plt.subplot(1, 5, i + 1)

    plt.imshow(sample_images[i], cmap="gray")

    plt.title(
        f"Actual: {actual_labels[i]}\n"
        f"Predicted: {predicted_labels[i]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
```

**Insert the five-image prediction output from the Jupyter Notebook here.**

### Prediction Results

| Image | Actual Label | Predicted Label | Result |
|---|---:|---:|---|
| 1 | ____ | ____ | ____ |
| 2 | ____ | ____ | ____ |
| 3 | ____ | ____ | ____ |
| 4 | ____ | ____ | ____ |
| 5 | ____ | ____ | ____ |

The predicted labels were compared with the actual labels to determine whether each classification was correct.

---

# 18. Experiment

## Experiment: Changing the Number of Hidden Neurons

The original model contains:

```text
128 neurons
```

in the hidden Dense layer.

For the experiment, the number of neurons was increased to:

```text
256 neurons
```

All other parameters were kept the same.

### Original Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

### Experimental Model

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
```

---

## 19. Experimental Model Evaluation

```python
experiment_loss, experiment_accuracy = experiment_model.evaluate(
    x_test,
    y_test
)

print("Experiment Test Loss:", experiment_loss)
print("Experiment Test Accuracy:", experiment_accuracy)
```

### Comparison

| Model | Hidden Neurons | Epochs | Test Accuracy | Test Loss |
|---|---:|---:|---:|---:|
| Original | 128 | 10 | ______ | ______ |
| Experimental | 256 | 10 | ______ | ______ |

> Fill the table using the actual results produced by the notebook.

### Observation

Increasing the number of neurons from 128 to 256 increases the capacity of the hidden layer. The experimental model can therefore learn a larger number of patterns from the input data.

The actual effect on test accuracy should be determined from the obtained experimental results. Increasing the number of neurons does not automatically guarantee a large improvement in test accuracy.

---

# 20. Results

The neural network successfully learned to classify handwritten digits from the MNIST dataset.

The main results obtained were:

- Training dataset: 60,000 images
- Testing dataset: 10,000 images
- Image dimensions: 28 × 28 pixels
- Hidden layer: 128 neurons
- Hidden activation: ReLU
- Output neurons: 10
- Output activation: Softmax
- Number of epochs: 10
- Optimizer: Adam
- Loss function: Sparse Categorical Cross-Entropy
- Test accuracy: **__________**
- Experimental hidden neurons: 256
- Experimental test accuracy: **__________**

---

# 21. Discussion

The experiment demonstrates the basic workflow of a deep learning classification problem.

The MNIST images were first normalized to make the input values suitable for neural network training. The Flatten layer converted the two-dimensional images into one-dimensional input vectors. A Dense hidden layer with ReLU activation was then used to learn patterns in the images.

The final Softmax layer produced probabilities for all ten digit classes. The class with the highest probability was selected as the predicted digit.

The training and validation graphs were used to observe how model performance changed across epochs. The model was also evaluated on the test dataset, which was not used during training.

Finally, the number of hidden neurons was changed from 128 to 256 to observe the effect of increasing the model's capacity.

---

# 22. Advantages

1. Simple and easy to implement.
2. High classification accuracy on the MNIST dataset.
3. TensorFlow/Keras provides a straightforward model-building interface.
4. The model trains relatively quickly.
5. The experiment demonstrates how neural network architecture can affect performance.

---

# 23. Limitations

1. The model uses a fully connected network rather than a Convolutional Neural Network (CNN).
2. Flattening the image removes some of the spatial structure of the image.
3. Performance on more complex real-world handwriting may be lower than on MNIST.
4. Only one architectural parameter was changed in the experiment.
5. The five handwritten test samples are selected from the MNIST test dataset rather than separately created handwritten images.

---

# 24. Future Improvements

The model could be improved by:

- Using a Convolutional Neural Network (CNN).
- Adding dropout layers to reduce overfitting.
- Increasing or tuning the number of hidden layers.
- Applying image augmentation.
- Testing different activation functions.
- Using batch normalization.
- Performing hyperparameter tuning.
- Testing the model on handwritten images created outside the MNIST dataset.

---

# 25. Conclusion

A simple neural network for handwritten digit classification was successfully implemented using TensorFlow/Keras and the MNIST dataset.

The dataset was explored, normalized, and used to train a neural network consisting of a Flatten layer, a Dense hidden layer with ReLU activation, and a Softmax output layer with ten classes.

The model was evaluated using test accuracy, and training and validation accuracy and loss were visualized. Five randomly selected images from the MNIST test dataset were also classified and their actual labels were compared with the predicted labels.

An additional experiment was performed by increasing the number of neurons in the hidden layer from 128 to 256. The results of both configurations were compared to observe the effect of changing the network architecture.

The experiment provides a practical understanding of the basic deep learning workflow, including dataset preparation, neural network design, training, evaluation, prediction, visualization, and experimentation.

---

# 26. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab

---

# 27. GitHub Repository Structure

The assignment repository is organized as follows:

```text
MNIST-Digit-Classification/
│
├── MNIST_Digit_Classification.ipynb
├── MNIST_Report.md
├── README.md
├── requirements.txt
│
└── handwritten_images/
```

Since the five test images are selected directly from the MNIST dataset, the `handwritten_images` folder is optional and does not need to contain external images.

---

# 28. References

1. TensorFlow Documentation — MNIST Dataset and Keras.
2. Keras Documentation — Sequential Models and Dense Layers.
3. MNIST Database of Handwritten Digits.
4. Python NumPy Documentation.
5. Matplotlib Documentation.

---

## End of Report
