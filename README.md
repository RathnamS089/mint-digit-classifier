# MNIST Digit Classifier: From Linear to Convolutional Networks

A PyTorch-based deep learning project showcasing the evolution of a handwritten digit classifier trained on the classic MNIST dataset. This repository documents the performance jump achieved by moving from a spatially blind Multi-Layer Perceptron (MLP) to a spatially aware Convolutional Neural Network (CNN).

---

## 📈 Performance & Results

By preserving spatial structure and extracting local features with visual filters, the CNN model successfully cut the error rate of the linear network **by more than half**.

| Model Architecture | Test Accuracy (10,000 Unseen Images) | Estimated Mistakes |
| :--- | :--- | :--- |
| **Multi-Layer Perceptron (Linear)** | 96.67% | ~333 |
| **Convolutional Neural Network (CNN)** | **98.56%** | **~144** |

---

## 🧠 Architectural Breakdown

### 1. The Linear Baseline (MLP)
* **Approach:** Flattens the $28 \times 28$ input image into a single 1D vector of 784 pixels.
* **Limitation:** Spatially blind. If an image is shifted by even a few pixels, the model struggles to generalize because it relies on absolute pixel positioning.
* **Structure:** `Linear(784 -> 128) -> ReLU -> Linear(128 -> 64) -> ReLU -> Linear(64 -> 10)`

### 2. The Convolutional Upgrade (CNN)
* **Approach:** Preserves the 2D matrix structure of the image. Slides a $3 \times 3$ convolutional kernel across the image matrix to extract spatial structural features (edges, curves, loops).
* **Downsampling:** Uses Max Pooling ($2 \times 2$ filters) to compress feature dimensions and retain only the most intense structural signatures, reducing overfitting.
* **Structure:** * `Conv2d(1 -> 16 channels, Kernel=3, Padding=1) -> ReLU -> MaxPool2d(2x2)`
  * `Conv2d(16 -> 32 channels, Kernel=3, Padding=1) -> ReLU -> MaxPool2d(2x2)`
  * `Flatten -> Linear(1568 -> 64) -> ReLU -> Linear(64 -> 10)`

---

## 🛠️ Tech Stack & Concepts Covered

* **Framework:** PyTorch (with `torchvision`)
* **Optimization:** Adam Optimizer (`lr=0.001`), Cross-Entropy Loss
* **Data Streaming:** Automated batching, shuffling, and transformation pipelines using `Dataset` and `DataLoader`
* **Deep Learning Paradigms:** Backpropagation, Gradient Descent, Multi-class Classification, Feature Extraction, Max Pooling, Spatial Generalization

---

## 🚀 How to Run It Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/RathnamS089/mnist-digit-classifier.git](https://github.com/RathnamS089/mnist-digit-classifier.git)
cd mnist-digit-classifier
