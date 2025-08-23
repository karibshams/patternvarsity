# Mango Classification Project using BYOL and ResNet-18

## Overview

This project demonstrates **self-supervised learning** using **BYOL (Bootstrap Your Own Latent)** and fine-tuning a **ResNet-18** model for mango classification. The dataset includes images of mangoes, and the goal is to train the model to classify different varieties of mangoes. The project consists of pre-training using BYOL, followed by supervised fine-tuning to improve classification accuracy.

## Project Versions

### **Old Version:**

In the earlier version of this project, we used a basic **supervised learning pipeline** for mango classification. The steps involved:

* **Pre-processing and augmentation** of mango images.
* Training a **ResNet-18** model with labeled data.
* Fine-tuning the model with basic **transfer learning** techniques.

### **New Version:**

In the updated version, we use **BYOL (Bootstrap Your Own Latent)** for **self-supervised learning** followed by **supervised fine-tuning** on the **ResNet-18** model:

* **BYOL** pre-training (without labels) to learn meaningful representations of mango images.
* **Fine-tuning** the model with labeled data to improve classification accuracy.
* **Silhouette score** and **t-SNE visualization** to evaluate feature quality.

## Files and Directories

* **`mango10.ipynb, mango20.ipynb, mango30.ipynb, etc.`**: These notebooks represent different models or experiments related to mango classification at various stages, from pre-training to fine-tuning.
* **`ResNet-18`**: The model architecture used for both pre-training and fine-tuning.
* **`BYOL_ResNet`**: Implementation of BYOL using the ResNet-18 backbone.

## Steps

### 1. Pre-training (BYOL)

In the new version, the first part of the project involves **pre-training the ResNet-18 model** using **BYOL** (Bootstrap Your Own Latent). This technique is a **self-supervised learning** method that uses no labeled data and helps the model learn meaningful representations. The pre-training process is done on the full dataset with augmentations applied to the images.

### 2. Data Augmentation

Data augmentations for training and validation include:

* **Random Resized Crop**
* **Random Horizontal Flip**
* **Random Vertical Flip**
* **Color Jitter**
* **Random Grayscale**
* **Random Gaussian Blur**
* **Random Rotation**
* **Random Perspective and Affine Transformations**
* **Random Erasing** for robustness.

### 3. Dataset Setup

The dataset is divided into:

* **Training set (70%)**: For both pre-training and supervised fine-tuning.
* **Validation set (15%)**: For monitoring the model's performance during training.
* **Test set (15%)**: For final evaluation after training is completed.

### 4. Fine-tuning with Supervision

After pre-training, the model undergoes **supervised fine-tuning** using a **ResNet-18 encoder**. This step utilizes labeled data to fine-tune the model for the specific task of mango classification. The model is trained using a **cross-entropy loss** and optimized with **AdamW** optimizer.

### 5. Model Evaluation

During the fine-tuning process:

* **Training Loss** and **Validation Accuracy** are tracked.
* **Confusion Matrix** and **Classification Report** are generated for detailed model evaluation.
* **Silhouette Score** and **t-SNE** analysis are performed for visualizing the quality of learned features.

### 6. Hyperparameters

* **Pre-training epochs**: 100
* **Fine-tuning epochs**: 150
* **Batch size for pre-training**: 4
* **Batch size for fine-tuning**: 4
* **Learning rate**: 3e-4 for the encoder, 1e-3 for projector and predictor.
* **EMA decay**: 0.996 for target network update.
* **Gradient accumulation steps**: 32
* **Linear probe**: Option to fine-tune only the final classifier or the whole model.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/karibshams/mango-classification.git
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the **Mango Dataset** (if not already available). Make sure to place it in the correct directory.

### 7. Dependencies

* `torch`
* `torchvision`
* `seaborn`
* `matplotlib`
* `sklearn`
* `tqdm`
* `PIL`
* `numpy`

## Usage

1. **Pre-training**: Start by training the model with **BYOL** on the dataset:

   * `BYOL_ResNet` model pre-trains on the dataset using augmentations for self-supervised learning.

2. **Fine-tuning**: Once the pre-training is complete, use the labeled data to fine-tune the model:

   * Supervised fine-tuning on **ResNet-18** model using the cross-entropy loss.

3. **Evaluation**: After fine-tuning, evaluate the model on the test set and view the results in terms of accuracy, confusion matrix, silhouette score, and t-SNE plots.

## Results

### Performance Metrics

* **Final Test Accuracy**: Measures the model's classification accuracy on the test set.
* **Silhouette Score**: Measures the quality of clustering (feature learning) in the pre-training and post-training steps.
* **Classification Report**: Detailed metrics like precision, recall, and F1-score.
* **Confusion Matrix**: Visualizes how well the model performs across different classes.

## Contributions

Feel free to fork this repository, file issues for bugs or feature requests, and contribute improvements to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Summary:

* **New Version**: Self-supervised learning using BYOL and fine-tuning with ResNet-18. This version includes **t-SNE** and **Silhouette Score** analysis for feature quality visualization.


