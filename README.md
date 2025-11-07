# 🐾 Cats vs Dogs Image Classifier

This project trains a **Convolutional Neural Network (CNN)** to classify images of **cats 🐱** and **dogs 🐶** using TensorFlow and Keras.  
It focuses on building, training, and evaluating a simple deep learning model for binary image classification.

---

## 🧠 Project Overview

The notebook includes:
- Image preprocessing using `ImageDataGenerator`
- CNN architecture built with TensorFlow/Keras
- Training and validation visualization
- Model evaluation metrics
- Saved model for reuse (`.keras` format)

---

## 📁 Project Structure

```
cats-dogs-classifier/
│
├── requirements.txt         # Dependencies
├── .gitignore                       
├── README.md                        
│
├── models/
│   └── cats_dogs_classification_model.keras    # Trained CNN model
│
└── data/
    ├── train.zip
    ├── train_split.zip
    └── test1.zip
```

---

## 🧠 Model Summary

| Parameter               | Value                                        |
| ----------------------- | -------------------------------------------- |
| **Architecture**        | Custom CNN (3 Conv layers + Dense + Dropout) |
| **Image Size**          | 224 × 224 px                                 |
| **Optimizer**           | Adam                                         |
| **Loss Function**       | Binary Crossentropy                          |
| **Batch Size**          | 32                                           |
| **Epochs**              | 12                                           |
| **Training Accuracy**   | ~76%                                         |
| **Validation Accuracy** | ~77%                                         |

✅ The model generalizes well and performs reliably on unseen images.

---

## 🧩 Dataset Details

* **Source:** [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
* **Total images:** 25,000
* **Split:**

  * `train.zip` → original train dataset
  * `train_split.zip` → train validation split dataset
  * `test1.zip` → test images
* **Preprocessing:** Resizing, normalization, and augmentation

---


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/berilkln/cats-dogs-classifier.git
cd cats-dogs-classifier
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Unzip the datasets

```bash
unzip data/train.zip -d data/train
unzip data/train_split.zip -d data/train_split
unzip data/test1.zip -d data/test1
```


---

## 💾 Model File (Git LFS)

The trained model (`.keras`) and dataset ZIPs are managed via **Git LFS** for large file support.
Make sure to install LFS before cloning:

```bash
git lfs install
git clone https://github.com/berilkln/cats-dogs-classifier.git
```

---

## 🧰 Requirements
```
tensorflow==2.16.2
numpy
matplotlib
Pillow
tqdm
```
💡 Note: Jupyter Notebook is not included in this file.  
You can open and run the notebook in your own Jupyter environment.

---

## 📘 Educational Purpose

> This project was developed **for educational use** only.
> It demonstrates how to build and train a CNN for image classification

---
