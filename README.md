# 🐾 Cats vs Dogs Image Classifier

A deep learning project built using **TensorFlow** and **Streamlit**, designed to classify images as **Cat 🐱** or **Dog 🐶**.
This project was developed **for learning purposes**, demonstrating the full workflow of building, training, and deploying a CNN model.

---

## 🚀 Features

* Upload any image (JPG, JPEG, PNG) via a simple Streamlit UI.
* Classifies whether it’s a **cat** or a **dog** in real time.
* Built with a **custom Convolutional Neural Network (CNN)** trained from scratch.

---

## 📁 Project Structure

```
cats-dogs-classifier/
│
├── apps/
│   └── app.py               # Streamlit web app
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

* **Source:** [Kaggle Dogs vs Cats Dataset (Microsoft Research)](https://www.kaggle.com/c/dogs-vs-cats/data)
* **Total images:** 25,000
* **Split:**

  * `train.zip` → original train dataset
  * `train_split.zip` → train validation split dataset
  * `test1.zip` → test images
* **Preprocessing:** Resizing, normalization, and augmentation

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/cats-dogs-classifier.git
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

### 4️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

Then open the app in your browser at ...

---

## 🖼️ How to Use

1. Launch the app (`streamlit run app.py`).
2. Upload an image of a **cat** or **dog**.
3. Wait for the model to analyze it.
4. See the prediction and model confidence instantly.

If the model is unsure or the image doesn’t look like a cat or dog,
it displays:

> ❌ “Not a cat or dog (model uncertain)”

---

## 💾 Model File (Git LFS)

The trained model (`.keras`) and dataset ZIPs are managed via **Git LFS** for large file support.
Make sure to install LFS before cloning:

```bash
git lfs install
git clone https://github.com/<your-username>/cats-dogs-classifier.git
```

---

## 🧰 Requirements

* Python 3.9+
* TensorFlow 2.x
* Streamlit
* Pillow
* NumPy

(Already listed in `requirements.txt`)

---

## 📘 Educational Purpose

> This project was developed **for educational use** only.
> It demonstrates:
>
> * How to build and train a CNN for image classification
> * How to deploy a trained model using Streamlit

---

## ✨ Future Improvements

* Integrate **Transfer Learning (VGG16, ResNet50)** for better accuracy

---
