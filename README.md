# Iris Flower Species Classification 🌸

A machine learning classification project that predicts the species of an iris flower based on its sepal and petal measurements using a Decision Tree Classifier.

---

## 📌 Project Overview
The objective of this project is to build and evaluate a supervised learning model that classifies iris flowers into one of three species:
- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

The model is trained on the classic Iris dataset using four morphological features:
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

---

## ⚙️ Methodology & Pipeline

1. **Data Loading:** Fetched directly via `sklearn.datasets.load_iris`.
2. **Data Splitting:** 80% training set and 20% testing set with stratified sampling (`stratify=y`) to maintain balanced class distributions.
3. **Model Selection:** `DecisionTreeClassifier` (scikit-learn).
4. **Performance Evaluation:** Evaluated using Accuracy Score, Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.
5. **Inference / Testing:** Tested with new sample measurements to predict flower class.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install scikit-learn
