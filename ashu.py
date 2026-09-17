from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
#          IRIS FLOWER SPECIES CLASSIFICATION
# ==========================================================

print("\n" + "=" * 60)
print("          IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 60)

# 1. Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("\n[1] DATASET INFORMATION")
print("-" * 60)
print(f"Features : {', '.join(iris.feature_names)}")
print(f"Classes  : {', '.join(iris.target_names)}")
print(f"Total Samples : {len(X)}")

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n[2] DATASET SPLITTING")
print("-" * 60)
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# 3. Create Model
model = DecisionTreeClassifier(random_state=42)

# 4. Train Model
model.fit(X_train, y_train)

print("\n[3] MODEL TRAINING")
print("-" * 60)
print("Model : Decision Tree Classifier")
print("Status: Training Completed Successfully")

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n[4] MODEL EVALUATION")
print("-" * 60)
print(f"Accuracy          : {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

# 7. Classification Report
print("\n[5] CLASSIFICATION REPORT")
print("-" * 60)
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# 8. Confusion Matrix
print("[6] CONFUSION MATRIX")
print("-" * 60)
print(confusion_matrix(y_test, y_pred))

# 9. New Flower Prediction
print("\n[7] NEW FLOWER PREDICTION")
print("-" * 60)

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Input Measurements:")
print(f"Sepal Length : {new_flower[0][0]} cm")
print(f"Sepal Width  : {new_flower[0][1]} cm")
print(f"Petal Length : {new_flower[0][2]} cm")
print(f"Petal Width  : {new_flower[0][3]} cm")

print(f"\nPredicted Species : {iris.target_names[prediction[0]].upper()}")

print("\n" + "=" * 60)
print("       PROJECT EXECUTED SUCCESSFULLY")
print("=" * 60 + "\n")
