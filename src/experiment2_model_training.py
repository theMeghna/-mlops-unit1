import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load Dataset (Local CSV)
# -----------------------------
data = pd.read_csv("train.csv")   # use Titanic dataset

print("First 5 rows:")
print(data.head())

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
# Drop unnecessary columns
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Handle missing values
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Convert categorical to numeric
data = pd.get_dummies(data, drop_first=True)

# -----------------------------
# 3. Features & Target
# -----------------------------
X = data.drop("Survived", axis=1)
y = data["Survived"]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# -----------------------------
# 5. Train Model
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------
# 6. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8. Save Model
# -----------------------------
joblib.dump(model, "titanic_model.pkl")
print("\nModel saved as titanic_model.pkl")