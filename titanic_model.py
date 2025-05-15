import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODEL_DIR = "."
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")

df = pd.read_csv("train.csv")

sex_encoder = LabelEncoder()
df["Sex"] = sex_encoder.fit_transform(df["Sex"])

df["Embarked"] = df["Embarked"].fillna("S")
embarked_encoder = LabelEncoder()
df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])

median_age = df["Age"].median()
median_fare = df["Fare"].median()
df["Age"] = df["Age"].fillna(median_age)
df["Fare"] = df["Fare"].fillna(median_fare)

features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
X = df[features]
y = df["Survived"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
