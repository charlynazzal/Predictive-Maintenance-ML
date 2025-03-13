import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ✅ Ensure the script can access the parent directory (for config.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py
from load_data_v4modified import df_balanced  # ✅ Import balanced dataset

print("\n🚀 Running: logreg_model_1modified.py")

# ✅ Define Features (X) and Target (y)
X = df_balanced.drop(columns=["Target"])  # Features
y = df_balanced["Target"]  # Target variable

# ✅ Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Train the Logistic Regression Model
model = LogisticRegression(solver="newton-cg", C=0.01, max_iter=500)
model.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = model.predict(X_test)

# ✅ Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

