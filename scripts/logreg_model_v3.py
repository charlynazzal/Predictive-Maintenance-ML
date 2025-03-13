import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ✅ Ensure script can access parent directory (for config)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py
from load_data_v5 import X_train, X_test, y_train, y_test  # ✅ Import the split data

# ✅ Try different values of C
C_values = [0.01, 0.1, 1, 10, 100]

for C in C_values:
    print(f"\n🔹 Training Logistic Regression with C = {C}")
    
    # ✅ Train model with newton-cg solver and different C values
    model = LogisticRegression(solver="newton-cg", C=C, max_iter=500)
    model.fit(X_train, y_train)
    
    # ✅ Make Predictions
    y_pred = model.predict(X_test)
    
    # ✅ Evaluate Performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy with C = {C}: {accuracy:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

