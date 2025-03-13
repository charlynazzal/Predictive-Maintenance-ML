import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ✅ Ensure the script can access the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py
from load_data_v4modified import df_balanced  # ✅ Import balanced dataset

print("\n🚀 TRAINING XGBOOST CLASSIFIER\n")

# ✅ Define Features (X) and Target (y)
X = df_balanced.drop(columns=["Target"])  # Features
y = df_balanced["Target"]  # Target variable

# ✅ Rename Features for XGBoost Compatibility
X = X.rename(columns=lambda col: col.replace("[", "").replace("]", "").replace(" ", "_"))

# ✅ Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Initialize XGBoost Model
model = xgb.XGBClassifier(
    n_estimators=300,  
    max_depth=20,  
    learning_rate=0.1,  
    objective="binary:logistic",  
    eval_metric="logloss",
    random_state=42
)

# ✅ Train Model
model.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = model.predict(X_test)

# ✅ Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 XGBoost Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Save the trained model using the configured path
model.save_model(config.FINAL_XGB_MODEL)
print(f"\n✅ Model saved as {config.FINAL_XGB_MODEL}")
