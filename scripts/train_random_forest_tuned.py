import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# ✅ Ensure the script can access the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # ✅ Import paths from config.py
from load_data_v4modified import df_balanced  # ✅ Import balanced dataset

print("\n🚀 FINE-TUNING RANDOM FOREST CLASSIFIER\n")

# ✅ Define Features (X) and Target (y)
X = df_balanced.drop(columns=["Target"])  # Features
y = df_balanced["Target"]  # Target variable

# ✅ Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 300, 500],  # Number of trees
    'max_depth': [10, 20, None],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Min samples needed to split
    'min_samples_leaf': [1, 2, 4],  # Min samples in leaf node
}

# ✅ Initialize Random Forest Model
rf = RandomForestClassifier(random_state=42)

# ✅ Grid Search to Find Best Hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# ✅ Best Parameters
best_params = grid_search.best_params_
print(f"\n🎯 Best Hyperparameters: {best_params}")

# ✅ Train Final Model with Best Parameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = best_rf.predict(X_test)

# ✅ Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Fine-Tuned Random Forest Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Save the fine-tuned Random Forest model
import joblib

os.makedirs(config.MODELS_FOLDER, exist_ok=True)
rf_model_path = os.path.join(config.MODELS_FOLDER, "best_rf_model.pkl")
joblib.dump(best_rf, rf_model_path)

print(f"\n✅ Fine-tuned Random Forest model saved at: {rf_model_path}")
