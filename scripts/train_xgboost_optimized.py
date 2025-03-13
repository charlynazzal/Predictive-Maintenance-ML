import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from load_data_v4modified import df_balanced  # ✅ Import balanced dataset

# ✅ Ensure script can access the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # ✅ Import paths from config.py

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

print("\n🚀 FINE-TUNING XGBOOST CLASSIFIER\n")

# ✅ Debugging: Print paths to check correctness
print(f"🔍 Checking File Paths:")
print(f"TRAIN_DATA: {config.TRAIN_DATA} (Exists: {os.path.exists(config.TRAIN_DATA)})")
print(f"TEST_DATA: {config.TEST_DATA} (Exists: {os.path.exists(config.TEST_DATA)})")
print(f"TRAIN_NN_EMBEDDINGS: {config.TRAIN_NN_EMBEDDINGS} (Exists: {os.path.exists(config.TRAIN_NN_EMBEDDINGS)})")
print(f"TEST_NN_EMBEDDINGS: {config.TEST_NN_EMBEDDINGS} (Exists: {os.path.exists(config.TEST_NN_EMBEDDINGS)})\n")

# ✅ Check if required files exist before loading
if not os.path.exists(config.TRAIN_DATA) or not os.path.exists(config.TEST_DATA):
    raise FileNotFoundError("🚨 ERROR: One or more required data files are missing!")

# ✅ Load dataset files
train_data = pd.read_csv(config.TRAIN_DATA)
test_data = pd.read_csv(config.TEST_DATA)

# ✅ Load NN embeddings
train_embeddings = pd.read_csv(config.TRAIN_NN_EMBEDDINGS)
test_embeddings = pd.read_csv(config.TEST_NN_EMBEDDINGS)

# ✅ Ensure the target column is aligned
y_train = train_embeddings["Target"]
y_test = test_embeddings["Target"]

# ✅ Drop target column from embeddings to avoid duplication
train_embeddings = train_embeddings.drop(columns=["Target"])
test_embeddings = test_embeddings.drop(columns=["Target"])

# ✅ Merge original features with embeddings
X_train = pd.concat([train_data.drop(columns=["Target"]), train_embeddings], axis=1)
X_test = pd.concat([test_data.drop(columns=["Target"]), test_embeddings], axis=1)

print("✅ Merged original features with NN embeddings for XGBoost training!")

# ✅ Define Features (X) and Target (y)
X = df_balanced.drop(columns=["Target"])  # Features
y = df_balanced["Target"]  # Target variable

# ✅ Rename Features for XGBoost Compatibility
X = X.rename(columns=lambda col: col.replace("[", "").replace("]", "").replace(" ", "_"))

# ✅ Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Define XGBoost Model
model = xgb.XGBClassifier(
    objective="binary:logistic",  # Binary classification
    eval_metric="logloss",
    random_state=42
)

# ✅ Define Parameter Grid for Hyperparameter Tuning
param_grid = {
    "n_estimators": [300, 500, 700],
    "max_depth": [10, 15, 20],
    "learning_rate": [0.01, 0.05, 0.1],
    "gamma": [0, 0.1, 0.2],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0]
}

# ✅ Grid Search for Hyperparameter Optimization
grid_search = GridSearchCV(model, param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# ✅ Best Parameters
best_params = grid_search.best_params_
print("\n🏆 Best Parameters:", best_params)

# ✅ Train Best Model
best_model = xgb.XGBClassifier(**best_params, objective="binary:logistic", eval_metric="logloss", random_state=42)
best_model.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = best_model.predict(X_test)

# ✅ Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Optimized XGBoost Accuracy: {accuracy:.4f}")

# ✅ Save XGBoost Predictions for Ensemble
xgb_test_predictions = pd.DataFrame({"XGBoost_Pred": y_pred})
xgb_test_predictions.to_csv(config.XGB_TEST_PREDICTIONS, index=False)

print(f"✅ XGBoost test predictions saved as {config.XGB_TEST_PREDICTIONS}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Save the trained XGBoost model
os.makedirs(os.path.dirname(config.FINAL_XGB_MODEL), exist_ok=True)
best_model.save_model(config.FINAL_XGB_MODEL)

print(f"✅ XGBoost model saved successfully at: {config.FINAL_XGB_MODEL}")
