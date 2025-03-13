import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Ensure parent directory is accessible

import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import config  # Import configured paths

print("\n🚀 ENSEMBLE MODEL TRAINING\n")

# ✅ Load Trained Models
print("🔄 Loading trained models...")
nn_model = tf.keras.models.load_model(config.FINAL_NN_MODEL)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(config.FINAL_XGB_MODEL)  # Ensure XGBoost model is saved and loaded correctly

# ✅ Load test data
test_data = pd.read_csv(config.TEST_DATA_COMBINED)  # Ensure correct file
X_test = test_data.drop(columns=["Target"])  # Drop target column
y_test = test_data["Target"]  # Ground truth labels

# ✅ Standardizing Column Names for XGBoost
X_test = X_test.rename(columns={
    "Air temperature [K]": "Air_temperature_K",
    "Process temperature [K]": "Process_temperature_K",
    "Rotational speed [rpm]": "Rotational_speed_rpm",
    "Torque [Nm]": "Torque_Nm",
    "Tool wear [min]": "Tool_wear_min",
    "Failure Type": "Failure_Type"
})

print(f"✅ Standardized X_test Columns: {list(X_test.columns)[:10]} ...")

# ✅ Debugging Feature Shapes
print("🔍 Checking Feature Shapes...")
print(f"NN Model Expected Input Shape: {nn_model.input_shape}")
print(f"Actual X_test Shape: {X_test.shape}")

# ✅ Ensure the correct number of features for the Neural Network
nn_feature_count = nn_model.input_shape[1]  # Get expected input size
X_test_nn = X_test.iloc[:, :nn_feature_count]  # Select only the first 'n' features for the NN model

print(f"✅ Using first {nn_feature_count} features for NN model")
print(f"Modified X_test_nn Shape: {X_test_nn.shape}")

# ✅ Make Predictions
nn_probs = nn_model.predict(X_test_nn)
print(f"✅ Neural Network Predictions Shape: {nn_probs.shape}")

# ✅ Ensure XGBoost features match what it was trained on
xgb_feature_names = xgb_model.get_booster().feature_names
X_test_xgb = X_test[xgb_feature_names]  # Select only columns that match XGBoost training
print(f"✅ Matching X_test to XGBoost features: {X_test_xgb.shape}")

xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]  # Take probability of class 1
print(f"✅ XGBoost Predictions Shape: {xgb_probs.shape}")

# ✅ Merge Predictions into a New Feature Set for Meta-Learner
ensemble_features = pd.DataFrame({
    "NN_Prob": nn_probs.flatten(),  # Flatten to match dimensions
    "XGB_Prob": xgb_probs
})

print(f"✅ Ensemble Features Shape: {ensemble_features.shape}")

# ✅ Train Meta-Classifier (Random Forest as meta-learner)
meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
meta_model.fit(ensemble_features, y_test)

# ✅ Make Final Predictions
final_preds = meta_model.predict(ensemble_features)

# ✅ Evaluate Performance
accuracy = accuracy_score(y_test, final_preds)
print(f"\n🎯 Final Ensemble Model Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, final_preds))
