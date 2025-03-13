import os
import sys
import joblib  # To save/load models
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ✅ Ensure the script can access the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py
from load_data_v4modified import X_train, X_test, y_train, y_test

print("\n🚀 TRAINING RANDOM FOREST CLASSIFIER\n")

# ✅ Train the Random Forest Model
best_rf_model = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
best_rf_model.fit(X_train, y_train)

# ✅ Evaluate the Model
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Random Forest Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Save the trained model using the configured path
joblib.dump(best_rf_model, config.BEST_RF_MODEL)
print(f"\n✅ Model saved as {config.BEST_RF_MODEL}")
