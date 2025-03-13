import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # To load the trained model
import os
import sys

# ✅ Ensure the script can access the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py

print("\n🚀 FEATURE IMPORTANCE ANALYSIS\n")

# ✅ Load the trained Random Forest model
best_rf_model = joblib.load(config.BEST_RF_MODEL)

# ✅ Load training data to get feature names
train_data = pd.read_csv(config.TRAIN_DATA)
X_train = train_data.drop(columns=["Target"])  # Drop Target to get features

# ✅ Extract feature importance scores
feature_importance = best_rf_model.feature_importances_

# ✅ Create DataFrame for visualization
feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# ✅ Display top features
print("\n📊 Top Features Contributing to Machine Failures:\n")
print(feature_importance_df)

# ✅ Plot Feature Importance
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance_df["Importance"], y=feature_importance_df["Feature"], hue=feature_importance_df["Feature"], palette="viridis", legend=False)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest Model")
plt.show()
