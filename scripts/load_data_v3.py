import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ✅ Add parent directory to sys.path to access config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py

# ✅ Load dataset using config.py path
df = pd.read_csv(config.PREDICTIVE_MAINTENANCE)

# ✅ Drop unnecessary columns (UDI & Product ID)
df_cleaned = df.drop(columns=["UDI", "Product ID"])

# ✅ Convert categorical variables into numbers
df_cleaned["Type"] = df_cleaned["Type"].map({"L": 0, "M": 1, "H": 2})
df_cleaned["Failure Type"] = df_cleaned["Failure Type"].astype("category").cat.codes

# ✅ Select features to scale (keep original column names)
features_to_scale = ["Torque [Nm]", "Tool wear [min]", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]"]

# ✅ Apply StandardScaler
scaler = StandardScaler()
df_cleaned[features_to_scale] = scaler.fit_transform(df_cleaned[features_to_scale])

# ✅ Show scaled data preview
print("\n📌 First 5 rows of Scaled Data:")
print(df_cleaned.head())

# ✅ Compute correlation matrix (for verification)
correlation_matrix = df_cleaned.corr()

# ✅ Display correlation values with Target
print("\n📊 Feature Correlation with Failures (After Scaling):")
print(correlation_matrix["Target"].sort_values(ascending=False))

# ✅ Visualize Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap (After Scaling)")
plt.show()
