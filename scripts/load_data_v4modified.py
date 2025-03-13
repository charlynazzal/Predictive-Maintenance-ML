import os
import sys
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ✅ Ensure script can access parent directory (for config)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py

# ✅ Load dataset
df = pd.read_csv(config.PREDICTIVE_MAINTENANCE)

# ✅ Drop unnecessary columns (UDI & Product ID)
df_cleaned = df.drop(columns=["UDI", "Product ID"])

# ✅ Convert categorical variables into numbers
df_cleaned["Type"] = df_cleaned["Type"].map({"L": 0, "M": 1, "H": 2})
df_cleaned["Failure Type"] = df_cleaned["Failure Type"].astype("category").cat.codes

# ✅ Select features to scale
features_to_scale = ["Torque [Nm]", "Tool wear [min]", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]"]

# ✅ Apply StandardScaler
scaler = StandardScaler()
df_cleaned[features_to_scale] = scaler.fit_transform(df_cleaned[features_to_scale])

# ✅ Define features (X) and target (y)
X = df_cleaned.drop(columns=["Target"])  # Features (without Target)
y = df_cleaned["Target"]  # Target (Failure / No Failure)

# ✅ Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Creates synthetic failure cases
X_resampled, y_resampled = smote.fit_resample(X, y)

# ✅ Convert back to DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced["Target"] = y_resampled  # Add Target column back

# ✅ Train-Test Split (Move this here so we can import it)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# ✅ Export these variables so other scripts can use them
__all__ = ["X_train", "X_test", "y_train", "y_test", "df_balanced"]

# ✅ Only print when running this script directly
if __name__ == "__main__":
    print("\n🚀 LOADING SCRIPT: load_data_v4modified.py\n")

    print("\n🔹 Training Set Size:", X_train.shape[0])
    print("🔹 Testing Set Size:", X_test.shape[0])
