import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from load_data_v4modified import df_balanced  # ✅ Import balanced dataset

# ✅ Ensure script can access parent directory (for config)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py

# ✅ Define Features (X) and Target (y)
X = df_balanced.drop(columns=["Target"])  # Features
y = df_balanced["Target"]  # Target variable

# ✅ Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Save the split data (optional, in case we need it later)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# ✅ Export these variables so other scripts can use them
__all__ = ["X_train", "X_test", "y_train", "y_test", "train_data", "test_data"]

# ✅ Only print when running this script directly
if __name__ == "__main__":
    print("\n🚀 LOADING SCRIPT: load_data_v5.py\n")
    
    print(f"\n🔹 Training Set Size: {X_train.shape[0]} samples")
    print(f"🔹 Testing Set Size: {X_test.shape[0]} samples")

    print("\n📌 First 5 rows of Training Data:")
    print(train_data.head())  # ✅ Now this will work because train_data is defined
