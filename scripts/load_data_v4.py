import os
import sys
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

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

# ✅ Debug: Check Failure Type count before balancing
print("\n🛠️ DEBUG: First 10 rows of Failure Type column (before balancing):")
print(df_cleaned["Failure Type"].head(10))

print("\n📊 Count of Each Failure Type Before Balancing:")
print(df_cleaned["Failure Type"].value_counts())

# ✅ Define features (X) and target (y) BEFORE applying SMOTE
X = df_cleaned.drop(columns=["Target"])  # Features (without Target)
y = df_cleaned["Target"]  # Target (Failure / No Failure)

# ✅ Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Creates synthetic failure cases
X_resampled, y_resampled = smote.fit_resample(X, y)

# ✅ Check new class distribution
print("\n📊 Class Distribution After SMOTE:")
print(Counter(y_resampled))  # Should show more balanced numbers

# ✅ Convert back to DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced["Target"] = y_resampled  # Add Target column back

# ✅ Show first few rows
print("\n📌 First 5 rows of Balanced Data:")
print(df_balanced.head())

print("\n📌 Checking if Target = 1 exists in Balanced Data:")
print(df_balanced["Target"].value_counts())  # Should show both 0s and 1s



