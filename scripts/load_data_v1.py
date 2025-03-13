import os
import sys
import pandas as pd

# ✅ Ensure Python can find config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config  # Import paths from config.py

# ✅ Load dataset
df = pd.read_csv(config.PREDICTIVE_MAINTENANCE)

# ✅ Drop unnecessary columns (UDI & Product ID)
df_cleaned = df.drop(columns=["UDI", "Product ID"])  # Save to a new DataFrame

# ✅ Convert categorical variables into numbers
df_cleaned["Type"] = df_cleaned["Type"].map({"L": 0, "M": 1, "H": 2})  # Encoding Machine Type
df_cleaned["Failure Type"] = df_cleaned["Failure Type"].astype("category").cat.codes  # Convert failure labels to numbers

# ✅ Check class distribution (important for ML training)
print("\n📊 Class Distribution in 'Target' Column:")
print(df_cleaned["Target"].value_counts(normalize=True))  # Shows % of failures vs. non-failures

# ✅ Show cleaned dataset info
print("\n🔍 Cleaned Dataset Info:")
print(df_cleaned.info())

# ✅ Show updated first few rows
print("\n📌 First 5 rows of Cleaned Data:")
print(df_cleaned.head())  # Using the modified dataset
