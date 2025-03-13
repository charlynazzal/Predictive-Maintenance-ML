import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Add the parent directory to sys.path to access config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # Import paths from config.py

# ✅ Load dataset
df = pd.read_csv(config.PREDICTIVE_MAINTENANCE)

# ✅ Drop unnecessary columns (UDI & Product ID)
df_cleaned = df.drop(columns=["UDI", "Product ID"])

# ✅ Convert categorical variables into numbers
df_cleaned["Type"] = df_cleaned["Type"].map({"L": 0, "M": 1, "H": 2})  # Encoding Machine Type
df_cleaned["Failure Type"] = df_cleaned["Failure Type"].astype("category").cat.codes  # Convert failure labels to numbers

# ✅ Compute correlation matrix
correlation_matrix = df_cleaned.corr()

# ✅ Display correlation values (sorted by impact on failures)
print("\n📊 Feature Correlation with Failures:")
print(correlation_matrix["Target"].sort_values(ascending=False))  # Sort from highest to lowest

# ✅ Visualize Correlation as a Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()




