import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Ensure parent directory is accessible

import pandas as pd
import config  # Import configured paths

# ✅ Load Original Training and Test Data
train_data = pd.read_csv(config.TRAIN_DATA)
test_data = pd.read_csv(config.TEST_DATA)

# ✅ Load Neural Network Embeddings
train_embeddings = pd.read_csv(config.TRAIN_NN_EMBEDDINGS)
test_embeddings = pd.read_csv(config.TEST_NN_EMBEDDINGS)

# ✅ Ensure the Target Column is Aligned
y_train = train_embeddings["Target"]
y_test = test_embeddings["Target"]

# ✅ Drop the Target Column from Embeddings to Avoid Duplication
train_embeddings = train_embeddings.drop(columns=["Target"])
test_embeddings = test_embeddings.drop(columns=["Target"])

# ✅ Merge Original Features with Embeddings
X_train = pd.concat([train_data.drop(columns=["Target"]), train_embeddings], axis=1)
X_test = pd.concat([test_data.drop(columns=["Target"]), test_embeddings], axis=1)

# ✅ Save the Merged Data
X_train["Target"] = y_train  # Re-add target column
X_test["Target"] = y_test  # Re-add target column

# ✅ Save to configured paths
X_train.to_csv(config.TRAIN_DATA_COMBINED, index=False)
X_test.to_csv(config.TEST_DATA_COMBINED, index=False)

print(f"✅ Merged dataset saved as:\n  - {config.TRAIN_DATA_COMBINED}\n  - {config.TEST_DATA_COMBINED}")
