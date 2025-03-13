import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# ✅ Ensure the script can access the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # ✅ Import paths from config.py
from load_data_v4modified import X_train, X_test, y_train, y_test

# ✅ Define the Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# ✅ Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model
print("\n🚀 Training Neural Network...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# ✅ Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Neural Network Accuracy: {test_acc:.4f}")

# ✅ Save the trained model
os.makedirs(config.MODELS_FOLDER, exist_ok=True)
nn_model_path = os.path.join(config.MODELS_FOLDER, "best_nn_model.keras")
model.save(nn_model_path)  

print(f"\n✅ Neural Network model saved at: {nn_model_path}")
