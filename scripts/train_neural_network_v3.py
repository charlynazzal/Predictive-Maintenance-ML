import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ✅ Ensure script can access parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # Import paths from config.py

print("\n🚀 TRAINING IMPROVED NEURAL NETWORK\n")

# ✅ Load preprocessed training and test data
train_data = pd.read_csv(config.TRAIN_DATA)
test_data = pd.read_csv(config.TEST_DATA)

# ✅ Split features and target
X_train = train_data.drop(columns=["Target"]).values
y_train = train_data["Target"].values

X_test = test_data.drop(columns=["Target"]).values
y_test = test_data["Target"].values

# ✅ Define the improved neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# ✅ Compile the model with an improved optimizer
model.compile(
    optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ✅ Learning rate scheduler
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# ✅ Train the optimized model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,  # Increased epochs for better learning
    batch_size=32,
    callbacks=[lr_scheduler],
    verbose=1
)

# ✅ Evaluate the optimized model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

# ✅ Save the improved model
nn_model_path = os.path.join(config.MODELS_FOLDER, "optimized_nn_model_v2.keras")
model.save(nn_model_path)

# ✅ Save training history
history_df = pd.DataFrame(history.history)
history_log_path = os.path.join(config.TRAINING_LOGS_FOLDER, "training_history_v2.csv")
history_df.to_csv(history_log_path, index=False)

# ✅ Plot accuracy and loss curves
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.show()

# ✅ Show training curves
plot_training_curves(history)

# ✅ Print final accuracy
print(f"\n🎯 Final Neural Network Accuracy: {test_accuracy:.4f}")

print(f"\n✅ Model saved at: {nn_model_path}")
print(f"✅ Training history saved at: {history_log_path}")
