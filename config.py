import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data files
DATA_FOLDER = os.path.join(BASE_DIR, "data")
TRAIN_DATA = os.path.join(DATA_FOLDER, "train_data.csv")
TEST_DATA = os.path.join(DATA_FOLDER, "test_data.csv")
TRAIN_DATA_COMBINED = os.path.join(DATA_FOLDER, "train_data_combined.csv")
TEST_DATA_COMBINED = os.path.join(DATA_FOLDER, "test_data_combined.csv")
TRAIN_NN_EMBEDDINGS = os.path.join(DATA_FOLDER, "train_nn_embeddings.csv")
TEST_NN_EMBEDDINGS = os.path.join(DATA_FOLDER, "test_nn_embeddings.csv")
XGB_TEST_PREDICTIONS = os.path.join(DATA_FOLDER, "xgb_test_predictions.csv")
PREDICTIVE_MAINTENANCE = os.path.join(DATA_FOLDER, "predictive_maintenance.csv")

# Models
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
BEST_NN_MODEL = os.path.join(MODELS_FOLDER, "best_nn_model.h5")
BEST_RF_MODEL = os.path.join(MODELS_FOLDER, "best_rf_model.pkl")
FINAL_NN_MODEL = os.path.join(MODELS_FOLDER, "final_nn_model.keras")
FINAL_XGB_MODEL = os.path.join(MODELS_FOLDER, "final_xgb_model.json")
OPTIMIZED_NN_MODEL = os.path.join(MODELS_FOLDER, "optimized_nn_model.keras")
OPTIMIZED_NN_MODEL_V2 = os.path.join(MODELS_FOLDER, "optimized_nn_model_v2.keras")

# Scripts
SCRIPTS_FOLDER = os.path.join(BASE_DIR, "scripts")

# Training logs
TRAINING_LOGS_FOLDER = os.path.join(BASE_DIR, "training_logs")
LATEST_TRAINING_HISTORY = os.path.join(TRAINING_LOGS_FOLDER, "latest_training_history.csv")
TRAINING_HISTORY_1 = os.path.join(TRAINING_LOGS_FOLDER, "training_history_20250302_182621.csv")
TRAINING_HISTORY_2 = os.path.join(TRAINING_LOGS_FOLDER, "training_history_20250303_184911.csv")
TRAINING_HISTORY_3 = os.path.join(TRAINING_LOGS_FOLDER, "training_history_20250303_191504.csv")
TRAINING_HISTORY_CSV = os.path.join(TRAINING_LOGS_FOLDER, "training_history.csv")

# Visualizations (Previously "figures/")
VISUALS_FOLDER = os.path.join(BASE_DIR, "visuals")
FEATURES_FOR_FAILURE = os.path.join(VISUALS_FOLDER, "Features for Failure.png")
FIGURE_1 = os.path.join(VISUALS_FOLDER, "Figure_1.png")
FIGURE_1_SCALED = os.path.join(VISUALS_FOLDER, "Figure_1 (scaled).png")
NN_FIGURE_V2 = os.path.join(VISUALS_FOLDER, "nn_figure_v2.png")
NN_COMPARISON_CHART = os.path.join(VISUALS_FOLDER, "nn_comparison_chart.png")
TRAIN_NN_COMPARISON = os.path.join(VISUALS_FOLDER, "train_neural_network_optimized_comparison_graph.png")

