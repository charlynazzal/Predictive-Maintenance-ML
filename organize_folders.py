import os
import shutil

# Define the base directory
base_dir = "C:/Users/charl/OneDrive/Desktop/AI Projects/Construction AI"

# Define folder paths
folders = {
    "data": ["train_data.csv", "test_data.csv", "train_data_combined.csv", "test_data_combined.csv",
             "train_nn_embeddings.csv", "test_nn_embeddings.csv", "xgb_test_predictions.csv", "predictive_maintenance.csv"],
    
    "models": ["final_nn_model.keras", "final_xgb_model.json", "best_nn_model.h5", "best_rf_model.pkl",
               "optimized_nn_model.keras", "optimized_nn_model_v2.keras"],
    
    "training_logs": ["latest_training_history.csv", "training_history.csv",
                      "training_history_20250302_182621.csv", "training_history_20250303_184911.csv",
                      "training_history_20250303_191504.csv"],
    
    "scripts": ["ensemble_final.py", "feature_importance.py", "merge_embeddings.py",
                "load_data_v1.py", "load_data_v2.py", "load_data_v3.py", "load_data_v4.py",
                "load_data_v4modified.py", "load_data_v5.py", "train_model_1.py", "train_model_1modified.py",
                "train_model_2.py", "train_model_3.py", "train_neural_network.py",
                "train_neural_network_final.py", "train_neural_network_optimized.py",
                "train_neural_network_v2.py", "train_neural_network_v2C.py", "train_random_forest.py",
                "train_random_forest_tuned.py", "train_xgboost.py", "train_xgboost_optimized.py"],
    
    "visuals": ["Features for Failure.png", "Figure_1 (scaled).png", "Figure_1.png",
                "neural network v2.png", "nn_comparison_chart.png", "nn_figure_v2.png",
                "train_neural_network_optimized_comparison_graph.png"]
}

# Create folders if they don't exist and move files
for folder, files in folders.items():
    folder_path = os.path.join(base_dir, folder)
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

    for file in files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            shutil.move(file_path, folder_path)
            print(f"✅ Moved {file} to {folder}/")

print("\n🎯 All files organized successfully!")
