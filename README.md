# Predictive Maintenance in Construction Using AI

## Project Overview

This project uses machine learning and deep learning to predict maintenance needs for construction equipment. It analyzes factors like failure type, torque, rotational speed, tool wear, and temperature to determine the likelihood of equipment failure. The dataset is sourced from Kaggle's Machine Predictive Maintenance Classification, with 10,000 rows and 10 columns. Data was cleaned, scaled, and balanced for accurate predictions.

## Project Structure
CONSTRUCTION AI/
├── data/
│   ├── predictive_maintenance.csv
│   ├── test_data_combined.csv
│   ├── test_data.csv
│   ├── test_nn_embeddings.csv
│   ├── train_data_combined.csv
│   ├── train_data.csv
│   ├── train_nn_embeddings.csv
│   ├── xgb_test_predictions.csv
├── models/
│   ├── best_nn_model.h5
│   ├── best_nn_model.keras
│   ├── best_rf_model.pkl
│   ├── final_nn_model.keras
│   ├── final_xgb_model.json
│   ├── optimized_nn_model_v2.keras
│   ├── optimized_nn_model.keras
├── scripts/
│   ├── ensemble_final.py
│   ├── feature_importance.py
│   ├── load_data_v1.py
│   ├── load_data_v2.py
│   ├── load_data_v3.py
│   ├── load_data_v4.py
│   ├── load_data_v4modified.py
│   ├── load_data_v5.py
│   ├── logreg_model_v1.py
│   ├── logreg_model_v1modified.py
│   ├── logreg_model_v2.py
│   ├── logreg_model_v3.py
│   ├── merge_embeddings.py
│   ├── train_neural_network_v1.py
│   ├── train_neural_network_v2.py
│   ├── train_neural_network_v3.py
│   ├── train_neural_network_v4.py
│   ├── train_random_forest.py
│   ├── train_random_forest_tuned.py
│   ├── train_xgboost.py
│   ├── train_xgboost_optimized.py
├── training_logs/
│   ├── latest_training_history.csv
│   ├── training_history_20250302_182621.csv
│   ├── training_history_20250303_184911.csv
│   ├── training_history_20250303_191504.csv
│   ├── training_history_v2.csv
│   ├── training_history.csv
├── visuals/
│   ├── Features for Failure.png
│   ├── Figure_1 (scaled).png
│   ├── Figure_1.png
│   ├── neural network v2.png
│   ├── neural network v3.png
│   ├── neural network v2 (iterated).png
│   ├── nn_comparison_chart.png
│   ├── nn_figure_v2.png
│   ├── train_neural_network_optimized_comparison_graph.png
├── config.py
├── organize_folders.py
├── README.md
├── requirements.txt

## Table of Contents


1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Data Preprocessing & Cleaning](#data-preprocessing--cleaning)
4. [Model Training & Iteration Process](#model-training--iteration-process)
   - [Logistic Regression](#logistic-regression)
   - [Random Forest](#random-forest)
   - [XGBoost](#xgboost)
   - [Neural Networks](#neural-networks)
   - [Ensemble Model](#ensemble-model)
5. [Key Findings](#key-findings)
6. [How to Run the Project](#how-to-run-the-project)
7. [Future Improvements](#future-improvements)
8. [Author & Contact](#author--contact)

## Data Preprocessing & Cleaning

- Removed unnecessary columns (e.g., **UDI, Product ID**)
- Checked for class imbalance and applied **SMOTE**
- Scaled numerical features using **StandardScaler**
- Encoded categorical variables
- Saved preprocessed datasets for training and testing

## Model Training & Iteration Process
### Logistic Regression
Baseline accuracy: 86.75%.
Best solver: newton-cg (88.48% accuracy).
Regularization (C values) tested but showed minimal improvement.
SMOTE increased failure representation from 33% to 41%, reducing accuracy to 85.8%.
Conclusion: Limited performance potential.

### Random Forest
Achieved 99.4% accuracy.
Fine-tuning raised it to 99.5%, but further gains were limited.
Feature importance identified key failure indicators (see visuals).

### XGBoost
Reached 99.6% accuracy, slightly better than Random Forest.
Optimized hyperparameters:
{
    "colsample_bytree": 0.8,
    "gamma": 0,
    "learning_rate": 0.05,
    "max_depth": 15,
    "n_estimators": 300,
    "subsample": 0.8
}

### Neural Networks
Initial accuracy: 99.5%.
Optimization included dynamic learning rate, batch normalization, dropout layers, and ReduceLROnPlateau.
Final accuracy: 99.72%, with some fluctuations.

### Ensemble Model
Combined Neural Network and XGBoost predictions.
Used Random Forest as a meta-learner.
Final accuracy: 100.00%.

## Key Findings
Failure Type (53.46%) was the most critical feature.
Rotational Speed and Torque (~31%) were significant.
Tool Wear and Temperature (~8%) had minor roles.
Machine Type (~0.52%) showed negligible impact.

![Feature Importance](visuals/Features-for-Failure.png)


## How to Run the Project
Install dependencies:

cd "CONSTRUCTION AI"
pip install -r requirements.txt

pip install -r requirements.txt

***Run the final ensemble model with:***

python scripts/ensemble_final.py

***For the full pipeline, execute in order:***

load_data_v1.py → logreg_model_v1.py → train_random_forest.py → 
train_xgboost.py → train_neural_network_v1.py → ensemble_final.py
 
## Future Improvements
- Validate on new, unseen data.
- Test deployment in real-world settings.
- Enhance neural network interpretability.
- Reduce false positives in failure type predictions.

## Author & Contact
Email: charlynazzalofficial@gmail.com

LinkedIn: Charly Nazzal
