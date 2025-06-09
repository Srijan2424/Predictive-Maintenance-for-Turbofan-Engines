🚀 Predictive Maintenance for Turbofan Engines

This project implements a comprehensive predictive maintenance framework for turbofan engines, leveraging machine learning to predict engine degradation, estimate Remaining Useful Life (RUL), and calculate a dynamic risk score. By integrating various analytical phases—from data preprocessing and clustering to advanced regression and risk assessment—this system aims to enhance operational efficiency and prevent unexpected failures.

✨ Features
Robust Data Preprocessing: Handles diverse CMaps datasets (FD001, FD002, FD003, FD004) for both clustering and regression tasks.
Intelligent Degradation Clustering: Groups engine operational cycles into distinct degradation stages (0-4), identifying healthy to highly degraded states.
Adaptive Clustering: Supports both K-Means and Agglomerative Clustering.
Context-Aware Processing: Differentiates between datasets with constant and varying operational conditions.
Failure Probability Prediction: (Phase 2, integrated) Calculates the probability of engine failure.
Remaining Useful Life (RUL) Prediction: Utilizes a Support Vector Regression (SVR) model to forecast RUL, incorporating powerful features like rolling statistics.
Stage-Based RUL Mapping: Maps numerical RUL predictions to interpretable degradation stages.
Cycles-to-Next-Stage Estimation: Predicts the number of cycles until an engine transitions to a more critical degradation stage.
Dynamic Risk Scoring: Combines failure probability and predicted RUL into a single, normalized risk score for each engine.
Insightful Visualizations: Generates plots to visualize degradation stages and overall engine risk, complete with customizable high-risk thresholds.

🛠️ Project Architecture & Phases
The project is structured into modular phases, each handled by dedicated Python classes:

Phase 0: Data Preprocessing (Pre class)
This class is the entry point for all raw data. It loads and cleans CMaps datasets, preparing them specifically for either clustering or regression tasks. It handles different data combinations (e.g., all datasets for general clustering, specific combinations for targeted clustering or regression).

Phase 1: Engine Degradation Clustering (A, B, C classes)
This phase focuses on identifying and categorizing distinct operational and degradation states of the engines.

A class: Performs clustering across all combined CMaps datasets, providing a holistic view of engine health.
B class: Specializes in clustering data from FD001 and FD003 (constant operational conditions), identifying degradation stages specific to these scenarios.
C class: Handles clustering for FD002 and FD004 (varying operational conditions), adapting to the complexities introduced by dynamic operational parameters. Each clustering class employs data normalization, Principal Component Analysis (PCA) for visualization, and maps clusters to meaningful degradation stages (0-4).

Phase 2: Failure Probability Prediction (Two class)
(As observed from the risk_score_calculator usage) This module is responsible for calculating the failure probability of each engine. Its output (unit_number and failure_probability) is a critical input for the final risk assessment.

Phase 3: Remaining Useful Life (RUL) Prediction (Three class)
This is where the predictive power for RUL comes to life. The Three class:

Preprocesses data by normalizing operational settings and calculating rolling mean and standard deviation for sensor features.
Trains a Support Vector Regression (SVR) model on the prepared training data.
Predicts RUL for the test set.
Maps predicted RUL to discrete degradation stages.
Estimates cycles until an engine enters the next degradation stage.
Evaluates performance using Root Mean Squared Error (RMSE).

Phase 4: Risk Score Calculation & Visualization
The final phase brings it all together. The risk_score_calculator function merges the failure probability (from Phase 2) and predicted RUL (from Phase 3). It then calculates a normalized risk score for each engine, providing a unified metric for maintenance prioritization. A bar chart visually represents these risk scores, highlighting high-risk engines against a defined threshold.

💻 Technologies Used
Python 3.x
Pandas: For robust data manipulation and analysis.
NumPy: For numerical operations.
Scikit-learn: For machine learning algorithms (StandardScaler, PCA, KMeans, AgglomerativeClustering, SVR, mean_squared_error).
Matplotlib: For plotting and data visualization.
Seaborn: For enhanced statistical data visualization.

💡 Future Enhancements
Hyperparameter Tuning: Implement more sophisticated hyperparameter tuning for clustering and SVR models.
Real-time Integration: Explore integrating with real-time data streams for continuous monitoring.
Advanced Visualization: Develop interactive dashboards for deeper insights.
Model Explainability: Incorporate techniques to explain model predictions, especially for critical risk scores.
