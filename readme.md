# Three-Layer Fraud Detection Framework

🔗 **Live App:** [https://finxai.streamlit.app/](https://finxai.streamlit.app/)
📂 **GitHub Repo:** [https://github.com/jyotii616/Fraud-detection-framework](https://github.com/jyotii616/Fraud-detection-framework)

Dataset:
IEEE-CIS Fraud Detection Dataset
train_transcation.csv
train_identity.csv
https://www.kaggle.com/competitions/ieee-fraud-detection/data

# Dynamic Multi-Layer Fraud Detection in Financial Transactions

## Overview

Financial fraud has become increasingly sophisticated, making traditional rule-based systems and single-model approaches less effective. Fraudsters continuously adapt their behavior by changing transaction patterns, reusing devices, and operating through interconnected accounts.

This project presents a **Dynamic Multi-Layer Fraud Detection Framework** that combines temporal behavioral analysis, graph intelligence, device trust modeling, sequence learning, anomaly detection, and explainable AI into a single leakage-free pipeline. Instead of relying on one perspective, the framework analyzes fraud from multiple dimensions to improve early detection and interpretability.

---

## Repository Structure

```
JML/
│
├── .streamlit/
│   └── config.toml                  # App configuration (upload size, etc.)
├── models/                          
├── outputs/                         
│   ├── ablation_study.csv
│   ├── baseline_comparison.csv
│   ├── best_hyperparameters.json
│   ├── cv_fold_metrics.csv
│   ├── cv_summary.csv
│   ├── dormancy_risk_projection.png
│   ├── dormancy_risk_projections.csv
│   ├── early_warning_model_metrics.txt
│   ├── feature_importance.png
│   ├── feature_importance_shap_ranking.csv
│   ├── metrics_final_model.txt
│   ├── model_early_warning.json
│   ├── model_xgboost.json
│   ├── pr_curve.png
│   ├── risk_evolution_example.png
│   ├── roc_curve.png
│   ├── shap_local_example.png
│   ├── shap_summary_plot.png
│   └── threshold_comparison.csv
│
├── fraud_detection.ipynb            # Complete implementation
├── preprocessing.py                 # Preprocessing pipeline
├── app.py                           # Deployment script
├── README.md
├── requirements.txt
├── runtime.txt                      # Python version for deployment
├── train_identity.csv
└── train_transaction.csv
```

---

# Framework

The proposed framework consists of five complementary layers.

### Layer 1 — Temporal Behavioral Intelligence

Captures how user behavior evolves over time.

Features include:

- Transaction velocity
- Spending shock ratio
- Time since previous transaction
- Time since previous fraud
- Device switching rate
- IP switching rate
- Spending trend
- Rolling historical statistics

---

### Layer 2 — Dynamic Social Graph Intelligence

Builds a transaction graph connecting users through shared devices, IP addresses, emails, and other relationships.

Graph features include:

- Degree Centrality
- PageRank
- Clustering Coefficient
- Connected Components
- Label Propagation Community Detection
- Dynamic Neighbor Risk Propagation
- Graph-Based Risk Aggregation

---

### Layer 3 — Behavioral Device Trust

Instead of treating devices as fixed identifiers, the framework estimates evolving device reliability using historical behavior.

Features include:

- Bayesian Device Trust Score
- Historical Fraud Rate
- Recent Fraud Rate
- Device Sharing Frequency
- Device Consistency
- Dormancy Risk Projection

---

### Layer 4 — Sequential Behavior Modeling

Customer behavior is modeled as a sequence rather than isolated events.

Sequence features include:

- LSTM-based Sequential Embeddings
- Rolling Historical Statistics
- Transaction History Representation
- Behavioral Evolution

---

### Layer 5 — Hybrid Anomaly Detection

Detects suspicious activity that supervised models may not recognize.

Includes:

- Isolation Forest
- Deep Autoencoder Reconstruction Error

---

# Models Used

## Primary Fraud Detection

- XGBoost (Final Model)

## Sequential Learning

- LSTM

## Unsupervised Learning

- Isolation Forest
- Deep Autoencoder

## Baseline Models

- Random Forest
- Logistic Regression

---

# Graph & Network Analysis

The project uses **NetworkX** to model hidden relationships between accounts.

Graph analytics include:

- Dynamic Social Graph Construction
- PageRank
- Degree Centrality
- Clustering Coefficient
- Connected Components
- Label Propagation Community Detection
- Betweenness Centrality
- Eigenvector Centrality

---

# Feature Engineering

The project introduces several engineered feature groups.

- Temporal Behavioral Drift Features
- Bayesian Behavioral Device Trust Score
- Dynamic Graph Features
- Sequence Embeddings from LSTM
- Graph Neighbor Risk Features
- Early-Warning Lookahead Features
- Dormancy Risk Projection
- Hybrid Anomaly Scores

---

# Explainability

Model decisions are interpreted using **SHAP**.

The explainability pipeline includes:

- SHAP Summary Plot
- Local SHAP Explanations
- Feature Ranking
- SHAP-guided Feature Selection

---

# Hyperparameter Optimization

Hyperparameters are optimized using

- RandomizedSearchCV

to improve generalization while reducing computational cost.

---

# Results

The project generates comprehensive evaluation artifacts including:

- ROC Curve
- Precision-Recall Curve
- Feature Importance Plot
- SHAP Summary Plot
- Local SHAP Explanations
- Threshold Comparison
- Cross Validation Metrics
- Ablation Study
- Dormancy Risk Projection
- Early Warning Model Performance

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- NetworkX
- SHAP
- Matplotlib
- Joblib


# Key Contributions

- Leakage-free fraud detection pipeline
- Dynamic temporal behavioral modeling
- Bayesian device trust estimation
- Dynamic graph-based intelligence
- LSTM-based sequence representation
- Hybrid anomaly detection
- Explainable AI with SHAP
- Early-warning fraud prediction
- Dormancy risk projection
- Comprehensive evaluation with ablation studies

---

# Future Work

- Multi-hop graph risk propagation
- Graph Neural Network comparison
- Online learning for streaming fraud detection
- Real-time dashboard for fraud analysts

---
Final Results:

AUC = 0.9588
Precision = 0.5785
Recall = 0.7365
F1 Score = 0.6480
# Author

**Jyoti Kumari**

B.Tech Artificial Intelligence & Machine Learning

Indira Gandhi Delhi Technical University for Women (IGDTUW)
02601192025

---
