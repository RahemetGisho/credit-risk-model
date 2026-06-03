# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops a credit risk scoring system using alternative transactional data to support Buy Now, Pay Later (BNPL) and other digital lending products. Since the dataset does not contain a direct default label, customer risk behavior is inferred using an RFM (Recency, Frequency, Monetary) based proxy target.

The project follows a complete machine learning workflow, including business understanding, exploratory data analysis, feature engineering, proxy target creation, model development, model explainability, and deployment preparation.

---

# Business Understanding

## 1. Basel II and Model Interpretability

The Basel II Accord emphasizes accurate risk measurement, transparency, and regulatory compliance in credit risk management. Credit scoring models directly influence lending decisions, making interpretability a critical requirement.

An interpretable and well-documented model enables:

* Transparent credit decisions
* Regulatory compliance and auditing
* Effective model validation and monitoring
* Easier identification of bias and model weaknesses
* Improved stakeholder trust

Therefore, model development must balance predictive performance with explainability.

## 2. Why a Proxy Target Is Necessary

The dataset does not contain an explicit default indicator. Because supervised machine learning requires labeled outcomes, a proxy target must be created.

Customer behavior is summarized using RFM metrics:

* Recency: How recently a customer transacted
* Frequency: How often a customer transacts
* Monetary Value: Total spending activity

Customers exhibiting low engagement and spending behavior are treated as higher-risk segments and used as a proxy for potential default behavior.

### Risks of Proxy-Based Prediction

* The proxy may not perfectly represent true default behavior.
* Some low-activity customers may still be creditworthy.
* High-activity customers may still default.
* Business decisions may be influenced by assumptions rather than observed outcomes.

These limitations should be acknowledged when interpreting model outputs.

## 3. Model Trade-Offs in a Regulated Environment

### Logistic Regression + WoE

#### Advantages

* Highly interpretable
* Easy to explain to regulators
* Well-established in financial institutions
* Simple validation and auditing process

#### Limitations

* May miss complex patterns
* Lower predictive performance compared to advanced models

### Gradient Boosting Models

#### Advantages

* Strong predictive performance
* Captures nonlinear relationships
* Handles feature interactions effectively

#### Limitations

* Less transparent
* Requires additional explainability methods
* More difficult to justify in regulated environments

---

# Exploratory Data Analysis (EDA)

## Objective

The goal of EDA is to understand customer transaction behavior, assess data quality, identify potential risk indicators, and guide downstream feature engineering.

## Key Findings

### Dataset Quality

* Dataset is well-structured and suitable for modeling.
* Strong overall completeness.
* Minimal missing-value concerns.

### Numerical Features

* Transaction variables exhibit significant positive skewness.
* Customer activity is concentrated among lower transaction ranges.
* High-value transactions contribute substantial variance.
* Several features contain meaningful outliers.

### Categorical Features

* Activity is concentrated within a small number of products and providers.
* Some categories are infrequent and may require grouping.
* Transaction channels demonstrate distinct behavioral patterns.

### Correlation Analysis

* Strong relationships exist among several transaction features.
* No severe multicollinearity issues were identified.
* Correlation analysis supports future feature selection decisions.

### Outlier Analysis

* Several variables contain extreme observations.
* Outliers may represent high-value customers or unusual transaction behavior.
* Outliers should be reviewed carefully before removal.

### Top EDA Insights

1. Data quality is strong and suitable for modeling.
2. Customer activity is highly uneven.
3. Fraudulent transactions are rare, creating class imbalance challenges.
4. Customer behavior varies significantly across products and providers.
5. Transaction behavior contains meaningful risk-related signals.

---

# Task 3: Feature Engineering

## Objective

Transform raw transaction data into customer-level features suitable for credit risk modeling.

## Engineered Features

### RFM Features

* Recency
* Frequency
* Monetary Value

These features capture customer engagement and financial behavior.

### Aggregation Features

Customer transaction history was aggregated to generate:

* Total transaction amount
* Average transaction amount
* Transaction count
* Transaction amount standard deviation
* Maximum transaction value
* Minimum transaction value

### Temporal Features

Transaction timestamps were transformed into:

* Transaction hour
* Transaction day
* Transaction month
* Transaction weekday

These features help capture behavioral patterns over time.

### Categorical Encoding

Categorical variables were encoded using:

* One-Hot Encoding
* Label Encoding where appropriate

### Scaling

Numerical variables were standardized to ensure consistent model behavior.

---

# Task 4: Proxy Target Engineering

## Objective

Create a proxy credit-risk label in the absence of actual default information.

## Methodology

### RFM Calculation

For each customer:

* Recency score calculated
* Frequency score calculated
* Monetary score calculated

### Customer Segmentation

K-Means clustering was applied to RFM features to identify behavioral groups.

Clusters exhibiting:

* Low frequency
* Low monetary value
* Long inactivity periods

were classified as higher-risk customers.

### Proxy Target Definition

| Proxy Label | Description        |
| ----------- | ------------------ |
| 1           | High-Risk Customer |
| 0           | Low-Risk Customer  |

This proxy target serves as the dependent variable for supervised learning.

---

# Task 5: Model Development and Evaluation

## Models Trained

### Logistic Regression

Chosen as a baseline due to:

* Simplicity
* Interpretability
* Regulatory acceptance

### Random Forest

Used to capture nonlinear patterns and feature interactions.

### Gradient Boosting

Evaluated for maximum predictive performance.

---

## Evaluation Metrics

Models were assessed using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Validation Strategy

* Train/Test Split
* Cross-Validation
* Hyperparameter Tuning

---

## Model Selection

Model comparison was performed using ROC-AUC as the primary evaluation metric.

Selection criteria included:

* Predictive performance
* Stability
* Explainability
* Regulatory suitability

The final selected model balances business requirements and predictive power.

---

# Task 6: Model Explainability

## Importance of Explainability

Credit decisions require transparency and fairness. Model explainability helps stakeholders understand why customers are classified as low or high risk.

## Feature Importance

Feature importance analysis was conducted to identify the strongest predictors of customer risk behavior.

Typical influential features included:

* Recency
* Frequency
* Monetary Value
* Average transaction amount
* Transaction count

## SHAP Analysis

SHAP (SHapley Additive Explanations) was used to provide:

* Global model interpretation
* Local prediction explanations
* Feature contribution analysis

This improves trust, transparency, and regulatory readiness.

---

# Project Structure
```text
credit-risk-model/

├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI pipeline
│
├── data/                         # Raw and processed datasets
│
├── mlruns/                       # MLflow experiment tracking
│
├── notebooks/
│   ├── eda.ipynb                 # Exploratory Data Analysis
│   └── .ipynb_checkpoints/
│
├── src/
│   ├── api/                      # FastAPI application
│   │
│   ├── __init__.py
│   ├── data_processing.py        # Data cleaning and feature engineering
│   ├── train.py                  # Model training pipeline
│   ├── predict.py                # Prediction pipeline
│   └── run_pipeline.py           # End-to-end workflow execution
│
├── tests/                        # Unit and integration tests
│
├── .gitignore
├── docker-compose.yml            # Multi-container orchestration
├── Dockerfile                    # Application containerization
├── README.md
└── requirements.txt
```
---

# Code Quality Improvements

To improve maintainability and scalability:

* Modular project structure implemented
* Reusable utility functions developed
* Unit tests added for critical functionality
* Automated validation checks introduced
* CI/CD pipeline configured through GitHub Actions
* Error handling added to preprocessing workflows

---

# Future Improvements

* Incorporate real default labels when available
* Explore advanced ensemble methods
* Implement model monitoring pipelines
* Introduce fairness and bias testing
* Deploy model using Docker and cloud infrastructure
* Integrate continuous retraining workflows

---

# Conclusion

This project demonstrates a complete end-to-end credit risk modeling workflow using alternative transactional data. Through proxy target engineering, customer behavioral analysis, machine learning, and explainability techniques, the solution provides a foundation for transparent and scalable credit risk assessment suitable for modern digital lending environments.
