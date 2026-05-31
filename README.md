# Credit Scoring Business Understanding

## 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord emphasizes accurate risk measurement, transparency, and regulatory compliance in credit risk management. Because credit scoring models influence lending decisions, banks must be able to explain how predictions are generated and justify those decisions to regulators and stakeholders.

An interpretable and well-documented model supports:

- Transparency in credit decisions
- Regulatory compliance and auditing
- Effective risk monitoring and validation
- Easier identification of model weaknesses and biases

Therefore, model development should prioritize not only predictive performance but also explainability and proper documentation.

---

## 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The provided dataset contains customer transaction records but does not include a direct default label indicating whether a customer failed to repay credit. Since supervised machine learning requires a target variable, a proxy variable must be created.

In this project, customer behavior can be analyzed using Recency, Frequency, and Monetary (RFM) metrics. Customers with low engagement and spending activity may be considered higher risk and used as a proxy for potential default behavior.

However, proxy-based prediction introduces several risks:

- The proxy may not accurately represent actual default behavior.
- Good customers may be classified as high risk and vice versa.
- Behavioral patterns do not always reflect repayment ability.
- Business decisions may be affected by assumptions rather than true outcomes.

These limitations should be clearly acknowledged when interpreting model results.

---

## 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

### Logistic Regression with WoE

**Advantages**

- Highly interpretable and easy to explain
- Widely accepted in financial institutions
- Easier to validate and audit

**Limitations**

- May not capture complex relationships in data
- Usually provides lower predictive performance

### Gradient Boosting

**Advantages**

- Strong predictive accuracy
- Captures nonlinear relationships and feature interactions

**Limitations**

- Less transparent and harder to interpret
- More difficult to explain to regulators
- Requires additional explainability techniques
