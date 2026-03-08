# DeepCSAT: E-Commerce Customer Satisfaction Score Prediction

## Project Overview

DeepCSAT is a machine learning project developed to predict Customer Satisfaction (CSAT) scores in an e-commerce environment using customer interaction and service-related features.

The goal of this project is to help businesses identify important operational factors affecting customer satisfaction and support proactive service improvement.

---

## Problem Statement

Customer satisfaction is a critical business metric in e-commerce.
Traditional survey-based monitoring often delays insights and limits immediate corrective action.

This project predicts CSAT score using machine learning models trained on customer support interaction data.

---

## Dataset Features Used

The final deployed model uses the following five important features:

* response_time
* category
* customer_city
* agent_shift
* tenure_bucket

---

## Project Workflow

1. Data Cleaning and Preprocessing
2. Handling Missing Values
3. Categorical Encoding
4. Feature Selection using Random Forest Feature Importance
5. Data Scaling
6. PCA for Dimensionality Reduction
7. Model Training using:

   * Linear Regression
   * Random Forest Regressor
   * Gradient Boosting Regressor
8. Hyperparameter Tuning using GridSearchCV
9. Model Deployment using Streamlit

---

## Final Model Selected

Gradient Boosting Regressor was selected as the final model because it provided:

* Better R2 Score
* Lower RMSE
* Stable generalization performance

---

## Evaluation Metrics Used

* R2 Score
* RMSE (Root Mean Squared Error)

---

## Deployment

The final model is deployed using Streamlit for real-time CSAT prediction.

Users can enter feature values and obtain predicted customer satisfaction score instantly.

---

## Files Included

* app.py
* csat_model_5features.pkl
* scaler_5features.pkl
* requirements.txt

---

## Run Locally

```bash
streamlit run app.py
```

---

## GitHub Repository

Add your GitHub link here.

---

## Future Scope

* Live dashboard integration
* Real-time customer support analytics
* Advanced explainability using SHAP

---

## Author

Vedant
BTech Student | VIT Bhopal University
