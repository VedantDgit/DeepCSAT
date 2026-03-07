import joblib
import numpy as np

model = joblib.load("csat_gradient_boosting_model.pkl")
print(f"Features in: {model.n_features_in_}")
if hasattr(model, "feature_names_in_"):
    print(f"Feature names: {model.feature_names_in_}")
else:
    print("No feature names in model.")
