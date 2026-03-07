import joblib
import os

model_path = 'csat_gradient_boosting_model.pkl'

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully. Type: {type(model)}")
        if hasattr(model, 'feature_names_in_'):
            print(f"Features in model: {list(model.feature_names_in_)}")
        else:
            print("No feature names in model.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found.")
