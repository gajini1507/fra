import joblib
import numpy as np

# -------------------------------
# 1️⃣ Load your trained model
# -------------------------------
model_path = "C:/Users/anbus/OneDrive/Desktop/fra demo/fra demo/fra_fault_model.pkl"  # change to your .pkl file
model = joblib.load(model_path)

# Print model summary
print("Loaded model:")
print(model)

# -------------------------------
# 2️⃣ Test the model with example input
# -------------------------------
# Replace this with your actual feature vector(s)
# Example: if your model expects 3 features, use 1x3 array
X_new = np.array([[0.5, 1.2, 3.3]])

# Make predictions
y_pred = model.predict(X_new)
print("Prediction:", y_pred)

# Optionally, get prediction probabilities if classifier supports it
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_new)
    print("Prediction probabilities:", y_prob)