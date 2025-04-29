# ### MODEL1 incremental training to 100 estimators ###

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load data
X_train = pd.read_parquet("../Test-Train-Validation Data/X_train.parquet")
X_test = pd.read_parquet("../Test-Train-Validation Data/X_test.parquet")
X_val = pd.read_parquet("../Test-Train-Validation Data/X_val.parquet")
y_train = pd.read_parquet("../Test-Train-Validation Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("../Test-Train-Validation Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("../Test-Train-Validation Data/y_val.parquet").squeeze()
print("Training/Validation/Test data loaded")

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model training completed")

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# Evaluation Metrics
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Test Set Performance:")
print(f"R-squared (R²): {r2_score(y_test, y_pred_test):.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_test):.4f}")

print("Validation Set Performance:")
print(f"R-squared (R²): {r2_score(y_val, y_pred_val):.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_val, y_pred_val):.4f}")