import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

'''
Variables(Final)
X = df[['airline_iata','acft_class', 'departure_country', 'departure_continent', 
        'arrival_country', 'arrival_continent', 'domestic', 'ask', 'rpk', 
        'fuel_burn', 'iata_departure', 'iata_arrival', 'acft_icao']]
y = df['co2_per_distance']
'''

X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_train.parquet")
X_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_test.parquet")
X_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_val.parquet")
y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_val.parquet").squeeze()

X_train.head(5)

print("Training/Validation/Test data loaded")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model training completed")

# # Save model and feature columns
# joblib.dump(rf_model, "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Models/rf_model.pkl")
# joblib.dump(X_train.columns.tolist(), "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Models/feature_columns.pkl")
# print("Model and feature columns saved")

# Evaluate on Test Set
y_pred_test = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)

print("Test Set Performance:")
print(f"R-squared (R²): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Evaluate on Validation Set
y_pred_val = rf_model.predict(X_val)
r2_val = r2_score(y_val, y_pred_val)
mse_val = mean_squared_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(y_val, y_pred_val)

print("Validation Set Performance:")
print(f"R-squared (R²): {r2_val:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")
print(f"Mean Absolute Error (MAE): {mae_val:.4f}")