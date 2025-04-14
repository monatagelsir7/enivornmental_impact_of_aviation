# -------------------- Training --------------------
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_train.parquet")
X_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_test.parquet")
X_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_val.parquet")
y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/y_val.parquet").squeeze()

print("Training/Validation/Test data loaded")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model training completed")

# Save model and feature columns
joblib.dump(rf_model, "/Users/ilseoplee/enivornmental_impact_of_aviation/Models/rf_model.pkl")
joblib.dump(X_train.columns.tolist(), "/Users/ilseoplee/enivornmental_impact_of_aviation/Models/feature_columns.pkl")
print("Model and feature columns saved")

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

# -------------------- Prediction(Aircraft) + Visualization --------------------

# Load full dataset
df = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/cleaned_aviation_data_v3.parquet")
print("Full dataset loaded!")

# Handle missing values
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
print("Missing values handled!")

# Load model and feature columns
rf_model = joblib.load("/Users/ilseoplee/enivornmental_impact_of_aviation/Models/rf_model.pkl")
feature_columns = joblib.load("/Users/ilseoplee/enivornmental_impact_of_aviation/Models/feature_columns.pkl")
print("Model and feature columns loaded!")

# Prepare features
# If columns are missing, add them with default value (0)
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0 
X_full = df[feature_columns]

# 10. Predict
y_pred_full = rf_model.predict(X_full)
df['predicted_co2_per_distance'] = y_pred_full
print("Prediction completed!")

# 11. Analyze by Aircraft Type
aircraft_analysis = df.groupby('acft_class')['predicted_co2_per_distance'].mean().sort_values()
print("Predicted CO₂ per Distance by Aircraft Type:")
print(aircraft_analysis)

# 12. Visualization
plt.figure(figsize=(10, 6))
aircraft_analysis.plot(kind='barh', color='steelblue')
plt.title('Predicted CO₂ per Distance by Aircraft Type', fontsize=16)
plt.xlabel('CO₂ per Distance (kg/km)', fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("/Users/ilseoplee/enivornmental_impact_of_aviation/aircraft_co2_analysis.png")
plt.show()


# -------------------- Prediction(Continent) + Visualization --------------------

# Load full dataset
df = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/cleaned_aviation_data_v3.parquet")
print("Full dataset loaded!")

# Handle missing values
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
print("Missing values handled!")

# Load model and feature columns (trained for continent prediction)
rf_model = joblib.load("/Users/ilseoplee/enivornmental_impact_of_aviation/Models/rf_model.pkl")
feature_columns = joblib.load("/Users/ilseoplee/enivornmental_impact_of_aviation/Models/feature_columns.pkl")
print("Model and feature columns loaded!")

# Prepare features
# If columns are missing, add them with default value (0)
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0
X_continent = df[feature_columns]

# Predict
y_pred_continent = rf_model.predict(X_continent)
df['predicted_co2_per_distance_continent'] = y_pred_continent
print("Prediction completed!")

# Analyze by Continent Pair (Departure -> Arrival)
continent_analysis = df.groupby(['departure_continent', 'arrival_continent'])['predicted_co2_per_distance_continent'].mean().sort_values()
print("Predicted CO₂(kg)per Distance(km) by Departure and Arrival Continent:")
print(continent_analysis)

# Visualization
plt.figure(figsize=(12, 8))
continent_analysis.plot(kind='barh', color='seagreen')
plt.title('Predicted CO₂(kg)per Distance(km) by Departure and Arrival Continent', fontsize=16)
plt.xlabel('CO₂ per Distance (kg/km)', fontsize=14)
plt.ylabel('Continent Pair (Departure -> Arrival)', fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("/Users/ilseoplee/enivornmental_impact_of_aviation/continent_co2_analysis.png")
plt.show()