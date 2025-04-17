import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df=pd.read_parquet("../0.Data_after_cleaning/cleaned_aviation_data_v3.parquet")
df.head()

domestic_df=df[df["domestic"]==1]
intl_df=df[df["domestic"]==0]

domestic_df=domestic_df.drop_duplicates()
intl_df=intl_df.drop_duplicates()

X_test = intl_df[['airline_iata','acft_class', 'departure_country', 'departure_continent',
        'arrival_country', 'arrival_continent', 'domestic', 'ask', 'rpk', 'fuel_burn', 'iata_departure', 'iata_arrival', 'acft_icao']]
y_test = intl_df['co2_per_distance']
X_test = pd.get_dummies(X_test, drop_first=True)



X = domestic_df[['airline_iata','acft_class', 'departure_country', 'departure_continent',
        'arrival_country', 'arrival_continent', 'domestic', 'ask', 'rpk', 'fuel_burn', 'iata_departure', 'iata_arrival', 'acft_icao']]
y = domestic_df['co2_per_distance']
X = pd.get_dummies(X, drop_first=True)

X = X.loc[:, X.columns.isin(X_test.columns)]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# _________________________________________________________________

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model training completed")

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