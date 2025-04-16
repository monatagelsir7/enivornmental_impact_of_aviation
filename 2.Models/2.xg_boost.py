# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import pandas as pd
# import numpy as np

# df = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/cleaned_aviation_data_v3.parquet")

# # X = df[['acft_class', 'seats', 'n_flights', 'departure_country', 'departure_continent',
# #         'arrival_country', 'arrival_continent', 'domestic', 'ask', 'rpk', 'fuel_burn']]

# X = df[['acft_class', 'n_flights', 'departure_continent',
#         'arrival_continent', 'domestic', 'ask', 'rpk', 'fuel_burn']]

# y = df['co2_per_distance']

# X_encoded = pd.get_dummies(X, drop_first=True)

# X_train = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet")
# X_test = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_test.parquet")
# X_val = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_val.parquet")
# y_train = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_train.parquet").squeeze()
# y_test = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_test.parquet").squeeze()
# y_val = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_val.parquet").squeeze()

# # X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# xgb_model.fit(X_train, y_train)

# y_pred = xgb_model.predict(X_test)

# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)

# print(f"XGBoost R-squared (R²): {r2:.4f}")
# print(f"XGBoost Mean Squared Error (MSE): {mse:.4f}")
# print(f"XGBoost Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"XGBoost Mean Absolute Error (MAE): {mae:.4f}")


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/cleaned_aviation_data_v3.parquet")

# Define features and target
X = df[['airline_iata','acft_class', 'departure_country', 'departure_continent',
        'arrival_country', 'arrival_continent', 'domestic', 'ask', 'rpk', 'fuel_burn', 'iata_departure', 'iata_arrival', 'acft_icao']]
y = df['co2_per_distance']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Load pre-split datasets
X_train = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet")
X_test = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_test.parquet")
X_val = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_val.parquet")
y_train = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_val.parquet").squeeze()

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred_test = xgb_model.predict(X_test)

# Predict on validation set
y_pred_val = xgb_model.predict(X_val)

# Define a function to evaluate
def evaluate(y_true, y_pred, dataset_name="Dataset"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n {dataset_name} Performance:")
    print(f"R² (R-squared): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")

# Evaluate on validation set
evaluate(y_val, y_pred_val, "Validation Set (XGBoost)")

# Evaluate on test set
evaluate(y_test, y_pred_test, "Test Set (XGBoost)")
