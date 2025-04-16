from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_column_names(df):
    df.columns = df.columns.str.replace(r'\[', '(', regex=True)
    df.columns = df.columns.str.replace(r'\]', ')', regex=True)
    df.columns = df.columns.str.replace(r'<', 'lt_', regex=True)
    df.columns = df.columns.str.replace(r'>', 'gt_', regex=True)
    return df

df = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/cleaned_aviation_data_v3.parquet")

X_raw = df[['airline_iata','acft_class', 'departure_country', 'departure_continent',
            'arrival_country', 'arrival_continent', 'domestic', 'ask', 'rpk', 'fuel_burn',
            'iata_departure', 'iata_arrival', 'acft_icao']]
y_raw = df['co2_per_distance']

X_encoded_base = pd.get_dummies(X_raw, drop_first=True)
X_encoded_base = clean_column_names(X_encoded_base)

# Load pre-split data
X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_train.parquet")
X_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_test.parquet")
X_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_val.parquet")
y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_val.parquet").squeeze()

# Clean and align
X_train = clean_column_names(X_train)
X_test = clean_column_names(X_test)
X_val = clean_column_names(X_val)

def align_columns(df, ref_columns):
    for col in ref_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[ref_columns]
    return df

X_train = align_columns(X_train, X_encoded_base.columns)
X_test = align_columns(X_test, X_encoded_base.columns)
X_val = align_columns(X_val, X_encoded_base.columns)

# Initialize model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Train with evaluation sets to enable staged performance tracking
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    eval_metric="rmse",
    verbose=False
)

# Get staged results from internal logs
results = xgb_model.evals_result()

train_rmse = results['validation_0']['rmse']
val_rmse = results['validation_1']['rmse']
test_rmse = results['validation_2']['rmse']

# Plot RMSE over boosting rounds
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_rmse, label="Train RMSE", marker='o')
plt.plot(range(1, 101), val_rmse, label="Validation RMSE", marker='s')
plt.plot(range(1, 101), test_rmse, label="Test RMSE", marker='^')
plt.xlabel("Number of Trees")
plt.ylabel("RMSE")
plt.title("XGBoost Performance (RMSE vs. Number of Trees)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate final model
def evaluate(y_true, y_pred, dataset_name="Dataset"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{dataset_name} Performance:")
    print(f"R² (R-squared): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")

y_pred_val = xgb_model.predict(X_val)
y_pred_test = xgb_model.predict(X_test)
evaluate(y_val, y_pred_val, "Validation Set (XGBoost)")
evaluate(y_test, y_pred_test, "Test Set (XGBoost)")
