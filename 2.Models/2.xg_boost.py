from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Functions ---
def clean_column_names(df):
    df.columns = df.columns.str.replace(r'\[', '(', regex=True)
    df.columns = df.columns.str.replace(r'\]', ')', regex=True)
    df.columns = df.columns.str.replace(r'<', 'lt_', regex=True)
    df.columns = df.columns.str.replace(r'>', 'gt_', regex=True)
    return df

def evaluate(y_true, y_pred, dataset_name="Dataset"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{dataset_name} Performance:")
    print(f"R² (R-squared): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")

# --- Load Data ---
X_train = pd.read_parquet("../Test-Train-Validation Data/X_train.parquet")
X_test = pd.read_parquet("../Test-Train-Validation Data/X_test.parquet")
X_val = pd.read_parquet("../Test-Train-Validation Data/X_val.parquet")
y_train = pd.read_parquet("../Test-Train-Validation Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("../Test-Train-Validation Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("../Test-Train-Validation Data/y_val.parquet").squeeze()

# --- Clean Columns ---
X_train = clean_column_names(X_train)
X_test = clean_column_names(X_test)
X_val = clean_column_names(X_val)

# --- Train Model ---
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    eval_metric="rmse"
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    verbose=False
)

# --- Track Training ---
results = xgb_model.evals_result()
train_rmse = results['validation_0']['rmse']
val_rmse = results['validation_1']['rmse']
test_rmse = results['validation_2']['rmse']

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_rmse, label="Train RMSE")
plt.plot(range(1, 101), val_rmse, label="Validation RMSE")
plt.plot(range(1, 101), test_rmse, label="Test RMSE")
plt.xlabel("Number of Trees")
plt.ylabel("RMSE")
plt.title("XGBoost Performance (RMSE vs. Number of Trees)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Evaluate Model ---
y_pred_val = xgb_model.predict(X_val)
y_pred_test = xgb_model.predict(X_test)
evaluate(y_val, y_pred_val, "Validation Set (XGBoost)")
evaluate(y_test, y_pred_test, "Test Set (XGBoost)")

# --- Save Plot ---
plt.savefig("results_img/xgboost_rmse_plot.png", dpi=300)
