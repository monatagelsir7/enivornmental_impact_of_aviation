from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
X_train = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/Shawn/Test-Train-Validation%20Data/X_train.parquet"
)
X_test = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/Shawn/Test-Train-Validation%20Data/X_test.parquet"
)
X_val = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/Shawn/Test-Train-Validation%20Data/X_val.parquet"
)

y_train = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/Shawn/Test-Train-Validation%20Data/y_train.parquet"
).squeeze()
y_test = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/Shawn/Test-Train-Validation%20Data/y_test.parquet"
).squeeze()
y_val = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/Shawn/Test-Train-Validation%20Data/y_val.parquet"
).squeeze()

# Define pools
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)
test_pool = Pool(X_test, y_test)

# Train CatBoost model
cat_model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    eval_metric="RMSE",
    custom_metric=["R2"],
    verbose=0,
)

# Fit model with early stopping
cat_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=10, plot=False)

# Predictions
y_pred_val_cat = cat_model.predict(val_pool)
y_pred_test_cat = cat_model.predict(test_pool)


# Evaluation function
def evaluate(y_true, y_pred, dataset_name="Dataset"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{dataset_name} Performance:")
    print(f"R² (R-squared): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")


# Evaluate model
evaluate(y_val, y_pred_val_cat, "Validation Set (CatBoost)")
evaluate(y_test, y_pred_test_cat, "Test Set (CatBoost)")

# Get the actual number of iterations used (accounting for early stopping)
best_iteration = cat_model.get_best_iteration()
iterations_used = best_iteration + 1 if best_iteration is not None else 100
print(f"\nModel used {iterations_used} iterations")

# Manual metric calculation
train_rmse = []
val_rmse = []
test_rmse = []
train_r2 = []
val_r2 = []
test_r2 = []

for i in range(1, iterations_used + 1):
    # Predictions at iteration i
    y_train_pred_i = cat_model.predict(X_train, ntree_end=i)
    y_val_pred_i = cat_model.predict(X_val, ntree_end=i)
    y_test_pred_i = cat_model.predict(X_test, ntree_end=i)

    # Calculate metrics
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred_i)))
    val_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred_i)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred_i)))
    train_r2.append(r2_score(y_train, y_train_pred_i))
    val_r2.append(r2_score(y_val, y_val_pred_i))
    test_r2.append(r2_score(y_test, y_test_pred_i))

# Create DataFrames
df_eval = pd.DataFrame(
    {
        "Iteration": range(1, iterations_used + 1),
        "Train RMSE": train_rmse,
        "Validation RMSE": val_rmse,
        "Test RMSE": test_rmse,
    }
)

df_r2 = pd.DataFrame(
    {
        "Iteration": range(1, iterations_used + 1),
        "Train R2": train_r2,
        "Validation R2": val_r2,
        "Test R2": test_r2,
    }
)

# Create plot
fig, ax1 = plt.subplots(figsize=(14, 6))

# RMSE Plotting
ax1.set_xlabel("Iteration")
ax1.set_ylabel("RMSE", color="blue")
line1 = ax1.plot(
    df_eval["Iteration"],
    df_eval["Train RMSE"],
    label="Train RMSE",
    color="blue",
    linewidth=2,
)
line2 = ax1.plot(
    df_eval["Iteration"],
    df_eval["Validation RMSE"],
    label="Validation RMSE",
    color="orange",
    linewidth=2,
)
line3 = ax1.plot(
    df_eval["Iteration"],
    df_eval["Test RMSE"],
    label="Test RMSE",
    color="green",
    linestyle="--",
    linewidth=2,
)
ax1.tick_params(axis="y", labelcolor="blue")

# Set y-axis limits based on data
rmse_max = max(df_eval[["Train RMSE", "Validation RMSE", "Test RMSE"]].max())
ax1.set_ylim(0, rmse_max * 1.1)

# R2 Plotting
ax2 = ax1.twinx()
ax2.set_ylabel("R2 Score", color="teal")
line4 = ax2.plot(
    df_r2["Iteration"], df_r2["Train R2"], label="Train R2", color="teal", linewidth=2
)
line5 = ax2.plot(
    df_r2["Iteration"],
    df_r2["Validation R2"],
    label="Validation R2",
    color="darkorange",
    linewidth=2,
)
line6 = ax2.plot(
    df_r2["Iteration"],
    df_r2["Test R2"],
    label="Test R2",
    color="deepskyblue",
    linestyle="--",
    linewidth=2,
)
ax2.tick_params(axis="y", labelcolor="teal")
ax2.set_ylim(
    min(0.9, df_r2[["Train R2", "Validation R2", "Test R2"]].min().min() - 0.05),
    max(1.0, df_r2[["Train R2", "Validation R2", "Test R2"]].max().max() + 0.05),
)

# Combine legends
lines = line1 + line2 + line3 + line4 + line5 + line6
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)

plt.title("CatBoost: RMSE and R2 Over Iterations")
plt.grid(True)
plt.tight_layout()

# Save plot to directory
save_dir = (
    r"C:\Users\DELL\ML Project\enivornmental_impact_of_aviation\2.Models\results_img"
)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "catboost_performance_metrics.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
print(f"\nPlot saved to: {save_path}")

plt.show()
