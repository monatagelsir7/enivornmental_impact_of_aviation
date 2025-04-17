from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# def clean_column_names(df):
#     df.columns = df.columns.str.replace(r"\[", "(", regex=True)
#     df.columns = df.columns.str.replace(r"\]", ")", regex=True)
#     df.columns = df.columns.str.replace(r"<", "lt_", regex=True)
#     df.columns = df.columns.str.replace(r">", "gt_", regex=True)
#     return df


# df = pd.read_parquet(
#     "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/cleaned_aviation_data_v3.parquet"
# )

# X_raw = df[
#     [
#         "airline_iata",
#         "acft_class",
#         "departure_country",
#         "departure_continent",
#         "arrival_country",
#         "arrival_continent",
#         "domestic",
#         "ask",
#         "rpk",
#         "fuel_burn",
#         "iata_departure",
#         "iata_arrival",
#         "acft_icao",
#     ]
# ]
# y_raw = df["co2_per_distance"]

# LOAD SPLIT DATA
X_train = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/Shawn/Test-Train-Validation%20Data/X_train.parquet"
)
X_train.columns
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

# -- CLEAN COLUMNS (if necessary for consistency) --
# X_train = clean_column_names(X_train)
# X_test = clean_column_names(X_test)
# X_val = clean_column_names(X_val)

# -- IDENTIFY CATEGORICAL FEATURES --
# categorical_features = [
#     "airline_iata",
#     "acft_class",
#     "departure_country",
#     "departure_continent",
#     "arrival_country",
#     "arrival_continent",
#     "domestic",
#     "iata_departure",
#     "iata_arrival",
#     "acft_icao",
# ]

# -- DEFINE POOLS FOR TRAINING & EVALUATION --
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)
test_pool = Pool(X_test, y_test)

# -- TRAIN CATBOOST MODEL --
cat_model = CatBoostRegressor(
    iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0
)
cat_model.fit(train_pool, eval_set=val_pool)


# -- PREDICT --
y_pred_val_cat = cat_model.predict(val_pool)
y_pred_test_cat = cat_model.predict(test_pool)


# -- EVALUATION FUNCTION --
def evaluate(y_true, y_pred, dataset_name="Dataset"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{dataset_name} Performance:")
    print(f"R² (R-squared): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")


# -- EVALUATE CATBOOST PERFORMANCE --
evaluate(y_val, y_pred_val_cat, "Validation Set (CatBoost)")
evaluate(y_test, y_pred_test_cat, "Test Set (CatBoost)")

# -- Plotting --

eval_results = cat_model.get_evals_result()
print(eval_results.keys())
iterations = len(eval_results["validation"]["RMSE"])
######## For RMSE
test_rmse = []
for i in range(1, iterations + 1):
    y_test_pred_i = cat_model.predict(X_test, ntree_end=i)
    rmse_i = np.sqrt(mean_squared_error(y_test, y_test_pred_i))
    test_rmse.append(rmse_i)

df_eval = pd.DataFrame(
    {
        "Iteration": range(1, iterations + 1),
        "Train RMSE": eval_results["learn"]["RMSE"],
        "Validation RMSE": eval_results["validation"]["RMSE"],
        "Test RMSE": test_rmse,
    }
)

######## For R2
train_r2 = []
val_r2 = []
test_r2 = []

# Loop through each iteration and compute R2

for i in range(1, iterations + 1):
    # Predict at current iteration
    y_train_pred_i = cat_model.predict(X_train, ntree_end=i)
    y_val_pred_i = cat_model.predict(X_val, ntree_end=i)
    y_test_pred_i = cat_model.predict(X_test, ntree_end=i)

    # Compute R2
    train_r2.append(r2_score(y_train, y_train_pred_i))
    val_r2.append(r2_score(y_val, y_val_pred_i))
    test_r2.append(r2_score(y_test, y_test_pred_i))

# Build DataFrame
df_r2 = pd.DataFrame(
    {
        "Iteration": range(1, iterations + 1),
        "Train R2": train_r2,
        "Validation R2": val_r2,
        "Test R2": test_r2,
    }
)
fig, ax1 = plt.subplots(figsize=(14, 6))

# RMSE on Left Axis
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
ax1.set_ylim(0, 2500)  # Adjusting y limits based on RMSE range we want to see

# R2 on Right Axis
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
ax2.set_ylim(0.98, 1.0)  # Adjusting y limits based on R2 range we want to see

lines = line1 + line2 + line3 + line4 + line5 + line6
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)

plt.title("CatBoost: RMSE and R2 Over Iterations")
plt.grid(True)
plt.tight_layout()
plt.show()
