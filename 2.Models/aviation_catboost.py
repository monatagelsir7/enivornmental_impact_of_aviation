from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np


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
cat_model.fit(train_pool)

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
