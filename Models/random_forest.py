# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np

# df = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/cleaned_aviation_data_v3.parquet")

# # Load pre-split datasets
# X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_train.parquet")
# X_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_test.parquet")
# X_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_val.parquet")
# y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/y_train.parquet").squeeze()
# y_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_test.parquet/y_test.parquet").squeeze()
# y_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_test.parquet/y_val.parquet").squeeze()

# # Train the model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Predict on test set
# y_pred_test = rf_model.predict(X_test)

# # Evaluate on test set
# r2 = r2_score(y_test, y_pred_test)
# mse = mean_squared_error(y_test, y_pred_test)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred_test)

# print(f"R-squared (R²): {r2:.4f}")
# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")

# # Predict on validation set (optional)
# y_pred_val = rf_model.predict(X_val)

# # If needed, evaluate using y_val
# r2_val = r2_score(y_val, y_pred_val)
# mse_val = mean_squared_error(y_val, y_pred_val)
# rmse_val = np.sqrt(mse_val)
# mae_val = mean_absolute_error(y_val, y_pred_val)

# print("\nValidation Set Performance:")
# print(f"R-squared (R²): {r2_val:.4f}")
# print(f"Mean Squared Error (MSE): {mse_val:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")
# print(f"Mean Absolute Error (MAE): {mae_val:.4f}")


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load datasets
X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_train.parquet")
X_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_test.parquet")
X_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/X_val.parquet")
y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation/Test-Train-Validation Data/y_val.parquet").squeeze()

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # keep numerical columns as they are
)

# Build pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred_test = rf_pipeline.predict(X_test)

# Evaluate on test set
r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)

print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Predict and evaluate on validation set
y_pred_val = rf_pipeline.predict(X_val)

r2_val = r2_score(y_val, y_pred_val)
mse_val = mean_squared_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(y_val, y_pred_val)

print("\nValidation Set Performance:")
print(f"R-squared (R²): {r2_val:.4f}")
print(f"Mean Squared Error (MSE): {mse_val:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")
print(f"Mean Absolute Error (MAE): {mae_val:.4f}")
