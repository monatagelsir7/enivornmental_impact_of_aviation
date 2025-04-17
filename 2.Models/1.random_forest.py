# ### MODEL1 incremental training to 100 estimators ###

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# # Load data
# X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_train.parquet")
# X_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_test.parquet")
# X_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_val.parquet")
# y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_train.parquet").squeeze()
# y_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_test.parquet").squeeze()
# y_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_val.parquet").squeeze()
# print("Training/Validation/Test data loaded")

# # Initialize model
# rf_model = RandomForestRegressor(n_estimators=1, warm_start=True, random_state=42, n_jobs=-1)

# # Track RMSEs and actual tree counts
# train_rmse_list = []
# val_rmse_list = []
# test_rmse_list = []
# tree_counts = []

# # Iterative training and error tracking
# for n_trees in range(1, 101, 10):  # Trees: 1, 11, 21, ..., 91
#     rf_model.set_params(n_estimators=n_trees)
#     rf_model.fit(X_train, y_train)

#     y_train_pred = rf_model.predict(X_train)
#     y_val_pred = rf_model.predict(X_val)
#     y_test_pred = rf_model.predict(X_test)

#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

#     train_rmse_list.append(train_rmse)
#     val_rmse_list.append(val_rmse)
#     test_rmse_list.append(test_rmse)
#     tree_counts.append(n_trees)

#     print(f"Trees: {n_trees:3} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(tree_counts, train_rmse_list, label='Train RMSE', linewidth=1)
# plt.plot(tree_counts, val_rmse_list, label='Validation RMSE', linewidth=1)
# plt.plot(tree_counts, test_rmse_list, label='Test RMSE', linewidth=1)
# plt.xlabel('Number of Trees')
# plt.ylabel('RMSE')
# plt.title('Random Forest Performance (RMSE vs. Number of Trees)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("results_img/randomforest_rmse_plot.png", dpi=300)
# plt.show()


# '''
# Training/Validation/Test data loaded
# Trees:   1 | Train RMSE: 1967.0906 | Val RMSE: 3207.8019 | Test RMSE: 3275.7662
# Trees:  11 | Train RMSE: 1090.9325 | Val RMSE: 2326.3931 | Test RMSE: 2643.6891
# Trees:  21 | Train RMSE: 997.6132 | Val RMSE: 2310.0992 | Test RMSE: 2644.6761
# Trees:  31 | Train RMSE: 970.5103 | Val RMSE: 2269.1709 | Test RMSE: 2575.8214
# Trees:  41 | Train RMSE: 986.0953 | Val RMSE: 2271.1918 | Test RMSE: 2539.3715
# Trees:  51 | Train RMSE: 988.5807 | Val RMSE: 2278.8249 | Test RMSE: 2536.5934
# Trees:  61 | Train RMSE: 972.7973 | Val RMSE: 2258.4468 | Test RMSE: 2508.1547
# Trees:  71 | Train RMSE: 973.5421 | Val RMSE: 2264.8556 | Test RMSE: 2525.2312
# Trees:  81 | Train RMSE: 976.7938 | Val RMSE: 2264.3685 | Test RMSE: 2526.2335
# Trees:  91 | Train RMSE: 985.8692 | Val RMSE: 2275.7229 | Test RMSE: 2533.0360

# Hyperparameters: n_estimators = 610


## MODEL2 BEFORE Tuning, estimators is 100 ###

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load data
X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_train.parquet")
X_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_test.parquet")
X_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_val.parquet")
y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_val.parquet").squeeze()

print("Training/Validation/Test data loaded")

# Train model
rf_model = RandomForestRegressor(n_estimators=61, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model training completed")

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# Evaluation Metrics
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Test Set Performance:")
print(f"R-squared (R²): {r2_score(y_test, y_pred_test):.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_test):.4f}")

print("Validation Set Performance:")
print(f"R-squared (R²): {r2_score(y_val, y_pred_val):.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_val:.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_val, y_pred_val):.4f}")

### Saving the model for prediction
# Save the trained model to a file
joblib.dump(rf_model, "../2.Models/random_forest_model.pkl")
print("Model saved to random_forest_model.pkl")
