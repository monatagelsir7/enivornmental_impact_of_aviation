
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
X_train=pd.read_parquet("Test-Train-Validation Data/X_train.parquet")
X_val=pd.read_parquet("Test-Train-Validation Data/X_val.parquet")
X_test=pd.read_parquet("Test-Train-Validation Data/X_test.parquet")
y_train=pd.read_parquet("Test-Train-Validation Data/y_train.parquet")
y_val=pd.read_parquet("Test-Train-Validation Data/y_val.parquet")
y_test=pd.read_parquet("Test-Train-Validation Data/y_test.parquet")
y_train = y_train.squeeze()
y_val = y_val.squeeze()
y_test = y_test.squeeze()
categorical_columns = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
numerical_columns = [col for col in X_train.columns if col not in categorical_columns]
scaler = StandardScaler()

X_train_scaled_num = pd.DataFrame(scaler.fit_transform(X_train[numerical_columns]), 
                                  columns=numerical_columns, 
                                  index=X_train.index)
X_val_scaled_num = pd.DataFrame(scaler.transform(X_val[numerical_columns]), 
                                columns=numerical_columns, 
                                index=X_val.index)
X_test_scaled_num = pd.DataFrame(scaler.transform(X_test[numerical_columns]), 
                                 columns=numerical_columns, 
                                 index=X_test.index)

X_train_scaled = pd.concat([X_train_scaled_num, X_train[categorical_columns]], axis=1)
X_val_scaled = pd.concat([X_val_scaled_num, X_val[categorical_columns]], axis=1)
X_test_scaled = pd.concat([X_test_scaled_num, X_test[categorical_columns]], axis=1)
# Train the model on scaled training data
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# Predict on scaled validation set
y_val_pred = model.predict(X_val_scaled)
# Predict on scaled test set
y_test_pred = model.predict(X_test_scaled)
# Evaluation on validation
print("Validation Set Evaluation")
print("MSE (val):", mean_squared_error(y_val, y_val_pred))
print("R² (val):", r2_score(y_val, y_val_pred))
# Evaluation on test
print("Test Set Evaluation")
print("MSE (test):", mean_squared_error(y_test, y_test_pred))
print("R² (test):", r2_score(y_test, y_test_pred))
