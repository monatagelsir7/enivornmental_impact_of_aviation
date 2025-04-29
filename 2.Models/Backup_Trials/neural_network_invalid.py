from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

df = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/cleaned_aviation_data_v3.parquet")

X = df[['acft_class', 'seats', 'n_flights', 'departure_country', 'departure_continent',
        'arrival_country', 'arrival_continent', 'domestic', 'ask', 'rpk', 'fuel_burn']]
y = df['co2_per_distance']

X_encoded = pd.get_dummies(X, drop_first=True)

X_train = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet")
X_test = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_test.parquet")
X_val = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_val.parquet")
y_train = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_train.parquet").squeeze()
y_test = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_test.parquet").squeeze()
y_val = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_val.parquet").squeeze()

# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # 1 hidden layer
model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

y_pred = model.predict(X_test).flatten()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Baseline Neural Network R-squared (R²): {r2:.4f}")
print(f"Baseline Neural Network Mean Squared Error (MSE): {mse:.4f}")
print(f"Baseline Neural Network Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Baseline Neural Network Mean Absolute Error (MAE): {mae:.4f}")


# Hidden layers	1	Keep it simple
# Neurons	64 neurons	Moderate size, not too small or too large
# Activation	ReLU	Default choice for tabular data
# Optimizer	Adam	Reliable default optimizer
# Loss	MSE	Standard for regression tasks
# Epochs	50	Enough for small to medium datasets
# Batch size	32	Balanced: small enough to learn, large enough for speed