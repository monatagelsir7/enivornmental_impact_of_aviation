import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_parquet("cleaned_aviation_data_v3.parquet")
df["same_continent"] = df["departure_continent"] == df["arrival_continent"]
df["same_country"] = df["departure_country"] == df["arrival_country"]


df["domestic"] = df["domestic"].astype("bool")
X = df[
    [
        "acft_class",
        # "seats",
        "n_flights",
        "departure_country",
        "departure_continent",
        "arrival_country",
        "arrival_continent",
        "domestic",
        "ask",
        "rpk",
        "fuel_burn",
        "same_continent",
        "same_country",
    ]
]
y = df["co2_per_distance"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split into train + validation and test sets (80% train_val, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split train + validation into separate train and validation sets (now 64% train, 16% val)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the validation and test data based on the training data scaling
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train the model on scaled training data
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on scaled validation set
y_val_pred = model.predict(X_val_scaled)

# Predict on scaled test set
y_test_pred = model.predict(X_test_scaled)

# Evaluation on validation
print("📊 Validation Set Evaluation")
print("MSE (val):", mean_squared_error(y_val, y_val_pred))
print("R² (val):", r2_score(y_val, y_val_pred))

# Evaluation on test
print("\n🧪 Test Set Evaluation")
print("MSE (test):", mean_squared_error(y_test, y_test_pred))
print("R² (test):", r2_score(y_test, y_test_pred))
