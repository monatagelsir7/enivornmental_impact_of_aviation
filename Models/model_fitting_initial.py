# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_parquet("cleaned_aviation_data_v3.parquet")

# Feature engineering
df["same_continent"] = df["departure_continent"] == df["arrival_continent"]
df["same_country"] = df["departure_country"] == df["arrival_country"]
df["domestic"] = df["domestic"].astype("bool")

# Define features and target
X = df[
    [
        "acft_class",
        # "seats",  # commented out
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

# Identify numeric and categorical features
numeric_features = ["n_flights", "ask", "rpk", "fuel_burn"]
categorical_features = [col for col in X.columns if col not in numeric_features]

# Build ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", "passthrough", categorical_features),
    ]
)

# Build pipeline with preprocessing and linear regression
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

# Split the data into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict and evaluate on validation and test sets
y_val_pred = pipeline.predict(X_val)
y_test_pred = pipeline.predict(X_test)

print("📊 Validation Set Evaluation")
print("MAE (val):", mean_absolute_error(y_val, y_val_pred))
print("R² (val):", r2_score(y_val, y_val_pred))

print("\n🧪 Test Set Evaluation")
print("MAE (test):", mean_absolute_error(y_test, y_test_pred))
print("R² (test):", r2_score(y_test, y_test_pred))
