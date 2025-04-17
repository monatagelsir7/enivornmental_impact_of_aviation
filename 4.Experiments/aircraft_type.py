import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("../2.Models/random_forest_model.pkl")

# Training columns (from X_train)
X_train = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet"
)
expected_columns = X_train.columns.tolist()


# Input preparation function
def prepare_input(raw_dict, expected_columns):
    df = pd.DataFrame([raw_dict])
    df_encoded = pd.get_dummies(df)
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_columns]
    return df_encoded


# Aircrafts to test
aircraft_list = ["A320", "B738", "E190", "A321", "B77W", "A359", "CRJ9", "A388"]
predictions = []

# Base sample input
base_input = {
    "airline_iata": "AF",
    "acft_class": "NB",
    "departure_country": "France",
    "departure_continent": "Europe",
    "arrival_country": "Germany",
    "arrival_continent": "Europe",
    "domestic": 0,
    "ask": 200000,
    "rpk": 150000,
    "fuel_burn": 12000,
    "iata_departure": "CDG",
    "iata_arrival": "FRA",
    "acft_icao": "A320",
}

# Run predictions
for acft in aircraft_list:
    row = base_input.copy()
    row["acft_icao"] = acft
    df_row = prepare_input(row, expected_columns)
    pred = model.predict(df_row)[0]
    predictions.append(pred)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(aircraft_list, predictions)
plt.xlabel("Aircraft Type (ICAO Code)")
plt.ylabel("Predicted CO₂ per km")
plt.title("CO₂/km by Aircraft Type")
plt.xticks(rotation=45)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()
