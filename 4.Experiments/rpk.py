'''
Experiment idea
1. rpk (Revenue Passenger Kilometers) -> co2_per_distance
2. rpk/ask (Revenue Passenger Kilometers / Available Seat Kilometers) -> co2_per_distance
  - find optimum rpk/ask ratio (full capacity may affect the co2_per_distance)

Test Summary : 
The CO2_per_distance(km) increase steadily as the passenger load factor(RPK/ASK) ratio rises, up to approximately 0.5. 
However, beyond this threshold, CO2_per_distance(km) begin to decline consistently. 
This indicates that aircraft operating below a 50% the load factor may benefit from maximizing the load factor 
ratio—such as through codeshare agreements with other carriers—to improve overall environmental efficiency.
'''

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Model
model = joblib.load('../2.Models/random_forest_model.pkl')
X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_train.parquet")
expected_columns = X_train.columns.tolist()

# Prepare input data
def prepare_input(raw_dict, expected_columns):
    df = pd.DataFrame([raw_dict])
    df_encoded = pd.get_dummies(df)
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_columns]
    return df_encoded

# Base input data(sample)
base_input = {
    'airline_iata': 'AF',
    'acft_class': 'NB',
    'departure_country': 'France',
    'departure_continent': 'Europe',
    'arrival_country': 'Germany',
    'arrival_continent': 'Europe',
    'domestic': 0,
    'ask': 200000,
    'fuel_burn': 12000,
    'iata_departure': 'CDG',
    'iata_arrival': 'FRA',
    'acft_icao': 'A320'
}

# RPK test
rpk_values = np.linspace(50000, 300000, 50)
predictions = []

for rpk in rpk_values:
    row = base_input.copy()
    row['rpk'] = rpk
    df_row = prepare_input(row, expected_columns)
    pred = model.predict(df_row)[0]
    predictions.append(pred)

plt.figure(figsize=(8, 5))
plt.plot(rpk_values, predictions, marker='o')
plt.xlabel('RPK (Revenue Passenger Kilometers)')
plt.ylabel('Predicted CO₂ per km')
plt.title('Relationship between RPK and CO₂/km')
plt.grid(True)
plt.show()

# Load Factor (RPK/ASK) test
ask_fixed = 200000
load_factors = np.linspace(0.25, 1.0, 50)
predictions = []

for lf in load_factors:
    row = base_input.copy()
    row['rpk'] = lf * ask_fixed
    df_row = prepare_input(row, expected_columns)
    pred = model.predict(df_row)[0]
    predictions.append(pred)

plt.figure(figsize=(8,5))
plt.plot(load_factors, predictions, marker='o')
plt.xlabel('Load Factor (RPK / ASK)')
plt.ylabel('Predicted CO₂ per km')
plt.title('Effect of Load Factor on CO₂/km')
plt.grid(True)
plt.show()