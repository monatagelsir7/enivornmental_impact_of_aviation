'''
Experiment idea
1. rpk (Revenue Passenger Kilometers) -> co2_per_distance
2. rpk/ask (Revenue Passenger Kilometers / Available Seat Kilometers) -> co2_per_distance
  - Passenger load factor (RPK/ASK) is a measure of how efficiently an airline is filling seats and generating revenue.
  - find optimum rpk/ask ratio (full capacity may affect the co2_per_distance)

Test Summary
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

# Base input data(sample) # This case is from rpk_sampler(real_observations) based on the median values
base_input = {
    'airline_iata': 'AF', # Air France
    'acft_class': 'NB', # Narrow Body
    'acft_icao': 'A321', # Airbus A321
    'departure_continent': 'EU', # Europe 
    'departure_country': 'CH', # Switzerland
    'iata_departure': 'ZRH', # Zurich Airport
    'arrival_continent': 'EU', # Europe
    'arrival_country': 'FR', # France
    'iata_arrival': 'CDG', # Charles de Gaulle Airport
    'domestic': 0, # 0 = intl, 1 = domestic
    'ask': 1400814.103, # Index 94298
    'fuel_burn': 45330.2042, # # Index 94298
}


# RPK test
load_factors = np.linspace(0.25, 1.0, 50)
rpk_values = load_factors * base_input['ask']
predictions_rpk = []

for rpk in rpk_values:
    row = base_input.copy()
    row['rpk'] = rpk
    df_row = prepare_input(row, expected_columns)
    pred = model.predict(df_row)[0]
    predictions_rpk.append(pred)

# RPK vs CO₂/km
plt.figure(figsize=(8, 5))
plt.plot(rpk_values, predictions_rpk, marker='o')
plt.xlabel('RPK (Revenue Passenger Kilometers)')
plt.ylabel('Predicted CO₂ per km')
plt.title('RPK vs Predicted CO₂/km')
plt.grid(True)
plt.show()

# Passenger Load Factor → co2_per_distance
predictions_lf = []

for lf in load_factors:
    row = base_input.copy()
    row['rpk'] = lf * row['ask']
    df_row = prepare_input(row, expected_columns)
    pred = model.predict(df_row)[0]
    predictions_lf.append(pred)

# Load Factor vs CO₂/km
plt.figure(figsize=(8, 5))
plt.plot(load_factors, predictions_lf, marker='o')
plt.xlabel('Passenger Load Factor (RPK / ASK)')
plt.ylabel('Predicted CO₂ per km')
plt.title('Effect of Load Factor on CO₂/km')
plt.grid(True)
plt.show()