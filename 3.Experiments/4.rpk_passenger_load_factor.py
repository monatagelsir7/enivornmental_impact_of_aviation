'''
Experiment idea
1. rpk (Revenue Passenger Kilometers) -> co2_per_distance
2. rpk/ask (Revenue Passenger Kilometers / Available Seat Kilometers) -> co2_per_distance
  - Passenger load factor (RPK/ASK) is a measure of how efficiently an airline is filling seats and generating revenue.
  - Find the optimum RPK/ASK ratio to minimize CO₂ emissions per distance traveled.
  - Full capacity may not always lead to minimum CO₂/km due to aircraft operational characteristics.

Test Summary
A real-world case from the most frequently operated aircraft class (Boeing 737, Narrow Body) was selected, 
representing median operational metrics within this subset. 
The sensitivity analysis shows that CO₂ emissions per kilometer decline significantly when the passenger load factor 
exceeds 60%, with diminishing improvements observed beyond approximately 85%. 
This indicates that maintaining load factors above 60% should be prioritized to maximize environmental efficiency, 
while chasing near-100% occupancy offers limited additional benefits.
'''

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import textwrap
import shap
shap.initjs()


# Model
model = joblib.load('../2.Models/random_forest_model.pkl')
X_train = pd.read_parquet("../Test-Train-Validation Data/X_train.parquet")
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
    'airline_iata': 'DL', # Delta Airlines
    'acft_class': 'NB', # Narrow Body
    'acft_icao': 'B737', # Boeing 737
    'departure_continent': 'NA', # North America 
    'departure_country': 'US', # United States
    'iata_departure': 'JFK', # John F Kennedy International Airport
    'arrival_continent': 'NA', # North America
    'arrival_country': 'AG', # Antigua and Barbuda
    'iata_arrival': 'ANU', # V.C. Bird International Airport
    'domestic': 0, # 0 = intl, 1 = domestic
    'ask': 2119478.265, # from rpk_sampler index# 118081 
    'fuel_burn': 54221.87646, # from rpk_sampler index# 118081 
}

# Real RPK of case above Index 118081
rpk = 1746450.09
real_passenger_load_factor = rpk / base_input['ask']
print(f"Real Passenger Load Factor: {real_passenger_load_factor:.2f}")


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
plt.plot(rpk_values, predictions_rpk)
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


# Load Factor vs CO₂/km with Real Passenger Load Factor Line
plt.figure(figsize=(8, 5))
plt.plot(load_factors, predictions_lf, label='Predicted CO₂/km')
plt.axvline(x=real_passenger_load_factor, color='red', linestyle='--', label='Actual Passenger Load Factor')
plt.xlabel('Passenger Load Factor (RPK / ASK)')
plt.ylabel('Predicted CO₂ per km')
plt.title('Effect of Passenger Load Factor on CO₂/km')
plt.legend()
plt.grid(True)
plt.show()



note_text = (
    "Note: This prediction is based on a real-world flight with the following parameters:\n"
    "Airline: Delta Airlines (DL), Aircraft: Boeing 737 (Narrow Body),\n"
    "Route: New York John F. Kennedy (JFK), United States to V.C. Bird International (ANU), Antigua and Barbuda,\n"
    "International flight, ASK = 2,119,478.265, Fuel burn = 54,221.87646.\n"
    "Passenger Load Factor (RPK/ASK) indicates how efficiently seat capacity is used; "
    "higher values mean more seats are filled with passengers.\n"
    "RPK (Revenue Passenger Kilometers) is the total number of kilometers flown by paying passengers.\n"
    "ASK (Available Seat Kilometers) is a measure of the number of kilometers available for passengers to fly."
)

wrapped_text = "\n".join(textwrap.wrap(note_text, width=85))
plt.figtext(0.5, -0.1, wrapped_text, horizontalalignment='center', fontsize=9)