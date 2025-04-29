import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and training schema
model = joblib.load('../2.Models/random_forest_model.pkl')
X_train = pd.read_parquet("../Test-Train-Validation Data/X_train.parquet")
expected_columns = X_train.columns.tolist()

# Helper function: prepare input
def prepare_input(raw_dict, expected_columns):
    df = pd.DataFrame([raw_dict])
    df_encoded = pd.get_dummies(df)
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[expected_columns]

# Base input (RDU to ORD route)
base_input = {
    'airline_iata': 'AA',
    'acft_class': 'NB',
    'acft_icao': 'B738',
    'departure_country': 'US',
    'departure_continent': 'NA',
    'arrival_country': 'US',
    'arrival_continent': 'NA',
    'iata_departure': 'RDU',
    'iata_arrival': 'ORD',
    'domestic': 1,
    'ask': 185405250.3,
    'rpk': 152773926.3,
    'fuel_burn': 375569
}

# Aircraft variants (with both codes and full class names) ## fuel_burn assumption ia based on the average fuel burn per km (EDA folder: avr_fuelburn_by_acft_type.py)
aircraft_variants = [
    {'acft_class': 'NB', 'acft_class_full': 'Narrow Body', 'acft_icao': 'B738', 'fuel_burn': 375569},
    {'acft_class': 'WB', 'acft_class_full': 'Wide Body', 'acft_icao': 'B763', 'fuel_burn': 483025},
    {'acft_class': 'RJ', 'acft_class_full': 'Regional Jet', 'acft_icao': 'CRJ2', 'fuel_burn': 316015},
    {'acft_class': 'TP', 'acft_class_full': 'Turbo Propeller', 'acft_icao': 'C208', 'fuel_burn': 178624},
    {'acft_class': 'PP', 'acft_class_full': 'Piston Propeller', 'acft_icao': 'C172', 'fuel_burn': 47060},
    {'acft_class': 'PJ', 'acft_class_full': 'Private Jet', 'acft_icao': 'GL5T', 'fuel_burn': 7183}
]

# Step 1: Model prediction
results = []
for variant in aircraft_variants:
    row = base_input.copy()
    row.update(variant)
    df_prepared = prepare_input(row, expected_columns)
    pred = model.predict(df_prepared)[0]
    results.append({
        'acft_class': variant['acft_class'],
        'acft_class_full': variant['acft_class_full'],
        'co2_per_distance': pred
    })

df_results = pd.DataFrame(results)

# Step 2: Passenger-based emission
seats_info = {
    'NB': 160,
    'WB': 218,
    'RJ': 50,
    'TP': 9,
    'PP': 4,
    'PJ': 13
}

load_factor = 0.824  # fixed
df_results['seats'] = df_results['acft_class'].map(seats_info)
df_results['passenger_count'] = df_results['seats'] * load_factor
df_results['co2_per_passenger_per_km'] = df_results['co2_per_distance'] / df_results['passenger_count']

# Step 3: Plotting (Dual Y-axis)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar: CO2 per km
color = 'skyblue'
ax1.bar(df_results['acft_class_full'], df_results['co2_per_distance'], color=color)
ax1.set_xlabel('Aircraft Class')
ax1.set_ylabel('Predicted CO₂ per Distance (kg/km)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title('Predicted CO₂ per Distance and per Passenger per km (RDU → ORD)')
ax1.grid(axis='y')

# Line: CO2 per passenger per km
ax2 = ax1.twinx()
color = 'darkgreen'
ax2.plot(df_results['acft_class_full'], df_results['co2_per_passenger_per_km'],
         marker='o', color=color, label='CO₂ per Passenger per km')
ax2.set_ylabel('Predicted CO₂ per Passenger per km (kg/passenger/km)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Final layout
fig.tight_layout()
plt.xticks(rotation=20)
plt.show()
