import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import textwrap

# Load trained model
model = joblib.load('../2.Models/random_forest_model.pkl')
X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_train.parquet")
expected_columns = X_train.columns.tolist()

# Function to prepare input data
def prepare_input(raw_dict, expected_columns):
    df = pd.DataFrame([raw_dict])
    df_encoded = pd.get_dummies(df)
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_columns]
    return df_encoded

# Base input data (Air France flight: ZRH → CDG)
base_input = {
    'airline_iata': 'AF',              # Air France
    'acft_class': 'NB',                # Narrow Body
    'departure_continent': 'EU',
    'departure_country': 'CH',
    'iata_departure': 'ZRH',
    'arrival_continent': 'EU',
    'arrival_country': 'FR',
    'iata_arrival': 'CDG',
    'domestic': 0,                     # International
    'ask': 1400814.103,
    'fuel_burn': 45330.2042,          # Fixed for all unless you want to vary per aircraft
}

# Aircraft types to compare
aircraft_list = ['B752', 'B737', 'B738','A320', 'A321', 'A319']  # Add more aircraft types if needed

# Load factor range (0.25 to 1.0)
load_factors = np.linspace(0.25, 1.0, 50)

# Plotting
plt.figure(figsize=(10, 6))

for acft in aircraft_list:
    predictions_lf = []

    for lf in load_factors:
        row = base_input.copy()
        row['acft_icao'] = acft
        row['rpk'] = lf * row['ask']  # RPK = Load Factor * ASK
        df_row = prepare_input(row, expected_columns)
        pred = model.predict(df_row)[0]
        predictions_lf.append(pred)

    plt.plot(load_factors, predictions_lf, marker='o', label=f'{acft}')

# Final plot setup
plt.xlabel('Passenger Load Factor (RPK / ASK)')
plt.ylabel('Predicted CO₂ per km')
plt.title('Effect of Passenger Load Factor on CO₂/km by Aircraft Type')
plt.legend(title="Aircraft Type")
plt.grid(True)
plt.tight_layout()

# Note text
note_text = (
    "Note: Each line represents a different aircraft type for the same Air France route\n"
    "(ZRH to CDG). All input features were held constant except aircraft type and RPK."
)
wrapped_text = "\n".join(textwrap.wrap(note_text, width=85))
plt.figtext(0.5, -0.1, wrapped_text, horizontalalignment='center', fontsize=9)

plt.show()

