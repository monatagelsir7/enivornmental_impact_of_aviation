import pandas as pd
import numpy as np


## CASE 1
df = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/0.Data_after_cleaning/cleaned_aviation_data_v3.parquet")

X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_train.parquet")
y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_train.parquet")
y_train.describe()
X_train[['ask', 'rpk', 'fuel_burn']].describe() ## Check the data distribution to avoid corner case

# base_input
target = {
    'ask': 1413504.5545,     # Median
    'rpk': 1164727.753,      # Median
    'fuel_burn': 40375.0506, # Median
}

# Filter
allowed_airlines = ['DL', 'AF', 'BA']
df_filtered = df[
    (df['domestic'] == 0) &
    (df['acft_class'] == 'NB') &
    (df['airline_iata'].isin(allowed_airlines))
].copy()

# Similarity
def compute_distance(row):
    return np.sqrt(
        (row['ask'] - target['ask'])**2 +
        (row['rpk'] - target['rpk'])**2 +
        (row['fuel_burn'] - target['fuel_burn'])**2
    )

# Similar Observations
df_filtered['distance'] = df_filtered[['ask', 'rpk', 'fuel_burn']].apply(compute_distance, axis=1)
nearest_row = df_filtered.sort_values(by='distance').iloc[0]
nearest_index = nearest_row.name  


print(f"Most similar real case index: {nearest_index}\n")
print(nearest_row[['airline_iata', 'acft_class', 'acft_icao',
                   'departure_country', 'arrival_country',
                   'iata_departure', 'iata_arrival',
                   'ask', 'rpk', 'fuel_burn']])


base_input_candidate = nearest_row[[
    'airline_iata', 'acft_class', 'acft_icao',
    'departure_country', 'departure_continent',
    'arrival_country', 'arrival_continent',
    'iata_departure', 'iata_arrival', 'domestic',
    'ask', 'rpk', 'fuel_burn'
]].to_dict()

print(base_input_candidate)


# checker = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/0.Data_after_cleaning/cleaned_aviation_data_v3.parquet")
# checker.head(5)


## CASE 2 'Delta Air Lines'

import pandas as pd
import numpy as np

df = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/0.Data_after_cleaning/cleaned_aviation_data_v3.parquet")

X_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/X_train.parquet")
y_train = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/Test-Train-Validation Data/y_train.parquet")
y_train.describe()
X_train[['ask', 'rpk', 'fuel_burn']].describe() ## Check the data distribution to avoid corner case

# base_input
target = {
    'ask': 1413504.5545,     # Median
    'rpk': 1164727.753,      # Median
    'fuel_burn': 40375.0506, # Median
}

# Filter
allowed_airlines = ['DL']
df_filtered = df[
    (df['domestic'] == 0) &
    (df['acft_class'] == 'NB') &
    (df['airline_iata'].isin(allowed_airlines))
].copy()

# Similarity
def compute_distance(row):
    return np.sqrt(
        (row['ask'] - target['ask'])**2 +
        (row['rpk'] - target['rpk'])**2 +
        (row['fuel_burn'] - target['fuel_burn'])**2
    )

# Similar Observations
df_filtered['distance'] = df_filtered[['ask', 'rpk', 'fuel_burn']].apply(compute_distance, axis=1)
nearest_row = df_filtered.sort_values(by='distance').iloc[0]
nearest_index = nearest_row.name  


print(f"Most similar real case index: {nearest_index}\n")
print(nearest_row[['airline_iata', 'acft_class', 'acft_icao',
                   'departure_country', 'arrival_country',
                   'iata_departure', 'iata_arrival',
                   'ask', 'rpk', 'fuel_burn']])


base_input_candidate = nearest_row[[
    'airline_iata', 'acft_class', 'acft_icao',
    'departure_country', 'departure_continent',
    'arrival_country', 'arrival_continent',
    'iata_departure', 'iata_arrival', 'domestic',
    'ask', 'rpk', 'fuel_burn'
]].to_dict()

print(base_input_candidate)