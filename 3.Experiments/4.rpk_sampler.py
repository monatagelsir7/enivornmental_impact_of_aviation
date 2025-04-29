import pandas as pd
import numpy as np

df = pd.read_parquet("../0.Data_after_cleaning/cleaned_aviation_data_v3.parquet")
X_train = pd.read_parquet("../Test-Train-Validation Data/X_train.parquet")
y_train = pd.read_parquet("../Test-Train-Validation Data/y_train.parquet")
y_train.describe()
X_train[['ask', 'rpk', 'fuel_burn']].describe() ## Check the data distribution to avoid corner case

# Check the distribution of acft_class
df_acft_class_count = df['acft_class'].value_counts().reset_index()
df_acft_class_count.columns = ['acft_class', 'count']
print(df_acft_class_count)

'''
acft_class   count
0         NB  143804 !!! Pick this one
1         RJ   44404
2    Unknown   37080
3         WB   26456
4         TP   14904
5         PJ   13876
6         PP    5220
7         HE     238
8      OTHER      94
'''

# Check the distribution of domestic and international flights
nb_only = df[df['acft_class'] == 'NB']
median_values =nb_only[['ask', 'rpk', 'fuel_burn']].median()
print(median_values)

'''
ask          2.118280e+06
rpk          1.745463e+06
fuel_burn    5.092985e+04
'''

# base_input referring to the median of the NB aircraft class
target = {
    'ask': median_values['ask'],
    'rpk': median_values['rpk'],
    'fuel_burn': median_values['fuel_burn'],
}
print(target)

# The most frequent airline
airline_counts = nb_only['airline_iata'].value_counts()
print(airline_counts)

# Filter
allowed_airlines = ['DL', 'UA', 'AA', 'WN', 'FR']
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