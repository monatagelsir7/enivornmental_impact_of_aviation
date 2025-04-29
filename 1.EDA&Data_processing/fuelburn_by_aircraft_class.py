import pandas as pd

df = pd.read_parquet("../0.Data_after_cleaning/cleaned_aviation_data_v3.parquet")

df = df[
    (df['fuel_burn'] > 0) &
    (df['distance_km'] > 0) &
    (df['ask'] > 0) &
    (df['rpk'] > 0)
].copy()

# Fuel burn per km
df['fuel_burn_per_km'] = df['fuel_burn'] / df['distance_km']

# average fuel burn per km by aircraft class
fuel_burn_per_km_by_class = df.groupby('acft_class')['fuel_burn_per_km'].mean()

print(" Aircraft Class: Fuel Burn per km (kg/km):")
print(fuel_burn_per_km_by_class)




# experiment case for aircraft class (RDU→ORD)
route_distance_km = 1039.95

# expected fuel burn by aircraft class
expected_fuel_burn_by_class = fuel_burn_per_km_by_class * route_distance_km

print("Fuel Burn estimation for RDU→ORD:")
print(expected_fuel_burn_by_class)
