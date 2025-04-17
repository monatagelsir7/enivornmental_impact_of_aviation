'''
## Counterfactual Simulation ##

Predictor

'''

import joblib

# Loading the Prediction Model
model = joblib.load('../2.Models/random_forest_model.pkl')


# Target Variable 


base_input = pd.DataFrame([{
    'airline_iata': 'AF',
    'acft_class': 'Narrow',
    'departure_country': 'France',
    'departure_continent': 'Europe',
    'arrival_country': 'Germany',
    'arrival_continent': 'Europe',
    'domestic': 0,
    'ask': 200000,
    'rpk': 180000,
    'fuel_burn': 12000,
    'iata_departure': 'CDG',
    'iata_arrival': 'FRA',
    'acft_icao': 'A320'
}])


fuel_burn_values = range(8000, 20001, 2000)

samples = []
for fb in fuel_burn_values:
    sample = base_input.copy()
    sample['fuel_burn'] = fb
    samples.append(sample)

experiment_df = pd.concat(samples, ignore_index=True)