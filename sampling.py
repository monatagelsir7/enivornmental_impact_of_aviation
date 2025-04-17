import pandas as pd

sample = pd.read_parquet("/Users/ilseoplee/enivornmental_impact_of_aviation-2/0.Data_after_cleaning/cleaned_aviation_data_v3.parquet")  
sample.head(5)


sample_row = sample.iloc[43126]

# sample_row = sample[
#     (sample['airline_iata'] == 'DL') &
#     (sample['acft_icao'] == 'B752') &
#     (sample['seats'] == 20128.5)
# ]

print(sample_row)