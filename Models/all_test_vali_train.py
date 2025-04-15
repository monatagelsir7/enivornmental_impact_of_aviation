import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/cleaned_aviation_data_v3.parquet")

# Define features and target
X = df[['acft_class', 'seats', 'n_flights', 'departure_country', 'departure_continent',
        'arrival_country', 'arrival_continent', 'domestic', 'ask', 'rpk', 'fuel_burn']]
y = df['co2_per_distance']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train + validation into separate train and validation sets (now 64% train, 16% val)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

X_train.to_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet")
X_val.to_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_val.parquet")
X_test.to_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_test.parquet")
y_train.to_frame().to_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_train.parquet")
y_val.to_frame().to_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_val.parquet")
y_test.to_frame().to_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/y_test.parquet")