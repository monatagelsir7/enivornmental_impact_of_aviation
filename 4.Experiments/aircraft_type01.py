import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("../2.Models/random_forest_model.pkl")
X_train = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet"
)
expected_columns = X_train.columns.tolist()

sample_rows = X_train.sample(n=3, random_state=42)
aircraft_list = [
    "PA32",
    "B712",
    "A320",
    "B737",
    "B752",
    "C402",
    "B739",
    "DH8D",
    "BN2P",
    "B738",
    "E75L",
    "A321",
    "E190",
    "C208",
    "MD87",
    "AT76",
    "DH8A",
    "LJ75",
    "A20N",
    "DHC6",
    "A319",
    "E170",
    "RJ1H",
    "A388",
    "E145",
    "CRJ2",
    "CRJ9",
    "C172",
    "E135",
    "CRJ7",
    "AT75",
    "CRJX",
    "MD90",
    "SF34",
    "B190",
    "B753",
    "PC12",
    "RJ85",
    "DHC3",
    "AT43",
    "PA31",
    "A318",
    "AC90",
    "E195",
    "CRJ1",
    "AT72",
    "B763",
    "B77W",
    "B772",
    "E75S",
    "DH8C",
    "B788",
    "A332",
    "B744",
    "AT45",
    "BCS3",
    "A333",
    "A346",
    "A359",
    "B736",
    "B764",
    "BCS1",
    "GA8",
    "B735",
    "A21N",
    "B789",
    "D328",
    "SB20",
    "DHC2",
    "B77L",
    "B748",
    "B78X",
    "B733",
    "B38M",
    "J328",
    "CL60",
    "DH8B",
    "A343",
    "B734",
    "B74R",
    "JS31",
    "SU95",
    "A342",
    "B762",
    "A35K",
    "A339",
    "B39M",
    "BE20",
    "BE99",
    "A306",
    "R44",
    "MD11",
    "C180",
    "F100",
    "A310",
    "B350",
    "C30J",
    "A400",
    "T37",
    "A139",
    "C56X",
    "BE35",
    "C212",
    "E45X",
    "H500",
    "AS50",
    "GLEX",
    "A345",
    "GL5T",
    "BE9L",
    "C441",
    "FA7X",
    "PA30",
    "CL30",
    "A119",
    "BE55",
    "F2TH",
    "P180",
    "F900",
    "LJ45",
    "GLF6",
    "DC10",
    "SW4",
    "H25B",
    "B732",
    "B722",
    "FA8X",
    "TBM9",
    "GA5C",
    "C500",
    "G150",
    "BE10",
    "LJ60",
    "MU2",
    "GALX",
    "GLF4",
    "C130",
    "KODI",
    "E120",
    "BE40",
    "LJ35",
    "TBM7",
    "DC93",
    "D228",
    "E35L",
    "FA5X",
    "DC91",
    "P46T",
    "MD88",
    "TBM8",
    "SW3",
    "BE30",
    "DH2T",
    "C425",
]


def prepare_input(raw_dict, expected_columns):
    df = pd.DataFrame([raw_dict])
    df_encoded = pd.get_dummies(df)

    # Ensure only expected columns are present
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Drop any columns not in expected_columns
    df_encoded = df_encoded[[col for col in expected_columns]]

    # Reorder columns exactly to match training
    df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)

    return df_encoded


results = []

for idx, row in sample_rows.iterrows():
    for acft in aircraft_list:
        modified_row = row.copy()
        modified_row["acft_icao"] = acft
        df_input = prepare_input(modified_row.to_dict(), expected_columns)

        try:
            pred = model.predict(df_input)[0]
            results.append(
                {"sample_id": idx, "aircraft": acft, "predicted_co2_km": pred}
            )
        except Exception as e:
            print(f"Skipping {acft} due to error: {e}")
