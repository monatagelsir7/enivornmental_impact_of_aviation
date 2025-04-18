import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and data
model = joblib.load("../2.Models/random_forest_model.pkl")
X_train = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet"
)
expected_columns = X_train.columns.tolist()

# Configuration
N_SAMPLES = 3
RANDOM_STATE = 42
aircraft_classes = ["NB", "WB", "RJ", "TP", "PJ", "PP", "OTHER", "Unknown"]

# Prepare samples
sample_rows = X_train.sample(n=N_SAMPLES, random_state=RANDOM_STATE)


def prepare_input(raw_dict, expected_columns):
    """Create input DataFrame with exact expected columns"""
    df = pd.DataFrame(0, index=[0], columns=expected_columns)
    for col, val in raw_dict.items():
        if col in expected_columns:
            df[col] = val
    return df


# Run experiments
results = []
for idx, row in sample_rows.iterrows():
    try:
        original_class = next(
            c.replace("acft_class_", "")
            for c in expected_columns
            if c.startswith("acft_class_") and row[c] == 1
        )
    except StopIteration:
        print(f"No aircraft class found for sample {idx}")
        continue

    for new_class in aircraft_classes:
        modified_row = row.copy()
        # Reset all aircraft class flags
        for col in expected_columns:
            if col.startswith("acft_class_"):
                modified_row[col] = 0
        # Set new class
        modified_row[f"acft_class_{new_class}"] = 1

        # Predict
        try:
            input_df = prepare_input(modified_row, expected_columns)
            pred = model.predict(input_df)[0]
            results.append(
                {
                    "sample_id": idx,
                    "original_class": original_class,
                    "new_class": new_class,
                    "predicted_co2_km": float(pred),  # Ensure numeric value
                    "distance": float(row["ask"]),  # Ensure numeric value
                }
            )
        except Exception as e:
            print(f"Error testing {original_class}→{new_class}: {str(e)}")
            print("Problematic features:")
            print(modified_row[modified_row != 0])  # Show non-zero features
            continue

# Convert results to DataFrame
if not results:
    raise ValueError("No successful predictions - check error messages above")

df_results = pd.DataFrame(results)

# Data validation
print("\nData Validation:")
print(f"Total samples: {len(df_results)}")
print("Unique classes in results:", df_results["new_class"].unique())
print("Data types:\n", df_results.dtypes)

# Filter to only include classes with data
valid_classes = [c for c in aircraft_classes if c in df_results["new_class"].unique()]

# Visualization 1: Class Comparison
plt.figure(figsize=(14, 6))
if len(valid_classes) > 0:
    sns.boxplot(
        data=df_results, x="new_class", y="predicted_co2_km", order=valid_classes
    )
    plt.title("CO₂/km Distribution by Aircraft Class")
    plt.xlabel("Aircraft Class")
    plt.ylabel("kg CO₂ per km")
    plt.tight_layout()
else:
    print("Warning: No valid classes for boxplot")

# Visualization 2: Change Analysis
plt.figure(figsize=(14, 6))
if len(df_results["original_class"].unique()) > 0:
    for orig_class in sorted(df_results["original_class"].unique()):
        subset = df_results[df_results["original_class"] == orig_class]
        if len(subset) > 0:
            sns.lineplot(
                data=subset,
                x="new_class",
                y="predicted_co2_km",
                label=f"Originally {orig_class}",
                marker="o",
                sort=False,
                ci=None,
            )
    plt.title("Impact of Changing Aircraft Class")
    plt.xlabel("New Aircraft Class")
    plt.ylabel("kg CO₂ per km")
    plt.legend(title="Original Class")
    plt.tight_layout()
else:
    print("Warning: No data for change analysis plot")

plt.show()

# Additional debug output
print("\nSample predictions:")
print(
    df_results.groupby(["original_class", "new_class"])["predicted_co2_km"]
    .mean()
    .unstack()
)
