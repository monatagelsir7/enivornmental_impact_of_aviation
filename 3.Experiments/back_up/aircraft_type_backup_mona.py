import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and data
model = joblib.load("../2.Models/random_forest_model.pkl")
X_train = pd.read_parquet(
    "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet"
)

# ✅ Use actual trained features from the model
if hasattr(model, "feature_names_in_"):
    expected_columns = model.feature_names_in_.tolist()
else:
    expected_columns = X_train.columns.tolist()

# Configuration
N_SAMPLES = 3
RANDOM_STATE = 42
aircraft_classes = ["NB", "WB", "RJ", "TP", "PJ", "PP", "OTHER", "Unknown"]

# Sample selection
sample_rows = X_train.sample(n=N_SAMPLES, random_state=RANDOM_STATE)

# ✅ Safe input constructor with dtype handling
def prepare_input_safe(modified_row, expected_columns):
    """Safely construct input DataFrame with float-compatible dtypes"""
    input_df = pd.DataFrame(0.0, index=[0], columns=expected_columns)
    for col in expected_columns:
        if col in modified_row:
            val = modified_row[col]
            # Normalize types: bool → float, int/float → float
            if isinstance(val, bool):
                val = float(int(val))
            elif isinstance(val, (int, float)):
                val = float(val)
            input_df.at[0, col] = val
    return input_df

# Run experiments
results = []

for idx, row in sample_rows.iterrows():
    # Detect original aircraft class
    original_class = next(
        (c.replace("acft_class_", "") for c in expected_columns if c.startswith("acft_class_") and row.get(c, 0) == 1),
        "Unknown"
    )

    for new_class in aircraft_classes:
        modified_row = row.copy()

        # Reset all aircraft class one-hot flags
        for col in expected_columns:
            if col.startswith("acft_class_"):
                modified_row[col] = 0

        # Set new aircraft class flag
        class_col = f"acft_class_{new_class}"
        if class_col in expected_columns:
            modified_row[class_col] = 1
        else:
            continue  # Skip unknown class combinations

        # Prepare input safely
        input_df = prepare_input_safe(modified_row, expected_columns)

        # Predict
        try:
            pred = model.predict(input_df)[0]
            results.append(
                {
                    "sample_id": idx,
                    "original_class": original_class,
                    "new_class": new_class,
                    "predicted_co2_km": pred,
                    "distance": row["ask"],  # Preserve original feature
                }
            )
        except Exception as e:
            print(f"❌ Prediction failed: {original_class} → {new_class}")
            print("   Error:", str(e))

# Convert to DataFrame
df_results = pd.DataFrame(results)

# ✅ Safety check before plotting
if df_results.empty:
    print("⚠️ No results to visualize – check model and input feature compatibility.")
else:
    # Visualization 1: Boxplot
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=df_results,
        x="new_class",
        y="predicted_co2_km",
        order=aircraft_classes
    )
    plt.title("CO₂/km Distribution by Aircraft Class")
    plt.xlabel("Aircraft Class")
    plt.ylabel("kg CO₂ per km")
    plt.tight_layout()

    # Visualization 2: Lineplot
    plt.figure(figsize=(14, 6))
    for orig_class in df_results["original_class"].unique():
        subset = df_results[df_results["original_class"] == orig_class]
        sns.lineplot(
            data=subset,
            x="new_class",
            y="predicted_co2_km",
            label=f"Originally {orig_class}",
            marker="o",
            sort=False,
        )
    plt.title("Impact of Changing Aircraft Class")
    plt.xlabel("New Aircraft Class")
    plt.ylabel("kg CO₂ per km")
    plt.legend(title="Original Class")
    plt.tight_layout()

    plt.show()
