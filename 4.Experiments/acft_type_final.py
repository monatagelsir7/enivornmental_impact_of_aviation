import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from warnings import filterwarnings

filterwarnings("ignore")  # Suppress sklearn warnings about feature names


class AircraftClassExperiment:
    def __init__(self):
        # Configuration
        self.N_SAMPLES = 3
        self.RANDOM_STATE = 42
        self.aircraft_classes = ["NB", "WB", "RJ", "TP", "PJ", "PP", "OTHER", "Unknown"]

        # output directory
        self.output_dir = r"C:\Users\DELL\ML Project\enivornmental_impact_of_aviation\4.Experiments\plotting"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load resources
        self.model = joblib.load(r"C:\Users\DELL\Downloads\random_forest_model.pkl")
        self.X_train = pd.read_parquet(
            "https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/Test-Train-Validation%20Data/X_train.parquet"
        )

        # Get expected features
        self.expected_columns = getattr(
            self.model, "feature_names_in_", self.X_train.columns.tolist()
        )
        self.aircraft_class_cols = [
            c for c in self.expected_columns if c.startswith("acft_class_")
        ]
        self.icao_cols = [
            c for c in self.expected_columns if c.startswith("acft_icao_")
        ]

        # Prepare sample data
        self.sample_data = self.X_train[
            self.X_train.columns.intersection(self.expected_columns)
        ]
        self.sample_rows = self.sample_data.sample(
            n=self.N_SAMPLES, random_state=self.RANDOM_STATE
        )

    def prepare_input(self, input_data):
        """Ensure input has exactly the features the model expects"""
        input_df = pd.DataFrame(0, index=[0], columns=self.expected_columns)
        for col in input_data.index:
            if col in self.expected_columns:
                input_df[col] = input_data[col]
        return input_df

    def get_aircraft_class(self, row):
        """Identify the aircraft class from one-hot encoded columns"""
        for col in self.aircraft_class_cols:
            if row[col] == 1:
                return col.replace("acft_class_", "")
        return "Unknown"

    def run_experiments(self):
        """Run all aircraft class change scenarios"""
        results = []

        for idx, row in self.sample_rows.iterrows():
            original_class = self.get_aircraft_class(row)

            for new_class in self.aircraft_classes:
                modified_row = row.copy()
                modified_row[self.aircraft_class_cols] = 0
                modified_row[f"acft_class_{new_class}"] = 1

                try:
                    input_df = self.prepare_input(modified_row)
                    pred = self.model.predict(input_df)[0]

                    results.append(
                        {
                            "sample_id": idx,
                            "original_class": original_class,
                            "new_class": new_class,
                            "predicted_co2_km": float(pred),
                            "distance": float(row.get("ask", 0)),
                            "fuel_burn": float(row.get("fuel_burn", 0)),
                        }
                    )
                except Exception as e:
                    print(f"Error testing {original_class}→{new_class}: {str(e)}")
                    continue

        return pd.DataFrame(results)

    def save_plot(self, fig, filename, dpi=300):
        """Save plot to the specified directory"""
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Graph saved to: {save_path}")
        plt.close(fig)

    def visualize_results(self, df_results):
        """Create and save visualization plots"""
        if len(df_results) == 0:
            print("No results to visualize")
            return

        # Plot 1: Boxplot of CO2 by aircraft class
        plt.figure(figsize=(14, 6))
        sns.boxplot(
            data=df_results,
            x="new_class",
            y="predicted_co2_km",
            order=self.aircraft_classes,
        )
        plt.title("CO₂/km Distribution by Aircraft Class")
        plt.xlabel("Aircraft Class")
        plt.ylabel("kg CO₂ per km")
        plt.tight_layout()
        self.save_plot(plt.gcf(), "co2_by_aircraft_class.png")

        # Plot 2: Impact of changing aircraft class
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
        self.save_plot(plt.gcf(), "aircraft_class_impact.png")

    def save_results_csv(self, df_results):
        """Save results to CSV file in the specified directory"""
        csv_path = os.path.join(self.output_dir, "experiment_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

    def run(self):
        """Execute full experiment pipeline"""
        print(f"\nStarting experiment - graphs will be saved to:\n{self.output_dir}")

        results = self.run_experiments()

        if len(results) > 0:
            print("\nExperiment completed successfully!")
            print("Summary statistics:")
            print(
                results.groupby(["original_class", "new_class"])["predicted_co2_km"]
                .mean()
                .unstack()
            )

            self.visualize_results(results)
            self.save_results_csv(results)
        else:
            print("\nExperiment failed - no valid results generated")

        return results


# Execute the experiment
if __name__ == "__main__":
    experiment = AircraftClassExperiment()
    results_df = experiment.run()
