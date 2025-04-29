# ✈️ Predicting Aviation CO₂ Emissions with Machine Learning

## 📚 Project Summary
Air travel is critical for global mobility but contributes significantly to climate change.  
In this project, we predict flight-level CO₂ emissions per kilometer using machine learning models trained on the AeroSCOPE dataset.  
By analyzing emissions patterns across continents, aircraft types, and operational factors, we provide actionable insights for improving aviation sustainability.

## 📊 Problem Statement
- **Goal**: 
    - Build predictive models for CO₂ emissions intensity (g/km) at the flight level.
    - Predict CO₂ emissions per kilometer using flight-level features.
    - Compare model performance across Linear Regression, Random Forest, XGBoost, and CatBoost.
    - Analyze emissions by departure continent, aircraft type, and passenger load factor.
    - Test model generalizability between international and domestic flights.
    
- **Challenges**: Capture non-linear relationships, handle high-cardinality categorical data, and ensure model interpretability.
- **Impact**: 
    - Support policy recommendations with counterfactual scenarios.
    - Provide insights for airlines to optimize fleet and route planning.
    - Contribute to the understanding of aviation's environmental impact.

## 🗃️ Dataset
- **Source**: [AeroSCOPE Dataset (ICCT)](https://zenodo.org/records/10143773)
- **Size**: 200,000+ flight records from 2019.
- **Variables include**:
  - Departure & arrival country/continent
  - Aircraft class & ICAO code
  - Fuel burn, seat capacity, distance
  - Derived metrics like ASK, RPK, and CO₂ per distance

## 🧪 Experiments
| No. | Name | Description |
|-----|------|-------------|
| 1 | Model Comparison | Evaluated Linear Regression, Random Forest, XGBoost, CatBoost |
| 2 | Emissions by Continent | Analyzed CO₂/km differences across departure continents |
| 3 | Aircraft Type Analysis | Estimated emissions per km across 6 aircraft classes (e.g., NB, WB, RJ) |
| 4 | Load Factor Sensitivity | Quantified emissions variation across different passenger load factors |
| 5 | Generalization Check | Trained model only on international flights and tested on domestic ones |

## 📁 Repository Structure

```
enivornmental_impact_of_aviation-2
├─ 0.Data_after_cleaning
│  ├─ AeroSCOPE_global_aviation_traffic_dataset_16_11_0320.csv
│  ├─ cleaned_aviation_data_v1_INVALID.parquet
│  ├─ cleaned_aviation_data_v2_INVALID.parquet
│  ├─ cleaned_aviation_data_v3.parquet 
│  ├─ cleaned_aviation_data_with_outliers_v4_INVALID.parquet
│  └─ compressed_aviation_traffic_data.parquet
├─ 1.EDA&Data_processing
│  ├─ 1.EDA_for_v1.ipynb
│  ├─ 2.datacleaning_duplication_checker_for_v2.py
│  ├─ 3.datacleaning_co2_per_distance_converter_for_v3.ipynb
│  ├─ 4.all_test_vali_train_seperator.py
│  ├─ 4_continent_co2_analysis.png
│  ├─ acft_class.ipynb
│  ├─ continets.ipynb
│  ├─ domestic_international.py
│  ├─ fuelburn_by_aircraft_class.py
│  └─ others_frequent_flights_by_types.xlsx
├─ 2.Models
│  ├─ 0.linear_regression.py
│  ├─ 1.random_forest_baseline.py
│  ├─ 1.random_forest_finetuned.py
│  ├─ 2.xg_boost.py
│  ├─ 3.cat_boost.py
│  ├─ random_forest_model.pkl
├─ 3.Experiments
│  ├─ 1.Experiement_Continent.ipynb
│  ├─ 2.dom_vs_intl.py
│  ├─ 3.aircraft_type_RDU-ORD.py
│  ├─ 4.rpk_passenger_load_factor.py
│  ├─ 4.rpk_sampler.py
├─ Code_Dictionary.csv
└─ Test-Train-Validation Data
   ├─ X_test.parquet
   ├─ X_train.parquet
   ├─ X_val.parquet
   ├─ y_test.parquet
   ├─ y_train.parquet
   └─ y_val.parquet
```

## 🧠 Models Used
- **Linear Regression** (baseline model)
- **Random Forest** (best performer, R² = 0.89)
- **XGBoost**
- **CatBoost**

Evaluation metrics:
- R² (Coefficient of Determination)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

## 📊 Key Results
- **Random Forest** achieved highest test R²: **0.8953**.
- Emissions from **Asia** and **Africa** were significantly higher per kilometer than from Europe and North America.
- **Wide-body aircraft** showed better per-passenger emissions when operated at high capacity.
- Maintaining passenger **load factors above 60%** drastically improved emission efficiency.
- Model trained only on **international data** performed worse on domestic flights (R² ≈ 0.66).

## ⚖️ Ethical Considerations
- The AeroSCOPE dataset was anonymized and publicly available; no personal data was used.
- Regional biases due to underrepresentation of certain areas are acknowledged.
- Interpretability was prioritized: tree-based models (Random Forest, XGBoost, CatBoost) and feature importance analyses were used.
- No black-box models were deployed to ensure transparent and responsible analysis.

## 👥 Team & Contributions
- [Adil Gazder](https://github.com/ag826)
- [Ilseop Lee](https://github.com/ISL-0111)
- [Yirang Liu](https://github.com/cathylyirang)
- [Mona Saeed](https://github.com/monatagelsir7)
- [Afag Ramazanova](https://github.com/Afag-Ramazanova)


---



DOCUMENTATION : 
https://docs.google.com/document/d/1iT7NYoRJLByU2Qa7XF5TS5aoM-Fwoir1jlD7ZWd7GKI/edit?tab=t.neu634jcgns0#heading=h.e0yki8jznmcx
