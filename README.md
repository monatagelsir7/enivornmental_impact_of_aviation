# enivornmental_impact_of_aviation


ORIGINAL REPO LINK: https://github.com/AeroMAPS/AeroSCOPE/tree/main



FOR EDA:
EDA -> Create cleaned_aviation_data_v1
additional_eda -> Create cleaned_aviation_data_v2
extra_works -> Create cleaned_aviation_data_v3

FOR DATA SPLIT:
run all_test_vali_train -> Create Test-Train_Validation Data (6 parquet files in folder)

FOR MODELS:
run all files in Models

DOCUMENTATION : 
https://docs.google.com/document/d/1iT7NYoRJLByU2Qa7XF5TS5aoM-Fwoir1jlD7ZWd7GKI/edit?tab=t.neu634jcgns0#heading=h.e0yki8jznmcx

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
│  ├─ 1.random_forest.py
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