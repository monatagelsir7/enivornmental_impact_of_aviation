import pandas as pd

df=pd.read_parquet("https://github.com/monatagelsir7/enivornmental_impact_of_aviation/raw/refs/heads/main/cleaned_aviation_data_v1.parquet")
df["departure_continent"].value_counts()

# For Departures
a=df[["departure_continent","departure_country"]].drop_duplicates()
a["departure_country"].value_counts()

two_counts = a["departure_country"].value_counts()
indices_with_two = two_counts[two_counts == 2].index
print(indices_with_two)
a[a["departure_country"].isin(indices_with_two)].sort_values(by="departure_country")




# For Arrivals
b=df[["arrival_continent","arrival_country"]].drop_duplicates()
b["arrival_country"].value_counts()

two_counts = b["arrival_country"].value_counts()
indices_with_two = two_counts[two_counts == 2].index
print(indices_with_two)
b[b["arrival_country"].isin(indices_with_two)].sort_values(by="arrival_country")



updated_country_to_continent={"EG":"AF","ES":"EU","KZ":"AS","RU":"EU","TR":"AS","CO":"SA","US":"NA",}
df.loc[df["departure_country"].isin(updated_country_to_continent.keys()), "departure_continent"] = df["departure_country"].map(updated_country_to_continent)