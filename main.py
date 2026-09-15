import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

data["price"] = data["price"].astype(str).replace(r'[^\d.]', '', regex=True).astype(float).astype(int)
data['milage'] = data['milage'].astype(str).replace(r'[^\d.]', '', regex=True).astype(float).astype(int)

data['hp'] = data['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
data['liters'] = data['engine'].str.extract(r'(\d+\.?\d*)(?:L| Liter)').astype(float)
data = data.drop(columns=['engine'])

for col in ['hp', 'liters']:
    data[col] = data.groupby(['brand', 'model'])[col].transform(lambda x: x.fillna(x.median()))
    data[col] = data.groupby('brand')[col].transform(lambda x: x.fillna(x.median()))
    data[col] = data[col].fillna(data[col].median())

valid_fuels = ['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Diesel', 'Plug-In Hybrid', 'Electric']
data['fuel_type'] = data['fuel_type'].apply(lambda x: x if pd.notna(x) and x in valid_fuels else np.nan)
data['fuel_type'] = data['fuel_type'].fillna(data['fuel_type'].mode()[0])

data['accident'] = data['accident'].fillna('None reported')
data['accident'] = data['accident'].apply(lambda x: 1 if 'At least 1 accident' in str(x) else 0)

data['clean_title'] = data['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)

print(data.isna().sum())