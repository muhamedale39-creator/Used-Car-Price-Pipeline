import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

df["price"] = df["price"].replace(r'\D', '', regex=True).astype(int)
df['milage'] = df['milage'].replace(r'\D', "", regex=True).astype(int)

df['hp'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
df['liters'] = df['engine'].str.extract(r'(\d+\.?\d*)(?:L| Liter)').astype(float)
df['cylinders'] = df['engine'].str.extract(r'(V\d+|I\d+|Flat \d+)')
df = df.drop(columns=['engine'])

df['hp'] = df['hp'].fillna(df['hp'].median())
df['liters'] = df['liters'].fillna(df['liters'].median())
df['cylinders'] = df['cylinders'].fillna(df['cylinders'].mode()[0])



valid_fuels = ['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Diesel', 'Plug-In Hybrid', 'Electric']
df['fuel_type'] = df['fuel_type'].apply(lambda x: x if pd.notna(x) and x in valid_fuels else np.nan)
df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])

df['accident'] = df['accident'].fillna('None reported')
df['accident'] = df['accident'].apply(lambda x: 1 if 'At least 1 accident' in str(x) else 0)

df['clean_title'] = df['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)
print(df.isna().sum())