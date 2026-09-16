import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os 

data = pd.read_csv("data.csv")

data["price"] = data["price"].astype(str).replace(r'[^\d.]','', regex=True).astype(float).astype(int)
data['milage'] = data['milage'].astype(str).replace(r'[^\d.]','', regex=True).astype(float).astype(int)

data['hp'] = data['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
data['liters'] = data['engine'].str.extract(r'(\d+\.?\d*)(?:L| Liter)').astype(float)
data = data.drop(columns=['engine'])

for col in ['hp', 'liters']:
    data[col] = data.groupby(['brand', 'model'])[col].transform(lambda x: x.fillna(x.median()))
    data[col] = data.groupby('brand')[col].transform(lambda x: x.fillna(x.median()))
    data[col] = data[col].fillna(data[col].median())

valid_fuels = ['Gasoline','Hybrid','E85 Flex Fuel','Diesel','Plug-In Hybrid','Electric']
data['fuel_type']= data['fuel_type'].apply(lambda x: x if pd.notna(x) and x in valid_fuels else np.nan)
data['fuel_type'] =data['fuel_type'].fillna(data['fuel_type'].mode()[0])

data['accident'] =data['accident'].fillna('None reported')
data['accident'] = data['accident'].apply(lambda x: 1 if "At least 1 accident" in str(x) else 0)

data['clean_title']= data['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)

print(data.isna().sum())

q1_price = data['price'].quantile(0.25)
q3_price = data['price'].quantile(0.75)
iqr_price = q3_price - q1_price
data['price'] = np.where(data['price'] > (q3_price +1.5 * iqr_price), q3_price +1.5 *iqr_price, data['price'])
data['price'] = np.where(data['price'] < (q1_price  - 1.5 * iqr_price), q1_price -1.5 * iqr_price, data['price'])

plt.figure(figsize=(8, 4))
sns.histplot(data['price'], kde=True)
plt.title("Car Prices Distribution")
plt.show()

plt.figure(figsize=(8, 4))
sns.scatterplot(x=data['milage'], y=data['price'], alpha=0.5)
plt.title("Mileage vs Price")
plt.show()

plt.figure(figsize=(10, 4))
data['brand'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title("Top 10 Car Brands")
plt.show()

data.to_csv('cleaned_cars_data.csv',index=False)
print("EDA Complete Clean dataset saved as cleaned_cars_data.csv")


if not os.path.exists('images'):
    os.makedirs('images')


plt.figure(figsize=(8, 4)) 
sns.histplot(data['price'], kde=True)
plt.title("Car Prices Distribution")
plt.savefig("images/price_distribution.png") 
plt.close() 


plt.figure(figsize=(8, 4))
sns.scatterplot(x=data['milage'], y=data['price'], alpha=0.5)
plt.title("Mileage vs Price")
plt.savefig("images/mileage_vs_price.png") 
plt.close() 


plt.figure(figsize=(10, 4))
data['brand'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title("Top 10 Car Brands")
plt.savefig("images/top_10_brands.png") 
plt.close() 


print(data["brand"].unique())


print(data["brand"].unique())