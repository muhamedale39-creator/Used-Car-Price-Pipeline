import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,r2_score , mean_squared_error ,  root_mean_squared_error

df = pd.read_csv("cleaned_cars_data.csv")

text_columns = ['brand', 'model', 'fuel_type', "transmission", 'ext_col', 'int_col']
saved_encoders = {}

for col in text_columns:
    le = LabelEncoder()

    df[col] = le.fit_transform(df[col]).astype(str)

    saved_encoders[col] = le

X = df.drop(columns = ['price'])
y = df['price']

X_train ,X_test, y_train,y_test = train_test_split(X,y, test_size=0.2 , random_state=1)

model = RandomForestClassifier()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test,predictions)

r2 = r2_score(y_test, predictions)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f" R² Accuracy Score : {r2:.4f} ({r2*100:.1f}% of data variance explained)")

print(f" Mean Absolute Error (MAE) : ${mae:.2f} (Average guess deviation)")
print(f" Root Mean Squared Error (RMSE): ${rmse:.2f} (Penalizes larger mistakes)")

