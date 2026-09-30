import numpy as np
import pandas as pd 
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split , GridSearchCV  ,StratifiedKFold , KFold
from sklearn.ensemble import RandomForestRegressor ,VotingRegressor
from sklearn.metrics import mean_absolute_error,r2_score , mean_squared_error ,  root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer

df = pd.read_csv("cleaned_cars_data.csv")

text_columns = ['brand', 'model', 'fuel_type', "transmission", 'ext_col', 'int_col']

X = df.drop(columns = ['price'])
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
cv_splitter = KFold(n_splits=5,random_state=1,shuffle=True)


transformer  = ColumnTransformer(transformers=[
    ('Target' , TargetEncoder(target_type='continuous') , text_columns)
] , remainder='passthrough')

X_train_processed = transformer.fit_transform(X_train ,y_train)
X_test_processed = transformer.transform(X_test)

model = RandomForestRegressor(random_state=1)

param_grid = {
    'n_estimators' :[100,150,200],
    'max_depth' : [5,7,10,15]
}

grid = GridSearchCV(estimator=model , cv=cv_splitter ,scoring='r2' , param_grid=param_grid)
grid.fit(X_train_processed,y_train)

best_model_R = grid.best_estimator_

predictions = best_model_R.predict(X_test_processed)

model_xgb = XGBRegressor(random_state=42)

param_grid_x = {
    'n_estimators' : [150,200,300,],
    'learning_rate' : [0.1,0.4,0.6],
    'max_depth' : [4,6,10,15]
}

grid_x = GridSearchCV(estimator=model_xgb , cv=cv_splitter , scoring='r2' , param_grid=param_grid_x)
grid_x.fit(X_train_processed,y_train)

best_model_X = grid_x.best_estimator_

y_pred_xgb = best_model_X.predict(X_test_processed)

vote = VotingRegressor(estimators=[('Random' , best_model_R) , ('XGBoost' , best_model_X)])

vote.fit(X_train_processed,y_train)
predictions_ensemble = vote.predict(X_test_processed)

print("\n=== Combined Ensemble Performance ===")
mae_ens = mean_absolute_error(y_test, predictions_ensemble)
r2_ens = r2_score(y_test, predictions_ensemble)
rmse_ens = np.sqrt(mean_squared_error(y_test, predictions_ensemble))

print(f" R² Accuracy Score             : {r2_ens:.4f} ({r2_ens*100:.1f}% of data variance explained)")
print(f" Mean Absolute Error (MAE)     : ${mae_ens:.2f}")
print(f" Root Mean Squared Error (RMSE): ${rmse_ens:.2f}")

