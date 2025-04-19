from statistics import linear_regression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.random import standard_t
# import scikitlearn as sc
from sklearn.preprocessing import LabelEncoder

data=r'C:\Users\Jitendra\PycharmProjects\ML PROJECT\.venv\CAR DETAILS FROM CAR DEKHO.csv'
df=pd.read_csv(data)
print(df)
print(df.isnull().sum())

print(df.duplicated().sum())
cleaned_dta = df.drop_duplicates(inplace=True)
print(cleaned_dta )
print(df.shape)

# if null values df.dropna()
# df.fillna('na' , implace = 'true')
print(df.head())
print(df.shape)
print(df.describe())

# print(df.columns)
print(df.info)

# checking outliers
print(df.describe())
sns.boxplot(x=df["selling_price"])
plt.show()
sns.boxplot(x=df["km_driven"])
plt.show()

encoder = LabelEncoder()
df['name'] = encoder.fit_transform (df['name'])
print(df['name'])

encoder = LabelEncoder()
df['fuel'] = encoder.fit_transform (df['fuel'])
print(df['fuel'])

encoder = LabelEncoder()
df['seller_type'] = encoder.fit_transform(df['seller_type'])
print(df['seller_type'])

encoder = LabelEncoder()
df['transmission']= encoder.fit_transform(df['transmission'])
print(df['transmission'])

encoder = LabelEncoder()
df['owner']= encoder.fit_transform(df['owner'])
print(df['owner'])

print(df.head())

input_data = df.iloc[:,:-1]
output_data = df['selling_price']

from sklearn.preprocessing import StandardScaler

Standardization = StandardScaler()
input_data=pd.DataFrame(Standardization.fit_transform(input_data),columns=input_data.columns)
print(input_data)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(input_data,output_data,test_size=0.2)
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train, Y_train)

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, Y_train)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, Y_train)

# Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, Y_train)

# K-Nearest Neighbors Regressor
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, Y_train)


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predictions
lr_pred = lr_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# Linear Regression Evaluation
lr_mse = mean_squared_error(Y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(Y_test, lr_pred)
print(f"Linear Regression Evaluation:")
print(f"MSE: {lr_mse}")
print(f"RMSE: {lr_rmse}")
print(f"R²: {lr_r2}\n")

# Lasso Regression Evaluation
# lasso_mse = mean_squared_error(Y_test, lasso_pred)
# lasso_rmse = np.sqrt(lasso_mse)
# lasso_r2 = r2_score(Y_test, lasso_pred)
# print(f"Lasso Regression Evaluation:")
# print(f"MSE: {lasso_mse}")
# print(f"RMSE: {lasso_rmse}")
# print(f"R²: {lasso_r2}\n")

# Ridge Regression Evaluation
# ridge_mse = mean_squared_error(Y_test, ridge_pred)
# ridge_rmse = np.sqrt(ridge_mse)
# ridge_r2 = r2_score(Y_test, ridge_pred)
# print(f"Ridge Regression Evaluation:")
# print(f"MSE: {ridge_mse}")
# print(f"RMSE: {ridge_rmse}")
# print(f"R²: {ridge_r2}\n")

# Decision Tree Regressor Evaluation
# dt_mse = mean_squared_error(Y_test, dt_pred)
# dt_rmse = np.sqrt(dt_mse)
# dt_r2 = r2_score(Y_test, dt_pred)
# print(f"Decision Tree Regressor Evaluation:")
# print(f"MSE: {dt_mse}")
# print(f"RMSE: {dt_rmse}")
# print(f"R²: {dt_r2}\n")

# Random Forest Regressor Evaluation
# rf_mse = mean_squared_error(Y_test, rf_pred)
# rf_rmse = np.sqrt(rf_mse)
# rf_r2 = r2_score(Y_test, rf_pred)
# print(f"Random Forest Regressor Evaluation:")
# print(f"MSE: {rf_mse}")
# print(f"RMSE: {rf_rmse}")
# print(f"R²: {rf_r2}\n")

# K-Nearest Neighbors Evaluation
# knn_mse = mean_squared_error(Y_test, knn_pred)
# knn_rmse = np.sqrt(knn_mse)
# knn_r2 = r2_score(Y_test, knn_pred)
# print(f"K-Nearest Neighbors Evaluation:")
# print(f"MSE: {knn_mse}")
# print(f"RMSE: {knn_rmse}")
# print(f"R²: {knn_r2}\n")


print(lr_model.coef_)
print(lr_model.intercept_)


# visualization of predicted vs actual value
plt.scatter(Y_test,lr_pred)
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.title('actual vs predicted')
print(plt.show())



# checking model performance by making difference group
from sklearn.model_selection import cross_val_score
cv_score=cross_val_score(lr_model,input_data,output_data,cv=10,scoring="r2")
print("cross validated r2 score",cv_score)
print("mean cross validated r2", cv_score.mean())

# deployment preparation
import os
print(os.getcwd())
import joblib
joblib.dump(lr_model,"linear_regression_model.pkl")


sample_data = pd.read_csv(r'C:\Users\Jitendra\PycharmProjects\ML PROJECT\.venv\CAR DETAILS FROM CAR DEKHO sample.csv' )
print(sample_data)

