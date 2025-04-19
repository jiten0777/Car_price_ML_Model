from statistics import linear_regression
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL.GimpGradientFile import linear
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
encoder = LabelEncoder()



data = pd.read_csv(r'C:\Users\Jitendra\PycharmProjects\ML PROJECT\.venv\Cardetails.csv')
print(data)
print(data.head(5))
print(data.duplicated().sum())
print(data.isnull().sum())
# print(data.drop('mileage','engine','max_power','torque','seats',axis=1,inplace=False))
# Assuming 'data' is a pandas DataFrame
data = data.drop_duplicates()
print(data)
data.drop(['mileage', 'engine', 'max_power', 'torque', 'seats'], axis=1, inplace=True)
print(data.isnull().sum())
print(data)


encoder = LabelEncoder()
data['name'] = encoder.fit_transform(data['name'])
data['fuel'] = encoder.fit_transform(data['fuel'])
data['seller_type'] = encoder.fit_transform(data['seller_type'])
data['transmission'] = encoder.fit_transform(data['transmission'])
data['owner'] = encoder.fit_transform(data['owner'])
print(data[['name', 'fuel', 'seller_type', 'transmission', 'owner']])
print(data.info())

input_data=data.iloc[:,:-1]
output_data=data['selling_price']
from sklearn.preprocessing import StandardScaler
standardization=StandardScaler()
input_data=pd.DataFrame(standardization.fit_transform(input_data),columns=input_data.columns)
print(input_data)

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression, ridge_regression, Lasso
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# linear_regression()
lr_model=LinearRegression()
lr_model.fit(x_train,y_train)

# lasss_regression
lasso_model=Lasso()
lasso_model.fit(x_train,y_train)

# ridge_regression
ridge_regression_model=ridge_regression()
ridge_regression_model.fit(x_train,y_train)

# decision_tree_regression tree
decision_tree_model=decision_tree_regression()
decision_tree_model.fit(x_train,y_train)

# random_forest_regression
RandomForestRegressor_model=RandomForestRegressor()
RandomForestRegressor_model.fit(x_train,y_train)

# KNeighborsRegressor
KNeighborsRegressor_model=KNeighborsRegressor()
KNeighborsRegressor_model.fit(x_train,y_train)

from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

# prediction
lr_prediction=lr_model.predict(x_test)
lasso_prediction=lasso_model.predict(x_test)
ridge_prediction=ridge_regression.predict(x_test)
decision_tree_prediction=DecisionTree.predict(x_test)
RandomForest_prediction_=RandomForest_model.predict(x_test)
KNeighbors_prediction=KNeighborsRegressor.predict(x_test)

# evaluation
# linear_regression()
lr_mse=mean_squared_error(y_test,lr_prediction)
lr_rmse=np.sqrt(lr_mse)
lr_r2=r2_score(y_test,lr_prediction)
print("linear regression evaluation")
print(f"mse: {lr_mse}")
print(f"rmse: {lr_rmse}")
print(f"r2: {lr_r2}")

# lasso regression
lasso_mse=mean_squared_error(y_test,lasso_prediction)
lasso_rmse=np.sqrt(lasso_mse)
lasso_r2=r2_score(y_test,lasso_prediction)
print("lasso regression evaluation")
print(f"mse: {lasso_mse}")
print(f"rmse: {lasso_rmse}")
print(f"r2: {lasso_r2}")

# ridge regression evaluation
ridge_prediction_mse=mean_squared_error(y_test,ridge_prediction)
ridge_prediction_rmse=np.sqrt(ridge_mse)
ridge_prediction_r2=r2_score(y_test,ridge_prediction)
print("ridge prediction regression evaluation")
print(f"mse: {ridge_mse}")
print(f"rmse: {ridge_rmse}")
print(f"r2: {ridge_r2}")

# decision tree regression evaluation
decision_tree_prediction_mse=mean_squared_error(y_test,decision_tree_prediction)
decision_tree_prediction_rmse=np.sqrt(decision_tree_mse)
decision_tree_prediction_r2=r2_score(y_test,lasso_prediction)
print("decision tree prediction regression evaluation")
print(f"mse: {decision_tree_mse}")
print(f"rmse: {decision_tree_rmse}")
print(f"r2: {decision_tree_r2}")

# random forest
RandomForest_prediction_mse=mean_squared_error(y_test,RandomForest_prediction)
RandomForest_prediction_rmse=np.sqrt(RandomForest_prediction_mse)
RandomForest_prediction_r2=r2_score(y_test,RandomForest_prediction)
print(" RandomForest prediction evaluation")
print(f"mse: {random_forest_mse}")
print(f"rmse: {random_forest_rmse}")
print(f"r2: {random_forest_r2}")

# KNeighbors_prediction_evaluation
KNeighbors_prediction_mse=mean_squared_error(y_test,KNeighbors_prediction)
KNeighbors_prediction_rmse=np.sqrt(KNeighbors_prediction_mse)
KNeighbors_prediction_r2=r2_score(y_test,KNeighbors_prediction)
print(" KNeighbors  prediction evaluation")
print(f"mse: {KNeighbors_prediction_mse}")
print(f"rmse: {KNeighbors_prediction_rmse}")
print(f"r2: {KNeighbors_prediction_r2}")

print(lr_model.coef_)
print(lr_model.intercept)
print(lr_model.coef_)

# visualization of predicted vs actual value
plt.scatter(y_test,lr_prediction)












