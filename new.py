import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load Dataset
data_path = r'C:\Users\Jitendra\PycharmProjects\ML PROJECT\.venv\CAR DETAILS FROM CAR DEKHO.csv'  # Update with your file path
df = pd.read_csv(data_path)
print("Dataset Loaded Successfully")

# Step 2: Data Cleaning
print("Missing Values:\n", df.isnull().sum())
df.drop_duplicates(inplace=True)

# Step 3: Encode Categorical Variables
categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# Step 4: Feature Selection
X = df.drop(columns=['selling_price'])  # Input Features
y = df['selling_price']  # Target Variable

# Step 5: Standardize Data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Step 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Machine Learning Models
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Step 8: Evaluate Models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {model.__class__.__name__}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}\n")

# Evaluate Linear Regression
evaluate_model(lr_model, X_test, y_test)

# Evaluate Random Forest Regressor
evaluate_model(rf_model, X_test, y_test)

# Step 9: Save the Best Model
joblib.dump(rf_model, "car_price_prediction_model.pkl")
print("Model Saved Successfully")

# Step 10: Predict for New Data (Handle unseen labels)

# Load Sample Data
sample_data_path = r'C:\Users\Jitendra\PycharmProjects\ML PROJECT\.venv\CAR DETAILS FROM CAR DEKHO sample.csv'  # Update with your file path
sample_data = pd.read_csv(sample_data_path)

# Load Encoders used during training
encoders = {}
for col in categorical_columns:
    encoders[col] = LabelEncoder()
    encoders[col].fit(df[col])  # Fit on the training data

# Function to handle unseen labels in test data
def handle_unseen_labels(encoder, series):
    return series.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

# Apply encoding for categorical columns in the sample data
for col in categorical_columns:
    sample_data[col] = handle_unseen_labels(encoders[col], sample_data[col])

# Standardize the sample data
sample_data = scaler.transform(sample_data)

# Step 11: Predict Car Prices using the saved model
model = joblib.load("car_price_prediction_model.pkl")
predictions = model.predict(sample_data)
print("Predicted Prices for Sample Data:\n", predictions)
