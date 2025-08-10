import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

# Example: Load your dataset
df = pd.read_csv("features.csv")  

# Select ONE attribute for regression
X = df[['f1']]   # Feature
y = df['f2']       # Target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create & Train Model
reg = LinearRegression().fit(X_train, y_train)

# Predictions
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Function to compute metrics
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# Train metrics
train_mse, train_rmse, train_mape, train_r2 = regression_metrics(y_train, y_train_pred)

# Test metrics
test_mse, test_rmse, test_mape, test_r2 = regression_metrics(y_test, y_test_pred)

# Display results
print(" Train Metrics")
print(f"MSE:  {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"MAPE: {train_mape:.4f}")
print(f"R²:   {train_r2:.4f}")

print("\nTest Metrics ")
print(f"MSE:  {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAPE: {test_mape:.4f}")
print(f"R²:   {test_r2:.4f}")
