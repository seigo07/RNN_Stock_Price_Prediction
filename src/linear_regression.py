import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Input feature
y = np.array([2, 4, 6, 8, 10])               # Target variable

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict using the trained model
x_test = np.array([6, 7, 8]).reshape(-1, 1)  # New input data
y_pred = model.predict(x_test)

print("Predicted values:", y_pred)
