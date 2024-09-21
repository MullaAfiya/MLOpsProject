# # Import necessary libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Step 1: Load Data from CSV
# # Read the CSV file (ensure 'house_data.csv' is in the same directory or provide the full path)
# data = pd.read_csv('Data.csv')

# # Step 2: Data Preprocessing
# # Display the first few rows of the data to ensure it's loaded correctly
# print("First 5 rows of the dataset:")
# print(data.head())

# # Step 3: Train-Test Split
# # Split the data into features (X) and target (y)
# X = data[['Number_of_Rooms']]
# y = data['Price']

# # Split into training and test sets (80% training, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Model Training
# # Initialize the Linear Regression model
# model = LinearRegression()

# # Train the model using the training data
# model.fit(X_train, y_train)

# # Step 5: Model Testing (Prediction on Test Data)
# # Predict the house prices for the test set
# y_pred = model.predict(X_test)

# # Step 6: Model Evaluation
# # Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R²)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Print evaluation results
# print("\nModel Evaluation:")
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"R-squared (R²): {r2:.2f}")

# # Step 7: Model Visualization
# # Plot the actual vs predicted house prices
# plt.figure(figsize=(8, 6))
# plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
# plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
# plt.xlabel('Number of Rooms')
# plt.ylabel('House Price')
# plt.title('House Price Prediction: Actual vs Predicted')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Step 8: Summary Report
# print("\n### Model Summary ###")
# print(f"Intercept (b0): {model.intercept_:.2f}")
# print(f"Coefficient (b1): {model.coef_[0]:.2f}")
# print(f"The predicted equation is: Price = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Number_of_Rooms")

# # Step 9: Predictions on New Data
# new_data = np.array([[1.5], [3.0], [4.5]])  # Predict for 1.5, 3.0, and 4.5 rooms
# predicted_prices = model.predict(new_data)

# print("\nPredictions on New Data:")
# for i, price in enumerate(predicted_prices):
#     print(f"House with {new_data[i][0]} rooms -> Predicted Price: ${price:.2f}")






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the data from CSV
data = pd.read_csv('Data.csv')

# Assuming 'Number_of_Rooms' and 'Price' are the relevant columns
X = data[['Number_of_Rooms']]
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('house_price_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")