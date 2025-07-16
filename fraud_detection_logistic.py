
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("fraud-data.csv")

# Display first few rows
print(df.head())

# Feature Scaling (Min-Max Normalization)
def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

df[['Transaction_Amount', 'Transaction_Time', 'Account_Age']] = df[['Transaction_Amount', 'Transaction_Time', 'Account_Age']].apply(normalize)

# Extract features (X) and target (y)
X = df[['Transaction_Amount', 'Transaction_Time', 'Account_Age']].values
y = df['Fraudulent'].values

# Add Bias Term (X0 = 1)
X = np.c_[np.ones(X.shape[0]), X]  # Adding ones column for bias

# Sigmoid Activation Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hypothesis function (Logistic Model)
def predict_proba(X, theta):
    return sigmoid(np.dot(X, theta))

# Cost Function (Log Loss)
def compute_cost(X, y, theta):
    m = len(y)
    h = predict_proba(X, theta)
    return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient Descent Algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = predict_proba(X, theta)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient  # Update theta
        cost_history.append(compute_cost(X, y, theta))

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost_history[-1]}")

    return theta, cost_history

# Initialize Parameters
theta = np.zeros(X.shape[1])  # Initialize weights with zeros
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Train the model
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

plt.plot(range(len(cost_history)), cost_history, color='blue')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()

# Example: Predict if a new transaction is fraudulent
new_transaction = np.array([1, 0.8, 0.5, 0.6])  # Bias term + Normalized Features
fraud_probability = predict_proba(new_transaction, theta)
print("Fraud Probability:", fraud_probability)

# Convert probability to class label (0 or 1)
prediction = 1 if fraud_probability >= 0.5 else 0
print("Predicted Class:", prediction)

# Convert probabilities to class labels
y_pred = predict_proba(X, theta) >= 0.5
y_pred = y_pred.astype(int)

# Accuracy
accuracy = np.mean(y_pred == y)
print("Model Accuracy:", accuracy)

# Precision and Recall
true_positive = np.sum((y_pred == 1) & (y == 1))
false_positive = np.sum((y_pred == 1) & (y == 0))
false_negative = np.sum((y_pred == 0) & (y == 1))

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print("Precision:", precision)
print("Recall:", recall)
