import numpy as np
import math


def gradient_descent(x, y):
    m_curr = b_curr = 0  # Initial parameters
    learning_rate = 0.008  # Learning rate
    n = len(x)  # Number of data points
    iterations = 1000  # Number of iterations for gradient descent
    cost_previous = 0  # To store the previous cost

    for i in range(iterations):
        y_pred = m_curr * x + b_curr  # Predicted values
        cost = (1 / n) * sum((y - y_pred) ** 2)  # Mean squared error

        # Partial derivatives
        md = -(2 / n) * sum(x * (y - y_pred))
        bd = -(2 / n) * sum(y - y_pred)

        # Update parameters
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        # Check for convergence
        if math.isclose(cost, cost_previous, rel_tol=1e-9):
            break
        cost_previous = cost
        print(f"Iteration {i + 1}: m = {m_curr}, b = {b_curr}, cost = {cost}")

    return m_curr, b_curr


# Sample data
x = np.array([92, 92, 88, 80, 80, 49, 65, 35, 66, 67])
y = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])

# Perform gradient descent
m_opt, b_opt = gradient_descent(x, y)
print(f"\nOptimized parameters: m = {m_opt}, b = {b_opt}")

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m_opt * x + b_opt, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
