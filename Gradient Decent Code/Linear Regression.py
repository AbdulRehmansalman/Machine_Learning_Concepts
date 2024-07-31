import numpy as np
import math
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0
        self.b = 0

    def compute_cost(self, x, y):
        n = len(x)
        y_pred = self.m * x + self.b
        cost = (1 / n) * sum((y - y_pred) ** 2)
        return cost

    def fit(self, x, y):
        n = len(x)
        cost_previous = 0

        for i in range(self.iterations):
            y_pred = self.m * x + self.b
            cost = self.compute_cost(x, y)

            # Partial derivatives
            md = -(2 / n) * sum(x * (y - y_pred))
            bd = -(2 / n) * sum(y - y_pred)

            # Update parameters
            self.m = self.m - self.learning_rate * md
            self.b = self.b - self.learning_rate * bd

            # Check for convergence
            if math.isclose(cost, cost_previous, rel_tol=1e-10):
                break

            cost_previous = cost
            print(f"Iteration {i + 1}: m = {self.m}, b = {self.b}, cost = {cost}")

    def predict(self, x):
        return self.m * x + self.b

# Sample data
x = np.array([92, 92, 88, 80, 80, 49, 65, 35, 66, 67])
y = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])

# Create and train the model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(x, y)

# Make predictions
predictions = model.predict(x)
print(predictions)

# Output the optimized parameters
print(f"\nOptimized parameters: m = {model.m}, b = {model.b}")

# Plot the results
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, predictions, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
