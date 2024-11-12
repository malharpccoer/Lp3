"Implement K-Nearest Neighbors algorithm on diabetes.cs dataset. Compute confusion matrix,
accuracy, error rate, precision and recall on the given dataset."


import numpy as np
import matplotlib.pyplot as plt
def gradient_descent(learning_rate, max_iterations, initial_x):
x = initial_x
x_history = [] 
for_in range(max_iterations):
gradient = 2 * (x + 3) 
x = x - learning_rate * gradient # Update x using the gradient and␣
↪learning rate
x_history.append(x) # Append the current x to the history list
return x, x_history
# Parameters for Gradient Descent
learning_rate = 0.1
max_iterations = 1000
initial_x = 2
# Run Gradient Descent to find the local minimum
local_minimum, x_history = gradient_descent(learning_rate, max_iterations,initial_x)
print(f"Local Minimum at x = {local_minimum}")
# Plot the graph to visualize the convergence
x_values = np.linspace(-10, 10, 400) # Generate x values for the graph
y_values = (x_values + 3)**2 # Calculate corresponding y values
plt.plot(x_values, y_values, label='y = (x + 3)^2', color='blue')
plt.scatter(x_history, [(x + 3)**2 for x in x_history], label='Gradient DescentPath', color='red', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gradient Descent Convergence')
plt.grid(True)

