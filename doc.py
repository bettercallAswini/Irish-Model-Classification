import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Fetch the data from the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(url, names=names)

# Step 2: Explore the data
print(iris_data.head())  # Print the first few rows to understand the structure

# Step 3: Visualize the data
# Example: Scatter plot of sepal length vs sepal width
plt.figure(figsize=(8, 6))
plt.scatter(iris_data['sepal_length'], iris_data['sepal_width'], c='blue', marker='o', label='Sepal')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Example: Scatter plot of petal length vs petal width
plt.scatter(iris_data['petal_length'], iris_data['petal_width'], c='red', marker='x', label='Petal')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.title('Iris Dataset: Sepal and Petal Measurements')
plt.legend()
plt.grid(True)
plt.show()

