from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

app = Flask(__name__)
iris = load_iris()
X = iris.data[:, :2]  # Only take first two features for simplicity
y = iris.target

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']

    prediction = knn.predict([[sepal_length, sepal_width]])
    target_names = iris.target_names
    predicted_class = target_names[prediction[0]]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
