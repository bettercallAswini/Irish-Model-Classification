from flask import Flask, render_template, request
import numpy as np
import joblib
import plotly.graph_objs as go
import plotly.express as px

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_classification_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve inputs from form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction
    prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))

    # Determine the predicted class
    if prediction == 0:
        predicted_class = 'Setosa'
    elif prediction == 1:
        predicted_class = 'Versicolor'
    else:
        predicted_class = 'Virginica'

    # Create a Pie chart for predicted probabilities
    probabilities = model.predict_proba(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))[0]
    labels = ['Setosa', 'Versicolor', 'Virginica']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    pie_chart = go.Figure(data=[go.Pie(labels=labels, values=probabilities, marker_colors=colors)])
    pie_chart.update_layout(title='Predicted Probabilities')

    # Create a Scatter plot for feature comparison
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    scatter_data = []
    for i in range(4):
        scatter_data.append(go.Scatter(x=[feature_names[i]], y=features[0][i], mode='markers', name=feature_names[i]))

    scatter_plot = go.Figure(data=scatter_data)
    scatter_plot.update_layout(title='Input Features Comparison', xaxis_title='Feature', yaxis_title='Value')

    # Convert figures to HTML
    pie_chart_html = pie_chart.to_html(full_html=False, default_height=400, default_width=500)
    scatter_plot_html = scatter_plot.to_html(full_html=False, default_height=400, default_width=500)

    return render_template('result.html', prediction=predicted_class, pie_chart=pie_chart_html, scatter_plot=scatter_plot_html)

if __name__ == '__main__':
    app.run(debug=True)
