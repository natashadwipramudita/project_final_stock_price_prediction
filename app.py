from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import base64
import os

app = Flask(__name__)

# Set the upload folder to the same directory as the script
app.config['UPLOAD_FOLDER'] = os.path.dirname(os.path.abspath(__file__))


def predict_stock_prices(df, prediction_duration):
    # Convert the 'tanggal' column to datetime
    df['tanggal'] = pd.to_datetime(df['tanggal'])

    # Sort the DataFrame by date
    df.sort_values('tanggal', inplace=True)

    # Create the feature matrix X (using the index) and the target variable y (using the 'harga_current' column)
    X = df.index.to_frame()
    y = df['harga_current']

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model with the data
    model.fit(X, y)

    # Get the last date in the DataFrame
    last_date = df['tanggal'].max()

    # Calculate the prediction date based on the selected duration
    if prediction_duration == '1 Week':
        prediction_date = last_date + timedelta(weeks=1)
    elif prediction_duration == '1 Month':
        prediction_date = last_date + timedelta(days=30)
    elif prediction_duration == '1 Year':
        prediction_date = last_date + timedelta(days=365)
    else:
        return None, None, None, None

    # Make prediction for the selected duration
    prediction = model.predict([[df.index.max() + (prediction_date - last_date).days]])[0]

    # Generate the list of prices for the selected duration
    prices = []
    date_range = pd.date_range(start=last_date + timedelta(days=1), end=prediction_date, freq='D')
    for date in date_range:
        index = df.index.max() + (date - last_date).days
        price = model.predict([[index]])[0]
        prices.append(price)

    return prediction, prices


@app.route('/', methods=['GET', 'POST'])
def predict_stock_price():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Save the file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file.csv')
        file.save(file_path)

        # Read the file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Get the selected prediction duration from the dropdown
        prediction_duration = request.form['duration']

        # Predict stock prices for the selected duration
        prediction, prices = predict_stock_prices(df, prediction_duration)

        if prediction is not None:
            # Generate the price chart
            plt.figure(figsize=(10, 6))
            plt.plot(df['tanggal'], df['harga_current'], label='Actual')
            max_date = pd.to_datetime(df['tanggal']).max()
            date_range = pd.date_range(start=max_date + timedelta(days=1), end=max_date + timedelta(days=len(prices)), freq='D')
            plt.plot(date_range, prices, label=prediction_duration)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Stock Price Prediction')
            plt.legend()

            # Save the chart to a temporary file
            chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_chart.png')
            plt.savefig(chart_path)
            plt.close()

            # Convert the chart image to base64 encoding
            with open(chart_path, 'rb') as chart_file:
                chart_data = chart_file.read()
                encoded_chart = base64.b64encode(chart_data).decode('utf-8')

            # Remove the temporary chart file
            os.remove(chart_path)

            # Remove the temporary uploaded file
            os.remove(file_path)

            return render_template('index.html', prediction=prediction, prices=prices, chart_data=encoded_chart)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
