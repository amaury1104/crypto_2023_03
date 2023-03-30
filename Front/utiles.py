import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from tensorflow.keras import models
#from tensorflow.keras import layers
#from sklearn.model_selection import train_test_split

def get_crypto_data():

    #sources API coingecko
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=max&interval=daily"
    response = requests.get(url)
    data = response.json()['prices']

    #creates DataFrame with columns date and price
    df = pd.DataFrame(data, columns=['date', 'price'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    df.index = df.index.date

    df = df.groupby(df.index).max()
    return df

def get_x_y(df):
    price_data = df['price'].values

    #How many previous prices the model will use to predict the target
    past_days = 15

    X = []
    y = []
    for i in range(len(price_data) - past_days):
        X.append(price_data[i:i+past_days])
        y.append(price_data[i+past_days])

    X = np.array(X)
    y = np.array(y)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    return X, y

def price_prediction(df, X, past_days, model):
    # Get the last few prices, dependent on past_days
    latest_prices = df['price'].values[-past_days:]
    X = np.append(X, latest_prices)[-past_days:]

    # Reshape X to match the input shape of the LSTM model
    X = X.reshape((1, past_days, 1))

    #predict next 5 prices
    next_prices = []
    for i in range(5):
        next_price = model.predict(X, verbose = 0)[0][0]
        next_prices.append(next_price)
        X = np.append(X, next_price)[-past_days:]
        X = X.reshape((1, past_days, 1))

    return next_prices

def plot_last_month(df, next_prices):
        # Select the last year of data
        last_year = df['price'].iloc[-28:]

        # Create a plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the last month of data
        ax.plot(last_year.index, last_year.values, label='Last Month Prices')

        # Add the predicted prices
        next_price_dates = pd.date_range(start=df.index[-1], periods=6, freq='D')[1:]
        ax.plot(next_price_dates, next_prices, label='Predicted Prices', color='orange')

        # Set plot properties
        ax.set_title('Crypto Prices')
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')

        return fig, ax
