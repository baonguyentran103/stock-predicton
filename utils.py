from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10


def get_new_data_ticket(name):
    ticket = yf.Ticker(name)

    hist = ticket.history(period="1d")

    hist.drop('Volume', axis='columns', inplace=True)
    hist.drop('Dividends', axis='columns', inplace=True)
    hist.drop('Stock Splits', axis='columns', inplace=True)

    hist['Date'] = hist.index
    hist['Date'] = pd.to_datetime(hist['Date']).dt.date
    col = hist.pop("Date")
    hist.insert(0, col.name, col)
    hist.to_csv(name + '.csv', mode='a', index=False, header=False)


def get_all_ticket_new_data():
    get_new_data_ticket('BTC-USD')
    get_new_data_ticket('ETH-USD')
    get_new_data_ticket('ADA-USD')


def train_LTSM_close_name(name):
    df = pd.read_csv(name + '.csv')
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df.Date
    df.drop("Date", axis=1, inplace=True)
    df.drop("Open", axis=1, inplace=True)
    df.drop("High", axis=1, inplace=True)
    df.drop("Low", axis=1, inplace=True)
    final_dataset = df.values
    data_length = final_dataset.shape[0]
    validation_data_length = int(data_length * 0.1)
    train_data_length = data_length - validation_data_length
    train_data = final_dataset[0:train_data_length, :]
    valid_data = final_dataset[train_data_length:, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)
    x_train_data, y_train_data = [], []
    for i in range(validation_data_length, len(train_data)):
        x_train_data.append(scaled_data[i-validation_data_length:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    inputs_data = df[len(df)-len(valid_data)-validation_data_length:].values
    inputs_data = inputs_data.reshape(-1, 1)
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data,
                   epochs=1, batch_size=1, verbose=2)

    X_test = []
    inputs_data = scaler.transform(inputs_data)
    for i in range(validation_data_length, inputs_data.shape[0]):
        X_test.append(inputs_data[i-validation_data_length:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_closing_price = lstm_model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
    lstm_model.save("ltsm-" + name + ".h5")


def train_ltsm_close_all():
    train_LTSM_close_name('BTC-USD')
    train_LTSM_close_name('ETH-USD')
    train_LTSM_close_name('ADA-USD')


def ROC(data, n):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D, name='Rate of Change')
    data = data.join(ROC)
    return data


def train_LTSM_roc_name(name):
    df = pd.read_csv(name + '.csv')
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df.Date
    df.drop("Date", axis=1, inplace=True)
    df.drop("Open", axis=1, inplace=True)
    df.drop("High", axis=1, inplace=True)
    df.drop("Low", axis=1, inplace=True)
    df = df.pct_change(periods=1)
    df.drop([0])
    final_dataset = df.values
    data_length = final_dataset.shape[0]
    validation_data_length = int(data_length * 0.1)
    train_data_length = data_length - validation_data_length
    train_data = final_dataset[0:train_data_length, :]
    valid_data = final_dataset[train_data_length:, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)
    x_train_data, y_train_data = [], []
    for i in range(validation_data_length, len(train_data)):
        x_train_data.append(scaled_data[i-validation_data_length:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    inputs_data = df[len(df)-len(valid_data)-validation_data_length:].values
    inputs_data = inputs_data.reshape(-1, 1)
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data,
                   epochs=1, batch_size=1, verbose=2)

    X_test = []
    inputs_data = scaler.transform(inputs_data)
    for i in range(validation_data_length, inputs_data.shape[0]):
        X_test.append(inputs_data[i-validation_data_length:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_closing_price = lstm_model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
    lstm_model.save("ltsm-" + name + "-roc.h5")
