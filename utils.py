from keras.layers import LSTM, Dropout, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
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

    hist.drop(columns=['Volume', 'Dividends', 'Stock Splits'], axis=1, inplace=True)

    hist['Date'] = hist.index
    hist['Date'] = pd.to_datetime(hist['Date']).dt.date
    col = hist.pop("Date")
    hist.insert(0, col.name, col)
    hist.to_csv(name + '.csv', mode='a', index=False, header=False)


def get_all_ticket_new_data():
    get_new_data_ticket('BTC-USD')
    get_new_data_ticket('ETH-USD')
    get_new_data_ticket('ADA-USD')

def handle_data(name):
    df = pd.read_csv(name + '.csv')
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df.Date
    df.drop(columns=["Date", "Open", "High", "Low"], axis=1, inplace=True)
    
    final_dataset = df.values
    data_length = final_dataset.shape[0]
    validation_data_length = int(data_length * 0.1)
    train_data_length = data_length - validation_data_length
    train_data = final_dataset[0:train_data_length, :]
    valid_data = final_dataset[train_data_length:, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)
    x_train_data, y_train_data = [], []
    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i-60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
    X_test = []
    inputs_data = df[len(df)-len(valid_data)-60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return [x_train_data, y_train_data, X_test, valid_data, scaler]

def train_LTSM_close_name(x_train_data, y_train_data, name):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(x_train_data, y_train_data,
                   epochs=1, batch_size=1, verbose=2)
    lstm_model.save("ltsm-" + name + ".h5")

def ROC(data, n):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D, name='Rate of Change')
    data = data.join(ROC)
    return data


def handle_data_roc(name):
    df = pd.read_csv(name + '.csv')
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df.Date
    df.drop(columns=["Date", "Open", "High", "Low"], axis=1, inplace=True)
    df =df.pct_change(periods=1)
    final_dataset = df.values
    data_length = final_dataset.shape[0]
    validation_data_length = int(data_length * 0.1)
    train_data_length = data_length - validation_data_length
    train_data = final_dataset[0:train_data_length, :]
    valid_data = final_dataset[train_data_length:, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)
    x_train_data, y_train_data = [], []
    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i-60:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
    X_test = []
    inputs_data = df[len(df)-len(valid_data)-60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i-validation_data_length:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return [x_train_data, y_train_data, X_test, valid_data, scaler]

def train_RNN_close_name(x_train_data, y_train_data, name):
    my_rnn_model = Sequential()
    my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    #my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    #my_rnn_model.add(SimpleRNN(32, return_sequences=True))
    my_rnn_model.add(SimpleRNN(32))
    my_rnn_model.add(Dense(1)) # The time step of the output

    my_rnn_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    my_rnn_model.fit(x_train_data, y_train_data, epochs=100, batch_size=150, verbose=0)

    my_rnn_model.save("rnn" + name + ".h5")
def train_XGBoost_close_name(x_train_data, y_train_data, name):
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42, booster='gbtree')
    xgb.fit(x_train_data, y_train_data)
    xgb.save_model("xgb" + name + ".json")
