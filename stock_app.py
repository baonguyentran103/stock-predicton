import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import numpy as np
from utils import *

app = dash.Dash()
server = app.server

ltsm_model_btc = load_model("models/ltsm-BTC-USD.h5")
rnn_model_btc = load_model("models/rnnBTC-USD.h5")
xgb_model_btc = xgb.XGBRegressor()
xgb_model_btc.load_model("models/xgbBTC-USD.json")

ltsm_model_btc_roc = load_model("models/ltsm-BTC-USD-ROC.h5")
rnn_model_btc_roc = load_model("models/rnnBTC-USD-ROC.h5")
xgb_model_btc_roc = xgb.XGBRegressor()
xgb_model_btc_roc.load_model("models/xgbBTC-USD-ROC.json")

ltsm_model_eth = load_model("models/ltsm-ETH-USD.h5")
rnn_model_eth = load_model("models/rnnETH-USD.h5")
xgb_model_eth = xgb.XGBRegressor()
xgb_model_eth.load_model("models/xgbETH-USD.json")

ltsm_model_eth_roc = load_model("models/ltsm-ETH-USD-ROC.h5")
rnn_model_eth_roc = load_model("models/rnnETH-USD-ROC.h5")
xgb_model_eth_roc = xgb.XGBRegressor()
xgb_model_eth_roc.load_model("models/xgbETH-USD-ROC.json")

ltsm_model_ada = load_model("models/ltsm-ADA-USD.h5")
rnn_model_ada = load_model("models/rnnADA-USD.h5")
xgb_model_ada = xgb.XGBRegressor()
xgb_model_ada.load_model("models/xgbADA-USD.json")

ltsm_model_ada_roc = load_model("models/ltsm-ADA-USD-ROC.h5")
rnn_model_ada_roc = load_model("models/rnnADA-USD-ROC.h5")
xgb_model_ada_roc = xgb.XGBRegressor()
xgb_model_ada_roc.load_model("models/xgbADA-USD-ROC.json")

# btc
[x_train_data_btc, y_train_data_btc, X_test_btc,
    valid_data_btc, scaler_btc] = handle_data('BTC-USD')

[_, _, X_xgb_test_btc, _, _] = handle_data_xgboost('BTC-USD')

ltsm_closing_price_btc = ltsm_model_btc.predict(X_test_btc)
ltsm_closing_price_btc = scaler_btc.inverse_transform(ltsm_closing_price_btc)

rnn_closing_price_btc = rnn_model_btc.predict(X_test_btc)
rnn_closing_price_btc = scaler_btc.inverse_transform(rnn_closing_price_btc)

xgb_closing_price_btc = xgb_model_btc.predict(X_xgb_test_btc)
xgb_closing_price_btc = np.reshape(
    xgb_closing_price_btc, (xgb_closing_price_btc.shape[0], 1))
xgb_closing_price_btc = scaler_btc.inverse_transform(xgb_closing_price_btc)

valid_data_btc['Predictions-ltsm'] = ltsm_closing_price_btc
valid_data_btc['Predictions-rnn'] = rnn_closing_price_btc
valid_data_btc['Predictions-xgb'] = xgb_closing_price_btc

#btc - roc
[x_train_data_btc_roc, y_train_data_btc_roc, X_test_btc_roc,
    valid_data_btc_roc, scaler_btc_roc] = handle_data_roc('BTC-USD')

ltsm_closing_price_btc_roc = ltsm_model_btc_roc.predict(X_test_btc_roc)
ltsm_closing_price_btc_roc = scaler_btc_roc.inverse_transform(
    ltsm_closing_price_btc_roc)

A = valid_data_btc_roc.shift(1)['Close']
ltsm_closing_price_btc_roc = ltsm_closing_price_btc_roc.reshape(-1)
B = ltsm_closing_price_btc_roc * A
valid_data_btc_roc['Predictions-ltsm'] = valid_data_btc_roc['Close'] + B


rnn_closing_price_btc_roc = rnn_model_btc_roc.predict(X_test_btc_roc)
rnn_closing_price_btc_roc = scaler_btc_roc.inverse_transform(
    rnn_closing_price_btc_roc)

A = valid_data_btc_roc.shift(1)['Close']
rnn_closing_price_btc_roc = rnn_closing_price_btc_roc.reshape(-1)
B = rnn_closing_price_btc_roc * A
valid_data_btc_roc['Predictions-rnn'] = valid_data_btc_roc['Close'] + B

[_, _, X_xgb_test_btc, _, _] = handle_data_xgboost('BTC-USD')

xgb_closing_price_btc_roc = xgb_model_btc_roc.predict(X_xgb_test_btc)
xgb_closing_price_btc_roc = np.reshape(
    xgb_closing_price_btc_roc, (xgb_closing_price_btc_roc.shape[0], 1))
xgb_closing_price_btc_roc = scaler_btc_roc.inverse_transform(
    xgb_closing_price_btc_roc)

A = valid_data_btc_roc.shift(1)['Close']
xgb_closing_price_btc_roc = xgb_closing_price_btc_roc.reshape(-1)
B = xgb_closing_price_btc_roc * A
valid_data_btc_roc['Predictions-xgb'] = valid_data_btc_roc['Close'] + B

#
[x_train_data_eth, y_train_data_eth, X_test_eth,
    valid_data_eth, scaler_eth] = handle_data('ETH-USD')
[_, _, X_xgb_test_eth, _, _] = handle_data_xgboost('ETH-USD')

ltsm_closing_price_eth = ltsm_model_eth.predict(X_test_eth)
ltsm_closing_price_eth = scaler_eth.inverse_transform(ltsm_closing_price_eth)

rnn_closing_price_eth = rnn_model_eth.predict(X_test_eth)
rnn_closing_price_eth = scaler_eth.inverse_transform(rnn_closing_price_eth)

xgb_closing_price_eth = xgb_model_btc.predict(X_xgb_test_eth)
xgb_closing_price_eth = np.reshape(
    xgb_closing_price_eth, (xgb_closing_price_eth.shape[0], 1))
xgb_closing_price_eth = scaler_btc.inverse_transform(xgb_closing_price_eth)

valid_data_eth['Predictions-ltsm'] = ltsm_closing_price_eth
valid_data_eth['Predictions-rnn'] = rnn_closing_price_eth
valid_data_eth['Predictions-xgb'] = xgb_closing_price_eth

#eth - roc
[x_train_data_eth_roc, y_train_data_eth_roc, X_test_eth_roc,
    valid_data_eth_roc, scaler_eth_roc] = handle_data_roc('ETH-USD')

ltsm_closing_price_eth_roc = ltsm_model_eth_roc.predict(X_test_eth_roc)
ltsm_closing_price_eth_roc = scaler_eth_roc.inverse_transform(
    ltsm_closing_price_eth_roc)

A = valid_data_eth_roc.shift(1)['Close']
ltsm_closing_price_eth_roc = ltsm_closing_price_eth_roc.reshape(-1)
B = ltsm_closing_price_eth_roc * A
valid_data_eth_roc['Predictions-ltsm'] = valid_data_eth_roc['Close'] + B


rnn_closing_price_eth_roc = rnn_model_eth_roc.predict(X_test_eth_roc)
rnn_closing_price_eth_roc = scaler_eth_roc.inverse_transform(
    rnn_closing_price_eth_roc)

A = valid_data_eth_roc.shift(1)['Close']
rnn_closing_price_eth_roc = rnn_closing_price_eth_roc.reshape(-1)
B = rnn_closing_price_eth_roc * A
valid_data_eth_roc['Predictions-rnn'] = valid_data_eth_roc['Close'] + B

[_, _, X_xgb_test_eth, _, _] = handle_data_xgboost('ETH-USD')

xgb_closing_price_eth_roc = xgb_model_eth_roc.predict(X_xgb_test_eth)
xgb_closing_price_eth_roc = np.reshape(
    xgb_closing_price_eth_roc, (xgb_closing_price_eth_roc.shape[0], 1))
xgb_closing_price_eth_roc = scaler_eth_roc.inverse_transform(
    xgb_closing_price_eth_roc)

A = valid_data_eth_roc.shift(1)['Close']
xgb_closing_price_eth_roc = xgb_closing_price_eth_roc.reshape(-1)
B = xgb_closing_price_eth_roc * A
valid_data_eth_roc['Predictions-xgb'] = valid_data_eth_roc['Close'] + B

[x_train_data_ada, y_train_data_ada, X_test_ada,
    valid_data_ada, scaler_ada] = handle_data('ADA-USD')
[_, _, X_xgb_test_ada, _, _] = handle_data_xgboost('ADA-USD')


ltsm_closing_price_ada = ltsm_model_ada.predict(X_test_ada)
ltsm_closing_price_ada = scaler_ada.inverse_transform(ltsm_closing_price_ada)

rnn_closing_price_ada = rnn_model_ada.predict(X_test_ada)
rnn_closing_price_ada = scaler_ada.inverse_transform(rnn_closing_price_ada)

xgb_closing_price_ada = xgb_model_btc.predict(X_xgb_test_ada)
xgb_closing_price_ada = np.reshape(
    xgb_closing_price_ada, (xgb_closing_price_ada.shape[0], 1))
xgb_closing_price_ada = scaler_btc.inverse_transform(xgb_closing_price_ada)

valid_data_ada['Predictions-ltsm'] = ltsm_closing_price_ada
valid_data_ada['Predictions-rnn'] = rnn_closing_price_ada
valid_data_ada['Predictions-xgb'] = xgb_closing_price_ada

#ada - roc
[x_train_data_ada_roc, y_train_data_ada_roc, X_test_ada_roc,
    valid_data_ada_roc, scaler_ada_roc] = handle_data_roc('ADA-USD')

ltsm_closing_price_ada_roc = ltsm_model_ada_roc.predict(X_test_ada_roc)
ltsm_closing_price_ada_roc = scaler_ada_roc.inverse_transform(
    ltsm_closing_price_ada_roc)

A = valid_data_ada_roc.shift(1)['Close']
ltsm_closing_price_ada_roc = ltsm_closing_price_ada_roc.reshape(-1)
B = ltsm_closing_price_ada_roc * A
valid_data_ada_roc['Predictions-ltsm'] = valid_data_ada_roc['Close'] + B


rnn_closing_price_ada_roc = rnn_model_ada_roc.predict(X_test_ada_roc)
rnn_closing_price_ada_roc = scaler_ada_roc.inverse_transform(
    rnn_closing_price_ada_roc)

A = valid_data_ada_roc.shift(1)['Close']
rnn_closing_price_ada_roc = rnn_closing_price_ada_roc.reshape(-1)
B = rnn_closing_price_ada_roc * A
valid_data_ada_roc['Predictions-rnn'] = valid_data_ada_roc['Close'] + B

[_, _, X_xgb_test_ada, _, _] = handle_data_xgboost('ADA-USD')

xgb_closing_price_ada_roc = xgb_model_ada_roc.predict(X_xgb_test_ada)
xgb_closing_price_ada_roc = np.reshape(
    xgb_closing_price_ada_roc, (xgb_closing_price_ada_roc.shape[0], 1))
xgb_closing_price_ada_roc = scaler_ada_roc.inverse_transform(
    xgb_closing_price_ada_roc)

A = valid_data_ada_roc.shift(1)['Close']
xgb_closing_price_ada_roc = xgb_closing_price_ada_roc.reshape(-1)
B = xgb_closing_price_ada_roc * A
valid_data_ada_roc['Predictions-xgb'] = valid_data_ada_roc['Close'] + B

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='BTC-USD Stock Data', children=[
            html.Div([
                html.H1("LTSM Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Dropdown(id='btc-dropdown',
                             options=[{'label': 'Closed', 'value': 'Closed'},
                                      {'label': 'RoC', 'value': 'Roc'},
                                      ],
                             value='Closed',
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='btc-ltsm'),
                html.H1("models/rnn Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Graph(id='btc-rnn'),
                html.H1("XGB Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Graph(id='btc-xgb'),

            ], className="container"),
        ]),
        dcc.Tab(label='ETH-USD Stock Data', children=[
            html.Div([
                html.H1("LTSM Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Dropdown(id='eth-dropdown',
                             options=[{'label': 'Closed', 'value': 'Closed'},
                                      {'label': 'RoC', 'value': 'Roc'},
                                      ],
                             value='Closed',
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='eth-ltsm'),
                html.H1("models/rnn Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Graph(id='eth-rnn'),
                html.H1("XGB Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Graph(id='eth-xgb'),

            ], className="container"),
        ]),
        dcc.Tab(label='ADA-USD Stock Data', children=[
            html.Div([
                html.H1("LTSM Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Dropdown(id='ada-dropdown',
                             options=[{'label': 'Closed', 'value': 'Closed'},
                                      {'label': 'RoC', 'value': 'Roc'},
                                      ],
                             value='Closed',
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='ada-ltsm'),
                html.H1("models/rnn Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Graph(id='ada-rnn'),
                html.H1("XGB Predicted closing price",
                        style={'textAlign': 'center'}),
                dcc.Graph(id='ada-xgb'),

            ], className="container"),
        ]),
    ])
])


def getFigure(dropdown_value, valid_data, yKey, title, valid_data_roc):
    data = {"Closed": valid_data, "Roc": valid_data_roc}
    data_render = data[dropdown_value]
    trace1 = []
    trace2 = []
    trace1.append(
        go.Scatter(x=data_render.index,
                   y=data_render["Close"],
                   mode='lines', opacity=0.7,
                   name=f'Actual Close', textposition='bottom center'))
    trace2.append(
        go.Scatter(x=data_render.index,
                   y=data_render[yKey],
                   mode='lines', opacity=0.7,
                   name=f'Predict Close', textposition='bottom center'))

    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=title,
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                           'step': 'month',
                                                                            'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Close Price"})}
    return figure


@ app.callback(Output('btc-ltsm', 'figure'),
               [Input('btc-dropdown', 'value')])
def update_graph(selected_dropdown):  # btc-Roc btc-Closed
    return getFigure(selected_dropdown, valid_data_btc, "Predictions-ltsm", "BTC-USD LTSM Stock Data", valid_data_btc_roc)


@ app.callback(Output('btc-rnn', 'figure'),
               [Input('btc-dropdown', 'value')])
def update_graph(selected_dropdown):  # btc-Roc btc-Closed
    return getFigure(selected_dropdown, valid_data_btc, "Predictions-rnn", "BTC-USD RNN Stock Data", valid_data_btc_roc)


@ app.callback(Output('btc-xgb', 'figure'),
               [Input('btc-dropdown', 'value')])
def update_graph(selected_dropdown):  # btc-Roc btc-Closed
    return getFigure(selected_dropdown, valid_data_btc, "Predictions-xgb", "BTC-USD XGB Stock Data", valid_data_btc_roc)


@ app.callback(Output('eth-ltsm', 'figure'),
               [Input('eth-dropdown', 'value')])
def update_graph(selected_dropdown):  # eth-Roc eth-Closed
    return getFigure(selected_dropdown, valid_data_eth, "Predictions-ltsm", "ETH-USD LTSM Stock Data", valid_data_eth_roc)


@ app.callback(Output('eth-rnn', 'figure'),
               [Input('eth-dropdown', 'value')])
def update_graph(selected_dropdown):  # eth-Roc eth-Closed
    return getFigure(selected_dropdown, valid_data_eth, "Predictions-rnn", "ETH-USD RNN Stock Data", valid_data_eth_roc)


@ app.callback(Output('eth-xgb', 'figure'),
               [Input('eth-dropdown', 'value')])
def update_graph(selected_dropdown):  # eth-Roc eth-Closed
    return getFigure(selected_dropdown, valid_data_eth, "Predictions-xgb", "ETH-USD XGB Stock Data", valid_data_eth_roc)


@ app.callback(Output('ada-ltsm', 'figure'),
               [Input('ada-dropdown', 'value')])
def update_graph(selected_dropdown):  # ada-Roc ada-Closed
    return getFigure(selected_dropdown, valid_data_ada, "Predictions-ltsm", "ADA-USD LTSM Stock Data", valid_data_ada_roc)


@ app.callback(Output('ada-rnn', 'figure'),
               [Input('ada-dropdown', 'value')])
def update_graph(selected_dropdown):  # ada-Roc ada-Closed
    return getFigure(selected_dropdown, valid_data_ada, "Predictions-rnn", "ADA-USD RNN Stock Data", valid_data_ada_roc)


@ app.callback(Output('ada-xgb', 'figure'),
               [Input('ada-dropdown', 'value')])
def update_graph(selected_dropdown):  # ada-Roc ada-Closed
    return getFigure(selected_dropdown, valid_data_ada, "Predictions-xgb", "ADA-USD XGB Stock Data", valid_data_ada_roc)


if __name__ == '__main__':
    app.run_server()
