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

ltsm_model_btc = load_model("ltsm-BTC-USD.h5")
ltsm_model_btc_roc = load_model("ltsm-BTC-USD-ROC.h5")
rnn_model_btc = load_model("rnnBTC-USD.h5")
xgb_model_btc = xgb.XGBRegressor()
xgb_model_btc.load_model("xgbBTC-USD.json")
# xgb_model_btc = load_model("models/xgbBTC-USD.h5")

ltsm_model_eth = load_model("ltsm-ETH-USD.h5")
rnn_model_eth = load_model("rnnETH-USD.h5")
# xgb_model_eth = load_model("models/xgbETH-USD.h5")

ltsm_model_ada = load_model("ltsm-ADA-USD.h5")
rnn_model_ada = load_model("rnnADA-USD.h5")
# xgb_model_ada = load_model("models/xgbADA-USD.h5")


# inputs = new_data[len(new_data)-len(valid)-60:].values
# inputs = inputs.reshape(-1, 1)
# inputs = scaler.transform(inputs)


# inputs_btc = new_data_btc[len(new_data_btc)-len(valid_btc)-96:].values
# inputs_btc = inputs_btc.reshape(-1, 1)
# inputs_btc = scaler.transform(inputs_btc)

# inputs_eth = new_data_eth[len(new_data_eth)-len(valid_eth)-95:].values
# inputs_eth = inputs_eth.reshape(-1, 1)
# inputs_eth = scaler.transform(inputs_eth)

# inputs_ada = new_data_ada[len(new_data_ada)-len(valid_ada)-96:].values
# inputs_ada = inputs_ada.reshape(-1, 1)
# inputs_ada = scaler.transform(inputs_ada)

[x_train_data_btc, y_train_data_btc, X_test_btc,
    valid_data_btc, scaler_btc] = handle_data('BTC-USD')
[x_train_data_btc_roc, y_train_data_btc_roc, X_test_btc_roc,
    valid_data_btc_roc, scaler_btc_roc] = handle_data_roc('BTC-USD')


ltsm_closing_price_btc = ltsm_model_btc.predict(X_test_btc)
ltsm_closing_price_btc = scaler_btc.inverse_transform(ltsm_closing_price_btc)
ltsm_closing_price_btc_roc = ltsm_model_btc_roc.predict(X_test_btc_roc)
ltsm_closing_price_btc_roc = scaler_btc_roc.inverse_transform(ltsm_closing_price_btc_roc)

rnn_closing_price_btc = rnn_model_btc.predict(X_test_btc)
rnn_closing_price_btc = scaler_btc.inverse_transform(rnn_closing_price_btc)


X_test_eth = []
X_test_ada = []
# X_test_eth = np.reshape(
#     X_test_eth, (X_test_eth.shape[0], X_test_eth.shape[1], 1))
# closing_price_eth = model_eth.predict(X_test_eth)
# closing_price_eth = scaler_eth.inverse_transform(closing_price_eth)

# X_test_ada = np.reshape(
#     X_test_ada, (X_test_ada.shape[0], X_test_ada.shape[1], 1))
# closing_price_ada = model_ada.predict(X_test_ada)
# closing_price_ada = scaler_ada.inverse_transform(closing_price_ada)
valid_data_btc['Predictions-ltsm'] = ltsm_closing_price_btc
A =valid_data_btc_roc.shift(1)['Close']
ltsm_closing_price_btc_roc = ltsm_closing_price_btc_roc.reshape(-1)
B = ltsm_closing_price_btc_roc * A
valid_data_btc_roc['Predictions-ltsm-roc'] = valid_data_btc_roc['Close'] + B

# train_eth = new_data_eth[:1000]
# valid_eth = new_data_eth[1000:]
# valid_eth['Predictions'] = closing_price_eth

# train_ada = new_data_ada[:1000]
# valid_ada = new_data_ada[1000:]
# valid_ada['Predictions'] = closing_price_ada


app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='BTC-USD Stock Data', children=[
            html.Div([
                html.H1("LTSM Predicted closing price",
                         style={'textAlign': 'center'}),
                dcc.Dropdown(id='ltsm-btc-dropdown',
                             options=[{'label': 'Closed', 'value': 'btc-Closed'},
                                      {'label': 'RoC', 'value': 'btc-Roc'},
                                      ],
                             value='btc-Closed',
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='btc-ltsm'),
                # html.H1("",
                #         style={'textAlign': 'center'}),
                # dcc.Graph(id='btc-ltsm'),
                # html.H1("btc-rnn title", style={'textAlign': 'center'}),
                # dcc.Graph(id='btc-rnn'),
                # html.H1("btc-xgb title", style={'textAlign': 'center'}),
                # dcc.Graph(id='btc-xgb'),
            ], className="container"),
        ]),



        # dcc.Tab(label='BTC-USD Stock Data', children=[
        #     html.Div([
        #         html.H2("Actual closing price", style={"textAlign": "center"}),
        #         dcc.Graph(
        #             id="Actual Data BTC-USD",
        #             figure={
        #                 "data": [
        #                     go.Scatter(
        #                         x=valid_data_btc.index,
        #                         y=valid_data_btc["Close"],
        #                         mode='markers'
        #                     )

        #                 ],
        #                 "layout":go.Layout(
        #                     title='scatter plot',
        #                     xaxis={'title': 'Date'},
        #                     yaxis={'title': 'Closing Rate'}
        #                 )
        #             }

        #         ),
        #         html.H2("LSTM Predicted closing price",
        #                 style={"textAlign": "center"}),
        #         dcc.Graph(
        #             id="Predicted Data BTC-USD",
        #             figure={
        #                 "data": [
        #                     go.Scatter(
        #                         x=valid_btc.index,
        #                         y=valid_btc["Predictions"],
        #                         mode='markers'
        #                     )

        #                 ],
        #                 "layout":go.Layout(
        #                     title='scatter plot',
        #                     xaxis={'title': 'Date'},
        #                     yaxis={'title': 'Closing Rate'}
        #                 )
        #             }

        #         )
        #     ])


        # ]),
        # dcc.Tab(label='ETH-USD Stock Data', children=[
        #     html.Div([
        #         html.H2("Actual closing price", style={
        #             "textAlign": "center"}),
        #         dcc.Graph(
        #             id="Actual Data ETH-USD",
        #             figure={
        #                 "data": [
        #                     go.Scatter(
        #                         x=train_eth.index,
        #                         y=valid_eth["Close"],
        #                         mode='markers'
        #                     )

        #                 ],
        #                 "layout":go.Layout(
        #                     title='scatter plot',
        #                     xaxis={'title': 'Date'},
        #                     yaxis={'title': 'Closing Rate'}
        #                 )
        #             }

        #         ),
        #         html.H2("LSTM Predicted closing price",
        #                 style={"textAlign": "center"}),
        #         dcc.Graph(
        #             id="Predicted Data ETH-USD",
        #             figure={
        #                 "data": [
        #                     go.Scatter(
        #                         x=valid_eth.index,
        #                         y=valid_eth["Predictions"],
        #                         mode='markers'
        #                     )

        #                 ],
        #                 "layout":go.Layout(
        #                     title='scatter plot',
        #                     xaxis={'title': 'Date'},
        #                     yaxis={'title': 'Closing Rate'}
        #                 )
        #             }

        #         )
        #     ])


        # ]),
        # dcc.Tab(label='ADA-USD Stock Data', children=[
        #     html.Div([
        #         html.H2("Actual closing price", style={
        #             "textAlign": "center"}),
        #         dcc.Graph(
        #             id="Actual Data ADA-USD",
        #             figure={
        #                 "data": [
        #                     go.Scatter(
        #                         x=train_ada.index,
        #                         y=valid_ada["Close"],
        #                         mode='markers'
        #                     )

        #                 ],
        #                 "layout":go.Layout(
        #                     title='scatter plot',
        #                     xaxis={'title': 'Date'},
        #                     yaxis={'title': 'Closing Rate'}
        #                 )
        #             }

        #         ),
        #         html.H2("LSTM Predicted closing price",
        #                 style={"textAlign": "center"}),
        #         dcc.Graph(
        #             id="Predicted Data ADA-USD",
        #             figure={
        #                 "data": [
        #                     go.Scatter(
        #                         x=valid_ada.index,
        #                         y=valid_ada["Predictions"],
        #                         mode='markers'
        #                     )

        #                 ],
        #                 "layout":go.Layout(
        #                     title='scatter plot',
        #                     xaxis={'title': 'Date'},
        #                     yaxis={'title': 'Closing Rate'}
        #                 )
        #             }

        #         )
        #     ])


        # ]),
        # dcc.Tab(label='Facebook Stock Data', children=[
        #     html.Div([
        #         html.H1("Stocks High vs Lows",
        #                 style={'textAlign': 'center'}),

        #         dcc.Dropdown(id='my-dropdown',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple', 'value': 'AAPL'},
        #                               {'label': 'Facebook', 'value': 'FB'},
        #                               {'label': 'Microsoft', 'value': 'MSFT'}],
        #                      multi=True, value=['FB'],
        #                      style={"display": "block", "margin-left": "auto",
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='highlow'),
        #         html.H1("Stocks Market Volume", style={'textAlign': 'center'}),

        #         dcc.Dropdown(id='my-dropdown2',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple', 'value': 'AAPL'},
        #                               {'label': 'Facebook', 'value': 'FB'},
        #                               {'label': 'Microsoft', 'value': 'MSFT'}],
        #                      multi=True, value=['FB'],
        #                      style={"display": "block", "margin-left": "auto",
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='volume')
        #     ], className="container"),
        # ])


    ])
])


@ app.callback(Output('btc-ltsm', 'figure'),
               [Input('ltsm-btc-dropdown', 'value')])
def update_graph(selected_dropdown):
    trace1 = []
    trace2 = []
    trace1.append(
        go.Scatter(x=valid_data_btc.index,
                    y=valid_data_btc_roc["Close"],
                    mode='lines', opacity=0.7,
                    name=f'Actual Close', textposition='bottom center'))
    trace2.append(
        go.Scatter(x=valid_data_btc.index,
                    y=valid_data_btc_roc['Predictions-ltsm-roc'],
                    mode='lines', opacity=0.7,
                    name=f'Predict Close', textposition='bottom center'))

    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"BTC-USD LTSM Stock Data",
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


# @ app.callback(Output('highlow', 'figure'),
#                [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown):
#     dropdown = {"TSLA": "Tesla", "AAPL": "Apple",
#                 "FB": "Facebook", "MSFT": "Microsoft", }
#     trace1 = []
#     trace2 = []
#     for stock in selected_dropdown:
#         trace1.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["High"],
#                        mode='lines', opacity=0.7,
#                        name=f'High {dropdown[stock]}', textposition='bottom center'))
#         trace2.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["Low"],
#                        mode='lines', opacity=0.6,
#                        name=f'Low {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1, trace2]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
#                                             '#FF7400', '#FFF400', '#FF0056'],
#                                   height=600,
#                                   title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
#                                   xaxis={"title": "Date",
#                                          'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'count': 6, 'label': '6M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'step': 'all'}])},
#                                          'rangeslider': {'visible': True}, 'type': 'date'},
#                                   yaxis={"title": "Price (USD)"})}
#     return figure


# @ app.callback(Output('ltsm-btc', 'figure'),
#                [Input('dropdown-ltsm-btc', 'value')])
# def update_graph(selected_dropdown_value):
#     dropdown = {"TSLA": "Tesla", "AAPL": "Apple",
#                 "FB": "Facebook", "MSFT": "Microsoft", }
#     trace1 = []
#     for stock in selected_dropdown_value:
#         trace1.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["Volume"],
#                        mode='lines', opacity=0.7,
#                        name=f'Volume {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
#                                             '#FF7400', '#FFF400', '#FF0056'],
#                                   height=600,
#                                   title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
#                                   xaxis={"title": "Date",
#                                          'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'count': 6, 'label': '6M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'step': 'all'}])},
#                                          'rangeslider': {'visible': True}, 'type': 'date'},
#                                   yaxis={"title": "Transactions Volume"})}
#     return figure


if __name__ == '__main__':
    app.run_server()
