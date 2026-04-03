# app.py
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
import os
from flask import Flask

# ------------------------
# Flask Server for Vercel
# ------------------------
server = Flask(__name__)

# ------------------------
# Dash App
# ------------------------
app = dash.Dash(__name__, server=server, url_base_pathname='/')
app.title = "Premium Investing Dashboard"

# ------------------------
# Layout
# ------------------------
app.layout = html.Div(style={'backgroundColor': '#0e1117', 'color': '#fff', 'font-family': 'Arial'}, children=[
    html.H1("💹 Premium Investing Dashboard", style={'textAlign': 'center', 'color': 'cyan'}),
    
    html.Div([
        html.H3("Upload Your Portfolio CSV"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin-bottom': '20px', 'color':'#fff'
            },
            multiple=False
        ),
    ]),
    
    html.Div(id='portfolio-table'),
    html.Hr(style={'border-color':'#444'}),
    
    html.Div([
        html.H2("Portfolio Allocation Treemap", style={'color':'cyan'}),
        dcc.Graph(id='treemap')
    ]),
    
    html.Hr(style={'border-color':'#444'}),
    
    html.Div([
        html.H2("Monte Carlo Simulation", style={'color':'cyan'}),
        html.Label("Simulation Horizon (Years)"),
        dcc.Slider(id='years-slider', min=1, max=10, step=1, value=5),
        html.Label("Number of Simulations"),
        dcc.Slider(id='simulations-slider', min=1000, max=5000, step=500, value=2000),
        html.Label("Expected Annual Return (%)"),
        dcc.Input(id='mean-return', type='number', value=7, style={'margin-right':'20px'}),
        html.Label("Annual Volatility (%)"),
        dcc.Input(id='volatility', type='number', value=15),
        dcc.Graph(id='monte-carlo-graph'),
        html.Div(id='simulation-stats', style={'margin-top':'20px'})
    ]),
    
    html.Hr(style={'border-color':'#444'}),
    
    html.Div([
        html.H2("ETF Overlap & Sector/Country Exposure", style={'color':'cyan'}),
        dash_table.DataTable(id='overlap-table', style_table={'overflowX': 'auto'},
                             style_header={'backgroundColor':'#1f2937', 'color':'#fff'},
                             style_cell={'backgroundColor':'#0e1117', 'color':'#fff'}),
        dcc.Graph(id='sector-bar'),
        dcc.Graph(id='country-pie')
    ]),
    
    html.Hr(style={'border-color':'#444'}),
    
    html.Div([
        html.H2("Stock Research Tool", style={'color':'cyan'}),
        html.Label("Enter Ticker:"),
        dcc.Input(id='ticker-input', type='text', value='AAPL'),
        html.Div(id='stock-research')
    ])
])

# ------------------------
# Helper Functions
# ------------------------
def parse_contents(contents):
    import base64, io
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

def get_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except:
        return np.nan

# ------------------------
# Callbacks (same as before)
# ------------------------
# [Copy the previous callbacks for portfolio update and stock research here]
# Example:
@app.callback(
    Output('portfolio-table','children'),
    Input('upload-data','contents')
)
def dummy(contents):
    if contents is None:
        return html.Div("Upload CSV to see portfolio")
    return html.Div("Portfolio loaded!")

# ------------------------
# Run server for local debug
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
