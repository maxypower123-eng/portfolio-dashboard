import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
from flask import Flask
import os

# ------------------------
# Flask Server
# ------------------------
server = Flask(__name__)

# ------------------------
# Dash App
# ------------------------
app = dash.Dash(__name__, server=server)
app.title = "Premium Investing Dashboard"

# ------------------------
# DEMO PORTFOLIO (replace later)
# ------------------------
df_portfolio = pd.DataFrame({
    "Nome do Título": ["Apple", "Microsoft", "NVIDIA", "S&P 500 ETF"],
    "Ticker (Yahoo Finance)": ["AAPL", "MSFT", "NVDA", "SPY"],
    "Uni. / Nominal": [2, 1.5, 1, 3]
})

# ------------------------
# Helper
# ------------------------
def get_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except:
        return np.nan

# ------------------------
# Layout
# ------------------------
app.layout = html.Div(style={'backgroundColor': '#0e1117', 'color': '#fff'}, children=[

    html.H1("💹 Premium Investing Dashboard", style={'textAlign': 'center', 'color': 'cyan'}),

    html.Div(id='portfolio-table'),

    html.Hr(),

    dcc.Graph(id='treemap'),

    html.Hr(),

    html.H2("Monte Carlo Simulation"),

    dcc.Slider(id='years', min=1, max=10, value=5),
    dcc.Slider(id='sims', min=500, max=3000, step=500, value=1500),

    dcc.Graph(id='mc'),

    html.Div(id='stats')
])

# ------------------------
# Callback
# ------------------------
@app.callback(
    Output('portfolio-table','children'),
    Output('treemap','figure'),
    Output('mc','figure'),
    Output('stats','children'),
    Input('years','value'),
    Input('sims','value')
)
def update(years, sims):
    df = df_portfolio.copy()

    df['Price'] = df['Ticker (Yahoo Finance)'].apply(get_price)
    df['Value'] = df['Price'] * df['Uni. / Nominal']
    df['Weight'] = df['Value'] / df['Value'].sum()

    # Table
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_cell={'backgroundColor': '#0e1117', 'color': 'white'}
    )

    # Treemap
    treemap = go.Figure(go.Treemap(
        labels=df['Nome do Título'],
        values=df['Value']
    ))

    # Monte Carlo
    S0 = df['Value'].sum()
    mean = 0.07
    vol = 0.15

    sim = np.zeros((years, sims))
    sim[0] = S0

    for t in range(1, years):
        rand = np.random.standard_normal(sims)
        sim[t] = sim[t-1] * np.exp((mean - 0.5*vol**2) + vol*rand)

    mc_fig = go.Figure()
    for i in range(min(100, sims)):
        mc_fig.add_trace(go.Scatter(y=sim[:,i], mode='lines', opacity=0.2))

    stats = f"Mean: {sim[-1].mean():,.2f} | Median: {np.median(sim[-1]):,.2f}"

    return table, treemap, mc_fig, stats

# ------------------------
# Run (Vercel)
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
