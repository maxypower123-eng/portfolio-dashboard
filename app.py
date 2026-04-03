import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
import os

# ------------------------
# INIT
# ------------------------
app = dash.Dash(__name__)
server = app.server

# ------------------------
# LOAD CSV
# ------------------------
csv_path = "Portfolio_PowerBI_Ready.csv"

if os.path.exists(csv_path):
    df_portfolio = pd.read_csv(csv_path)
else:
    df_portfolio = pd.DataFrame(columns=[
        "Nome do Título",
        "Ticker (Yahoo Finance)",
        "Uni. / Nominal"
    ])

# ------------------------
# HELPER FUNCTION
# ------------------------
def get_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except:
        return np.nan

# ------------------------
# LAYOUT
# ------------------------
app.layout = html.Div(style={
    'backgroundColor': '#0e1117',
    'color': '#ffffff',
    'font-family': 'Arial',
    'padding': '20px'
}, children=[

    html.H1("💹 Premium Investing Dashboard", style={'textAlign': 'center', 'color': 'cyan'}),

    html.H3("Portfolio Overview"),

    html.Div(id='portfolio-table'),

    html.Hr(),

    html.H3("Portfolio Allocation"),
    dcc.Graph(id='treemap'),

    html.Hr(),

    html.H3("Monte Carlo Simulation"),

    html.Label("Years"),
    dcc.Slider(id='years', min=1, max=10, value=5),

    html.Label("Simulations"),
    dcc.Slider(id='sims', min=500, max=3000, step=500, value=1500),

    dcc.Graph(id='mc'),

    html.Div(id='stats', style={'marginTop': '20px'})
])

# ------------------------
# CALLBACK
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

    if df.empty:
        return "No portfolio data found.", go.Figure(), go.Figure(), ""

    # Fetch latest prices
    df['Price'] = df['Ticker (Yahoo Finance)'].apply(get_price)
    df['Value'] = df['Price'] * df['Uni. / Nominal']
    df['Weight'] = df['Value'] / df['Value'].sum()

    # Table
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_header={'backgroundColor':'#1f2937','color':'white'},
        style_cell={'backgroundColor':'#0e1117','color':'white'},
        style_table={'overflowX': 'auto'}
    )

    # Treemap
    treemap = go.Figure(go.Treemap(
        labels=df['Nome do Título'],
        values=df['Value'],
        hovertext=df['Ticker (Yahoo Finance)']
    ))

    treemap.update_layout(
        paper_bgcolor="#0e1117",
        font_color="#ffffff"
    )

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
        mc_fig.add_trace(go.Scatter(
            y=sim[:, i],
            mode='lines',
            opacity=0.2,
            line=dict(color='cyan')
        ))

    mc_fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#ffffff",
        xaxis_title="Years",
        yaxis_title="Portfolio Value"
    )

    stats = f"""
    Mean: {sim[-1].mean():,.2f} |
    Median: {np.median(sim[-1]):,.2f} |
    P10: {np.percentile(sim[-1],10):,.2f} |
    P90: {np.percentile(sim[-1],90):,.2f}
    """

    return table, treemap, mc_fig, stats

# ------------------------
# RUN LOCAL / RENDER
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
