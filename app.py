import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go

# --------------------------
# Dash App Initialization
# --------------------------
app = dash.Dash(__name__)
server = app.server
app.title = "Premium Investing Dashboard"

# --------------------------
# Layout
# --------------------------
app.layout = html.Div(style={'backgroundColor': '#0e1117', 'color': '#fff', 'font-family': 'Arial'}, children=[
    html.H1("💹 Premium Investing Dashboard", style={'textAlign': 'center', 'color': 'cyan'}),
    
    html.Div([
        html.H3("Upload Your Portfolio CSV"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ', html.A('Select CSV File')
            ]),
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

# --------------------------
# Callbacks
# --------------------------

# Parse uploaded CSV
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    import base64, io
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

# --------------------------
# Portfolio Callback
# --------------------------
@app.callback(
    Output('portfolio-table', 'children'),
    Output('treemap', 'figure'),
    Output('monte-carlo-graph', 'figure'),
    Output('simulation-stats', 'children'),
    Output('overlap-table', 'data'),
    Output('sector-bar', 'figure'),
    Output('country-pie', 'figure'),
    Input('upload-data', 'contents'),
    State('years-slider', 'value'),
    State('simulations-slider', 'value'),
    State('mean-return', 'value'),
    State('volatility', 'value')
)
def update_portfolio(contents, years, simulations, mean_return, volatility):
    if contents is None:
        return html.Div(), go.Figure(), go.Figure(), "", [], go.Figure(), go.Figure()

    df = parse_contents(contents)

    # Live prices
    def get_price(ticker):
        try:
            return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
        except:
            return np.nan
    df['Live Price'] = df['Ticker (Yahoo Finance)'].apply(get_price)
    df['Market Value'] = df['Uni. / Nominal'] * df['Live Price']
    df['Weight'] = df['Market Value'] / df['Market Value'].sum()

    table_div = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor':'#1f2937', 'color':'#fff'},
        style_cell={'backgroundColor':'#0e1117', 'color':'#fff'}
    )

    # Treemap
    treemap_fig = go.Figure(go.Treemap(
        labels=df['Nome do Título'],
        values=df['Market Value'],
        hovertext=df['Ticker (Yahoo Finance)'],
        marker=dict(colors=df['Weight'], colorscale='Viridis')
    ))

    # Monte Carlo
    T = years
    dt = 1
    mean = mean_return / 100
    sigma = volatility / 100
    S0 = df['Market Value'].sum()
    sim_matrix = np.zeros((T, simulations))
    sim_matrix[0] = S0
    for t in range(1,T):
        rand = np.random.standard_normal(simulations)
        sim_matrix[t] = sim_matrix[t-1] * np.exp((mean - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand)
    mc_fig = go.Figure()
    for i in range(min(simulations,200)):
        mc_fig.add_trace(go.Scatter(y=sim_matrix[:,i], mode='lines', line=dict(color='cyan', width=1), opacity=0.2))
    mc_fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fff",
                         xaxis_title="Years", yaxis_title="Portfolio Value")
    stats_div = f"""
    Mean final value: ${sim_matrix[-1].mean():,.2f} | Median: ${np.median(sim_matrix[-1]):,.2f} |
    10th percentile: ${np.percentile(sim_matrix[-1],10):,.2f} | 90th percentile: ${np.percentile(sim_matrix[-1],90):,.2f}
    """

    # ETF Overlap
    etf_rows = df[df['Nome do Título'].str.contains("ETF|Swap", case=False, regex=True)]
    overlap_dict = {}
    for idx,row in etf_rows.iterrows():
        try:
            etf = yf.Ticker(row['Ticker (Yahoo Finance)'])
            holdings = etf.fund_holdings
            if holdings is not None and not holdings.empty:
                holdings_df = pd.DataFrame(holdings)
                holdings_df.rename(columns={'symbol':'Ticker','holdingPercent':'Weight'}, inplace=True)
                for tck in holdings_df['Ticker']:
                    overlap_dict[tck] = overlap_dict.get(tck,0)+row['Weight']
        except: pass
    overlap_df = pd.DataFrame(list(overlap_dict.items()), columns=['Ticker','Portfolio_Weight']).sort_values('Portfolio_Weight', ascending=False)

    # Sector/Country Exposure
    sectors, countries = [], []
    for ticker in df['Ticker (Yahoo Finance)']:
        try:
            info = yf.Ticker(ticker).info
            sectors.append(info.get('sector','Unknown'))
            countries.append(info.get('country','Unknown'))
        except: 
            sectors.append('Unknown')
            countries.append('Unknown')
    df['Sector'] = sectors
    df['Country'] = countries

    sector_weights = df.groupby('Sector')['Weight'].sum().reset_index()
    sector_fig = go.Figure(go.Bar(x=sector_weights['Sector'], y=sector_weights['Weight'], marker_color='cyan'))
    sector_fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fff", title="Sector Exposure")

    country_weights = df.groupby('Country')['Weight'].sum().reset_index()
    country_fig = go.Figure(go.Pie(labels=country_weights['Country'], values=country_weights['Weight'], hole=0.4))
    country_fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fff", title="Country Exposure")

    return table_div, treemap_fig, mc_fig, stats_div, overlap_df.to_dict('records'), sector_fig, country_fig

# --------------------------
# Stock Research Callback
# --------------------------
@app.callback(
    Output('stock-research','children'),
    Input('ticker-input','value')
)
def stock_research(ticker):
    if ticker:
        try:
            t = yf.Ticker(ticker)
            price = t.history(period="1d")['Close'].iloc[-1]
            market_cap = t.info.get('marketCap','N/A')
            pe = t.info.get('trailingPE','N/A')
            rating = t.info.get('recommendationKey','N/A')
            return html.Div([
                html.P(f"**Current Price:** {price}"),
                html.P(f"**Market Cap:** {market_cap}"),
                html.P(f"**P/E Ratio:** {pe}"),
                html.P(f"**Analyst Ratings:** {rating}")
            ])
        except:
            return "Data not available."
    return ""

# --------------------------
# Run App
# --------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
