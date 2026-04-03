import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pdfplumber
import re

# Page config must be the first streamlit command
st.set_page_config(page_title="TERMINAL.ALPHA", layout="wide")

# --------------------------
# PDF PARSER ENGINE
# --------------------------
def parse_portfolio(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ["AAPL", "MSFT", "NVDA"]

    # Improved regex to find potential tickers/company names
    matches = re.findall(r"\b[A-Z]{2,5}\b", text) 
    found_tickers = list(set(matches))
    
    # Filter for known common tickers or return defaults if empty
    valid_defaults = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]
    selected = [t for t in found_tickers if t in valid_defaults]
    
    return selected if len(selected) > 0 else ["AAPL", "MSFT", "NVDA"]

# --------------------------
# SIDEBAR & INPUTS
# --------------------------
st.sidebar.title("TERMINAL.ALPHA v20.0")
st.sidebar.info("Upload your Trade Republic or Bank PDF to sync assets.")

uploaded = st.sidebar.file_uploader("Upload Portfolio PDF", type="pdf")

if uploaded:
    tickers = parse_portfolio(uploaded)
    st.sidebar.success(f"Extracted: {', '.join(tickers)}")
else:
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]

page = st.sidebar.selectbox("Navigation", ["Portfolio Dashboard", "Markowitz Optimization", "Strategy Comparison"])

# --------------------------
# DATA CACHING (Crucial for Streamlit)
# --------------------------
@st.cache_data(ttl=3600)
def load_data(ticker_list):
    df = yf.download(ticker_list, period="1y")["Close"]
    return df

data = load_data(tickers)
returns = data.pct_change().dropna()

# --------------------------
# PAGE 1 — PORTFOLIO
# --------------------------
if page == "Portfolio Dashboard":
    st.title("📊 Executive Dashboard")

    # Market Tickers
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    @st.cache_data(ttl=300)
    def get_market_stats():
        indices = {"S&P 500": "^GSPC", "VIX": "^VIX", "Gold": "GC=F", "Oil": "CL=F"}
        stats = {}
        for name, t in indices.items():
            stats[name] = yf.Ticker(t).history(period="1d")["Close"].iloc[-1]
        return stats

    m_data = get_market_stats()
    m_col1.metric("S&P 500", f"{m_data['S&P 500']:.2f}")
    m_col2.metric("VIX", f"{m_data['VIX']:.2f}")
    m_col3.metric("Gold", f"${m_data['Gold']:.2f}")
    m_col4.metric("Oil", f"${m_data['Oil']:.2f}")

    st.subheader("Historical Performance")
    st.line_chart(data)

# --------------------------
# PAGE 2 — MARKOWITZ
# --------------------------
elif page == "Markowitz Optimization":
    st.title("📉 Efficient Frontier")
    
    
    mean_returns = returns.mean()
    cov = returns.cov()
    num_portfolios = 2000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov * 252, weights)))
        
        results[0,i] = portfolio_std
        results[1,i] = portfolio_return
        results[2,i] = results[1,i] / results[0,i] # Sharpe Ratio

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('dark_background')
    scatter = ax.scatter(results[0,i], results[1,i], c=results[2,i], cmap='viridis', alpha=0.5)
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Returns')
    st.pyplot(fig)

# --------------------------
# PAGE 3 — STRATEGIES
# --------------------------
elif page == "Strategy Comparison":
    st.title("🎲 Monte Carlo & Strategy Drill-down")
    
    n = len(tickers)
    mean_rets = returns.mean()
    cov_mat = returns.cov()

    # Define Strategies
    weights_eq = np.ones(n)/n
    inv_vol = 1/returns.std()
    weights_iv = inv_vol / inv_vol.sum()

    strat_choice = st.radio("Select Allocation Engine", ["Equal Weight", "Inverse Volatility"])
    active_w = weights_eq if strat_choice == "Equal Weight" else weights_iv

    # Monte Carlo
    st.subheader(f"Monte Carlo Simulation: {strat_choice}")
    
    def run_sim(w):
        sim_paths = []
        for _ in range(30):
            prices = [6000]
            for _ in range(252):
                daily_ret = np.dot(w, np.random.multivariate_normal(mean_rets, cov_mat))
                prices.append(prices[-1] * (1 + daily_ret))
            sim_paths.append(prices)
        return np.array(sim_paths)

    paths = run_sim(active_w)
    
    fig2, ax2 = plt.subplots()
    for p in paths:
        ax2.plot(p, color='#00ff88', alpha=0.2)
    ax2.set_title("1-Year Predicted Value Path")
    st.pyplot(fig2)

    # Sector/Asset Breakdown
    st.subheader("Asset Allocation")
    df_w = pd.DataFrame({'Asset': tickers, 'Weight': active_w})
    st.bar_chart(df_w.set_index('Asset'))
