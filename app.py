import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pdfplumber
import re
from scipy.optimize import minimize

st.set_page_config(page_title="TERMINAL.ALPHA v22.0", layout="wide")

# --------------------------
# STRICT PDF PARSER (NO FAKES)
# --------------------------
def parse_portfolio_to_tickers(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    
    # 1. Regex for ISINs (Most reliable for Trade Republic/European Banks)
    # Example: LU1681043599 or US67066G1040
    isins = re.findall(r"[A-Z]{2}[A-Z0-9]{9}\d", text)
    
    # 2. Regex for Ticker Symbols (3-5 Uppercase letters usually followed by price)
    # This is a fallback but ISIN is preferred.
    potential_tickers = re.findall(r"\b[A-Z]{3,5}\b", text)
    
    # Filter out common non-ticker words found in PDFs
    blacklist = ['ISIN', 'PAGE', 'DATE', 'CASH', 'TOTAL', 'PORTFOLIO', 'TRADE', 'REPUBLIC']
    clean_tickers = [t for t in potential_tickers if t not in blacklist]
    
    # Combine and unique
    final_list = list(set(isins + clean_tickers))
    return final_list

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("TERMINAL.ALPHA")
uploaded = st.sidebar.file_uploader("Upload your Portfolio PDF", type="pdf")

# --------------------------
# CORE LOGIC
# --------------------------
if uploaded:
    user_tickers = parse_portfolio_to_tickers(uploaded)
    
    if not user_tickers:
        st.error("No tickers or ISINs identified in the PDF. Please ensure it is a standard statement.")
        st.stop()
        
    st.sidebar.success(f"Identified {len(user_tickers)} Assets")
    
    # Load Real Data from Yahoo Finance
    @st.cache_data(ttl=3600)
    def get_fin_data(t_list):
        # We fetch 2 years to get enough covariance data
        df = yf.download(t_list, period="2y", progress=False)["Close"]
        # Drop columns that failed to download
        return df.dropna(axis=1, how='all')

    df_prices = get_fin_data(user_tickers)
    
    if df_prices.empty:
        st.error("Could not fetch market data for the identified symbols.")
        st.stop()

    # Metrics Calculations
    returns = df_prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n_assets = len(df_prices.columns)
    tickers = df_prices.columns.tolist()

    # --------------------------
    # STRATEGY FUNCTIONS
    # --------------------------
    def get_weights(strat_name, personal_view=0.05):
        if strat_name in ["Buy & Hold", "Equally Weighted"]:
            return np.ones(n_assets) / n_assets

        elif strat_name == "Quintile Portfolio":
            top_n = max(1, n_assets // 5)
            top_indices = mean_returns.nlargest(top_n).index
            w = np.array([1.0 if t in top_indices else 0.0 for t in tickers])
            return w / w.sum()

        elif strat_name == "Markowitz MVP":
            # Minimize variance for a specific target return
            target = mean_returns.mean()
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: x @ mean_returns - target})
            res = minimize(lambda w: w.T @ cov_matrix @ w, n_assets*[1./n_assets], 
                           bounds=[(0,1)]*n_assets, constraints=cons)
            return res.x

        elif strat_name == "GMVP":
            # Global Minimum Variance
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            res = minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), n_assets*[1./n_assets], 
                           bounds=[(0,1)]*n_assets, constraints=cons)
            return res.x

        elif strat_name == "Max Sharpe (MSRP)":
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            def min_func_sharpe(w):
                p_ret = w @ mean_returns
                p_vol = np.sqrt(w.T @ cov_matrix @ w)
                return -(p_ret / p_vol)
            res = minimize(min_func_sharpe, n_assets*[1./n_assets], 
                           bounds=[(0,1)]*n_assets, constraints=cons)
            return res.x

        elif strat_name == "Inverse Volatility":
            vols = returns.std() * np.sqrt(252)
            inv_vols = 1 / vols
            return inv_vols / inv_vols.sum()

        elif strat_name == "Black-Litterman":
            # Equilibrium weights modified by personal view tilt
            w_eq = np.ones(n_assets) / n_assets
            # Tilt toward the asset with highest historical return
            tilt = np.zeros(n_assets)
            tilt[np.argmax(mean_returns)] = personal_view
            w = w_eq + tilt
            return w / w.sum()

    # --------------------------
    # TABS / UI
    # --------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Portfolio Dashboard", "📈 Markowitz Frontier", "🎲 Strategy Lab"])

    with tab1:
        st.title("Executive Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        
        # Real-time Market Context
        m_spx = yf.Ticker("^GSPC").history(period="1d")["Close"].iloc[-1]
        m_vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
        
        col1.metric("Your Assets", n_assets)
        col2.metric("S&P 500", f"{m_spx:,.2f}")
        col3.metric("VIX", f"{m_vix:.2f}")
        col4.metric("Current Value Base", "€6,000")
        
        st.subheader("Your Asset Relative Performance")
        st.line_chart(df_prices / df_prices.iloc[0])

    with tab2:
        st.subheader("Efficient Frontier Analysis")
        
        # Random portfolios for visualization
        n_sim = 1000
        sim_res = np.zeros((3, n_sim))
        for i in range(n_sim):
            w = np.random.random(n_assets)
            w /= np.sum(w)
            sim_res[0,i] = np.sqrt(w.T @ cov_matrix @ w)
            sim_res[1,i] = np.sum(mean_returns * w)
            sim_res[2,i] = sim_res[1,i] / sim_res[0,i]
            
        fig, ax = plt.subplots()
        plt.style.use('dark_background')
        ax.scatter(sim_res[0,:], sim_res[1,:], c=sim_res[2,:], cmap='viridis', s=5)
        ax.set_xlabel("Volatility (Risk)")
        ax.set_ylabel("Expected Return")
        st.pyplot(fig)

    with tab3:
        st.subheader("Strategy Lab")
        selected_strat = st.selectbox("Choose Optimization Model", [
            "Buy & Hold", "Equally Weighted", "Quintile Portfolio", 
            "Markowitz MVP", "GMVP", "Max Sharpe (MSRP)", 
            "Inverse Volatility", "Black-Litterman"
        ])
        
        view_tilt = 0.05
        if selected_strat == "Black-Litterman":
            view_tilt = st.slider("Personal View Bullish Tilt", 0.0, 0.3, 0.05)

        opt_weights = get_weights(selected_strat, view_tilt)
        
        # Display Results Table
        res_df = pd.DataFrame({
            "Asset": tickers,
            "Optimal Allocation": [f"{w*100:.2f}%" for w in opt_weights]
        })
        st.table(res_df)

        # Monte Carlo Simulation
        st.subheader("Future Value Projection (Monte Carlo)")
        def run_mc(w):
            sim_paths = []
            for _ in range(30):
                prices = [6000]
                for _ in range(252):
                    # Daily returns based on historical covariance
                    daily_ret = np.random.multivariate_normal(mean_returns/252, cov_matrix/252)
                    prices.append(prices[-1] * (1 + np.dot(w, daily_ret)))
                sim_paths.append(prices)
            return sim_paths

        paths = run_mc(opt_weights)
        fig2, ax2 = plt.subplots()
        for p in paths:
            ax2.plot(p, color='#00ff88', alpha=0.2)
        ax2.set_title(f"30 Simulated Outcomes for {selected_strat}")
        st.pyplot(fig2)

else:
    st.info("Waiting for PDF upload... Please upload your Trade Republic statement in the sidebar to begin analysis.")
