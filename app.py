import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pdfplumber
import re

st.set_page_config(layout="wide")

# --------------------------
# PDF PARSER
# --------------------------
def parse_portfolio(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()

    matches = re.findall(r"[A-Z][A-Za-z0-9\\s\\.\\-]{3,40}", text)
    assets = list(set(matches))[:15]

    mapping = {
        "NVIDIA": "NVDA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Tesla": "TSLA",
        "Amazon": "AMZN",
        "Visa": "V",
        "BYD": "BYDDF",
        "Deutsche": "DB",
        "Airbus": "AIR.PA"
    }

    tickers = []
    for a in assets:
        for key in mapping:
            if key.lower() in a.lower():
                tickers.append(mapping[key])

    return list(set(tickers)) if tickers else ["AAPL","MSFT","NVDA"]

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("TERMINAL.ALPHA")

uploaded = st.sidebar.file_uploader("Upload Portfolio PDF")

if uploaded:
    tickers = parse_portfolio(uploaded)
else:
    tickers = ["AAPL","MSFT","NVDA"]

page = st.sidebar.selectbox("Navigation", ["Portfolio", "Markowitz", "Strategies"])

# --------------------------
# LOAD DATA
# --------------------------
data = yf.download(tickers, period="1y")["Close"]
returns = data.pct_change().dropna()

# --------------------------
# PAGE 1 — PORTFOLIO
# --------------------------
if page == "Portfolio":
    st.title("📊 Portfolio Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    spx = yf.Ticker("^GSPC").history(period="1d")["Close"].iloc[-1]
    vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
    gold = yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1]
    oil = yf.Ticker("CL=F").history(period="1d")["Close"].iloc[-1]

    col1.metric("S&P 500", round(spx, 2))
    col2.metric("VIX", round(vix, 2))
    col3.metric("Gold", round(gold, 2))
    col4.metric("Oil", round(oil, 2))

    st.subheader("Your Portfolio")
    st.write(tickers)

    st.line_chart(data)

# --------------------------
# PAGE 2 — MARKOWITZ
# --------------------------
elif page == "Markowitz":
    st.title("📉 Markowitz Optimization")

    mean_returns = returns.mean()
    cov = returns.cov()

    results = []

    for _ in range(5000):
        w = np.random.random(len(tickers))
        w /= w.sum()

        ret = np.dot(w, mean_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))

        results.append([vol, ret])

    results = np.array(results)

    fig, ax = plt.subplots()
    ax.scatter(results[:,0], results[:,1], alpha=0.3)
    ax.set_xlabel("Risk")
    ax.set_ylabel("Return")
    st.pyplot(fig)

# --------------------------
# PAGE 3 — STRATEGIES
# --------------------------
elif page == "Strategies":
    st.title("🎲 Advanced Portfolio Strategies")

    mean_returns = returns.mean()
    cov = returns.cov()
    n = len(tickers)

    def equal_weight():
        return np.ones(n)/n

    def inverse_vol():
        vol = returns.std()
        w = 1/vol
        return w/w.sum()

    def gmvp():
        inv = np.linalg.inv(cov)
        ones = np.ones(n)
        w = inv @ ones
        return w/w.sum()

    def max_sharpe():
        best_w = None
        best_sr = -1
        for _ in range(3000):
            w = np.random.random(n)
            w /= w.sum()
            ret = w @ mean_returns
            vol = np.sqrt(w @ cov @ w)
            sr = ret/vol
            if sr > best_sr:
                best_sr = sr
                best_w = w
        return best_w

    def quintile():
        perf = mean_returns.sort_values()
        top = perf.index[int(0.8*len(perf)):]
        w = np.zeros(n)
        for i,t in enumerate(tickers):
            if t in top:
                w[i]=1
        return w/w.sum()

    def markowitz():
        best_w = None
        best_ret = -1
        for _ in range(3000):
            w = np.random.random(n)
            w /= w.sum()
            ret = w @ mean_returns
            vol = np.sqrt(w @ cov @ w)
            if vol < 0.2 and ret > best_ret:
                best_ret = ret
                best_w = w
        return best_w

    def black_litterman():
        w_eq = equal_weight()
        tilt = mean_returns.values
        w = w_eq + 0.1*tilt
        w = np.maximum(w,0)
        return w/np.sum(w)

    strategies = {
        "Buy & Hold": equal_weight(),
        "Equal Weight": equal_weight(),
        "Inverse Volatility": inverse_vol(),
        "GMVP": gmvp(),
        "Max Sharpe": max_sharpe(),
        "Quintile": quintile(),
        "Markowitz MVP": markowitz(),
        "Black-Litterman": black_litterman()
    }

    choice = st.selectbox("Select Strategy", list(strategies.keys()))
    weights = strategies[choice]

    st.subheader("Weights")
    for t,w in zip(tickers,weights):
        st.write(f"{t}: {round(w*100,2)}%")

    def simulate(w):
        paths=[]
        for _ in range(100):
            val=6000
            series=[]
            for _ in range(100):
                rand=np.random.normal(0,1,n)
                r = w @ (mean_returns + rand*returns.std())
                val *= (1+r)
                series.append(val)
            paths.append(series)
        return np.array(paths)

    paths = simulate(weights)

    st.subheader("Monte Carlo")

    fig, ax = plt.subplots()
    for p in paths[:30]:
        ax.plot(p, alpha=0.2)
    st.pyplot(fig)

    st.subheader("Distribution")

    final_vals = paths[:,-1]

    fig2, ax2 = plt.subplots()
    ax2.hist(final_vals, bins=20)
    st.pyplot(fig2)

    st.subheader("Strategy Comparison")

    names=[]
    values=[]

    for name,w in strategies.items():
        sim = simulate(w)
        names.append(name)
        values.append(sim[:,-1].mean())

    fig3, ax3 = plt.subplots()
    ax3.barh(names, values)
    st.pyplot(fig3)
