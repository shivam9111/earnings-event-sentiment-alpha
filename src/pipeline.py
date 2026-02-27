# =============================================================================
# Earnings Event Sentiment & Risk Analysis Pipeline
# =============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas_datareader import data as pdr

# =============================================================================
# 1. LOAD S&P 500 TICKERS
# =============================================================================

def load_sp500_constituents():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


# =============================================================================
# 2. LOAD EARNINGS DATES
# =============================================================================

def load_earnings_dates(tickers, limit=10):
    earnings_events = []

    for ticker in tickers[:limit]:
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.get_earnings_dates(limit=8)
            if earnings is not None:
                earnings = earnings.reset_index()
                earnings["ticker"] = ticker
                earnings_events.append(earnings)
        except Exception:
            continue

    if earnings_events:
        df = pd.concat(earnings_events, ignore_index=True)
        df["date"] = pd.to_datetime(df["Earnings Date"]).dt.date
        return df[["ticker", "date"]]

    return pd.DataFrame(columns=["ticker", "date"])


# =============================================================================
# 3. PRICE DATA
# =============================================================================

def load_price_data(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True
    )["Close"]

    returns = data.pct_change().dropna()

    return data, returns


# =============================================================================
# 4. FAMA-FRENCH 6 FACTORS
# =============================================================================

def load_ff_factors(start, end):
    ff5 = pdr.DataReader(
        "F-F_Research_Data_5_Factors_2x3_daily",
        "famafrench"
    )[0]

    mom = pdr.DataReader(
        "F-F_Momentum_Factor_daily",
        "famafrench"
    )[0]

    factors = ff5.join(mom, how="inner")
    factors.index = pd.to_datetime(factors.index)
    factors = factors.loc[start:end]
    factors = factors / 100.0

    return factors


# =============================================================================
# 5. ABNORMAL RETURNS (Rolling FF6)
# =============================================================================

def compute_abnormal_returns(returns, factors, estimation_window=252):
    aligned = returns.join(factors, how="inner")
    abnormal = []

    for ticker in returns.columns:
        df = aligned[[ticker]].join(factors, how="inner").dropna()
        df = df.rename(columns={ticker: "ret"})
        df["excess_ret"] = df["ret"] - df["RF"]

        for i in range(estimation_window, len(df)):
            window = df.iloc[i-estimation_window:i]

            y = window["excess_ret"]
            X = window[["Mkt-RF","SMB","HML","RMW","CMA","Mom   "]]
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            x_next = df.iloc[i][["Mkt-RF","SMB","HML","RMW","CMA","Mom   "]]
            x_next = sm.add_constant(x_next)

            expected = model.predict(x_next)
            actual = df.iloc[i]["excess_ret"]

            abnormal.append({
                "date": df.index[i].date(),
                "ticker": ticker,
                "abnormal_return": actual - expected.iloc[0]
            })

    return pd.DataFrame(abnormal)


# =============================================================================
# 6. RISK METRICS
# =============================================================================

def compute_risk_metrics(returns, window=5):
    risk_data = []

    for ticker in returns.columns:
        series = returns[ticker].dropna()

        for i in range(len(series) - window):
            forward = series.iloc[i+1:i+1+window]

            realised_vol = np.std(forward)
            downside = np.std(forward[forward < 0]) if (forward < 0).any() else 0

            risk_data.append({
                "date": series.index[i].date(),
                "ticker": ticker,
                "realised_vol_5d": realised_vol,
                "downside_vol_5d": downside
            })

    return pd.DataFrame(risk_data)


# =============================================================================
# 7. BUILD EVENT PANEL
# =============================================================================

def build_event_panel(earnings, abnormal_df, risk_df, sentiment_df):
    panel = earnings.merge(
        abnormal_df, on=["ticker","date"], how="left"
    ).merge(
        risk_df, on=["ticker","date"], how="left"
    ).merge(
        sentiment_df, on=["ticker","date"], how="left"
    )

    return panel


# =============================================================================
# 8. PANEL REGRESSION (Firm FE + Clustered SE)
# =============================================================================

def run_panel_regression(panel_df):
    panel_df = panel_df.dropna()

    panel_df["ticker_fe"] = panel_df["ticker"].astype("category")

    model_ret = smf.ols(
        formula="abnormal_return ~ sentiment + C(ticker_fe)",
        data=panel_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel_df["ticker"]}
    )

    model_risk = smf.ols(
        formula="realised_vol_5d ~ abs(sentiment) + C(ticker_fe)",
        data=panel_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel_df["ticker"]}
    )

    return model_ret, model_risk


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    print("Initialising pipeline...")

    tickers = load_sp500_constituents()[:10]

    earnings = load_earnings_dates(tickers)

    start = "2018-01-01"
    end   = "2024-01-01"

    prices, returns = load_price_data(tickers, start, end)
    factors = load_ff_factors(start, end)

    abnormal = compute_abnormal_returns(returns, factors)
    risk     = compute_risk_metrics(returns)

    # Synthetic sentiment for immediate working pipeline
    sentiment_df = pd.DataFrame({
        "ticker": abnormal["ticker"],
        "date": abnormal["date"],
        "sentiment": np.random.normal(0,1,len(abnormal))
    })

    panel = build_event_panel(earnings, abnormal, risk, sentiment_df)

    if len(panel) > 0:
        model_ret, model_risk = run_panel_regression(panel)

        print(model_ret.summary())
        print(model_risk.summary())
    else:
        print("No panel data constructed.")

if __name__ == "__main__":
    main()
