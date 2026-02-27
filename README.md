# Earnings Event Sentiment & Risk Analysis

Event-based sentiment analysis of S&P 500 earnings announcements using transformer models, testing the impact of sentiment on next-day abnormal returns and short-term risk dynamics.

---

## Research Objective

This project investigates whether earnings announcement sentiment contains predictive information about:

- Next-day abnormal returns (Fama-French 6-factor adjusted)
- Post-announcement realised volatility
- Downside risk exposure

The framework combines NLP with asset pricing and panel econometrics.

---

## Methodology

### Event Study Design
- Universe: S&P 500 constituents
- Estimation window: 252 trading days
- Event window: t+1 abnormal return
- Risk window: 5-day realised volatility

### Asset Pricing Model
- Fama-French 5 Factors + Momentum (daily)
- Rolling estimation for firm-specific betas
- Abnormal return = excess return − expected return

### Risk Metrics
- Forward 5-day realised volatility
- Downside semivariance

### Econometric Specification
- Firm fixed-effects panel regression
- Clustered standard errors (by firm)

---

## Architecture

- Modular pipeline (`src/pipeline.py`)
- Automated data ingestion (prices, factors, earnings)
- Rolling factor model estimation
- Event-level panel construction
- Panel regression with clustered inference

---

## Current Status

- End-to-end pipeline implemented
- Abnormal return and risk modules operational
- Panel regression functional
- Synthetic sentiment currently used for testing

Next step: integrate finance-tuned transformer model for real earnings sentiment extraction.

---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Execute:
   python src/pipeline.py
