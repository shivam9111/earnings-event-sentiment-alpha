# Earnings Event Sentiment & Risk Analysis

Event-based sentiment analysis of S&P 500 earnings announcements using a finance-tuned DeBERTa transformer model, testing the impact of sentiment on next-day abnormal returns and short-term risk dynamics.

---

## Research Objective

This project investigates whether earnings announcement sentiment contains predictive information about:

- Next-day abnormal returns  
- Post-announcement volatility  
- Downside risk exposure  

The study focuses on US equities (S&P 500 constituents) and constructs an event-based empirical pipeline combining NLP and asset pricing techniques.

---

## Methodology

### 1. Event Window Construction
- Identify earnings announcement dates.
- Define event window (t=0 announcement day, t+1 return horizon).

### 2. Sentiment Extraction
- Apply finance-tuned DeBERTa transformer model.
- Aggregate sentiment scores across headlines/transcripts.
- Construct continuous sentiment measure.

### 3. Abnormal Return Estimation
- Market-adjusted returns or CAPM-based abnormal returns.
- Cross-sectional analysis of sentiment vs abnormal returns.

### 4. Risk Analysis
- Realised volatility post-event.
- Downside semivariance.
- Conditional return dispersion.

---

## Architecture

- Modular pipeline (`src/pipeline.py`)
- Transformer-based NLP inference
- Financial econometrics module
- Event-study framework

---

## Expected Outputs

- Event-level sentiment scores  
- Abnormal return estimates  
- Risk metric panel dataset  
- Summary regression tables  
- Visualisations of sentiment-return relationship  

---

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Execute:
   python src/pipeline.py
