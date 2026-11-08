# Baldeh Scanner

A new Streamlit-based market research scanner inspired by the Baldeh_Forex_scanner Pro Max concept.

## Features
- SPY and QQQ market benchmarks
- 1-hour, 4-hour, and daily trend views
- 5 SMA, 10 SMA, 21 SMA, and 20 HMA
- Sector, theme ETF, and curated individual-stock relative-strength rankings
- Separate relative-strength measurements versus SPY and QQQ
- Green/bullish, red/bearish, and yellow/mixed trend labels
- CSV downloads
- Mobile-friendly Streamlit layout
- No email alerts, SMTP credentials, or email workflow

## Run locally
Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a new GitHub repository, for example `baldeh-scanner`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. In Streamlit Community Cloud, create a new app and select the repository.
4. Set **Main file path** to `app.py`.
5. Deploy.

No secrets are required for this initial Yahoo Finance version.

## Data notes
Yahoo Finance intraday availability and history are limited and can change. The app uses 60-minute data to construct four-hour bars; the final bar may be incomplete. Daily rankings use adjusted daily closes. Always check freshness and data sufficiency before relying on signals.

## Research disclaimer
For research and educational use only. Not financial advice or a trading system.
