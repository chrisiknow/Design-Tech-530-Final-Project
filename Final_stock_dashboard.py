import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from gemini_helper import summarize_stock

import yfinance as yf
YF_AVAILABLE = True

# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv("stocks_data.csv", parse_dates=["Date"], index_col="Date")
    return df

df = load_data()

# Page layout
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("Stock Prediction & AI Summary Dashboard")

st.write(
    "This dashboard uses a simple Linear Regression model on 5-day and 10-day "
    "moving averages to predict the **next-day closing price** for selected stocks "
    "(AAPL, NVDA, MS, JPM)."
)

st.sidebar.header("Controls")
ticker = st.sidebar.selectbox("Select a stock:", ["AAPL", "NVDA", "MS", "JPM"])
st.sidebar.write("Selected ticker:", ticker)

# Raw data preview
st.subheader("📊 Recent Data Snapshot")
st.dataframe(df.tail())

# Linear Regression model
data = df[[ticker]].copy()
data.rename(columns={ticker: "Close"}, inplace=True)

# Feature Engineering
data["SMA_5"] = data["Close"].rolling(window=5).mean()
data["SMA_10"] = data["Close"].rolling(window=10).mean()
data["Target"] = data["Close"].shift(-1)  # next day close price

data.dropna(inplace=True)

# Train/Test Split
split = int(len(data) * 0.8)
X_train = data[["SMA_5", "SMA_10"]].iloc[:split]
X_test = data[["SMA_5", "SMA_10"]].iloc[split:]
y_train = data["Target"].iloc[:split]
y_test = data["Target"].iloc[split:]

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

# Predict next-day closing price
latest_sma5 = data["SMA_5"].iloc[-1]
latest_sma10 = data["SMA_10"].iloc[-1]
next_input = pd.DataFrame([[latest_sma5, latest_sma10]], columns=["SMA_5", "SMA_10"])
next_day_pred = model.predict(next_input)[0]

# Metrics section
st.subheader(f"🔍 Model Results for {ticker}")

col1, col2, col3 = st.columns(3)
col1.metric("Next-Day Predicted Price", f"${next_day_pred:.2f}")
# Show RMSE as an investor-friendly average error in dollars
col2.metric("Average error (typical $ error)", f"${rmse:.2f}")
# Show R² as a simple reliability score
col3.metric("Model reliability (0-1, higher is better)", f"{r2:.3f}")

# Chart: Last 10 days actual vs predicted
st.subheader("Last 10 Days: Actual vs Predicted Close Price")

dates = y_test.index[-10:]
actual_last10 = y_test.values[-10:]
preds_last10 = preds[-10:]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dates, actual_last10, label="Actual", linewidth=2)
ax.plot(dates, preds_last10, label="Predicted", linestyle="--")
ax.set_title(f"{ticker} - Last 10 Days: Actual vs Predicted Close Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.caption(f"Next-day prediction (not on chart): **${next_day_pred:.2f}**")

# Statistics
st.subheader("📌 Historical Stats for Selected Stock")

desc = data["Close"].describe()
st.write(desc)

st.write(f"**Mean:** {data['Close'].mean():.2f}")
st.write(f"**Median:** {data['Close'].median():.2f}")
st.write(f"**Standard deviation:** {data['Close'].std():.2f}")

# Model Interpretation
st.subheader("Model Interpretation")

try:
    last_close = data["Close"].iloc[-1]
    price_change = next_day_pred - last_close
    pct_change = price_change / last_close if last_close != 0 else 0
    rmse_pct = (rmse / last_close) * 100 if last_close != 0 else 0

    # Confidence from R²
    if r2 >= 0.7:
        confidence = "high"
    elif r2 >= 0.4:
        confidence = "moderate"
    else:
        confidence = "low"

    # Volatility from coefficient of variation
    coef_var = data["Close"].std() / data["Close"].mean() if data["Close"].mean() != 0 else 0
    if coef_var < 0.01:
        volatility = "stable"
    elif coef_var < 0.03:
        volatility = "somewhat stable"
    else:
        volatility = "volatile"

    # Direction from percent change
    if pct_change > 0.01:
        direction = "rising"
    elif pct_change < -0.01:
        direction = "falling"
    else:
        direction = "flat"

    # Advice
    if confidence == "low":
        advice = (
            "The model's accuracy is low, so treat this prediction cautiously. "
            "Avoid making big moves based solely on this and consider combining it with other information."
        )
    else:
        if direction == "rising":
            if volatility == "stable":
                advice = "The model suggests a small short-term gain; a modest position or dollar-cost averaging could be reasonable."
            else:
                advice = "There may be upside, but price is volatile; smaller positions or staggered entries can help manage risk."
        elif direction == "falling":
            advice = "The model predicts a slight decline; waiting or using stop-losses may help limit downside."
        else:
            advice = "No clear short-term signal; holding or diversifying may be the safest choice for beginners."

    # BUY, SELL or HOLD
    if confidence == "low":
        signal = "HOLD"
    else:
        if direction == "rising" and pct_change > 0.01:
            signal = "BUY"
        elif direction == "falling" and pct_change < -0.01:
            signal = "SELL"
        else:
            signal = "HOLD"

    # Display interpretation
    st.markdown(f"- **Last close:** ${float(last_close):.2f}")
    st.markdown(f"- **Next-day prediction:** ${float(next_day_pred):.2f} ({pct_change*100:+.2f}% vs last close)")
    st.markdown(f"- **Average error (typical $ error):** ${float(rmse):.2f} ({rmse_pct:.2f}% of last close)")
    st.markdown(f"- **Model confidence:** `{str(confidence)}`")
    st.markdown(f"- **Historical volatility:** `{str(volatility)}`")
    st.markdown(f"- **Direction:** `{str(direction)}`")
    st.markdown(f"- **Suggested simple advice:** {str(advice)}")
    st.markdown(f"- **Actionable signal:** **{str(signal)}**")

except Exception as e:
    st.error(f"Model interpretation failed to compute: {e}")
    # Show some debugging info
    st.write("Debug: last few rows of data:")
    st.dataframe(data.tail())

# Fetch relevant news for the selected ticker by user
def fetch_news(ticker, top_n=5):
    articles = []
    # Will first try yfinance, if doesn't return articles will jump to RSS
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", []) or []
        for item in news[:top_n]:
            if isinstance(item, dict):
                title = (item.get("title") or "").strip()
                link = (item.get("link") or "").strip()
                publisher = (item.get("publisher") or "").strip()
                summary = (item.get("summary") or "").strip()
                # Capture publish time if available
                pubtime = item.get("providerPublishTime") or item.get("pubDate") or None
                if isinstance(pubtime, (int, float)):
                    try:
                        from datetime import datetime
                        pubtime = datetime.fromtimestamp(int(pubtime)).isoformat()
                    except Exception:
                        pubtime = str(pubtime)
                elif pubtime:
                    pubtime = str(pubtime)
                else:
                    pubtime = ""
            else:
                # defensive
                title = str(item).strip()
                link = ""
                publisher = ""
                summary = ""
                pubtime = ""
            articles.append({"title": title, "publisher": publisher, "link": link, "summary": summary, "time": pubtime})
    except Exception:
        articles = []

    # If yfinance returned nothing useful, try Yahoo RSS as fallback
    useful = any(a.get("title") for a in articles)
    if not useful:
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            r = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                import xml.etree.ElementTree as ET
                from datetime import datetime
                root = ET.fromstring(r.content)
                items = root.findall('.//item')
                articles = []
                for it in items[:top_n]:
                    title = it.findtext('title') or ''
                    link = it.findtext('link') or ''
                    desc = it.findtext('description') or ''
                    pubdate = it.findtext('pubDate') or ''
                    # Attempting to parse pubdate to ISO if present
                    try:
                        pubtime = ''
                        if pubdate:
                            pubtime = datetime.strptime(pubdate, '%a, %d %b %Y %H:%M:%S %Z').isoformat()
                    except Exception:
                        pubtime = pubdate
                    articles.append({
                        'title': title.strip(),
                        'publisher': '',
                        'link': link.strip(),
                        'summary': desc.strip(),
                        'time': pubtime
                    })
        except Exception:
            # Keep articles that are present (could possibly be empty)
            pass

    return articles

# Sidebar control for number of articles user wants to be presented with
top_n = st.sidebar.slider("Articles to fetch", 1, 8, 5)
news_list = fetch_news(ticker, top_n=top_n)

# Display Relevant News
st.subheader("Relevant News")
if news_list:
    st.write(f"Found {len(news_list)} articles.")
    for art in news_list:
        title = art.get("title") or "(no title)"
        link = art.get("link") or ""
        pub = art.get("publisher") or ""
        summary_snip = art.get("summary") or ""
        time_str = art.get("time") or ""

        # Summary to 220 chars
        max_len = 220
        if len(summary_snip) > max_len:
            summary_display = summary_snip[:max_len].rstrip() + "..."
        else:
            summary_display = summary_snip

        # Title as link that opens in new tab
        if link:
            # use HTML anchor to add target="_blank" and rel for safety
            st.markdown(f"<a href=\"{link}\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"font-weight:600;\">{title}</a>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{title}**")

        # Small metadata line
        meta_parts = []
        if pub:
            meta_parts.append(pub)
        if time_str:
            meta_parts.append(time_str)
        if meta_parts:
            st.markdown(f"<small>{' • '.join(meta_parts)}</small>", unsafe_allow_html=True)

        # Summary
        if summary_display:
            st.write(summary_display)

        # Open link text
        if link:
            st.markdown(f"<a href=\"{link}\" target=\"_blank\" rel=\"noopener noreferrer\">Open article</a>", unsafe_allow_html=True)
        st.write('')
else:
    st.info("No recent news found for this ticker.")
# News text for AI prompt
news_text = "\n".join([
    f"{i+1}. {a.get('title','')} ({a.get('publisher','')}) — {a.get('summary','')}. Link: {a.get('link','')}"
    for i, a in enumerate(news_list)
])

# AI short analysis by Gemini
st.subheader("🧠 AI Short Analysis (Gemini)")

if st.button("Generate AI Short Analysis"):
    if not news_list:
        st.warning("No articles available to summarize. Please fetch articles first or select a different ticker.")
    else:
        with st.spinner("Generating short analysis from articles..."):
            # Build a prompt that asks the model to summarize the provided articles only
            articles_text = "\n\n".join([f"Article {i+1}: {a.get('title','')} - {a.get('summary','')} (Link: {a.get('link','')})" for i, a in enumerate(news_list)])
            summary_prompt = f"""
You are a helpful assistant. Using ONLY the text of the news articles below, write a short, plain-language analysis for a beginner investor.
- Base your response strictly on the article content provided; do NOT use model metrics, historical prices, or any external data.
- Output exactly 3 to 5 plain sentences (no headings, no bullet lists).
- On a new final line after the sentences, include exactly one word: BUY, SELL, or HOLD.

Articles:
{articles_text}

Remember: use ONLY the article content above to form the analysis and the one-word action.
"""
            ai_response = summarize_stock(summary_prompt)
            ai_text = ai_response if isinstance(ai_response, str) else getattr(ai_response, "text", str(ai_response))
        st.markdown("**AI Short Analysis (from articles):**")
        st.write(ai_text)

# Longer-term Moving Averages section
st.markdown("---")
st.subheader("Longer-term Moving Averages")

# Compute 50 and 100 day simple moving averages
data["SMA_50"] = data["Close"].rolling(window=50).mean()
data["SMA_100"] = data["Close"].rolling(window=100).mean()

# Latest SMA values
latest_sma50 = data["SMA_50"].iloc[-1]
latest_sma100 = data["SMA_100"].iloc[-1]

col_a, col_b, col_c = st.columns(3)
col_a.metric("50-day average", f"${latest_sma50:.2f}" if not pd.isna(latest_sma50) else "N/A")
col_b.metric("100-day average", f"${latest_sma100:.2f}" if not pd.isna(latest_sma100) else "N/A")
# Simple status
price_status = ""
try:
    if not pd.isna(latest_sma50) and not pd.isna(latest_sma100):
        if data["Close"].iloc[-1] > latest_sma50 and data["Close"].iloc[-1] > latest_sma100:
            price_status = "Price is above both 50-day and 100-day averages — generally positive longer-term trend."
        elif data["Close"].iloc[-1] < latest_sma50 and data["Close"].iloc[-1] < latest_sma100:
            price_status = "Price is below both averages — generally negative longer-term trend."
        else:
            price_status = "Price sits between the 50-day and 100-day averages — mixed longer-term signals."
    else:
        price_status = "Not enough historical data to compute 50/100-day averages."
except Exception:
    price_status = "Unable to determine price vs averages."

st.write(price_status)