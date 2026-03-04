import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from plotly.subplots import make_subplots
import feedparser
from textblob import TextBlob
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image
import base64
from io import BytesIO

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="SMART_MARKET_AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply gradient background and custom text styles
st.markdown(
    """
    <style>
    /* Page background gradient */
    .stApp {
        
        color: white;
        font-family: 'Arial', sans-serif;
    }

    /* Header text styling */
    .stMarkdown h1 {
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: 2px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
    }

    .stMarkdown h4 {
        font-size: 22px;
        text-align: center;
        color: #cddc39;
        text-shadow: 1px 1px 6px rgba(0,0,0,0.6);
        margin-top: 0;
    }

    /* Buttons styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 25px;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
    }

   
    </style>
    """,
    unsafe_allow_html=True
)

# Header Text
st.markdown(
    """
    <h1>📊 SMART_MARKET_AI</h1>
    <h4>Intelligent Stock Market Prediction & Investment Insights</h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# ================================
# SIDEBAR MENU
# ==========================================
# Initialize default section in session_state
# ==========================================
if "section" not in st.session_state:
    st.session_state.section = "Stock Prediction"

# ==========================================
# Function to create styled buttons
# ==========================================
def sidebar_button(name, display_name):
    if st.sidebar.button(display_name, key=name):
        st.session_state.section = name

# ==========================================
# Sidebar Buttons (same size, attractive)
# ==========================================
st.sidebar.write("### Choose your section")

button_style = """
<style>
div.stButton > button {
    width: 220px;
    height: 50px;
    margin: 8px 0px;
    background-color: #2C3E50;
    color: #ECF0F1;
    font-size: 16px;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    text-align: left;
    padding-left: 15px;
}
div.stButton > button:hover {
    background-color: #34495E;
}
</style>
"""
st.sidebar.markdown(button_style, unsafe_allow_html=True)

# Sidebar buttons
if st.sidebar.button("📈 Stock Prediction"):
    st.session_state.section = "Stock Prediction"
if st.sidebar.button("💼 Portfolio Optimization"):
    st.session_state.section = "Portfolio Optimization"
if st.sidebar.button("📰 AI Market News"):
    st.session_state.section = "AI Market News"
if st.sidebar.button("💰 SIP Planner"):
    st.session_state.section = "SIP Planner"
if st.sidebar.button("🤖 AI Chat Assistant"):
    st.session_state.section = "AI Chat Assistant"

# Use the selected section
section = st.session_state.section

# st.write(f"Current Section: {section}")  # for testing


# ======================================================
# 1️⃣ STOCK PREDICTION MODULE
# ======================================================
if section == "Stock Prediction":

    st.header("📈 Smart Stock Prediction")

    stock_options = {
         
    # IT Sector
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro": "WIPRO.NS",
    "HCLTech": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",

    # Banking Sector
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Bank": "KOTAKBANK.NS",

    # FMCG Sector
    "HUL": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia": "BRITANNIA.NS",

    # Auto Sector
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "M&M.NS",

    # Infra / Energy
    "Reliance": "RELIANCE.NS",
    "Larsen & Toubro": "LT.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",

    # Finance / NBFC
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",

    # Telecom
    "Bharti Airtel": "BHARTIARTL.NS",

    # Pharma
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",

    # Adani Group
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS"
}
    
    
    selected_stock = st.selectbox("Select a Stock", list(stock_options.keys()))
    ticker = stock_options[selected_stock]

    if st.button("Analyze Stock"):

        # Download Data
        data = yf.download(ticker, period="2y", auto_adjust=False)

        if data.empty:
            st.error("No data found")
            st.stop()

        # Fix MultiIndex Issue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Open','High','Low','Close']].dropna()

        # Moving Averages
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()

        # Latest Values
        latest = data.iloc[-1]
        open_price = float(latest['Open'])
        high_price = float(latest['High'])
        low_price = float(latest['Low'])
        close_price = float(latest['Close'])

        # Simple Prediction Logic
        last_7_avg = data['Close'].tail(7).mean()
        last_30_avg = data['Close'].tail(30).mean()
        trend = last_7_avg - last_30_avg
        predicted_price = close_price + trend
        # ======================================================
        # 📉 CHART
        # ======================================================
        st.subheader("📉 Price Trend")

        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            name="MA20"
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            name="MA50"
        ))

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # ======================================================
        # 📊 ONE FRAME DASHBOARD (OHLC + Prediction + Suggestion)
        # ======================================================
        with st.container():

            st.subheader("📊 Market Summary & AI Insight")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Open", f"₹ {open_price:.2f}")
            col2.metric("High", f"₹ {high_price:.2f}")
            col3.metric("Low", f"₹ {low_price:.2f}")
            col4.metric("Close", f"₹ {close_price:.2f}")

            st.divider()

            col5, col6 = st.columns(2)
            col5.metric("Predicted Next Price",
                        f"₹ {predicted_price:.2f}",
                        f"{predicted_price-close_price:.2f}")

            if predicted_price > close_price:
                col6.success("🟢 BUY → Uptrend Detected")
            elif predicted_price < close_price:
                col6.error("🔴 SELL → Downtrend Detected")
            else:
                col6.warning("🟡 HOLD → Sideways Market")

            st.caption("Prediction based on short-term vs long-term trend analysis.")

        

# ======================================================
# 2️⃣ PORTFOLIO OPTIMIZATION MODULE
# ======================================================
elif section == "Portfolio Optimization":
    st.write("You are in Portfolio Optimization Module")

    st.header("💼 Portfolio Optimization")
    df = pd.read_csv("stocks.csv")

    # Remove unwanted spaces in column names
    df.columns = df.columns.str.strip()

    stocks = dict(zip(df["StockName"], df["Symbol"]))

    

    selected = st.multiselect(
        "Select Stocks",
        list(stocks.keys()),
        default=["Reliance", "TCS", "Infosys"]
    )

    # 🔹 Investment Frequency Option
    frequency = st.selectbox(
        "Investment Frequency",
        ["One-Time", "Monthly", "Yearly", "Weekly"]
    )

    investment_amount = st.number_input("Investment Amount (₹)", value=10000)
    years = st.slider("Investment Duration (Years)", 1, 20, 5)

    if st.button("Analyze Portfolio"):

        symbols = [stocks[s] for s in selected]
        data = yf.download(symbols, period="3y")["Close"]

        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Random Weights
        weights = np.random.random(len(symbols))
        weights /= np.sum(weights)

        portfolio_return = float(np.sum(mean_returns * weights))
        portfolio_volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

        # ---------------- Investment Logic ----------------
        total_periods = years

        if frequency == "Monthly":
            total_periods = years * 12
            rate = portfolio_return / 12
        elif frequency == "Weekly":
            total_periods = years * 52
            rate = portfolio_return / 52
        elif frequency == "Yearly":
            rate = portfolio_return
        else:  # One-Time
            rate = portfolio_return

        # Future Value Calculation
        if frequency == "One-Time":
            total_invested = investment_amount
            future_value = investment_amount * ((1 + rate) ** years)
        else:
            total_invested = investment_amount * total_periods
            future_value = investment_amount * (
                ((1 + rate) ** total_periods - 1) / rate
            ) * (1 + rate)

        profit = future_value - total_invested

        # ---------------- Results ----------------
        st.subheader("📊 Portfolio Results")

        st.write(f"💰 Total Amount Invested: ₹ {total_invested:,.0f}")
        st.write(f"📈 Expected Return: {portfolio_return*100:.2f}%")
        st.write(f"⚖ Risk Level: {portfolio_volatility*100:.2f}%")
        st.write(f"🏦 Future Value: ₹ {future_value:,.0f}")
        st.write(f"🔥 Total Profit Earned: ₹ {profit:,.0f}")

        # Risk Category
        if portfolio_return > 0.18:
            st.success("🚀 High Return Portfolio (Aggressive)")
        elif portfolio_return > 0.12:
            st.info("👍 Balanced Portfolio")
        else:
            st.warning("🛡 Safe Portfolio (Low Risk)")

        # ---------------- Growth Graph ----------------
        invested_list = []
        portfolio_list = []

        for t in range(1, total_periods + 1):

            if frequency == "One-Time":
                invested = investment_amount
                value = investment_amount * ((1 + rate) ** (t / total_periods * years))
            else:
                invested = investment_amount * t
                value = investment_amount * (
                    ((1 + rate) ** t - 1) / rate
                ) * (1 + rate)

            invested_list.append(invested)
            portfolio_list.append(value)

        graph_df = pd.DataFrame({
            "Period": range(1, total_periods + 1),
            "Invested Amount": invested_list,
            "Portfolio Value": portfolio_list
        })

        st.line_chart(graph_df.set_index("Period"))
# ======================================================
# 4️⃣ AI MARKET NEWS (ADVANCED VERSION)
# ======================================================
elif section == "AI Market News":
    st.write("You are in AI Market News Module")

    st.header("📰 AI Powered Market News & Decision Guide")

    import feedparser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    st.write("Fetching real-time financial news...")

    feed = feedparser.parse(
        "https://news.google.com/rss/search?q=stock+market+india&hl=en-IN&gl=IN&ceid=IN:en"
    )

    market_scores = []
    news_count = 0

    # -----------------------------
    # SHOW NEWS + SENTIMENT
    # -----------------------------
    for entry in feed.entries[:7]:

        title = entry.title
        link = entry.link

        sentiment = analyzer.polarity_scores(title)
        score = sentiment["compound"]

        market_scores.append(score)
        news_count += 1

        if score > 0.05:
            label = "🟢 Positive"
        elif score < -0.05:
            label = "🔴 Negative"
        else:
            label = "🟡 Neutral"

        st.subheader(title)
        st.write(f"Sentiment: {label}")
        st.markdown(f"[🔗 Read Full Article]({link})")
        st.divider()

    # -----------------------------
    # MARKET CONFIDENCE SCORE
    # -----------------------------
    if market_scores:

        avg_score = sum(market_scores) / news_count
        market_confidence = int((avg_score + 1) * 50)

        st.subheader("📊 Market Confidence Level")
        st.progress(market_confidence)

        st.info(f"AI Market Confidence Score: {market_confidence}/100")

        # -----------------------------
        # CLEAR DECISION GUIDE
        # -----------------------------
        st.subheader("🧭 CLEAR DECISION GUIDE")

        if market_confidence > 70:
            st.success("""
🟢 **Market Condition: Bullish**

✔ Good time for long-term investors  
✔ SIP investment recommended  
✔ Growth stocks may perform well  
✔ Can increase allocation gradually  
""")

        elif market_confidence > 40:
            st.warning("""
🟡 **Market Condition: Neutral / Uncertain**

✔ Invest slowly using SIP  
✔ Avoid investing full capital at once  
✔ Focus on strong companies  
✔ Monitor market weekly  
""")

        else:
            st.error("""
🔴 **Market Condition: Bearish / Risky**

✔ Avoid lump-sum investment  
✔ Continue small SIP only  
✔ Keep cash reserve  
✔ High risk for short-term trading  
""")

        # -----------------------------
        # INVESTOR TYPE GUIDE
        # -----------------------------
        st.subheader("👤 Investor Type Recommendation")

        col1, col2, col3 = st.columns(3)

        col1.success("""
**Long-Term Investor**
Best strategy: SIP + Hold
Ignore short-term volatility.
""")

        col2.warning("""
**Beginner Investor**
Learn first.
Invest small amount.
Avoid risky stocks.
""")

        col3.error("""
**Short-Term Trader**
Market is unpredictable.
High risk currently.
""")
# ======================================================
# 6️⃣ SIP PLANNER
# ======================================================
elif section == "SIP Planner":
    st.write("You are in SIP Planner Module")

    st.title("📊 Professional SIP Planner (Real-Time)")

    import yfinance as yf
    import pandas as pd
    import numpy as np

    # Stock List
    df = pd.read_csv("stocks.csv")

    # Remove unwanted spaces in column names
    df.columns = df.columns.str.strip()

    stocks = dict(zip(df["StockName"], df["Symbol"]))

    stock_name = st.selectbox("Select Stock", list(stocks.keys()))
    ticker = stocks[stock_name]

    stock = yf.Ticker(ticker)

    # Fetch 5 Year Data
    data = stock.history(period="5y")

    if not data.empty:

        # Current Price
        current_price = data["Close"].iloc[-1]
        st.success(f"Current Price: ₹ {round(current_price,2)}")

        # CAGR Calculation
        start_price = data["Close"].iloc[0]
        end_price = data["Close"].iloc[-1]
        years_data = 5

        cagr = ((end_price / start_price) ** (1/years_data) - 1) * 100
        cagr = round(cagr, 2)

        st.info(f"📈 Past 5 Year Average Annual Return (CAGR): {cagr}%")

        # User Input
        st.subheader("Enter SIP Details")

        monthly_investment = st.number_input("Monthly Investment (₹)", min_value=500, step=500)
        years = st.slider("Investment Duration (Years)", 1, 30, 5)

        if st.button("Calculate Professional SIP"):

            months = years * 12
            monthly_rate = cagr / 100 / 12

            future_value = monthly_investment * (
                ((1 + monthly_rate) ** months - 1) / monthly_rate
            ) * (1 + monthly_rate)

            total_investment = monthly_investment * months
            profit = future_value - total_investment

            st.subheader("📊 SIP Result")
            st.write(f"Total Investment: ₹ {round(total_investment,2)}")
            st.write(f"Estimated Future Value: ₹ {round(future_value,2)}")
            st.write(f"Estimated Profit: ₹ {round(profit,2)}")

            # Growth Chart
            values = []
            for m in range(1, months+1):
                value = monthly_investment * (
                    ((1 + monthly_rate) ** m - 1) / monthly_rate
                ) * (1 + monthly_rate)
                values.append(value)

            df = pd.DataFrame({
                "Month": range(1, months+1),
                "Investment Value": values
            })

            st.line_chart(df.set_index("Month"))

    else:
        st.error("Unable to fetch stock data.")

# ======================================================
# 🤖 FREE AI CHAT ASSISTANT (NO API)
# ======================================================
elif section == "AI Chat Assistant":
    st.write("You are in AI Chat Assistant Module")

    st.header("🤖 Smart Investment Assistant (Free AI)")

    st.write(
        "This assistant gives investment guidance using market logic. "
        "No internet or paid AI required."
    )

    user_question = st.text_input("Ask your question (example: Should I invest now?)")

    if user_question:

        q = user_question.lower()

        st.subheader("💬 AI Response")

        # ---------------------------
        # INVEST NOW
        # ---------------------------
        if "invest" in q or "buy" in q:
            st.success("""
📌 AI Advice:

✔ Invest slowly, not all at once  
✔ Prefer SIP over lump sum  
✔ Strong companies are safer  
✔ Market ups & downs are normal
""")

        # ---------------------------
        # RISK
        # ---------------------------
        elif "risk" in q:
            st.warning("""
📌 Risk Explanation (Simple):

🟢 Low Risk → Stable big companies  
🟡 Medium Risk → Growing companies  
🔴 High Risk → Fast price movement  

Beginners should avoid high risk.
""")

        # ---------------------------
        # MARKET CONDITION
        # ---------------------------
        elif "market" in q or "crash" in q:
            st.info("""
📌 Market Status:

✔ Market is uncertain  
✔ Avoid emotional buying  
✔ Learn and observe  
✔ Best time to invest is gradually
""")

        # ---------------------------
        # SIP
        # ---------------------------
        elif "sip" in q:
            st.success("""
📌 SIP Explained:

SIP = Invest fixed amount every month  
✔ Reduces risk  
✔ Builds discipline  
✔ Best for beginners
""")

        # ---------------------------
        # LONG TERM
        # ---------------------------
        elif "long" in q:
            st.success("""
📌 Long-Term Investing:

✔ Hold stocks for 5–10 years  
✔ Ignore short-term market noise  
✔ Quality companies grow with time
""")

        # ---------------------------
        # DEFAULT
        # ---------------------------
        else:
            st.info("""
I can help you with:

• Should I invest now?
• What is risk?
• What is SIP?
• Market condition
• Long-term investing
""")

    st.caption("This is an educational AI assistant for beginners.")