import requests
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Ensure nltk resources are downloaded
nltk.download('vader_lexicon')

# API Configuration
API_KEY = 'y9P6GQRglZUVHvqWSdnBnXVqduutrLkf'
BASE_URL = 'https://financialmodelingprep.com/api/v3/'

def fetch_data(symbol, statement_type):
    """
    Fetch financial data from Financial Modeling Prep API.
    """
    url = f"{BASE_URL}{statement_type}/{symbol}?apikey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Oops! Something went wrong while fetching {statement_type} data for {symbol}. Error: {e}")
        return None

def get_financial_statements(symbol):
    """
    Get all financial statements for a given symbol.
    """
    print(f"Fetching financial statements for {symbol}...")
    income_statement = fetch_data(symbol, 'income-statement')
    balance_sheet = fetch_data(symbol, 'balance-sheet-statement')
    cash_flow = fetch_data(symbol, 'cash-flow-statement')
    
    return income_statement, balance_sheet, cash_flow

def generate_signal(metric_name, metric_q1, metric_q2, symbol, statement_type):
    """
    Generate SIGNAL based on a metric's trend between two quarters.
    """
    trend = "increase" if metric_q2 > metric_q1 else "decrease"
    summary = f"{symbol} {statement_type} {metric_name} {trend} from Q1 to Q2."
    
    sentiment = "Buy" if trend == "increase" else "Sell"
    rating = 8 if sentiment == "Buy" else 6
    timeframe = "1Q"
    
    return {
        "summary": summary,
        "symbol": symbol,
        "topics": [statement_type, metric_name],
        "sentiment": sentiment,
        "rating": rating,
        "timeframe": timeframe
    }

def process_data(symbol, income_statement, balance_sheet, cash_flow):
    """
    Process the financial data to generate SIGNALs for each metric.
    """
    signals = []
    
    # Example: Net income (Income Statement)
    if income_statement and len(income_statement) >= 2:
        net_income_q1 = income_statement[0].get("netIncome", 0)
        net_income_q2 = income_statement[1].get("netIncome", 0)
        signals.append(generate_signal("net income", net_income_q1, net_income_q2, symbol, "income statement"))
    
    # Example: Total assets (Balance Sheet)
    if balance_sheet and len(balance_sheet) >= 2:
        total_assets_q1 = balance_sheet[0].get("totalAssets", 0)
        total_assets_q2 = balance_sheet[1].get("totalAssets", 0)
        signals.append(generate_signal("total assets", total_assets_q1, total_assets_q2, symbol, "balance sheet"))
    
    # Example: Operating cash flow (Cash Flow Statement)
    if cash_flow and len(cash_flow) >= 2:
        operating_cash_flow_q1 = cash_flow[0].get("operatingCashFlow", 0)
        operating_cash_flow_q2 = cash_flow[1].get("operatingCashFlow", 0)
        signals.append(generate_signal("operating cash flow", operating_cash_flow_q1, operating_cash_flow_q2, symbol, "cash flow"))
    
    return signals

def expansion_ideas():
    """
    Propose innovative ideas for expanding or scaling the solution.
    """
    return [
        "Use machine learning models to predict trends based on historical data.",
        "Integrate real-time market data for more dynamic and up-to-date SIGNALs.",
        "Use sentiment analysis on earnings calls transcripts for deeper insights."
    ]

def machine_learning_trend_prediction(symbol, data):
    """
    Implement machine learning to predict future trends based on historical data.
    """
    data = pd.DataFrame(data)
    
    # Check if there's sufficient data for prediction
    if len(data) < 2:
        print(f"Not enough data available for {symbol} to make a prediction.")
        return None  # Not enough data to make a prediction
    
    X = np.array(range(len(data))).reshape(-1, 1)  # Time (quarter index)
    y = data['value'].values  # Metric values
    
    # Scale the data for linear regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Predict the next quarter's value
    next_quarter = np.array([[len(data)]]).reshape(1, -1)
    next_quarter_scaled = scaler.transform(next_quarter)
    prediction = model.predict(next_quarter_scaled)
    
    return prediction[0]

def real_time_market_data(symbol):
    """
    Fetch real-time market data (e.g., stock price) for enhanced SIGNALs.
    """
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None
    except requests.exceptions.RequestException as e:
        print(f"Sorry, there was an issue fetching real-time data for {symbol}. Error: {e}")
        return None

def sentiment_analysis_earnings_call(symbol):
    """
    Perform sentiment analysis on earnings call transcripts for deeper insights.
    """
    # Fetch earnings call transcripts (sample URL)
    url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?apikey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return "Sorry, no transcripts available for the earnings call."
        
        transcript_text = " ".join([entry['content'] for entry in data])
        
        # Perform sentiment analysis
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(transcript_text)
        
        if sentiment['compound'] > 0:
            return "The sentiment of the earnings call is Positive."
        elif sentiment['compound'] < 0:
            return "The sentiment of the earnings call is Negative."
        else:
            return "The sentiment of the earnings call is Neutral."
    except requests.exceptions.RequestException as e:
        print(f"Oops! There was an error fetching the earnings call transcript for {symbol}. Error: {e}")
        return "Error fetching earnings call transcripts."

def main(symbol):
    """
    Main function to fetch data, process it, and generate SIGNALs.
    """
    print(f"\nProcessing data for {symbol}...\n")
    
    income_statement, balance_sheet, cash_flow = get_financial_statements(symbol)
    
    if not income_statement or not balance_sheet or not cash_flow:
        print("Sorry, we encountered an issue retrieving some data. Please try again later.")
        return
    
    # Process the financial data to generate SIGNALs
    signals = process_data(symbol, income_statement, balance_sheet, cash_flow)
    
    # Implement machine learning trend prediction (Example for Net Income)
    data = [{"value": entry.get("netIncome", 0)} for entry in income_statement[:2]]
    prediction = machine_learning_trend_prediction(symbol, data)
    if prediction is not None:
        print(f"Predicted next quarter net income: {prediction}")
    
    # Integrate real-time market data
    market_data = real_time_market_data(symbol)
    if market_data:
        print(f"Current stock price for {symbol}: {market_data['price']}")
    
    # Perform sentiment analysis on earnings call transcript
    sentiment = sentiment_analysis_earnings_call(symbol)
    print(f"Earnings Call Sentiment: {sentiment}")
    
    # Output the signals to a JSON file
    with open(f"{symbol}_signals.json", "w") as f:
        json.dump(signals, f, indent=4)
    
    # Output signals to console
    for signal in signals:
        print(json.dumps(signal, indent=4))
    
    # Print expansion ideas
    ideas = expansion_ideas()
    print("\nExpansion Ideas: Let's explore how we can make this solution even better:")
    for idea in ideas:
        print(f"- {idea}")

if __name__ == "__main__":
    symbol = input("Enter the stock symbol (e.g., AAPL): ").strip().upper()
    main(symbol)
