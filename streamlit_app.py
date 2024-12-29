import streamlit as st
import requests

# App Title
st.title("Rabiotic Trading Strategy Bot")

# Input Fields
license_key = st.text_input("License Key", "")
login = st.text_input("MT5/MT4 Login", "")
password = st.text_input("Password", type="password")
server = st.text_input("Server", "")
symbol = st.selectbox("Market Symbol", [
    "Volatility 10 Index", "Volatility 25 Index", "Volatility 50 Index",
    "Volatility 75 Index", "Volatility 100 Index", "Crash 1000 Index",
    "Crash 500 Index", "Boom 1000 Index", "Boom 500 Index", "Step Index",
    "Range Break 100 Index", "Range Break 200 Index", "EURUSD", "GBPUSD",
    "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY"
])
lot_size = st.number_input("Lot Size", min_value=0.01, value=0.01, step=0.01)
action = st.selectbox("Action", ["Buy", "Sell", "Both"])
take_profit = st.number_input("Take Profit (TP)", min_value=0.0, value=0.0, step=0.1)
stop_loss = st.number_input("Stop Loss (SL)", min_value=0.0, value=0.0, step=0.1)

# Submit Button
if st.button("Submit & Trade"):
    if not (license_key and login and password and server and symbol and lot_size and action and take_profit and stop_loss):
        st.error("Please fill all fields.")
    else:
        # Payload to send to the backend
        payload = {
            "license_key": license_key,
            "login": login,
            "password": password,
            "server": server,
            "symbol": symbol,
            "lot": lot_size,
            "action": action,
            "take_profit": take_profit,
            "stop_loss": stop_loss
        }

        # Replace with your actual backend API URL
        api_url = "https://your-backend-api-url.com/place_trade"

        try:
            response = requests.post(api_url, json=payload)
            if response.ok:
                st.success("Trade executed successfully!")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to the server: {e}")
