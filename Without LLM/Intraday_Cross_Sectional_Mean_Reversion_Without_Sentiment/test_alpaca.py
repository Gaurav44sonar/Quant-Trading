import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load your keys from the .env file
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Initialize the Trading Client (paper=True uses fake money)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# 1. Check your simulated account balance
account = trading_client.get_account()
print(f"Paper Trading Balance: ${account.cash}")

# 2. Prepare a market order to buy 10 shares of Apple
market_order_data = MarketOrderRequest(
    symbol="AAPL",
    qty=10,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

# 3. Submit the order
print("Submitting order for 10 shares of AAPL...")
market_order = trading_client.submit_order(order_data=market_order_data)
print(f"Order Status: {market_order.status}")
