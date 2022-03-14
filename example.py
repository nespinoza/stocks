import matplotlib.pyplot as plt
import utils

# Get stock data from Facebook (FB), Microsoft (MSFT), Apple (AAPL), Amazon (AMZN) and Google (GOOG and GOOGL):
tickers = ['FB', 'MSFT', 'AAPL', 'AMZN', 'GOOG', 'GOOGL']

# Extract data for last 10 years, each week:
data = utils.get_stock_data(tickers, period = '120mo', interval = '1wk')

# Get plot of weekly open value for all:
for ticker in tickers:

    ticker_data = data[ticker]
    plt.plot(ticker_data.index, ticker_data['Open'].values, label = ticker)

plt.legend()
plt.show()

# Now same plot, but normalizing by value of the stock on first week:

# Get plot of weekly open value for all:
for ticker in tickers:

    ticker_data = data[ticker]
    plt.plot(ticker_data.index, ticker_data['Open'].values / ticker_data['Open'].values[0], label = ticker)

plt.legend()
plt.show()
