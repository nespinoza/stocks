import yfinance as yf

def get_stock_data(tickers, period, interval):
    """
    Parameters

    tickers : list
        List of tickers, where each list of the element is a string with a ticker. For example, it can be 
        `tickers = ['FB', 'MSFT']`.

    period : string
        This needs to be a period of time since the current date on which you want to look at the data for 
        the tickers. Its format is `NumberLetter`, where `Number` is an integer and `Letter` is any of 
        `m` (minutes), `h` (hours), `d` (days), `wk` (weeks) and `mo` (months). For example, `1m` will get you 
        data a minute ago.

    interval : string
        Same as `period`, but for interval on needed data.

    """

    output = {}

    for ticker in tickers:

        output[ticker] = yf.download(tickers=ticker, period=period, interval=interval)

    return output
