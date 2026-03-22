import yfinance as yf
import pandas as pd

def load_data(tickers, period="2y"):
    dfs = []
    
    for ticker in tickers:
        df = yf.download(ticker, period=period)
        df = df[['Close']]
        df.columns = [ticker]
        dfs.append(df)
    
    combined = pd.concat(dfs, axis=1)
    combined.dropna(inplace=True)
    
    return combined