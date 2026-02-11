import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime



#for TLT vs Raffles
def fetch_and_rebase(ticker, start_date):
    data = yf.download(ticker, start=start_date, end='2026-01-01', interval='1mo', auto_adjust=False)['Adj Close']
    data = data.dropna()
    rebased_data = (data / data.iloc[0]) * 100
    rebased_data.drop(data.index[0], inplace=True)  
    return rebased_data

TICKERS = ['TLT', 'LQD', 'JNK', 'EMB', 'EMLC']
if __name__ == "__main__":

    for t in TICKERS:
        dt= fetch_and_rebase(t, '2025-01-01')
        dt = pd.DataFrame(dt)
        percent_return = (dt[t].iloc[-1]-dt[t].iloc[0])/dt[t].iloc[0]*100
        print(f'{t} percent return: {percent_return:.2f}%')

    
    
    



    



