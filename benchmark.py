import os
import yfinance as yf
import pandas as pd

"""
Repeat for 3 pairs (TLT vs Raffles), (LQD & JNK vs Raffles), (EMB & EMLC vs Raffles)
Step 1. Fetch historical monthly stock data using yfinance
Step 2. Rebase data to index at 100 for easier comparison
Step 3. Aggregate data into one dataframe (Raffles and benchmarks(s)), pull Raffles quarterly returns from raffles.csv
Step 4. Put csv into YEAR-MONTH, RAFFLES, TICKER* format. Save as CSV into final_data folder.
"""

def fetch_and_rebase_many(tickers, start_date):
    """
    Returns a monthly, month-end indexed DataFrame with each ticker rebased to 100.
    Aligns series to the common start (first month where ALL tickers have data),
    so the comparison is apples-to-apples.
    """
    adj = yf.download(tickers, start=start_date, interval='1mo', auto_adjust=False)['Adj Close'].dropna(how='all')

    if isinstance(adj, pd.Series):
        adj = adj.to_frame(tickers if isinstance(tickers, str) else tickers[0])

    adj.index = pd.to_datetime(adj.index).to_period('M').to_timestamp('M')
    adj = adj.sort_index()

    cols = [c for c in adj.columns if adj[c].notna().any()]
    adj = adj[cols]

    common_start_idx = adj.dropna().index.min()
    adj = adj.loc[common_start_idx:]

    base = adj.iloc[0]
    rebased = (adj / base) * 100.0
    return rebased


def read_raffles_monthly(raffles_csv, fill_inside_quarter=True):
    """
    Expects columns: YEAR-MONTH, Raffles.
    Converts to month-end index and (optionally) forward-fills inside quarters
    so you get a monthly line for plotting.
    """
    df = pd.read_csv(raffles_csv)
    if not {'YEAR-MONTH', 'Raffles'}.issubset(df.columns):
        raise ValueError(f"Expected columns YEAR-MONTH and Raffles in {raffles_csv}")

    dates = pd.to_datetime(df['YEAR-MONTH'], errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    raff = pd.Series(df['Raffles'].values, index=dates, name='Raffles').sort_index()

    if fill_inside_quarter:
        raff = raff.ffill()

    return raff


def build_csv_with_benchmarks(bench_tickers, start_date, raffles_csv, out_csv,
                              fill_inside_quarter=True):
    """
    bench_tickers: list like ['LQD','JNK'] or ['EMB','EMLC'] or ['TLT']
    out_csv: path to write final CSV with columns: Date, Raffles, <ticker1>, <ticker2>, ...
    """
    bench = fetch_and_rebase_many(bench_tickers, start_date)
    raff = read_raffles_monthly(raffles_csv, fill_inside_quarter=fill_inside_quarter)

    last_raff_date = raff.dropna().index.max()
    bench = bench.loc[:last_raff_date]
    raff = raff.loc[bench.index.min():last_raff_date]

    final = pd.concat([raff, bench], axis=1).sort_index()
    final = final.reset_index().rename(columns={'index': 'YEAR-MONTH'})

    # ✅ Format all numeric columns to 2 decimal places
    final = final.astype({col: 'int' for col in final.columns if col != 'YEAR-MONTH'})

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    final.to_csv(out_csv, index=False, float_format="%.2f")
    return final


if __name__ == "__main__":
    # Example 1: TLT vs Raffles
    build_csv_with_benchmarks(
        bench_tickers=['TLT'],
        start_date='2016-01-01',
        raffles_csv='raffles.csv',
        out_csv='final_data/TLT_vs_Raffles.csv'
    )

    # Example 2: LQD & JNK vs Raffles
    build_csv_with_benchmarks(
        bench_tickers=['LQD','JNK'],
        start_date='2016-01-01',
        raffles_csv='raffles.csv',
        out_csv='final_data/LQD_JNK_vs_Raffles.csv'
    )

    # Example 3: EMB & EMLC vs Raffles
    build_csv_with_benchmarks(
        bench_tickers=['EMB','EMLC'],
        start_date='2016-01-01',
        raffles_csv='raffles.csv',
        out_csv='final_data/EMB_EMLC_vs_Raffles.csv'
    )
