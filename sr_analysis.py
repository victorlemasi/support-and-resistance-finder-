import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
import argparse
from datetime import datetime, timedelta

def get_data(symbol, period='6mo'):
    """Fetch historical data from Yahoo Finance with intelligent interval selection."""
    ticker = yf.Ticker(symbol)
    
    # Select appropriate interval based on period
    if period in ['1d', '5d']:
        interval = '15m'
    elif period in ['1mo', '3mo']:
        interval = '1h'
    else:
        interval = '1d'
        
    df = ticker.history(period=period, interval=interval)
    return df, interval

def find_sr_levels(df, distance=None, prominence_pct=0.1):
    """
    Identify support and resistance levels using local peaks and troughs.
    - prominence_pct: percentage of the total price range in the data
    """
    highs = df['High'].values
    lows = df['Low'].values
    
    # Calculate price range for adaptive prominence
    price_range = highs.max() - lows.min()
    prominence = price_range * prominence_pct
    
    # Calculate dynamic distance if not provided (approx 5% of total bars)
    if distance is None:
        distance = max(1, len(df) // 20)
    
    # Find resistance (peaks in Highs)
    res_idx, _ = find_peaks(highs, distance=distance, prominence=prominence)
    resistance_levels = highs[res_idx]
    
    # Find support (peaks in -Lows)
    supp_idx, _ = find_peaks(-lows, distance=distance, prominence=prominence)
    support_levels = lows[supp_idx]
    
    return resistance_levels, support_levels

def cluster_levels(levels, threshold_pct=0.015):
    """
    Group nearby levels together to find significant zones.
    Returns a list of (average_price, hit_count) sorted by price.
    """
    if len(levels) == 0:
        return []
    
    levels = sorted(levels)
    clustered = []
    current_cluster = [levels[0]]
    
    for i in range(1, len(levels)):
        avg = sum(current_cluster) / len(current_cluster)
        if (levels[i] - avg) / avg < threshold_pct:
            current_cluster.append(levels[i])
        else:
            clustered.append((sum(current_cluster) / len(current_cluster), len(current_cluster)))
            current_cluster = [levels[i]]
    
    clustered.append((sum(current_cluster) / len(current_cluster), len(current_cluster)))
    return clustered

def plot_sr(df, symbol, resistance, support, interval):
    """Create an interactive Plotly chart with SR levels."""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price Action')])

    # Add Resistance Lines (Red)
    for price, count in resistance:
        fig.add_hline(y=price, line_dash="dash", line_color="red", 
                      annotation_text=f"Res: {price:.2f} ({count}x)", annotation_position="top right")

    # Add Support Lines (Green)
    for price, count in support:
        fig.add_hline(y=price, line_dash="dash", line_color="green", 
                      annotation_text=f"Supp: {price:.2f} ({count}x)", annotation_position="bottom right")

    fig.update_layout(
        title=f'{symbol} SR Levels ({interval} interval)',
        yaxis_title='Price',
        xaxis_title='Date/Time',
        template='plotly_dark'
    )
    
    fig.show()

def main():
    parser = argparse.ArgumentParser(description='Support and Resistance Analysis')
    parser.add_argument('--symbol', type=str, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, help='Data period (e.g., 6mo, 1y)')
    
    args = parser.parse_args()
    
    symbol = args.symbol
    period = args.period
    
    # Prompt user if not provided via command line
    if not symbol:
        print("\n--- Support and Resistance Analyzer ---")
        print("Tip: Use 'GC=F' for Gold, 'BTC-USD' for Bitcoin, 'AAPL' for Apple.")
        symbol = input("Enter Stock Symbol: ").strip().upper()
        if not symbol:
            print("Error: Symbol is required.")
            return
            
    if not period:
        print("\nValid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
        period = input("Enter Period (default '6mo'): ").strip().lower()
        if not period:
            period = '6mo'
    
    print(f"\nFetching data for {symbol} ({period})...")
    try:
        df, interval = get_data(symbol, period=period)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    if df.empty:
        print(f"Error: No data found for {symbol}.")
        if "XAU" in symbol:
            print("Tip: For Gold, try 'GC=F' or 'XAUUSD=X'.")
        return

    print(f"Identifying levels using {interval} data points...")
    raw_res, raw_supp = find_sr_levels(df)
    
    print("Clustering levels...")
    res_levels = cluster_levels(raw_res)
    supp_levels = cluster_levels(raw_supp)
    
    print(f"\nFound {len(res_levels)} Resistance levels and {len(supp_levels)} Support levels.")
    
    current_price = df['Close'].iloc[-1]
    print(f"Current Price ({symbol}): {current_price:.2f}")

    print("\n" + "="*45)
    print(f"{'TYPE':<12} | {'PRICE':<10} | {'CONFIRMATIONS':<13}")
    print("-" * 45)
    
    # Resistance Table
    if res_levels:
        for price, count in sorted(res_levels, key=lambda x: x[0], reverse=True):
            print(f"{'Resistance':<12} | {price:<10.2f} | {count:<13}")
    else:
        print(f"{'Resistance':<12} | {'None':<10} | {'-'*13}")
        
    print("-" * 45)
    
    # Support Table
    if supp_levels:
        for price, count in sorted(supp_levels, key=lambda x: x[0], reverse=True):
            print(f"{'Support':<12} | {price:<10.2f} | {count:<13}")
    else:
        print(f"{'Support':<12} | {'None':<10} | {'-'*13}")
    print("="*45)

    # Market Summary Logic
    nearest_res = min([r[0] for r in res_levels], key=lambda x: abs(x - current_price)) if res_levels else None
    nearest_supp = min([s[0] for s in supp_levels], key=lambda x: abs(x - current_price)) if supp_levels else None
    
    print("\n>>> Market Summary:")
    if nearest_res:
        dist = ((nearest_res - current_price) / current_price) * 100
        print(f" - Nearest Resistance: {nearest_res:.2f} ({dist:+.2f}% from current)")
    if nearest_supp:
        dist = ((nearest_supp - current_price) / current_price) * 100
        print(f" - Nearest Support:    {nearest_supp:.2f} ({dist:+.2f}% from current)")
    
    if not nearest_res and not nearest_supp:
        print(" - No significant levels detected in this timeframe.")
    elif nearest_res and nearest_supp:
        range_pct = ((nearest_res - nearest_supp) / nearest_supp) * 100
        print(f" - Current Trading Range: {range_pct:.2f}%")
    
    print("\nOpening interactive chart...")
    plot_sr(df, symbol, res_levels, supp_levels, interval)

if __name__ == "__main__":
    main()
