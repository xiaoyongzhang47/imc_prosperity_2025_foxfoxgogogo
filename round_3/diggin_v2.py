import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import time

# --- Data Loading Functions ---
def get_df(day):
    file_name = f"./round-3-island-data-bottle/prices_round_3_day_{day}.csv"
    return pd.read_csv(file_name, sep=';')

def get_product(df, product):
    return df[df['product'] == product].copy()

def get_first_two_dfs():
    first_df = get_df(1)
    second_df = get_df(2)
    second_df['timestamp'] = second_df['timestamp'] + 1000000
    return pd.concat([first_df, second_df])

# --- Black-Scholes and Greeks Functions ---
def black_scholes_call(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot / strike) + 0.5 * volatility ** 2 * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    return spot * norm.cdf(d1) - strike * norm.cdf(d2)

def delta(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot) - np.log(strike) + 0.5 * volatility ** 2 * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.cdf(d1)

def gamma(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot) - np.log(strike) + 0.5 * volatility ** 2 * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))

def vega(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot) - np.log(strike) + 0.5 * volatility ** 2 * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.pdf(d1) * (spot * np.sqrt(time_to_expiry)) / 100

def implied_volatility(call_price, spot, strike, time_to_expiry):
    def equation(vol):
        return black_scholes_call(spot, strike, time_to_expiry, vol) - call_price
    try:
        iv = brentq(equation, 1e-10, 3.0, xtol=1e-10)
    except ValueError:
        iv = np.nan
    return iv

# --- (Optional) Realized Volatility Calculation ---
def realized_vol(df, window, step_size):
    ret_col = f'log_return_{step_size}'
    vol_col = f'realized_vol_{step_size}'
    df[ret_col] = np.log(df['mid_price_volcanic_rock'] / df['mid_price_volcanic_rock'].shift(step_size))
    dt = step_size / (7 * 10000)
    df[vol_col] = df[ret_col].rolling(window=window).apply(lambda x: np.mean(x[::step_size] ** 2) / dt)
    df[vol_col] = np.sqrt(df[vol_col])
    return df

# --- Main Processing ---
df = get_first_two_dfs()
df_volcanic_rock = get_product(df, 'VOLCANIC_ROCK')



# List of all 5 voucher products
voucher_products = [
    "VOLCANIC_ROCK_VOUCHER_10000",
    "VOLCANIC_ROCK_VOUCHER_10250",
    "VOLCANIC_ROCK_VOUCHER_10500",
    "VOLCANIC_ROCK_VOUCHER_9500",
    "VOLCANIC_ROCK_VOUCHER_9750"
]

# Common parameters (adjust per product if needed)
strike_price = { 
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
    "VOLCANIC_ROCK_VOUCHER_9500" :  9500,
    "VOLCANIC_ROCK_VOUCHER_9750" :  9750
                }    # Can be customized per voucher if required


time_to_expiry = 7   
# in number of rounds

# Dictionaries to store processed DataFrames and summary stats
voucher_data = {}
summary = {}

# Process each voucher: merge with underlying data and compute greeks
for voucher in voucher_products:
    df_voucher = get_product(df, voucher)
    # Merge with underlying's mid_price using timestamp
    df_voucher = df_voucher.merge(
        df_volcanic_rock[['timestamp', 'mid_price']],
        on='timestamp',
        suffixes=('', '_volcanic_rock')
    )
    # Compute implied volatility and greeks (delta, gamma, vega)
    df_voucher['implied_vol'] = df_voucher.apply(
        lambda row: implied_volatility(row['mid_price'], row['mid_price_volcanic_rock'], strike_price[voucher], time_to_expiry),
        axis=1
    )
    df_voucher['delta'] = df_voucher.apply(
        lambda row: delta(row['mid_price_volcanic_rock'], strike_price[voucher], time_to_expiry, row['implied_vol']),
        axis=1
    )
    df_voucher['gamma'] = df_voucher.apply(
        lambda row: gamma(row['mid_price_volcanic_rock'], strike_price[voucher], time_to_expiry, row['implied_vol']),
        axis=1
    )
    df_voucher['vega'] = df_voucher.apply(
        lambda row: vega(row['mid_price_volcanic_rock'], strike_price[voucher], time_to_expiry, row['implied_vol']),
        axis=1
    )
    # Collect summary statistics
    mean_vol = df_voucher['implied_vol'].mean()
    std_vol = df_voucher['implied_vol'].std()
    summary[voucher] = {'mean_implied_vol': mean_vol, 'std_implied_vol': std_vol}
    voucher_data[voucher] = df_voucher

df_summary = pd.DataFrame.from_dict(summary, orient='index')
df_summary.to_csv("summary_parameters.csv", index=True)

# Output summary statistics for each voucher
for voucher, stats in summary.items():
    print(f"{voucher}: Mean Implied Vol = {stats['mean_implied_vol']:.6f}, Std = {stats['std_implied_vol']:.6f}")

# --- Backtesting Simulation (Optional) ---
def backtest_voucher(df_backtest, threshold_mult=1.0):
    # Set thresholds based on implied vol std
    upper_threshold = threshold_mult * df_backtest['implied_vol'].std()
    lower_threshold = -upper_threshold
    implied_vol_mean = df_backtest['implied_vol'].mean()
    position = 0
    pnl = 0
    trade_history = []
    
    for idx, row in df_backtest.iterrows():
        if idx == 0:
            continue
        implied_vol = row['implied_vol']
        mid_price_coupon = row['mid_price']
        mid_price_underlying = row['mid_price_volcanic_rock']
        d = row['delta']
        
        # Entry conditions (open position if no current exposure)
        if implied_vol > implied_vol_mean + upper_threshold and position == 0:
            position = -1
            entry_price_coupon = mid_price_coupon
            entry_price_underlying = mid_price_underlying
            trade_history.append(("Sell", entry_price_coupon, entry_price_underlying, implied_vol))
        elif implied_vol < implied_vol_mean + lower_threshold and position == 0:
            position = 1
            entry_price_coupon = mid_price_coupon
            entry_price_underlying = mid_price_underlying
            trade_history.append(("Buy", entry_price_coupon, entry_price_underlying, implied_vol))
        # Exit condition: price reverts toward the mean
        elif abs(implied_vol - implied_vol_mean) <= upper_threshold and position != 0:
            pnl += position * (mid_price_coupon - entry_price_coupon + d * (entry_price_underlying - mid_price_underlying))
            trade_history.append(("Close", mid_price_coupon, mid_price_underlying, implied_vol))
            position = 0
    
    if position != 0:
        pnl += position * (mid_price_coupon - entry_price_coupon + d * (entry_price_underlying - mid_price_underlying))
        
    return trade_history, pnl

# Run backtesting for each voucher and print trade history and PnL
do_backtest=True
if do_backtest:
    for voucher, df_voucher in voucher_data.items():
        # Rename columns for clarity in backtest: coupon price (option price) and underlying price
        df_backtest = df_voucher[['timestamp', 'mid_price', 'mid_price_volcanic_rock', 'implied_vol', 'delta', 'vega']].rename(
            columns={'mid_price': 'mid_price'}
        )
        trades, final_pnl = backtest_voucher(df_backtest)
        
        print(f"\nBacktest for {voucher}: Final PnL = {final_pnl}")
        print("Trade History:")
        for trade in trades:
            print(trade)


time.sleep(3)