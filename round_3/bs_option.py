import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import time 

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

df = get_first_two_dfs()


df_volcanic_rock = get_product(df, 'VOLCANIC_ROCK')
df_volcanic_rock_call = get_product(df, "VOLCANIC_ROCK_VOUCHER_10000")
df_volcanic_rock_call = df_volcanic_rock_call.merge(df_volcanic_rock[['timestamp', 'mid_price']], on='timestamp', suffixes=('', '_volcanic_rock'))

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_call(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    call_price = (spot * norm.cdf(d1) - strike * norm.cdf(d2))
    return call_price

def black_scholes_put(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot / strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    put_price = (strike * norm.cdf(-d2) - spot * norm.cdf(-d1))
    return put_price

def delta(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot) - np.log(strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.cdf(d1)

def gamma(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot) - np.log(strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.pdf(d1)/(spot * volatility * np.sqrt(time_to_expiry))

def vega(spot, strike, time_to_expiry, volatility):
    d1 = (np.log(spot) - np.log(strike) + (0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    return norm.pdf(d1) * (spot * np.sqrt(time_to_expiry)) / 100

def implied_volatility(call_price, spot, strike, time_to_expiry):
    # Define the equation where the root is the implied volatility
    def equation(volatility):
        estimated_price = black_scholes_call(spot, strike, time_to_expiry, volatility)
        return estimated_price - call_price

    # Using Brent's method to find the root of the equation
    implied_vol = brentq(equation, 1e-10, 3.0, xtol=1e-10)
    return implied_vol

def realized_vol(df_volcanic_rock_call, window, step_size):
    df_volcanic_rock_call[f'log_return_{step_size}'] = np.log(df_volcanic_rock_call['mid_price_volcanic_rock'].to_numpy()/df_volcanic_rock_call['mid_price_volcanic_rock'].shift(step_size).to_numpy())
    dt = step_size / 250 / 10000 
    df_volcanic_rock_call[f'realized_vol_{step_size}'] = df_volcanic_rock_call[f'log_return_{step_size}'].rolling(window=window).apply(lambda x: np.mean(x[::step_size]**2) / dt)
    df_volcanic_rock_call[f'realized_vol_{step_size}'] = np.sqrt(df_volcanic_rock_call[f'realized_vol_{step_size}'].to_numpy())
    return df_volcanic_rock_call



spot_price =   10000       # Spot price of the underlying asset
strike_price = 10000       # Strike price of the option
call_price = 637.5         # Market price of the call option
time_to_expiry = 7         # Time to expiry in years
initial_guess = 16
df_volcanic_rock_call['implied_vol'] = df_volcanic_rock_call.apply(lambda row: implied_volatility(row['mid_price'], row['mid_price_volcanic_rock'], strike_price, time_to_expiry), axis=1)
df_volcanic_rock_call['delta'] = df_volcanic_rock_call.apply(lambda row: delta(row['mid_price_volcanic_rock'], strike_price, time_to_expiry, row['implied_vol']), axis=1)
df_volcanic_rock_call['gamma'] = df_volcanic_rock_call.apply(lambda row: gamma(row['mid_price_volcanic_rock'], strike_price, time_to_expiry, row['implied_vol']), axis=1)
df_volcanic_rock_call['vega'] = df_volcanic_rock_call.apply(lambda row: vega(row['mid_price_volcanic_rock'], strike_price, time_to_expiry, row['implied_vol']), axis=1)

day = 1
df = get_df(day)

strike_price = 10000  
time_to_expiry = 7
initial_guess = 16

df_volcanic_rock = get_product(df, 'VOLCANIC_ROCK')
df_volcanic_rock_call = get_product(df, "VOLCANIC_ROCK_VOUCHER_10000")
df_volcanic_rock_call = df_volcanic_rock_call.merge(df_volcanic_rock[['timestamp', 'mid_price']], on='timestamp', suffixes=('', '_volcanic_rock'))

df_volcanic_rock_call['implied_vol'] = df_volcanic_rock_call.apply(lambda row: implied_volatility(row['mid_price'], row['mid_price_volcanic_rock'], strike_price, time_to_expiry), axis=1)
df_volcanic_rock_call['delta'] = df_volcanic_rock_call.apply(lambda row: delta(row['mid_price_volcanic_rock'], strike_price, time_to_expiry, row['implied_vol']), axis=1)
df_volcanic_rock_call['gamma'] = df_volcanic_rock_call.apply(lambda row: gamma(row['mid_price_volcanic_rock'], strike_price, time_to_expiry, row['implied_vol']), axis=1)

df_volcanic_rock_call['vega'] = df_volcanic_rock_call.apply(lambda row: vega(row['mid_price_volcanic_rock'], strike_price, time_to_expiry, row['implied_vol']), axis=1)

df_backtest = df_volcanic_rock_call[['timestamp', 'mid_price', 'mid_price_volcanic_rock', 'implied_vol', 'delta', 'vega']]
df_backtest = df_backtest.rename(columns={'mid_price': 'mid_price_coupon'})

implied_vol_mean = df_backtest['implied_vol'].mean()

th = df_backtest['implied_vol'].std()

print(implied_vol_mean, th)

print()
time.sleep(10)




import pandas as pd

# Set the threshold values
upper_threshold = th # Threshold for selling option
lower_threshold = -th  # Threshold for buying option
close_threshold = th  # Threshold for clearing position

# Initialize variables
position = 0
pnl = 0
vega_pnl = 0
trade_history = []

# Iterate over each row in the dataframe
for idx, row in df_backtest.iterrows():
    implied_vol = row['implied_vol']
    if idx == 0:
        continue
    prev_implied_vol = df_backtest.iloc[idx-1]['implied_vol']
    mid_price_coupon = row['mid_price_coupon']
    mid_price_volcanic_rock = row['mid_price_volcanic_rock']
    vega = row['vega']
    d = row['delta']

    # Check if implied vol is above the upper threshold and no current position
    if implied_vol > implied_vol_mean + upper_threshold and position == 0:
        # Sell 1 delta hedged option
        position = -1
        entry_price_coupon = mid_price_coupon
        entry_price_volcanic_rock = mid_price_volcanic_rock
        trade_history.append((-1, entry_price_coupon, entry_price_volcanic_rock, implied_vol))

    # Check if implied vol is below the lower threshold and no current position
    elif implied_vol < implied_vol_mean + lower_threshold and position == 0:
        # Buy 1 delta hedged option
        position = 1
        entry_price_coupon = mid_price_coupon
        entry_price_volcanic_rock = mid_price_volcanic_rock
        trade_history.append((1, entry_price_coupon, entry_price_volcanic_rock, implied_vol))

    # Check if implied vol is within the close threshold and there is a current position
    elif abs(implied_vol - implied_vol_mean) <= close_threshold and position != 0:
        # Clear the position
        pnl += position * (mid_price_coupon - entry_price_coupon + d * (entry_price_volcanic_rock - mid_price_volcanic_rock))
        position = 0
        trade_history.append((0, mid_price_coupon, mid_price_volcanic_rock, implied_vol))

    if position != 0:
        vega_pnl += position * vega * (implied_vol - prev_implied_vol) * 100
# Calculate final PnL if there is still an open position
if position != 0:
    pnl += position * (mid_price_coupon - entry_price_coupon + d * (entry_price_volcanic_rock - mid_price_volcanic_rock))

# Print the trade history and final PnL
print("Trade History:")
for trade in trade_history:
    print(f"Position: {trade[0]}, Option Price: {trade[1]}, Underlying Price: {trade[2]}, Implied Volatility: {trade[3]}")

print(f"\nFinal PnL: {pnl}")


import pandas as pd

# Set the threshold values
upper_threshold = 0.001  # Threshold for selling option
lower_threshold = -0.001  # Threshold for buying option

# Initialize variables
position = 0
pnl = 0
trade_history = []

# Iterate over each row in the dataframe
for _, row in df_backtest.iterrows():
    implied_vol = row['implied_vol']
    mid_price_coupon = row['mid_price_coupon']
    mid_price_volcanic_rock = row['mid_price_volcanic_rock']
    d = row['delta']

    # Check if implied vol is above the upper threshold
    if implied_vol > implied_vol_mean + upper_threshold:
        # Sell to target position of -1
        if position > -1:
            quantity = -1 - position
            position = -1
            entry_price_coupon = mid_price_coupon
            entry_price_volcanic_rock = mid_price_volcanic_rock
            trade_history.append((quantity, entry_price_coupon, entry_price_volcanic_rock, implied_vol))

    # Check if implied vol is below the lower threshold
    elif implied_vol < implied_vol_mean + lower_threshold:
        # Buy to target position of 1
        if position < 1:
            quantity = 1 - position
            position = 1
            entry_price_coupon = mid_price_coupon
            entry_price_volcanic_rock = mid_price_volcanic_rock
            trade_history.append((quantity, entry_price_coupon, entry_price_volcanic_rock, implied_vol))

# Calculate final PnL for the remaining position
if position != 0:
    pnl += position * (mid_price_coupon - entry_price_coupon + d * (entry_price_volcanic_rock - mid_price_volcanic_rock))

# Print the trade history and final PnL
print("Trade History:")
for trade in trade_history:
    print(f"Quantity: {trade[0]}, Option Price: {trade[1]}, Underlying Price: {trade[2]}, Implied Volatility: {trade[3]}")

print(f"\nFinal PnL: {pnl}")