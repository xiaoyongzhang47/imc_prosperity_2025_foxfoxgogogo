import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import time

# --- Data Loading Functions ---
def get_df(day: int) -> pd.DataFrame:
    file_name = f"./round-3-island-data-bottle/prices_round_3_day_{day}.csv"
    return pd.read_csv(file_name, sep=';')

def get_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    return df[df['product'] == product].copy()

def get_combined_data() -> pd.DataFrame:
    # Concatenate day 1 and day 2 (with adjusted timestamps)
    df1 = get_df(1)
    df2 = get_df(2)
    df2['timestamp'] += 1_000_000
    return pd.concat([df1, df2], ignore_index=True)

# --- Black‐Scholes and Implied‐Vol Functions ---
def black_scholes_call(S, K, t, vol):
    d1 = (np.log(S / K) + 0.5 * vol**2 * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    return S * norm.cdf(d1) - K * norm.cdf(d2)

def implied_volatility(C, S, K, t):
    # Brent‐q root‑find, fallback to NaN on failure
    def f(vol): return black_scholes_call(S, K, t, vol) - C
    try:
        return brentq(f, 1e-8, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan

# --- Backtest Strategy: volatility z‑score bands ---
def backtest_zscore(df, mean_iv, std_iv, threshold_mult):
    upper = mean_iv + threshold_mult * std_iv
    lower = mean_iv - threshold_mult * std_iv
    position = 0
    pnl = 0.0
    entry = None

    for _, row in df.iterrows():
        iv = row['implied_vol']
        opt_mid = row['mid_price']
        under_mid = row['mid_price_volcanic_rock']
        delta = row['delta']

        if position == 0:
            if iv > upper:
                position = -1
                entry = (opt_mid, under_mid, delta)
            elif iv < lower:
                position = 1
                entry = (opt_mid, under_mid, delta)
        else:
            if lower <= iv <= upper:
                e_opt, e_under, e_delta = entry
                pnl += position * (opt_mid - e_opt + e_delta * (e_under - under_mid))
                position = 0
                entry = None

    # Close any open position at last price
    if position != 0 and entry is not None:
        opt_mid, under_mid, delta = entry
        pnl += position * (opt_mid - opt_mid + delta * (under_mid - under_mid))

    return pnl

# --- Calibration Script ---
if __name__ == "__main__":
    # Load data
    df_all = get_combined_data()
    df_under = get_product(df_all, 'VOLCANIC_ROCK')

    voucher_list = [
        "VOLCANIC_ROCK_VOUCHER_10000",
        "VOLCANIC_ROCK_VOUCHER_10250",
        "VOLCANIC_ROCK_VOUCHER_10500",
        "VOLCANIC_ROCK_VOUCHER_9500",
        "VOLCANIC_ROCK_VOUCHER_9750"
    ]
    strikes = {
        "VOLCANIC_ROCK_VOUCHER_10000": 10000,
        "VOLCANIC_ROCK_VOUCHER_10250": 10250,
        "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        "VOLCANIC_ROCK_VOUCHER_9500": 9500,
        "VOLCANIC_ROCK_VOUCHER_9750": 9750
    }
    tte = 7  # time‐to‐expiry in same units as timestamps

    results = []
    for voucher in voucher_list:
        # Merge with underlying mid price
        df_v = get_product(df_all, voucher)
        df_v = df_v.merge(
            df_under[['timestamp', 'mid_price']],
            on='timestamp',
            suffixes=('', '_volcanic_rock')
        ).dropna()

        # Compute implied vol and delta
        df_v['implied_vol'] = df_v.apply(
            lambda r: implied_volatility(r['mid_price'],
                                         r['mid_price_volcanic_rock'],
                                         strikes[voucher], tte),
            axis=1
        )
        df_v['delta'] = df_v.apply(
            lambda r: norm.cdf((np.log(r['mid_price_volcanic_rock']/strikes[voucher])+
                                0.5*r['implied_vol']**2*tte)/
                               (r['implied_vol']*np.sqrt(tte))),
            axis=1
        )
        df_v.dropna(subset=['implied_vol'], inplace=True)

        # Fit expected‐vol quadratic: iv ≈ a*m^2 + b
        m = np.log(strikes[voucher] / df_v['mid_price_volcanic_rock']) / np.sqrt(tte)
        Y = df_v['implied_vol'].values
        X = np.vstack([m**2, np.ones_like(m)]).T
        a, b = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Optimize z‐score multiplier
        mean_iv = df_v['implied_vol'].mean()
        std_iv  = df_v['implied_vol'].std()
        best_pnl, best_thresh = -np.inf, None
        for mult in np.linspace(0.5, 3.0, 51):
            pnl = backtest_zscore(df_v, mean_iv, std_iv, mult)
            if pnl > best_pnl:
                best_pnl, best_thresh = pnl, mult

        results.append({
            'voucher': voucher,
            'a': a, 'b': b,
            'mean_iv': mean_iv,
            'std_iv': std_iv,
            'best_z_mult': best_thresh,
            'best_pnl': best_pnl
        })

    # Display & save results
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv("calibration_results.csv", index=False)

    time.sleep(1)