import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

# --- Your initial data loading and mid-price calculation ---
MERC_NAME = 'SQUID_INK'
day = -1

market_data = pd.read_csv(f"./round-1-island-data-bottle/prices_round_1_day_{day}.csv", sep=";", header=0)
trade_history = pd.read_csv(f"./round-1-island-data-bottle/trades_round_1_day_{day}.csv", sep=";", header=0)

merc_data = market_data[market_data['product'] == MERC_NAME].reset_index(drop=True)

def calculate_mm_mid(row):
    # Look for the best bid and ask with a minimum required volume.
    for i in range(1, 4):
        if row[f'bid_volume_{i}'] >= 15:
            best_bid = row[f'bid_price_{i}']
            break
    else:
        best_bid = None

    for i in range(1, 4):
        if row[f'ask_volume_{i}'] >= 15:
            best_ask = row[f'ask_price_{i}']
            break
    else:
        best_ask = None

    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    else:
        return None

merc_data['mm_mid'] = merc_data.apply(calculate_mm_mid, axis=1)
merc_fair_prices = merc_data[['timestamp', 'mm_mid']].rename(columns={'mm_mid': 'fair'})

merc_fair_prices = merc_fair_prices.sort_values('timestamp').reset_index(drop=True)


WINDOW_LENGTH = 40
FORWARD_WINDOW = 3


def compute_slope(series):
    """
    Computes the slope for a given 1-dimensional array of data.
    Uses np.polyfit to perform a simple linear regression.
    """
   
    x = np.arange(len(series))
    slope, _ = np.polyfit(x, series, 1)
    return slope

# --- Compute the moving slopes ---

slope_past = []    # slope [t - WINDOW_LENGTH, t)
slope_future = []  # slope [t, t + FORWARD_WINDOW)
timestamps  = []   # associated time for t (could be taken as the current row's time)

# Loop only over indices that can have a complete past and future window.
valid_indices = range(WINDOW_LENGTH, len(merc_fair_prices) - FORWARD_WINDOW)

for t in tqdm(valid_indices, desc="Computing slopes"):

    window_past = merc_fair_prices['fair'].iloc[t - WINDOW_LENGTH:t].dropna()
    window_future = merc_fair_prices['fair'].iloc[t:t + FORWARD_WINDOW].dropna()
    
    # Check that both windows contain the required number of data points.
    if len(window_past) < WINDOW_LENGTH or len(window_future) < FORWARD_WINDOW:
        continue

    past_slope = compute_slope(window_past)
    future_slope = compute_slope(window_future)

    slope_past.append(past_slope)
    slope_future.append(future_slope)
    timestamps.append(merc_fair_prices['timestamp'].iloc[t])

# Now create the DataFrame. All lists have the same length.
df_slopes = pd.DataFrame({
    'timestamp': timestamps,
    'slope_past': slope_past,
    'slope_future': slope_future
})



print(df_slopes.head())

# # --- Regression analysis: How does the past slope predict the near-future slope? ---
X = df_slopes['slope_past'].values.reshape(-1, 1)
y = df_slopes['slope_future'].values

# Create and train a linear regression model without an intercept (as in your original code)
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

predictions = model.predict(X)
train_r2 = r2_score(y, predictions)
train_mse = mean_squared_error(y, predictions)

print("Learned Equation:")
print("slope_future = {:.4f} * slope_past".format(model.coef_[0]))
print("R-squared:", train_r2)
print("MSE:", train_mse)

# --- Plot the relationship ---
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data Points')

# Generate a smooth line for the regression
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_range)

plt.plot(x_range, y_line, color='red', linewidth=2, label='Best Fit Line')
plt.xlabel('Past Slope (window = {})'.format(WINDOW_LENGTH))
plt.ylabel('Future Slope (window = {})'.format(FORWARD_WINDOW))
plt.title("Relationship between Past and Future Slopes")
plt.legend()
plt.show()