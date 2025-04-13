import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm


MERC_NAME = 'CROISSANTS'
CUT_OFFs = 85
day = 1



market_data = pd.read_csv(f"./round-2-island-data-bottle/prices_round_2_day_{day}.csv", sep=";", header=0)
trade_history = pd.read_csv(f"./round-2-island-data-bottle/trades_round_2_day_{day}.csv", sep=";", header=0)


merc_data = market_data[market_data['product'] == MERC_NAME].reset_index(drop=True)

def calculate_mm_mid(row):
    # Find the best bid with volume >= 20
    for i in range(1, 4):
        if row[f'bid_volume_{i}'] >= 15:
            best_bid = row[f'bid_price_{i}']
            break
    else:
        best_bid = None

    # Find the best ask with volume >= 20
    for i in range(1, 4):
        if row[f'ask_volume_{i}'] >= 15:
            best_ask = row[f'ask_price_{i}']
            break
    else:
        best_ask = None

    # Calculate the mid price if both best bid and ask are found
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    else:
        return None

merc_data['mm_mid'] = merc_data.apply(calculate_mm_mid, axis=1)

print(merc_data['mm_mid'])

merc_fair_prices = merc_data[['timestamp', 'mm_mid']]
merc_fair_prices = merc_fair_prices.rename(columns={'mm_mid': 'fair'})
iteration_counts = [1,2,5,10,50,100,500] 

for iterations in iteration_counts:
    merc_fair_prices[f"fair_in_{iterations}_its"] = merc_fair_prices['fair'].shift(-iterations)
    merc_fair_prices[f"fair_{iterations}_its_ago"] = merc_fair_prices['fair'].shift(iterations)


for iterations in iteration_counts:
    merc_fair_prices[f'returns_in_{iterations}_its'] = (merc_fair_prices[f'fair_in_{iterations}_its'] - merc_fair_prices['fair'])/merc_fair_prices['fair']

    merc_fair_prices[f'returns_from_{iterations}_its_ago'] = (merc_fair_prices['fair'] - merc_fair_prices[f'fair_{iterations}_its_ago'])/merc_fair_prices[f'fair_{iterations}_its_ago']


row_names = ['timestamp','fair']

for iterations in iteration_counts:
    row_names.append(f'returns_in_{iterations}_its')
    row_names.append(f'returns_from_{iterations}_its_ago')
    
merc_returns = merc_fair_prices[row_names]
merc_returns= merc_returns.dropna()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

# Perform train-test split
train_data, test_data = train_test_split(merc_returns, test_size=0.2, random_state=42)

# Iterate over each iteration count
for iterations in tqdm(iteration_counts):

    
    # Prepare the feature and target columns
    X_train = train_data[f'returns_from_{iterations}_its_ago'].values.reshape(-1, 1)
    y_train = train_data[f'returns_in_{iterations}_its']
    X_test = test_data[f'returns_from_{iterations}_its_ago'].values.reshape(-1, 1)
    y_test = test_data[f'returns_in_{iterations}_its']

    # Create and train the linear regression model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    # Make predictions on train and test data
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate R-squared and MSE for train and test data
    train_r2 = r2_score(y_train, train_predictions)
    train_mse = mean_squared_error(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    # Print the results
    print(f"Iteration Count: {iterations}")
    print(f"Learned Equation: returns_in_{iterations}_its = {model.coef_[0]:.4f} * returns_from_{iterations}_its_ago")
    print(f"Train R-squared: {train_r2:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test R-squared: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print()

for iteration in iteration_counts:

    # Prepare training data for the chosen iteration count
    X_train = train_data[f'returns_from_{iteration}_its_ago'].values.reshape(-1, 1)
    y_train = train_data[f'returns_in_{iteration}_its'].values

    # Create and train the linear regression model (without an intercept)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    # Generate a range of x values for plotting the best fit line
    x_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_range)

    # Plot the scatter plot for training data and the best fit line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')

    plt.plot(x_range, y_line, color='red', linewidth=2, 
            label=f'Best Fit Line: returns_in_{iteration}_its = {model.coef_[0]:.4f} * returns_from_{iteration}_its_ago')

    plt.xlabel(f'returns_from_{iteration}_its_ago')
    plt.ylabel(f'returns_in_{iteration}_its')
    plt.title(f'Linear Relationship for {iteration} Iteration(s)')
    plt.legend()
    plt.show()
