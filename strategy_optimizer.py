import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from scipy.stats import uniform, randint


class Backtester:
    def __init__(self, data, short_window=50, long_window=200):
        self.data = data.copy()
        self.short_window = short_window
        self.long_window = long_window
        self.positions = []
        self.capital = 100000.0
        self.holdings = 0

    def run_backtest(self):
        df = self.data
        df['SMA50'] = df['Close'].rolling(window=self.short_window).mean()
        df['SMA200'] = df['Close'].rolling(window=self.long_window).mean()
        df['ATR'] = df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()
        df['Signal'] = np.where(df['SMA50'] > df['SMA200'], 1, -1)
        df['Position'] = df['Signal'].diff()
        df['Position_Size'] = df['Close'] / (df['ATR'] * 10)

        for index, row in df.iterrows():
            if row['Position'] == 1:  # Buy signal
                self.holdings = self.capital / row['Close']
                self.capital = 0
            elif row['Position'] == -1:  # Sell signal
                self.capital = self.holdings * row['Close']
                self.holdings = 0
            self.positions.append({'capital': self.capital, 'holdings': self.holdings, 'price': row['Close']})

    def get_portfolio_value(self):
        return [p['capital'] + p['holdings'] * p['price'] for p in self.positions]

    def performance_metrics(self):
        portfolio = self.get_portfolio_value()
        returns = pd.Series(portfolio).pct_change().dropna()
        sharpe_ratio = np.sqrt(len(returns)) * returns.mean() / returns.std()
        return {
            'Sharpe Ratio': sharpe_ratio
        }


class StrategyEstimator(BaseEstimator):
    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window = long_window

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return None

    def score(self, X, y=None):
        backtest = Backtester(X, self.short_window, self.long_window)
        backtest.run_backtest()
        performance = backtest.performance_metrics()
        return -performance['Sharpe Ratio']  # Negative because we want to maximize Sharpe Ratio


def strategy_performance(df, short_window, long_window):
    estimator = StrategyEstimator(short_window, long_window)
    return estimator.score(df)


def optimize_strategy(df):
    param_dist = {
        'short_window': randint(20, 100),
        'long_window': randint(100, 300)
    }
    search = RandomizedSearchCV(
        estimator=StrategyEstimator(),
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring=make_scorer(lambda X, y=None, **kwargs: strategy_performance(X, **kwargs),
                            greater_is_better=False),
        n_jobs=-1
    )
    search.fit(df)
    print("Best parameters:", search.best_params_)
    return search.best_params_


# Load historical data
df = pd.read_csv('historical_data.csv', parse_dates=['Date'], index_col='Date')

# Run optimization
optimized_params = optimize_strategy(df)