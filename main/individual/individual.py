import numpy as np
import pandas as pd
import random
from numpy import ndarray


class Individual:
    counter = 0
    universe = None

    @classmethod
    def set_stock_universe(cls, universe: pd.DataFrame):
        """Add the universe of assets as class attribute."""
        cls.universe = universe

    @classmethod
    def create_random(cls):
        """Generates a new individual which represents a different allocation."""
        pfolio_length = random.randint(1, 20)
        portfolio_stocks = np.array(random.sample(range(1, cls.universe.shape[1]), pfolio_length))
        portfolio_weights = np.array(random.sample(range(1, 21), pfolio_length))
        portfolio_weights = portfolio_weights / portfolio_weights.sum()

        return Individual(portfolio_idx=portfolio_stocks, portfolio_weights=portfolio_weights)

    def __init__(self, portfolio_idx: ndarray, portfolio_weights: ndarray) -> None:
        self.portfolio_idx = portfolio_idx
        self.portfolio_weights = portfolio_weights
        self.__class__.counter += 1

    def get_sharpe(self):
        """Returns the sharpe ratio of the portfolio, also acts as the fitness function to maximize."""
        hist_ret = np.log(self.prices()).diff().dropna()
        cov_returns = hist_ret.cov()
        rent_cartera = hist_ret.mean().T @ self.portfolio_weights
        risk = np.sqrt(self.portfolio_weights @ cov_returns.values @ self.portfolio_weights)
        ratio_sharpe = rent_cartera.sum() / risk
        return ratio_sharpe

    def expected_return(self):
        """Returns the mean expected returns of the portfolio."""
        hist_ret = np.log(self.prices()).diff().dropna()
        ret = self.portfolio_weights @ hist_ret.mean().T
        return ret

    def prices(self):
        """Returns closing prices of this porfolio."""
        return self.universe.iloc[:, self.portfolio_idx]

    def risk(self):
        """Returns the risk for this portfolio."""
        hist_ret = np.log(self.prices()).diff().dropna()
        cov = hist_ret.cov()
        risk = np.sqrt(self.portfolio_weights @ cov.values @ self.portfolio_weights)
        return risk
