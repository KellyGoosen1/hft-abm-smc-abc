import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import statsmodels.tsa.api as smt
import os
import pingouin as pg
from hft_abm_smc_abc.config import PROCESSED_FOLDER, TIME_HORIZON

midprice = pd.read_csv(os.path.join(PROCESSED_FOLDER,
                                    "Log_Original_Price_Bars_2300.csv"), header=None)

midprice = midprice.rename(columns = {0:"Midprice"})



class StylisedFacts:

    # Return Attributes
    def __init__(self, log_midprice):
        self.midprice = np.exp(log_midprice)
        self.returns = self.midprice.pct_change().dropna()
        self.log_returns = np.log(self.midprice / self.midprice.shift(1)).dropna()
        self.time_horizon = midprice.shape[0]
        self.signs = self.tick_rule()

    def plot_returns(self):
        fig, ax = plt.subplots(3, 1, figsize=(11, 10))
        # price ----
        self.midprice.plot(ax=ax[0])
        ax[0].set_ylabel('Stock price ($)')
        ax[0].set_xlabel('')
        ax[0].set_title('Price vs. returns')
        # returns ----
        self.returns.plot(ax=ax[1])
        ax[1].set_ylabel('Simple returns (%)')
        ax[1].set_xlabel('')
        # log returns ----
        self.log_returns.plot(ax=ax[2])
        ax[2].set_ylabel('Log returns (%)')
        fig.show()

    def plot_dbn(self):
        # Plotting the distribution of the log returns ----
        ax = sns.distplot(self.log_returns, kde=False, norm_hist=True)

        xx = np.linspace(self.log_returns.min(), self.log_returns.max(), num=1000)
        yy = scs.norm.pdf(xx, loc=self.log_returns.mean(), scale=self.log_returns.std())
        ax.plot(xx, yy, 'r', lw=2)
        ax.set_title('Distribution of Log-Returns')
        plt.show(figsize=(5, 5))

    def qq_plot(self):
        ax = pg.qqplot(self.log_returns, dist='norm', confidence=False,
                       )
        plt.show()

    def autocorrelation_plot(self):
        # Autocorrelation plot of log returns ----
        acf_r = smt.graphics.plot_acf(self.log_returns, lags=40, alpha=0.05)
        acf_r.show()

    def all_autocorrelation_plots(self):
        # specify the max amount of lags
        lags = 20

        fig, ax = plt.subplots(4, 1, figsize=(12, 10))
        # returns ----
        smt.graphics.plot_acf(self.log_returns, lags=lags, alpha=0.05, ax=ax[0])
        ax[0].set_ylabel('Returns')
        ax[0].set_title('Autocorrelation Plots')
        # squared returns ----
        smt.graphics.plot_acf(self.log_returns ** 2, lags=lags, alpha=0.05, ax=ax[1])
        ax[1].set_ylabel('Squared Returns')
        ax[1].set_xlabel('')
        ax[1].set_title('')
        # absolute returns ----
        smt.graphics.plot_acf(np.abs(self.log_returns), lags=lags, alpha=0.05, ax=ax[2])
        ax[2].set_ylabel('Absolute Returns')
        ax[2].set_title('')
        ax[2].set_xlabel('Lag')
        fig.show()

        # order flow acf
        smt.graphics.plot_acf(self.signs, lags=lags, alpha=0.05, ax=ax[3])
        ax[3].set_ylabel('Order Flow Signs')
        ax[3].set_title('')
        ax[3].set_xlabel('Lag')
        fig.show()

    def descriptive_statistics(self):
        # Descriptive statistics ----
        print(f'Number of observations: {self.midprice.shape[0]}')
        print(f'Mean: {self.log_returns.values.mean()}')
        print(f'Median: {self.log_returns.median()}')
        print(f'Min: {self.log_returns.values.min()}')
        print(f'Max: {self.log_returns.values.max()}')
        print(f'Standard Deviation: {self.log_returns.values.std()}')
        print(f'Skewness: {self.log_returns.skew()}')
        print(f'Kurtosis: {self.log_returns.kurtosis()}')  # Kurtosis of std. Normal dist = 0
        print(f'Jarque-Bera statistic: {scs.jarque_bera(self.log_returns.values)[0]} '
              f'with p-value: {round(scs.jarque_bera(self.log_returns.values)[1], 5)}')

        # The Jarque-Bera Normality Test: with p-value small enough to reject the null hypothesis
        # stating that the data follows Gaussian distribution.

    def tick_rule(self):

        previous_tick_change = 0
        signs = np.zeros(self.time_horizon)
        for tstep in range(1, self.time_horizon):
            if self.midprice.iloc[tstep][0] == self.midprice.iloc[tstep - 1][0]:
                if previous_tick_change > 0:
                    signs[tstep] = 1
                elif previous_tick_change < 0:
                    self.signs[tstep] = -1

            elif self.midprice.iloc[tstep][0] > self.midprice.iloc[tstep - 1][0]:
                signs[tstep] = 1
                previous_tick_change = 1
            elif self.midprice.iloc[tstep][0] < self.midprice.iloc[tstep - 1][0]:
                signs[tstep] = -1
                previous_tick_change = 1

        return signs

    def order_flow_acf(self):

        acf_r = smt.graphics.plot_acf(self.signs, lags=40, alpha=0.05)
        acf_r.show(figsize=(4, 4))


S = StylisedFacts(midprice)

S.plot_returns()

# distribution, non gaussian, Leptokurtic returns
S.plot_dbn()
S.qq_plot()
S.descriptive_statistics()


# absense of autocorrelation or raw returns
# volatility clustering
# order flow clustering
S.all_autocorrelation_plots()

