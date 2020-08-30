"""Purpose of this script is to simlate observations from ARMA,
define model and its true parameters, give prior distribution"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pyabc.visualization
import logging
import statsmodels as sm
from statsmodels import tsa
from statsmodels.tsa import arima_process, arima_model, stattools
from pyabc import Distribution, RV, MedianEpsilon
from pyabc.visualization import plot_kde_matrix, plot_kde_2d
import pyabc
from scipy import stats
import pandas as pd
from hft_abm_smc_abc.config import SMCABC_DISTANCE, \
    SMCABC_TRANSITIONS, SMCABC_ACCEPTOR

np.random.seed(12345)

# for debugging
df_logger = logging.getLogger('Distance')
df_logger.setLevel(logging.DEBUG)


def summary_stats_extra(x):
    """outputs additional summary statistics: skewness, kurtosis, Hurst"""
    #
    # try:
    #     H, c, data = compute_Hc(x, kind='price', simplified=True)
    # except Exception as e:
    #     H = 0.25
    #     print(e)

    return {"skew": x.skew(),
            "kurt": x.kurt()  # ,
            # "hurst": H
            }


def all_summary_stats(price_sim, price_obs):
    """ouptuts all summary statistics of price path compared to true price path (path_obs)"""

    # count, mean, std, min, 25%, 50%, 75%, max
    s1 = price_sim[0].describe()

    # skew, kurt, hurst
    s2 = summary_stats_extra(price_sim[0])

    # Kolmogorov Smirnov 2 sample test statistic (if 0 - identical)
    ks_stat = {"KS": stats.ks_2samp(np.ravel(price_sim), np.ravel(price_obs))[0]}

    # acf
    acf = sm.tsa.stattools.acf(price_sim[0], unbiased=False, nlags=5, qstat=False, fft=None, alpha=None, missing='none')

    return {"mean": s1.loc["mean"],
            "std": s1.loc["std"],
            **s2,
            **ks_stat,
            "acf1": acf[1],
            "acf2": acf[2],
            "acf3": acf[3],
            "acf4": acf[4],
            "acf5": acf[5]
            }


def sum_stat_sim(parameters):
    price_path = arma_model(parameters)

    # summary statistics
    return all_summary_stats(price_path, price_obs)


# model definition
def arma_model(p):
    # Further, due to the conventions used in signal processing used in signal.lfilter vs.
    # conventions in statistics for ARMA processes, the AR parameters should have the opposite sign of what you might expect.
    arparams = np.array([p["AR"]])
    maparams = np.array([p["MA"]])

    # coefficients
    # coefficient on the zero-lag. This is typically 1.
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag

    y = sm.tsa.arima_process.arma_generate_sample(ar, ma, 500)

    return pd.DataFrame(y)


def plot_coonvergence(history, parameter, range_min, range_max, true_value, ax):
    # fig, ax = plt.subplots()
    for t in range(5):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df, w,
            xmin=range_min, xmax=range_max,
            x=parameter, ax=ax,
            label="PDF t={}".format(t))
    ax.axvline(true_value, color="k", linestyle="dashed");
    ax.axvline(true_value, color="k", linestyle="dashed");
    ax.legend(prop={'size': 6});


if __name__ == '__main__':

    # true model parameter
    param_dict = {"AR": 0.7, "MA": 0.8}

    # observation
    price_obs = arma_model(param_dict)
    price_obs[0].plot()
    plt.xlabel("Time-steps")
    plt.ylabel("ARMA(0.7, 0.8)")
    plt.show()

    # test optimisation:
    # Fits ARMA(p,q) model using exact maximum likelihood via Kalman filter.
    model_fit = sm.tsa.arima_model.ARMA(list(price_obs.values), (1, 1)).fit(trend='nc', disp=0)
    model_fit.params

    # prior distribution
    # Parameters as Random Variables
    prior = Distribution(AR=RV("uniform", 0, 1),
                         MA=RV("uniform", 0, 1))

    # database
    db_path = pyabc.create_sqlite_db_id(file_="arma_model1.db")

    abc = pyabc.ABCSMC(
        sum_stat_sim, prior, population_size=100,
        distance_function=SMCABC_DISTANCE,
        transitions=SMCABC_TRANSITIONS,
        eps=MedianEpsilon(1),
        acceptor=SMCABC_ACCEPTOR
    )

    ss_obs = all_summary_stats(price_obs, price_obs)
    abc.new(db_path, ss_obs)

    start_time = time.time()
    history1 = abc.run(minimum_epsilon=.001, max_nr_populations=5)
    print("--- %s seconds ---" % (time.time() - start_time))

    df, w = history1.get_distribution(m=0, t=4)
    plot_kde_matrix(df, w);
    plt.show()

    fig, axs = plt.subplots(2, 1)
    plot_coonvergence(history1, 'AR', 0, 1, 0.7, ax=axs[0])
    plot_coonvergence(history1, 'MA', 0, 1, 0.8, ax=axs[1])
    plt.show()

    plt.gcf().set_size_inches((4, 4))


    fig, axs = plt.subplots(5, 1)
    for t in range(5):
        df, w = abc.history.get_distribution(0, t)
        plot_kde_2d(df, w, "AR", "MA",
                    xmin=0, xmax=1,
                    ymin=0, ymax=1,
                    numx=100, numy=100, ax=axs[t])
        axs[t].scatter([0.7], [0.8],
                       edgecolor="black",
                       facecolor="white",
                       label="Observation");
        axs[t].legend();
        axs[t].set_title("PDF t={}".format(t + 1))
    fig.set_size_inches(4, 10, forward=True)

    plt.show()
