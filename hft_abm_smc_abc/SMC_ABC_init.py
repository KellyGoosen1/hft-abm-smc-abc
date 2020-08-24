"""Purpose of this script is to initialise all the model parameters"""

# Import libraries
import numpy as np
import pandas as pd
import os
from hurst import compute_Hc
from scipy import stats
import statsmodels as sm
from statsmodels import tsa
from statsmodels.tsa import stattools

from pyabc import Distribution, RV, ABCSMC
from hft_abm_smc_abc.config import DELTA_TRUE, MU_TRUE, ALPHA_TRUE, LAMBDA0_TRUE, C_LAMBDA_TRUE, DELTA_S_TRUE, \
    temp_output_folder, version_number, \
    DELTA_MIN, DELTA_MAX, MU_MIN, MU_MAX, ALPHA_MIN, ALPHA_MAX, LAMBDA0_MIN, LAMBDA0_MAX, \
    C_LAMBDA_MIN, C_LAMBDA_MAX, DELTAS_MIN, DELTAS_MAX, SMCABC_DISTANCE, SMCABC_POPULATION_SIZE, SMCABC_SAMPLER, \
    SMCABC_TRANSITIONS, SMCABC_EPS, SMCABC_ACCEPTOR


def summary_stats_extra(x):
    """outputs additional summary statistics: skewness, kurtosis, Hurst"""

    try:
        H, c, data = compute_Hc(x, kind='price', simplified=True)
    except Exception as e:
        H = 0.25
        print(e)

    return {"skew": x.skew(),
            "kurt": x.kurt(),
            "hurst": H
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
    if at_least_one_different(price_sim[0]):
        acf = sm.tsa.stattools.acf(price_sim[0], unbiased=False, nlags=5, qstat=False, fft=True, alpha=None,
                                   missing='drop')
    else:
        acf = [99999, 99999, 99999, 99999, 99999, 9999]

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


def accept_pos(x):
    """Outputs true if entire vector is positive"""

    if min(x) > 0:
        return True
    else:
        return False


def at_least_one_different(items):
    return any(x != items[0] for x in items)


def preisSim(parameters):
    """Outputs: summary statistics from Preis model,
     Inputs: dictionary with delta, mu, alpha, lambda0, C_lambda, delta
     Static parameters: L, p_0, MCSteps, N_A"""

    # Import libraries to be used in model simulation
    from hft_abm_smc_abc.preisSeed import PreisModel
    from hft_abm_smc_abc.config import PRICE_PATH_DIVIDER, TIME_HORIZON, P_0, MC_STEPS, N_A
    import pandas as pd

    # Initialize preis model class with specified parameters
    p = PreisModel(N_A=N_A,
                   delta=parameters["delta"],
                   mu=parameters["mu"],
                   alpha=parameters["alpha"],
                   lambda_0=parameters["lambda0"],
                   C_lambda=parameters["C_lambda"],
                   delta_S=parameters["delta_S"],
                   p_0=P_0,
                   T=TIME_HORIZON,
                   MC=MC_STEPS)

    # Start model
    p.simRun()
    p.initialize()

    # Simulate price path for L time-steps
    p.simulate()

    # ensure no negative prices
    positive_price_path = accept_pos(p.intradayPrice)

    # poor results - set to arbitrarily high number
    if not positive_price_path:
        price_path = pd.DataFrame([9999] * TIME_HORIZON)
    else:
        # Log and divide price path by 1000, Convert to pandas dataframe
        price_path = pd.DataFrame(np.log(p.intradayPrice / PRICE_PATH_DIVIDER))

    return price_path


def sum_stat_sim(parameters):
    price_path = preisSim(parameters)

    p_true = pd.read_csv(os.path.join(temp_output_folder, "p_true.csv"),
                         header=None)
    # p_true_real = pd.read_csv(os.path.join(PROCESSED_FOLDER,
    #                                        "Log_Original_Price_Bars_2300.csv"), header=None)

    # summary statistics
    return all_summary_stats(price_path, p_true)


if __name__ == '__main__':
    # Parameters as Random Variables
    prior = Distribution(delta=RV("uniform", DELTA_MIN, DELTA_MAX),
                         mu=RV("uniform", MU_MIN, MU_MAX),
                         alpha=RV("uniform", ALPHA_MIN, ALPHA_MAX),
                         lambda0=RV("uniform", LAMBDA0_MIN, LAMBDA0_MAX),
                         C_lambda=RV("uniform", C_LAMBDA_MIN, C_LAMBDA_MAX),
                         delta_S=RV("uniform", DELTAS_MIN, DELTAS_MAX))

    # define "true" parameters to calibrate
    param_true = {"delta": DELTA_TRUE,
                  "mu": MU_TRUE,
                  "alpha": ALPHA_TRUE,
                  "lambda0": LAMBDA0_TRUE,
                  "C_lambda": C_LAMBDA_TRUE,
                  "delta_S": DELTA_S_TRUE}

    # Simulate "true" summary statistics
    p_true = preisSim(param_true)
    p_true.to_csv(os.path.join(temp_output_folder, "p_true.csv"), header=False,
                  index=False)
    # p_true_real = pd.read_csv(os.path.join(PROCESSED_FOLDER,
    #                                        "Log_Original_Price_Bars_2300.csv"), header=None)

    p_true_SS = all_summary_stats(p_true, p_true)

    # Initialise ABCSMC model parameters
    abc = ABCSMC(models=sum_stat_sim,
                 parameter_priors=prior,
                 distance_function=SMCABC_DISTANCE,
                 population_size=SMCABC_POPULATION_SIZE,
                 sampler=SMCABC_SAMPLER,
                 transitions=SMCABC_TRANSITIONS,
                 eps=SMCABC_EPS,
                 acceptor=SMCABC_ACCEPTOR)

    # Set up SQL storage facility
    db = "sqlite:///" + os.path.join(temp_output_folder, "results" + version_number + ".db")

    # Input SMCABC SQL and observed summary stats
    abc.new(db, p_true_SS)
