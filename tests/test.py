"""Purpose of this script is to test whether the new price path equals the original price path"""

# Import libraries
from cProfile import Profile
import numpy as np
from hurst import compute_Hc
from scipy import stats
import pandas as pd
from hft_abm_smc_abc.config import PRICE_PATH_DIVIDER, TIME_HORIZON, P_0, MC_STEPS, N_A
from hft_abm_smc_abc.config import DELTA_TRUE, MU_TRUE, ALPHA_TRUE, LAMBDA0_TRUE,C_LAMBDA_TRUE, DELTA_S_TRUE



np.random.seed(123)
def accept_pos(x):
    """Outputs true if entire vector is positive"""

    if min(x) > 0:
        return True
    else:
        return False


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

    return {"mean": s1.loc["mean"],
            "std": s1.loc["std"],
            **s2,
            **ks_stat}



def preisSim(parameters):
    """Outputs: summary statistics from Preis model,
     Inputs: dictionary with delta, mu, alpha, lambda0, C_lambda, delta
     Static parameters: L, p_0, MCSteps, N_A"""

    # Import libraries to be used in model simulation
    from hft_abm_smc_abc.preisSeed import PreisModel
    import pandas as pd

    # Fixed Parameters
    PRICE_PATH_DIVIDER = 1000
    TIME_HORIZON = 100  # time horizon
    P_0 = 238.745 * PRICE_PATH_DIVIDER  # initial price
    MC_STEPS = 10 ** 5  # MC steps to generate variance
    N_A = 125  # no. market makers = no. liquidity providers

    # continue until price path simulated is all positive
    positive_price_path = False
    while not positive_price_path:

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

    # Log and divide price path by 1000, Convert to pandas dataframe
    price_path = pd.DataFrame(np.log(p.intradayPrice  / PRICE_PATH_DIVIDER))
    return price_path


def main():

    #define "true" parameters to calibrate
    param_true = {"delta": DELTA_TRUE,
                  "mu": MU_TRUE,
                  "alpha": ALPHA_TRUE,
                  "lambda0": LAMBDA0_TRUE,
                  "C_lambda": C_LAMBDA_TRUE,
                  "delta_S": DELTA_S_TRUE}


    price_path = preisSim(param_true)

    # p_true_SS = all_summary_stats(price_path, price_path)

    return(price_path)



p_true_SS = main()
p_true_SS.to_csv("new_test_case.csv", index=False, header=False)


original = pd.read_csv("original_test_case.csv", header=None)

new_case = pd.read_csv("new_test_case.csv", header=None)


print(all(round(original.apply(float, axis=1),0)==round(new_case.apply(float, axis=1),0)))


# profiler = Profile()
# profiler.runctx(
#      "main()",
#      locals(), globals())
# #
# from pyprof2calltree import convert, visualize
# visualize(profiler.getstats())                            # run kcachegrind
# convert(profiler.getstats(), 'profiling_results.kgrind')  # save for later


