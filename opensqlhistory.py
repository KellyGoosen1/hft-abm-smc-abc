from pyabc import History
import matplotlib.pyplot as plt
from hft_abm_smc_abc.config import DELTA_TRUE, MU_TRUE, ALPHA_TRUE, LAMBDA0_TRUE, C_LAMBDA_TRUE, DELTA_S_TRUE, \
    WORK_DIR, temp_output_folder, version_number, PROCESSED_FOLDER, \
    DELTA_MIN, DELTA_MAX, MU_MIN, MU_MAX, ALPHA_MIN, ALPHA_MAX, LAMBDA0_MIN, LAMBDA0_MAX, \
    C_LAMBDA_MIN, C_LAMBDA_MAX, DELTAS_MIN, DELTAS_MAX, SMCABC_DISTANCE, SMCABC_POPULATION_SIZE, SMCABC_SAMPLER, \
    SMCABC_TRANSITIONS, SMCABC_EPS
import pyabc

from scipy import stats

from scipy.special import btdtri

# load history
h_loaded = History("sqlite:///"
                   + "hft_abm_smc_abc/resultsReal_Data_Small_Test - Smaller Test - eps1_negfix_pop6_pop301597579353.943031.db")

# check that the history is not empty
print(h_loaded.all_runs())

from pyabc.visualization import plot_kde_matrix

df, w = h_loaded.get_distribution(m=0, t=4)
plot_kde_matrix(df, w);
plt.show()


def plot_coonvergence(history, parameter, range_min, range_max, true_value, ax):
    # fig, ax = plt.subplots()
    for t in range(history.max_t+1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df, w,
            xmin=range_min, xmax=range_max,
            x=parameter, ax=ax,
            label="PDF t={}".format(t))
    # ax.axvline(true_value, color="k", linestyle="dashed");
    ax.legend(prop={'size': 6});


fig, axs = plt.subplots(2, 3)
plot_coonvergence(h_loaded, 'mu', MU_MIN, MU_MAX, MU_TRUE, ax=axs[0, 0])
plot_coonvergence(h_loaded, 'lambda0', LAMBDA0_MIN, LAMBDA0_MAX, LAMBDA0_TRUE, ax=axs[0, 1])
plot_coonvergence(h_loaded, 'delta', DELTA_MIN, DELTA_MAX, DELTA_TRUE, ax=axs[0, 2])
plot_coonvergence(h_loaded, 'delta_S', DELTAS_MIN, DELTAS_MAX, DELTA_S_TRUE, ax=axs[1, 0])
plot_coonvergence(h_loaded, 'alpha', ALPHA_MIN, ALPHA_MAX, ALPHA_TRUE, ax=axs[1, 1])
plot_coonvergence(h_loaded, 'C_lambda', C_LAMBDA_MIN, C_LAMBDA_MAX, C_LAMBDA_TRUE, ax=axs[1, 2])
plt.gcf().set_size_inches((12, 8))
plt.gcf().tight_layout()
plt.show()

_, arr_ax = plt.subplots(1, 2)

pyabc.visualization.plot_sample_numbers(h_loaded, ax=arr_ax[0])
pyabc.visualization.plot_epsilons(h_loaded, ax=arr_ax[1])
pyabc.visualization.plot_credible_intervals(
    h_loaded, levels=[0.95], ts=[0, 1, 2, 3, 4, 5], par_names=['C_lambda'],
    show_mean=False, show_kde_max_1d=True,
    refval={'C_lambda': C_LAMBDA_TRUE}, arr_ax=arr_ax[1][0])
pyabc.visualization.plot_effective_sample_sizes(h_loaded, ax=arr_ax[1][1])

plt.gcf().set_size_inches((12, 8))
plt.gcf().tight_layout()
plt.show()


def posterior_mean_and_credibility_intervals(
        h_loaded, param="C_lambda", rounding=4):
    data = h_loaded.get_distribution(t=3)[0][param]

    res_mean, res_var, res_std = stats.bayes_mvs(data, alpha=0.90)

    mean_cred_dict = {'mean': res_mean[0].round(rounding),
                      'lower': res_mean[1][0].round(rounding),
                      'upper': res_mean[1][1].round(rounding),
                      'std/n': round(res_std[0]/SMCABC_POPULATION_SIZE, rounding)}

    return mean_cred_dict


posterior_mean_and_credibility_intervals(h_loaded, "mu")
posterior_mean_and_credibility_intervals(h_loaded, "lambda0")
posterior_mean_and_credibility_intervals(h_loaded, "delta")
posterior_mean_and_credibility_intervals(h_loaded, "delta_S")
posterior_mean_and_credibility_intervals(h_loaded, "alpha")
posterior_mean_and_credibility_intervals(h_loaded, "C_lambda")

