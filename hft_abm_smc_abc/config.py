"""Purpose of this script is to contain all the configurations"""
import os
import time

from pyabc import MedianEpsilon, MultivariateNormalTransition, \
    LocalTransition, Distribution, RV, ABCSMC, sge, \
    AdaptivePNormDistance, UniformAcceptor
from pyabc.sampler import MulticoreEvalParallelSampler
import pyabc

# Set version number each iteration
WORK_DIR = os.path.dirname(os.path.dirname(__file__)) + "\\"

temp_folder = "Fixthreads2_100_TH100_t=10_uniformAcceptor_eps0001_seed4"
version_number = temp_folder + str(time.time())

if os.path.dirname(WORK_DIR) == os.path.dirname('C:\\My Work Documents\\repo\\hft_abm_smc_abc\\'):
    WORK_DIR = 'C:/My Work Documents/Dissertation/Working_Directory'
    ncores = 7
else:
    WORK_DIR = '/home/gsnkel001/Working_Directory'
    ncores = 40

PROCESSED_FOLDER = WORK_DIR + '/' + '02_ProcessedData'

temp_output_folder = PROCESSED_FOLDER + '/' + version_number

# create folder if not exists
if not os.path.exists(temp_output_folder):
    os.mkdir(temp_output_folder)


# True price trajectory
DELTA_TRUE = 0.0250         # limit order cancellation rate
MU_TRUE = 0.0250            # rate of market orders
ALPHA_TRUE = 0.15           # rate of limit orders
LAMBDA0_TRUE = 100          # initial order placement depth
C_LAMBDA_TRUE = 10          # limit order placement depth coefficient
DELTA_S_TRUE = 0.0010       # mean reversion strength parameter

# prior range
DELTA_MIN, DELTA_MAX = 0, 0.05
MU_MIN, MU_MAX = 0, 0.05
ALPHA_MIN, ALPHA_MAX = 0.05, 0.5
LAMBDA0_MIN, LAMBDA0_MAX = 50, 300
C_LAMBDA_MIN, C_LAMBDA_MAX = 1, 50
DELTAS_MIN, DELTAS_MAX = 0, 0.005

# Fixed Parameters
PRICE_PATH_DIVIDER = 1000
TIME_HORIZON = 100                     # time horizon
P_0 = 238.745 * PRICE_PATH_DIVIDER      # initial price
MC_STEPS = 10 ** 5                      # MC steps to generate variance
N_A = 125                               # no. market makers = no. liquidity providers

# SMCABC parameters:
SMCABC_DISTANCE = AdaptivePNormDistance(
    p=2, scale_function=pyabc.distance.root_mean_square_deviation)
SMCABC_POPULATION_SIZE = 100
SMCABC_SAMPLER = MulticoreEvalParallelSampler(ncores)
SMCABC_TRANSITIONS = MultivariateNormalTransition()
SMCABC_EPS = MedianEpsilon(0.01)
SMCABC_ACCEPTOR = UniformAcceptor(use_complete_history=True)
smcabc_minimum_epsilon = 0.0001
smcabc_max_nr_populations = 10
smcabc_min_acceptance_rate = SMCABC_POPULATION_SIZE / 25000