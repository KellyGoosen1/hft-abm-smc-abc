from hft_abm_smc_abc.preisSeed import PreisModel
import matplotlib.pyplot as plt
import pandas as pd

# Import random numbers from Matlab generated csv
ran_seq_df = pd.read_csv('ran3.csv', header=None)  # seed =1
ran_seq = ran_seq_df[0].values

# Construct PreisModel object with default parameters
p = PreisModel(N_A=125, delta=0.025, lambda_0=100, C_lambda=10,
               delta_S=0.001, alpha=0.15, mu=0.025, p_0=100, T=250, MC=10 ** 5, ranSeq=ran_seq)

# Start model andc check match
p.simRun()  # matches Matlab
p.q_mean_dev  # matches Matlab
p.initialize()

# Simulate price path for T=250 time-steps
p.simulate()

# Check matchers Matlab intradayPrice and Limit order booko
p.intradayPrice
p.limitOrderBook.orderBook

# Plot intraday Price
df = pd.DataFrame(p.intradayPrice, columns=["Intraday Price"])

# Draw Plot
plt.figure(figsize=(10, 10), dpi=80)
plt.plot('Intraday Price', data=df, color='tab:blue')

# Decoration
plt.ylim(20, 140)
xtick_location = df.index.tolist()[::12]
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='both', alpha=.3)

plt.gca().set(xlabel='Time-steps', ylabel='Mid-price')
plt.ylabel("Mid-price", fontsize=12)
plt.xlabel("Ticks", fontsize=12)
plt.show()
