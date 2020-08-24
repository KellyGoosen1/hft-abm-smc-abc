% Generate Mersenne Twister Sequence of Random Numbers:
rand('twister',1)
r = rand(10^6,1)
csvwrite( "ran3.csv", r)


% Reproduce Intraday Price Path using default parameters and appropriate seed
rand('twister',1)
p=PreisModel(N_A = 125, delta = 0.025, lambda_0 =100, C_lambda = 10, 
                    delta_S=0.001, alpha=0.15, mu=0.025, p_0=100, T=250)
p.q_mean_dev % matched
p = p.initialize() % matched
p = p.simulate() % matched
p.intradayPrice % matched

% Plot Intraday Price
count1 = p.intradayPrice
hFig = figure(1);
set(hFig, 'Position', [16 16 width height])
plot(count1,':b')
xlabel('ticks')
ylabel('Mid-price')
grid on


