classdef PreisModel
% PreisModel: One complete Preis et al. (2006) model simulation.
%
% This class aims to encapsulate all the attributes and methods required 
% to implement a Monte Carlo simulation of trader behavior using the 
% Preis et al. (2006) model. Each object of type PreisModel represents one 
% complete simulation consisting of a number of Monte Carlo steps.
% 
% The class has attributes:
%
% Name:               Description:                                                       Type:
% N_A                 Number of traders of each type in the current simulation           (Integer)
% delta               Order cancellation probability                                     (Double)
% lambda_0            Initial limit order placement depth parameter                      (Integer)   
% lambda_t            Current limit order placement depth parameter                      (Integer)
% C_lambda            Placement depth coefficient                                        (Integer)
% q_provider          Buy probability of liquidity providers (constant)                  (Double)
% q_taker             Mean buy probability of liquidity takers                           (Double)
% q_mean_dev          Mean squared deviation of q_taker from 0.5 per simulation          (Double)
% delta_S             Liquidity taker buy probability random walk increment              (Double)
% limitOrderBook      PreisOrderBook object                                              (1 x 1 PreisOrderBook Object)
% intradayPrice       Time series of intraday prices                                     (T x 1 Vector)
% p_0                 Initial value for price time series                                (Double)
% alpha               Activation frequency for liquidity providers                       (Double) 
% mu                  Activation frequency for liquidity takers                          (Double) 
% T                   Total number of Monte Carlo steps                                  (Integer)
% tradeSigns          Series of trade signs                                              (Dynamic Size Vector)
%
% For method details, see also PreisModel/PreisModel, calcProviderLimitPrice,
% incrementPlacementDepth, incrementTakerRandomWalk, placeOrder, initialize,
% simulate and autocorrel.
%
% References:
% 1. Preis T, Golke S, Paul W, Schneider (2006) Multi-agent-based Order 
% Book Model of financial markets. Europhys Lett 75(3):510-516
%
% Donovan Platt
% School of Computer Science and Applied Mathematics
% University of the Witwatersrand, Johannesburg, South Africa

    %% Class Attributes
    properties            
        %% Trader Population Attributes
        N_A             %Number of traders of each type in the current simulation
        
        %% Trader Order Cancellation Attributes
        delta           %Order cancellation probability
             
        %% Liquidity Provider Order Placement Attributes
        lambda_0        %Initial limit order placement depth parameter
        lambda_t        %Current limit order placement depth parameter
        C_lambda        %Placement depth coefficient
        
        %% Trader Buy/Sell Decision Attributes
        q_provider      %Buy probability of liquidity providers (constant)
        q_taker         %Mean buy probability of liquidity takers
        q_mean_dev      %Mean squared deviation of q_taker from 0.5 per simulation
        delta_S         %Liquidity taker buy probability random walk increment
       
        
        %% Order Book Attributes
        limitOrderBook  %PreisOrderBook object
        
        %% Price Time Series Attributes
        intradayPrice   %Time series of intraday prices
        p_0             %Initial value for price time series
        
        %% Trader Activation Frequency Attributes
        alpha           %Activation frequency for liquidity providers
        mu              %Activation frequency for liquidity takers
        
        %% Simulation Attributes
        T               % Total number of Monte Carlo steps 
        tradeSigns      % Series of trade signs
        
    end
    
    %% Class Methods
    methods
        %% Class Constructor Method
        function simRun = PreisModel(delta, N_A, lambda_0, C_lambda, delta_S, alpha, mu, p_0, T)
        % PreisModel: Class constructor method.
        %
        % simRun = PreisModel(N_A, delta, lambda_0, C_lambda, delta_S, 
        % alpha, mu, p_0, T) takes in the number of traders of each type in 
        % the simulation, N_A, the cancelation probability, delta, 
        % the initial order placement depth parameter, lambda_0, the placement 
        % depth coefficient, C_lambda, the increment for the random walk 
        % defining the buy and sell probabilities for liquidity takers, 
        % delta_S, the trading frequency of liquidity providers, alpha, the 
        % trading frequency of liquidity takers, mu, an initial price, p_0, 
        % and the desired number of Monte Carlo steps, T. Thereafter, a 
        % PreisModel object with the desired parameter values is returned.
       
            
            % Set Trader Order Cancellation Attributes
            simRun.delta = delta;
            
            % Set Trader Population Attributes
            simRun.N_A = N_A;
            
            % Set Liquidity Provider Order Placement Attributes
            simRun.lambda_0 = lambda_0;
            simRun.lambda_t = lambda_0;
            simRun.C_lambda = C_lambda;
            
            % Set Trader Buy/Sell Decision Attributes
            simRun.q_provider = 1/2;
            simRun.q_taker = 1/2;
            simRun.delta_S = delta_S;
            
            % Create Data Structure to Store q_taker Simulations
            q_taker_sim = zeros(10 ^ 5, 1);
            
            % Simulate q_taker
            for i = 1 : 10 ^ 5
                simRun = simRun.incrementTakerRandomWalk();
                q_taker_sim(i) = simRun.q_taker;
            end
            
            % Determine the Mean Sqaure Deviation of q_taker from 0.5 for the
            % Simulation
            simRun.q_mean_dev = mean((q_taker_sim - 0.5) .^ 2);
            
            % Reset q_taker
            simRun.q_taker = 1/2;
            
            % Set Order Book Attributes
            simRun.limitOrderBook = PreisOrderBook();
            simRun.limitOrderBook.bestAsk = p_0;
            simRun.limitOrderBook.bestBid = p_0;
            
            % Set Price Time Series Attributes
            simRun.intradayPrice = zeros(T, 1);
            simRun.p_0 = p_0;
            
            % Set LF Trader Activation Frequency Attributes
            simRun.alpha = alpha;
            simRun.mu = mu;
            
            % Set Simulation Attributes
            simRun.T = T;
            simRun.tradeSigns = [];
            
        end
                   
        %% Determine Provider Limit Price
        function [limitPrice, buyFlag] = calcProviderLimitPrice(simRun)
        % calcProviderLimitPrice: Generate trade decision and limit price 
        % for a liquidity provider.
        % 
        % [limitPrice, buyFlag] = calcProviderLimitPrice(simRun) takes in a
        % PreisModel object, simRun, and proceeds to randomly determine the
        % decision to buy or sell for a given trader based on the probability 
        % q_provider stored in simRun. Thereafter, the corresponding
        % posting depth is calculated based upon the exponential
        % distribution parameter, lambda_t, stored in simRun. Finally, the
        % trade decision is returned as buyFlag, with a 1 corresponding to
        % a buy and a 0 a sell, and the limit price is returned as
        % limitPrice.
        
            % Determine Whether to Buy or Sell
            if rand <= simRun.q_provider
                
                % Set eta
                eta = floor(-simRun.lambda_t * log(rand));
                
                % Set Buy Limit Price
                limitPrice = simRun.limitOrderBook.bestAsk - 1 - eta;
                
                % Set Buy Flag
                buyFlag = 1;
                
            else 
                
                % Set eta
                eta = floor(-simRun.lambda_t * log(rand));
                
                % Set Sell Limit Price
                limitPrice = simRun.limitOrderBook.bestBid + 1 + eta;
                
                % Set Buy Flag
                buyFlag = 0;
                
            end
            
        end
        
        %% Increment Order Placement Depth
        function simRun = incrementPlacementDepth(simRun)
        % incrementPlacementDepth: Calculate new placement depth for liquidity 
        % providers.
        % 
        % simRun = incrementPlacementDepth(simRun) takes in a PreisModel
        % object, simRun, and proceeds to calculate the next value for
        % lambda_t based upon the values of q_taker, q_mean_dev, lambda_0
        % and C_lambda stored in simRun. lambda_0 is then modified in
        % simRun and the modified PreisModel object is then returned.
        
            % Determine Current Buy Probability for Liquidity Takers
            q_current = simRun.q_taker;
            
            % Calculate New Placement Depth
            simRun.lambda_t = simRun.lambda_0 * (1 + abs(q_current - 0.5) ...
                / sqrt(simRun.q_mean_dev) * simRun.C_lambda);
            
        end
        
        %% Increment Buy Probability of Liquidity Takers
        function simRun = incrementTakerRandomWalk(simRun)
        % incrementTakerRandomWalk: Increment buy probability for liquidity 
        % takers using a a mean-reverting random walk.
        %
        % simRun = incrementTakerRandomWalk(simRun) takes in a PreisModel
        % object, simRun, and proceeds to iterate q_taker, stored in
        % simRun, according to a mean-reverting random walk. q_taker is
        % then modified in simRun and the modified PreisModel object is
        % then returned.
        
            % Get Current Probability
            q_current = simRun.q_taker;
            
            % Determine Mean Reversion Probability
            p_revert = 0.5 + abs(q_current - 0.5);
            
            % Determine Up Probability
            if q_current < 1/2
                
                % Below Mean, Up Move More Likely
                p_up = p_revert;
                
            elseif q_current > 1/2
                
                % Above Mean, Down Move More Likely
                p_up = 1 - p_revert;
                
            else
                
                % At Mean, Each Move Equally Likely
                p_up = 0.5;
                
            end
            
            % Generate Next Buy Probability
            if rand <= p_up
                simRun.q_taker = q_current + simRun.delta_S;
            else
                simRun.q_taker = q_current - simRun.delta_S;
            end
            
        end
            
        %% Place an Order
        function simRun = placeOrder(simRun, traderType)
        % placeOrder: Add a limit order to the limit order book for a 
        % liquidity provider or execute a market order for a liquidity taker.
        %
        % simRun = placeOrder(simRun, traderType) takes in a PreisModel
        % object, simRun, and a trader Type, traderType (1 = provider, 2 =
        % taker), and proceeds to either determine the limit price, trade
        % decision and place a limit order for a provider or determine the
        % trade decision and place a market order for a taker. The
        % limitOrderBook object in simRun is then modified according to the
        % results of the placement of the desired order and the modified
        % PreisModel object is then returned.
        %
        % See also providerLimitPrice, PreisOrderBook.
        
            % Set Order Size
            orderSize = 1; %Model default
            
            % Differentiate Between Trader Types
            if traderType == 1 %Provider
                
                % Set Limit Price and Buy Flag
                [limitPrice, buyFlag] = simRun.calcProviderLimitPrice();
                
                disp(limitPrice)           
                % Place Limit Order
                simRun.limitOrderBook = simRun.limitOrderBook.limitOrder(...
                    buyFlag, orderSize, limitPrice); 
                                
            else %Taker
                
                % Generate Buy Flag
                if rand <= simRun.q_taker
                    
                    % Buyer
                    buyFlag = 1;
                    
                    % Update Trade Sign Series
                    simRun.tradeSigns = [simRun.tradeSigns; 1];
                    
                else
                    
                    % Seller
                    buyFlag = 0;
                    
                    % Update Trade Sign Series
                    simRun.tradeSigns = [simRun.tradeSigns; -1];
                    
                end
                
                % Place Market Order
                simRun.limitOrderBook = simRun.limitOrderBook.marketOrder(...
                    buyFlag, orderSize);
            
            end
            
        end
                
        %% Initialize Simulation
        function simRun = initialize(simRun)
        % initialize: Fill the order book before the commencement of actual 
        % trading.
        %
        % simRun = initialize(simRun) takes in a PreisModel object, simRun,
        % and proceeds to generate and insert initial limit orders for 10 
        % Monte Carlo steps into the PreisOrderBook object stored in simRun.
        % The number of orders inserted depends on the number of traders,
        % N_A, and frequency of limit orders, alpha, stored in simRun.
        % After the completion of initialization, the modified PreisModel
        % object is returned.
        
            % Generate Initial Orders for 10 Monte Carlo Steps
            for i = 1 : floor(simRun.N_A * 10 * simRun.alpha) %Number of traders * rate of orders * 10
                           
                % Place Orders
                simRun = simRun.placeOrder(1);
                
            end
            
        end
        
        %% Simulate Trading
        function simRun = simulate(simRun)
        % simulate: Perform a complete PreisModel simulation of T Monte
        % Carlo steps.
        %
        % simRun = simulate(simRun) takes in a PreisModel 
        % object, simRun, and proceeds to perform a complete 
        % PreislModel simulation of T Monte Carlo steps, where the parameter 
        % T is stored in the provided object. Thereafter, the modified
        % PreisModel object is returned.
        %
        % See also incrementTakerRandomWalk and incrementPlacementDepth.
            
            % Simulate for Desired Number of Monte Carlo Steps
            for i = 1 : simRun.T
                
                % Liquidity Providers Place New Limit Orders
                for j = 1 : floor(simRun.alpha * simRun.N_A)
                                 
                        % Place Order
                        simRun = simRun.placeOrder(1);                       
                end
                
                % Liquidity Takers Place New Market Orders
                for j = 1 : floor(simRun.mu * simRun.N_A)
                                        
                        % Place Order
                        simRun = simRun.placeOrder(2);                       
                    
                end
                
                % Determine Orders to be Kept
                keepIndices = rand(size(simRun.limitOrderBook.orderBook, 1), 1 ...
                ) > simRun.delta;
                
                % Update Number of Buy Orders
                simRun.limitOrderBook.numBuy = simRun.limitOrderBook.numBuy    ...
                    - sum(simRun.limitOrderBook.orderBook(keepIndices == 0, 3) == 2);
                
                % Update Number of Sell Orders
                simRun.limitOrderBook.numSell = simRun.limitOrderBook.numSell  ...
                    - sum(simRun.limitOrderBook.orderBook(keepIndices == 0, 3) == 1);
                
                % Cancel Orders
                simRun.limitOrderBook.orderBook = simRun.limitOrderBook.orderBook(...
                    keepIndices, :);
                
                % Update Best Bid
                if simRun.limitOrderBook.numBuy ~= 0   
                    simRun.limitOrderBook.bestBid = simRun.limitOrderBook.orderBook(simRun.limitOrderBook.numBuy, 2);
                else
                    simRun.limitOrderBook.bestBid = 0;
                end
                
                % Update Best Bid
                if simRun.limitOrderBook.numSell ~= 0  
                    simRun.limitOrderBook.bestAsk = simRun.limitOrderBook.orderBook(simRun.limitOrderBook.numBuy + 1, 2);
                else
                    simRun.limitOrderBook.bestAsk = 0;                
                end
                
                % Update Price
                if (simRun.limitOrderBook.bestBid ~= 0) && (simRun.limitOrderBook.bestAsk ~= 0)
                    simRun.intradayPrice(i) = (simRun.limitOrderBook.bestAsk + simRun.limitOrderBook.bestBid) / 2;
                elseif i > 1
                    simRun.intradayPrice(i) = simRun.intradayPrice(i - 1);
                else
                    simRun.intradayPrice(i) = simRun.p_0;
                end
                
                % Update Taker Buy Probability
                simRun = simRun.incrementTakerRandomWalk();
                
                % Update Order Placement Depth
                
                simRun = simRun.incrementPlacementDepth();
                
            end
            
        end
        
        %% ACF Calculation
        function acf = sampleAutoCorrel(simRun, series, lags)  
        % autocorrel: Calculate the autocorrelation function for a given 
        % series and desired number of lags.
        %
        % acf = sampleAutoCorrel(simRun, series, lags) takes in a PreisModel 
        % object, simrun, a time series, series, and a desired number of lags, 
        % lags, and returns the autocorrelation function for the inputted time 
        % series at lag 0 to lag (lags - 1), acf.
        
            % Create Data Structure to Store ACF
            acf = zeros(lags, 1);
            
            % Calculate Series Sample Mean
            muhat = mean(series);

            % Repeat for Desired Number of Lags
            for i = 0 : lags
    
                % Calculate Sample ACF
                acf(i + 1, 1) = sum((series(i + 1 : end) - muhat) .* ...
                    (series(1 : end - i) - muhat)) / sum((series - muhat) .^ 2);
    
            end
            
        end
        
    end
    
end


