classdef PreisOrderBook
% PreisOrderBook: Order book storage and related methods for the
% implementation of the Preis et al. (2006) model.
% 
% This class aims to encapsulate all the attributes and methods required 
% to implement the storage, update processes and execution of orders in a 
% virtual limit order book, with the aim of aiding the implementation of the 
% Preis et al. (2006) model.
%
% The class has attributes:
%
% Name:               Description:                                  Type:
% orderBook           Limit order book matrix                       (Dynamic Size Matrix)
% numSell             Number of sell orders in limit order book     (Integer)
% numBuy              Number of buy orders in limit order book      (Integer)
% bestBid             Current best bid in limit order book          (Integer)
% bestAsk             Current best ask in limit order book          (Integer) 
%
% Order Book Structure:
%
% The order book consists of 4 columns: Order Size, Order Price, Order Type 
% (Buy/Sell) and Trade Flag. Columns 1 and 2 are self explanatory, column 3 
% holds a 1 for sell orders and a 2 for buy orders, and column 4 is used to 
% mark orders for removal in the matching algorithm.
%
% References:
% 1. Preis T, Golke S, Paul W, Schneider (2006) Multi-agent-based Order Book 
% Model of financial markets. Europhys Lett 75(3):510-516
% 
% See also PreisModel, PreisOrderBook\PreisOrderBook, marketOrder,
% limitOrder.
%
% Donovan Platt
% School of Computer Science and Applied Mathematics
% University of the Witwatersrand, Johannesburg, South Africa
    
    %% Class Attributes
    properties    
        %% Limit Order Data Attributes
        orderBook   %Limit order book
        
        %% Limit Order Book Metadata Attributes
        numSell     %Number of sell orders in limit order book
        numBuy      %Number of buy orders in limit order book
        bestBid     %Current best bid in limit order book
        bestAsk     %Current best ask in limit order book
        
    end
    
    %% Class Methods
    methods
        %% Class Constructor Method
        function limitOrderBook = PreisOrderBook()
        % PreisOrderBook: Class constructor method.
        %
        % limitOrderBook = PreisOrderBook() requires no inputs and creates
        % and returns a PreisOrderBook object with all attributes set to 
        % their default value, 0.
        
            % Create Order Book Matrix
            limitOrderBook.orderBook = [];
            
            % Set Initial Order Book Metadata
            limitOrderBook.numSell = 0;
            limitOrderBook.numBuy = 0;
            limitOrderBook.bestBid = 0;
            limitOrderBook.bestAsk = 0;
         
        end
        
        %% Place Market Order
        function limitOrderBook = marketOrder(limitOrderBook, buyFlag, orderSize)
        % marketOrder: Execute a market order of a certain size.
        %
        % limitOrderBook = marketOrder(limitOrderBook, buyFlag, orderSize)
        % takes in a PreisOrderBook object, limitOrderBook, a binary value, 
        % buyFlag, with 1 corresponding to an agent who is buying, and a
        % desired order size, orderSize, and proceeds to execute the market
        % order of the specified size, modifying the bestBid, bestAsk,
        % numBuy and numSell attributes of limitOrderBook accordingly. The
        % modified PreisOrderBook object is then returned.    
        
            % Differentiate Between Buy and Sell Order Cases
            if buyFlag == 0
                
                % Determine Index of Best Bid
                bidIndex = limitOrderBook.numBuy;
                
                % Exit if One or Fewer Buy Orders
                if limitOrderBook.numBuy <= 1
                    return;
                end
                
                % Iterate Until the Market Order is Fully Executed
                while orderSize > 0
                    
                    % Size of Best Bid
                    bidSize = limitOrderBook.orderBook(bidIndex, 1);
                    
                    % Check if Best Bid is Large Enough to Satisfy Market Order
                    if bidSize > orderSize
                        
                        % Update Order Sizes     
                        orderSize = orderSize - bidSize;
                        limitOrderBook.orderBook(bidIndex, 1) = bidSize - orderSize;
                                               
                    else
                        
                        % Update Trade Flag
                        limitOrderBook.orderBook(bidIndex, 4) = 1;
                        
                        % Update Order Size
                        orderSize = orderSize - bidSize;
                        
                        % Update Index of Best Bid
                        bidIndex = bidIndex - 1;
                        
                        % Update Best Bid
                        limitOrderBook.bestBid = limitOrderBook.orderBook(bidIndex, 2);
                        
                        % Update Number of Buy Orders
                        limitOrderBook.numBuy = limitOrderBook.numBuy - 1;                   
                                                
                    end
                    
                end
                
            else
                
                % Determine Index of Best Ask
                askIndex = limitOrderBook.numBuy + 1;
                
                % Exit if One or Fewer Sell Orders
                if limitOrderBook.numSell <= 1
                    return;
                end
                
                % Iterate Until the Market Order is Fully Executed
                while orderSize > 0
                    
                    % Size of Best Ask
                    askSize = limitOrderBook.orderBook(askIndex, 1);
                    
                    % Check if Best Ask is Large Enough to Satisfy Market Order
                    if askSize > orderSize
                        
                        % Update Order Sizes     
                        orderSize = orderSize - askSize;
                        limitOrderBook.orderBook(askIndex, 1) = askSize - orderSize;
                        
                    else
                        
                        % Update Trade Flag
                        limitOrderBook.orderBook(askIndex, 4) = 1;
                        
                        % Update Order Size
                        orderSize = orderSize - askSize;
                        
                        % Update Index of Best Ask
                        askIndex = askIndex + 1;
                        
                        % Update Best Ask
                        limitOrderBook.bestAsk = limitOrderBook.orderBook(askIndex, 2);
                        
                        % Update Number of Sell Orders
                        limitOrderBook.numSell = limitOrderBook.numSell - 1;
                                                
                    end
                    
                end
                
            end
            
            % Remove Executed Orders
            limitOrderBook.orderBook = limitOrderBook.orderBook(...
                limitOrderBook.orderBook(:, 4) ~= 1, :);
                        
        end
            
        %% Place Limit Order
        function limitOrderBook = limitOrder(limitOrderBook, buyFlag, ...
                orderSize, limitPrice)
        % limitOrder: Insert a specified limit order into the correct
        % location in the limit order book.
        %
        % limitOrderBook = limitOrder(limitOrderBook, buyFlag, orderSize, 
        % limitPrice) takes in a PreisOrderBook object, limitOrderBook, a 
        % binary value, buyFlag, with 1 corresponding to an agent who is 
        % buying, the size of the order, orderSize and the limit price,
        % limitPrice, and proceeds to insert the corresponding order into
        % the orderBook attribute stored in the provided PreisOrderBook 
        % object. Thereafter, the bestBid, bestAsk, numBuy and numSell 
        % attributes are modified appropriately and the modified 
        % PreisOrderBook object is then returned.
        
            % Differentiate Between Buy and Sell Order Cases
            if buyFlag == 1
                
                % Create Order
                order = [orderSize, limitPrice, 2, 0];

                % Check if Orders Exist in Book
                if limitOrderBook.numBuy == 0
                    
                    % Insert Order
                    limitOrderBook.orderBook = [order; limitOrderBook.orderBook];
                    
                else
                
                    % Determine Number of Buy Orders with a Lower Price
                    lowerOrders = limitOrderBook.orderBook(:, 2) < limitPrice;
                    buyOrders = limitOrderBook.orderBook(:, 3) == 2;
                    lowerBuyOrders = sum(lowerOrders .* buyOrders);

                    % Place Order in Correct Location
                    if lowerBuyOrders ~= 0 

                        % Insert Order
                        limitOrderBook.orderBook = [limitOrderBook.orderBook(1 : ...
                            lowerBuyOrders, :); order; limitOrderBook.orderBook( ...
                            lowerBuyOrders + 1 : end, :)];

                    else

                        % Insert Order
                        limitOrderBook.orderBook = [order; limitOrderBook.orderBook];

                    end

                end
                
                % Update Number of Buy Orders
                limitOrderBook.numBuy = limitOrderBook.numBuy + 1;
                    
                % Check for New Best Bid
                if (limitPrice > limitOrderBook.bestBid) || (limitOrderBook.bestBid == 0)
                    
                    % Update Best Bid
                    limitOrderBook.bestBid = limitPrice;
                               
                end
                      
            else
                
                % Create Order
                order = [orderSize, limitPrice, 1, 0];
                
                % Check if Sell Orders Exist in Book
                if limitOrderBook.numSell == 0
                    
                    % Insert Order
                    limitOrderBook.orderBook = [limitOrderBook.orderBook; order];
                    
                else
                
                    % Determine Number of Sell Orders with a Higher Price
                    higherOrders = limitOrderBook.orderBook(:, 2) > limitPrice;
                    sellOrders = limitOrderBook.orderBook(:, 3) == 1;
                    higherSellOrders = sum(higherOrders .* sellOrders);

                    % Place Order in Correct Location
                    if higherSellOrders ~= 0 

                        % Insert Order
                        limitOrderBook.orderBook = [limitOrderBook.orderBook(1 : ...
                            end - higherSellOrders, :); order; limitOrderBook.orderBook( ...
                            end - higherSellOrders + 1 : end, :)];
                    else

                        % Insert Order
                        limitOrderBook.orderBook = [limitOrderBook.orderBook; order];                              

                    end
                    
                end
               
                % Update Number of Sell Orders
                limitOrderBook.numSell = limitOrderBook.numSell + 1;

                % Check for new Best Ask
                if (limitPrice < limitOrderBook.bestAsk) || (limitOrderBook.bestAsk == 0)
                    
                    % Update Best Ask
                    limitOrderBook.bestAsk = limitPrice;
                                        
                end
                
            end 
                        
        end
    
    end
    
end