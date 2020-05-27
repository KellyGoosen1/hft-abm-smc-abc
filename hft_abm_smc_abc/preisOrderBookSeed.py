import numpy as np
import pandas as pd

class PreisOrderBook:
    # Attributes
    def __init__(limitOrderBook):
        limitOrderBook.orderBook = pd.DataFrame(columns=["limitOrderSize", "limitOrderPrice",
                                                         "limitOrderType", "tradeFlag"])
        limitOrderBook.numSell = 0
        limitOrderBook.numBuy = 0
        limitOrderBook.bestBid = 0
        limitOrderBook.bestAsk = 0

        # buyFlag = None
        # orderSize = None

    # Functions
    # Initiate LOB
    def initLOB():
        # no inputs required
        # returns: PreisOrderBook object (attributes set to their default value, 0).

        # Create Order Book Matrix(orderSize, orderPrice, orderType, tradeFlag)
        orderBook = pd.DataFrame(columns=["limitOrderSize", "limitOrderPrice", "limitOrderType", "tradeFlag"])

        # Set Initial Order Book Metadata
        numSell = 0
        numBuy = 0
        bestBid = 0
        bestAsk = 0

        return ([orderBook, numSell, numBuy, bestBid, bestAsk])

    # Place Market Order
    def marketOrder(limitOrderBook, buyFlag, orderSize):
        # Execute market order of a given size
        #
        # takes in a PreisOrderBook object, limitOrderBook,
        # a binary value, buyFlag, with 1 corresponding to an agent who is buying, and a
        # desired order size, orderSize, and proceeds to execute the market
        # order of the specified size, modifying the bestBid, bestAsk,
        # numBuy and numSell attributes of limitOrderBook accordingly.
        # The modified PreisOrderBook object is then returned.

        # Differentiate Between Buy and Sell Order Cases
        if buyFlag == 0:  # MO wants to sell - need best bid to be high enough

            # Determine Index of Best Bid
            bidIndex = limitOrderBook.numBuy - 1

            # Exit if one or fewer have Buy orders

            if limitOrderBook.numBuy <= 1:
                return

            # Iterate until the Market Order is Fully executed
            while orderSize > 0:

                # size of best bid
                bidSize = limitOrderBook.orderBook.iloc[bidIndex, 0]

                # Check if Best Bid is Large Enough to Satisfy Market Order
                if bidSize > orderSize:
                    #
                    # Update Order Sizes
                    orderSize = orderSize - bidSize
                    limitOrderBook.orderBook.iloc[bidIndex, 0] = bidSize - 1

                else:

                    # Update trade flag
                    limitOrderBook.orderBook.iloc[bidIndex, 3] = 1

                    # Update order size
                    orderSize = orderSize - bidSize

                    # Update index of best bid
                    bidIndex = bidIndex - 1

                    # Update best bid
                    limitOrderBook.bestBid = limitOrderBook.orderBook["limitOrderPrice"].iloc[bidIndex]

                    # Update number of buy orders
                    limitOrderBook.numBuy = limitOrderBook.numBuy - 1

        else:  # market buy not working

            # Determine Index of Best Ask
            askIndex = limitOrderBook.numBuy

            # Exit if One or Fewer Sell Orders
            if limitOrderBook.numSell <= 1:
                return

            # Iterate Until the Market Order is Fully Executed
            while orderSize > 0:

                # Size of Best Ask
                askSize = limitOrderBook.orderBook.iloc[askIndex, 0]

                # Check if Best Ask is Large Enough to Satisfy Market Order
                if askSize > orderSize:

                    # Update Order Sizes
                    limitOrderBook.orderBook.iloc[askIndex, 0] = askSize - 1  # WHY?
                    orderSize = orderSize - askSize

                else:
                    # Update Trade Flag
                    limitOrderBook.orderBook.iloc[askIndex, 3] = 1

                    # Update Index of Best Ask
                    askIndex = askIndex + 1

                    # Update Best Ask
                    limitOrderBook.bestAsk = limitOrderBook.orderBook["limitOrderPrice"].iloc[askIndex]

                    # Update Number of Sell Orders
                    limitOrderBook.numSell = limitOrderBook.numSell - 1

                    # Update Order Size
                    orderSize = orderSize - askSize

        # Remove Executed Orders
        limitOrderBook.orderBook = limitOrderBook.orderBook.loc[limitOrderBook.orderBook["tradeFlag"] != 1]

    # Place Limit Order Function
    def limitOrder(limitOrderBook, buyFlag, orderSize, limitPrice):
        # insert LO in correct position in LOB

        # Differentiate between buy and sell order
        if buyFlag == 1:

            # Create buy LO (2-buy, 1-sell)
            order = np.array([orderSize, limitPrice, 2, 0])

            # Check if any orders exists LOB
            if limitOrderBook.numBuy == 0:

                # Insert into LOB
                limitOrderBook.orderBook = pd.DataFrame(np.insert(
                    limitOrderBook.orderBook.values, 0, values=order, axis=0),
                    columns=["limitOrderSize", "limitOrderPrice",
                             "limitOrderType", "tradeFlag"])
            else:

                # Obtain number of buy orders with a lower price
                lowerOrders = limitOrderBook.orderBook["limitOrderPrice"] < limitPrice
                buyOrders = limitOrderBook.orderBook["limitOrderType"] == 2

                lowerBuyOrders = int(np.floor(np.sum(lowerOrders.values * buyOrders.values)))  # WHY?

                # Insert Order
                if lowerBuyOrders > 0:
                    limitOrderBook.orderBook = pd.DataFrame(np.insert(
                        limitOrderBook.orderBook.values, lowerBuyOrders, order, axis=0),
                        columns=["limitOrderSize", "limitOrderPrice",
                                 "limitOrderType", "tradeFlag"])
                else:
                    store = np.concatenate([np.matrix([order]), limitOrderBook.orderBook.values], axis=0)
                    limitOrderBook.orderBook = pd.DataFrame(store,
                                                            columns=["limitOrderSize", "limitOrderPrice",
                                                                     "limitOrderType", "tradeFlag"])

            # Update number of buy orders
            limitOrderBook.numBuy = limitOrderBook.numBuy + 1

            # check for new best bid
            if (limitPrice > limitOrderBook.bestBid) | (limitOrderBook.bestBid == 0):
                # Update best bid:
                limitOrderBook.bestBid = limitPrice

        else:  # (limit sell order)

            # create sell LO (2-buy, 1-sell)
            order = np.array([orderSize, limitPrice, 1, 0])

            # check if any sell orders exist in LOB:
            if limitOrderBook.numSell == 0:

                # insert order
                limitOrderBook.orderBook = pd.DataFrame(np.insert(
                    limitOrderBook.orderBook.values, limitOrderBook.numBuy, order, axis=0),
                    columns=["limitOrderSize", "limitOrderPrice",
                             "limitOrderType", "tradeFlag"])

            else:

                # number of sell orders with a higher price:
                higherOrders = limitOrderBook.orderBook["limitOrderPrice"] > limitPrice
                sellOrders = limitOrderBook.orderBook["limitOrderType"] == 1

                higherSellOrders = int(np.floor(np.sum(higherOrders.values*sellOrders.values)))

                # place order in correct location
                # Insert order:
                if higherSellOrders > 0:
                    limitOrderBook.orderBook = pd.DataFrame(np.insert(
                        limitOrderBook.orderBook.values,
                        (limitOrderBook.orderBook.shape[0] - higherSellOrders), order, axis=0),
                        columns=["limitOrderSize", "limitOrderPrice",
                                 "limitOrderType", "tradeFlag"])
                else:
                    store = np.concatenate([limitOrderBook.orderBook.values, np.matrix([order])], axis=0)
                    limitOrderBook.orderBook = pd.DataFrame(store,
                                                            columns=["limitOrderSize", "limitOrderPrice",
                                                                     "limitOrderType", "tradeFlag"])

            # Update number of sell orders
            limitOrderBook.numSell = limitOrderBook.numSell + 1

            # check for new best ask
            if (limitPrice < limitOrderBook.bestAsk) | (limitOrderBook.bestAsk == 0):
                # Update best ask
                limitOrderBook.bestAsk = limitPrice
