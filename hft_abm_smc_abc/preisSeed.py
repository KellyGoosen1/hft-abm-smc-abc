# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:55:27 2019

@author: Kelly.Goosen
"""

import numpy as np
import math

from hft_abm_smc_abc.preisOrderBookSeed import PreisOrderBook


class PreisModel:
    # Single complete Preis model iteration

    # Attributes
    def __init__(self, N_A, delta, mu, alpha, lambda_0, C_lambda, delta_S, p_0, T, MC):
        # Trader population attributes:
        self.N_A = N_A  # number of liquidity providers = takers = N_A

        # Trade order cancellation attributes
        self.delta = delta  # Limit order probability cancellation

        # Liquidity provider order placement attributes
        self.lambda_0 = lambda_0  # initial order placement depth
        self.lambda_t = None  # current order placement depth
        self.C_lambda = C_lambda  # placement depth coefficient

        # Trader buy/sell decision attributes
        self.q_provider = None  # =0.5 (constant) buy probability of liquidity providers
        self.q_taker = None  # buy probability of liquidity takers (mean 0.5)
        self.q_mean_dev = None  # variance of q_taker
        self.delta_S = delta_S  # liquidity taker buy probability random walk increment

        # Order Book attributes
        self.limitOrderBook = None

        # Price time series attributes
        self.intradayPrice = np.zeros((T, 1))  # Timeseries of intraday prices (mid-prices)
        self.p_0 = p_0  # initial price value

        # Trader activation frequency attributes
        self.alpha = alpha  # activation frequency for liquidity providers
        self.mu = mu  # activation frequency for liquidity takers

        # Simulation attributes
        self.T = T  # timehorizon
        self.MC = MC  # MC steps to estimate standard dev of q_taker
        self.tradeSigns = []  # series of trade signs
        self.u01 = None

    # Update limit order buy probability
    def incrementTakerRandomWalk(self):
        q_current = self.q_taker

        # mean reversion probability
        p_revert = 0.5 + abs(q_current - 0.5)

        # Up probability
        if q_current < 0.5:

            # below the mean, up move more likely:
            p_up = p_revert

        else:

            # above the mean, down move more likely:
            p_up = 1 - p_revert

        if np.random.random(1) <= p_up:
            self.q_taker = q_current + self.delta_S
        else:
            self.q_taker = q_current - self.delta_S

    # Construct Preis
    # Requires incrementTalkerRandomWalk function and PreisOrderBook class
    def simRun(self):
        # requires the following methods:
        # incrementTalkerRandomWalk
        # Preis order book

        self.lambda_t = self.lambda_0
        self.C_lambda = self.C_lambda

        self.q_provider = 1 / 2
        self.q_taker = 1 / 2

        self.q_taker_sim = np.zeros((self.MC, 1))

        for i in range(self.MC):
            self.u01 = np.random.random(1)
            self.incrementTakerRandomWalk()
            self.q_taker_sim[i,] = self.q_taker - 0.5

        self.q_mean_dev = np.dot(self.q_taker_sim.T, self.q_taker_sim) / self.MC

        # reset q_taker
        self.q_taker = 1 / 2

        self.limitOrderBook = PreisOrderBook()

        self.limitOrderBook.bestAsk = self.p_0
        self.limitOrderBook.bestBid = self.p_0

        self.intradayPrice = np.zeros((self.T, 1))

    def calcProviderLimitPrice(self, u01, u02):
        # function to determine limit order price

        # buy rand <= q_provider then buy
        if u01 <= self.q_provider:

            eta = np.floor(-self.lambda_t * math.log(u02))

            # set buy limit price

            limitPrice = self.limitOrderBook.bestAsk - 1 - eta

            # set buy flag
            buyFlag = 1

        else:

            eta = np.floor(-self.lambda_t * math.log(u02))

            # set sell limit price
            limitPrice = self.limitOrderBook.bestBid + 1 + eta

            # set sell flag
            buyFlag = 0

        return ([limitPrice, buyFlag])

    def incrementPlacementDepth(self):
        # Increment order placement depth

        # current buy probability of liquidity taker
        q_current = self.q_taker

        # calculate new placement depth
        self.lambda_t = self.lambda_0 * (1 + abs(q_current - 0.5) / math.sqrt(self.q_mean_dev) * self.C_lambda)

    def placeOrder(self, traderType, u01, u02):
        # function to place a limit order

        orderSize = 1  # model default

        # differentiate between trader types (provider and taker)
        if traderType == 1:  # liquidity provider (limit order)
            [limitPrice, buyFlag] = self.calcProviderLimitPrice(u01, u02)
            # bestBid for LOs not updating properly silly
            # place limit order
            self.limitOrderBook.limitOrder(buyFlag, orderSize, limitPrice)


        else:  # taker (market order)

            # buyer
            if u01 <= self.q_taker:

                # buyer
                buyFlag = 1

                # Update trade sign series
                self.tradeSigns.append(1)

            else:  # seller

                # seller:
                buyFlag = 0

                # Update trade sign series
                self.tradeSigns.append(-1)

            # place market order
            self.limitOrderBook.marketOrder(buyFlag, orderSize)

    def initialize(self):
        # initialise simulation
        # fill orderbook before trading starts with LOs

        # Extract random numbers for placing limit orders
        numInit = int(np.floor(self.N_A * 10 * self.alpha))

        for i in range(numInit):  # num traders *10* rate of trading

            # place orders
            u01 = np.random.random(1)
            u02 = np.random.random(1)
            self.placeOrder(1, u01, u02)  # 1 being provider

    def simulate(self):
        # simulate trading for T MC steps 

        # simulate intial orders of:
        numLimitOrders = int(np.floor(self.N_A * self.alpha))
        numMarketOrders = int(np.floor(self.N_A * self.mu))

        # simulate time series of T MC steps
        for i in range(self.T):

            # providers place new Limit Orders
            for j in range(numLimitOrders):
                # Random numbers extract
                u01 = np.random.random(1)
                u02 = np.random.random(1)

                # place Limit Order

                self.placeOrder(1, u01, u02)

            # takers place new market orders
            for j in range(numMarketOrders):
                # Random numbers extract
                u01 = np.random.random(1)

                # place market order
                self.placeOrder(2, u01, u01)

            # Random numbers extract
            uSeq = np.random.random(self.limitOrderBook.orderBook.shape[0])

            # determine orders to be kept: (if greater than delta then avoids removing)
            keepIndices = uSeq > self.delta

            # update number of buy orders
            self.limitOrderBook.numBuy = self.limitOrderBook.numBuy - sum(
                np.array(self.limitOrderBook.orderBook["limitOrderType"].iloc[keepIndices == 0] == 2))

            # update number of sell orders
            self.limitOrderBook.numSell = self.limitOrderBook.numSell - sum(
                np.array(self.limitOrderBook.orderBook["limitOrderType"].iloc[keepIndices == 0] == 1))

            # cancel orders
            self.limitOrderBook.orderBook = self.limitOrderBook.orderBook.iloc[keepIndices == 1,]

            # update best bid
            if self.limitOrderBook.numBuy > 0:
                self.limitOrderBook.bestBid = self.limitOrderBook.orderBook["limitOrderPrice"].iloc[
                    (self.limitOrderBook.numBuy - 1)]
            else:
                self.limitOrderBook.bestBid = 0

            # update best ask
            if self.limitOrderBook.numSell > 0:
                self.limitOrderBook.bestAsk = self.limitOrderBook.orderBook["limitOrderPrice"].iloc[
                    self.limitOrderBook.numBuy]
            else:
                self.limitOrderBook.bestAsk = 0

                # Update price (mid-price)
            if (self.limitOrderBook.bestBid != 0) & (self.limitOrderBook.bestAsk != 0):
                self.intradayPrice[i,] = (self.limitOrderBook.bestBid + self.limitOrderBook.bestAsk) / 2
            elif i > 0:
                self.intradayPrice[i,] = self.intradayPrice[(i - 1),]
            else:
                self.intradayPrice[i,] = self.p_0

            self.incrementTakerRandomWalk()

            # update order placement depth
            self.incrementPlacementDepth()

    def sampleAutoCorrelation(self, series, lags):
        # autocorrelaion for a given series and a desired number of lags

        # data structure to store ACF
        acf = np.zeros(lags, 1)

        # series sample mean
        mu_hat = np.mean(series)

        # repeat for desired number of lags
        for i in range(lags):
            # calculate sample acf
            acf[i, 1] = sum((series[i:] - mu_hat) * (series[:-i] - mu_hat) / sum(series - mu_hat) ^ 2)
