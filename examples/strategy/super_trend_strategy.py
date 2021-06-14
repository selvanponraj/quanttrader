#!/usr/bin/env python
# -*- coding: utf-8 -*-
from quanttrader.strategy.strategy_base import StrategyBase
from quanttrader.data.tick_event import TickType
from quanttrader.order.order_event import OrderEvent
from quanttrader.order.order_status import OrderStatus
from quanttrader.order.order_type import OrderType
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

# TODO: remove the dataframe warnings
import warnings
warnings.filterwarnings('ignore')
from ib_insync import *

_logger = logging.getLogger('qtlive')


class SuperTrendStrategy(StrategyBase):
    """
    Dual time frame: 5sec and 15 sec SMA.
    No overnight positions

    Chapter Two Multiple Time Frame Momentum Strategy
    Miner, Robert C. High probability trading strategies: Entry to exit tactics for the forex, futures, and stock markets. Vol. 328. John Wiley & Sons, 2008.
    * Trade in the direction of the larger time frame momentum.
    * Execute the trade following the smaller time frame momentum reversals.
    """
    def __init__(self):
        super(SuperTrendStrategy, self).__init__()
        self.bar_start_time = '08:30:00'        # bar starts earlier
        self.bar_end_time = '16:15:00'          # 16:15; instead, stocks close at 16:00
        self.start_time = '09:30:00'     # trading starts
        self.end_time = '16:14:58'       # 16:14:58
        self.current_pos = 0               # flat
        self.lookback_5sec = 20            # lookback period
        self.lookback_60sec = 20           # lookback period           # lookback period
        self.sma_5sec = 0.0         # sma
        self.sma_60sec = 0.0        # sma

        self.sidx_5sec = 0         # df start idx
        self.eidx_5sec = 0         # df end idx
        self.nbars_5sec = 0            # current bars
        self.sidx_60sec = 0        # df start idx
        self.eidx_60sec = 0        # df end idx
        self.nbars_60sec = 0           # current bars

        _logger.info('SuperTrendStrategy initiated')

    def set_params(self, params_dict=None):
        super(SuperTrendStrategy, self).set_params(params_dict)

        today = datetime.today()
        self.bar_start_time = today.replace(hour=int(self.bar_start_time[:2]), minute=int(self.bar_start_time[3:5]), second=int(self.bar_start_time[6:]), microsecond=0)        
        self.bar_end_time = today.replace(hour=int(self.bar_end_time[:2]), minute=int(self.bar_end_time[3:5]), second=int(self.bar_end_time[6:]), microsecond=0)    
        self.start_time = today.replace(hour=int(self.start_time[:2]), minute=int(self.start_time[3:5]), second=int(self.start_time[6:]), microsecond=0)      
        self.end_time = today.replace(hour=int(self.end_time[:2]), minute=int(self.end_time[3:5]), second=int(self.end_time[6:]), microsecond=0)          

        dt_5sec = np.arange(0, (self.bar_end_time-self.bar_start_time).seconds, 5) 
        idx_5sec = self.bar_start_time + dt_5sec * timedelta(seconds=1)
        self.df_5sec_bar = pd.DataFrame(np.zeros_like(idx_5sec, dtype=[('Open', np.float64), ('High', np.float64), ('Low', np.float64), ('Close', np.float64), ('Volume', np.uint8)]))
        self.df_5sec_bar.index = idx_5sec

        dt_60sec = np.arange(0, (self.bar_end_time-self.bar_start_time).seconds, 60)
        idx_60sec = self.bar_start_time + dt_60sec * timedelta(seconds=1)
        self.df_60sec_bar = pd.DataFrame(np.zeros_like(idx_60sec, dtype=[('Open', np.float64), ('High', np.float64), ('Low', np.float64), ('Close', np.float64), ('Volume', np.uint8)]))

        self.df_60sec_bar.index = idx_60sec

        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=501)
        contract= Future('ES', '20210618', 'GLOBEX')
        # contract = Forex('EURUSD')
        if type(contract) in [CFD,Forex]:
            whatToShow = "MIDPOINT"
        else:
            whatToShow="TRADES"
        barSizeSetting='1 min'
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting=barSizeSetting,
            # whatToShow='ADJUSTED_LAST',
            whatToShow=whatToShow,
            useRTH=False,
            formatDate=1,
            keepUpToDate=True,
        )
        df = pd.DataFrame(bars[:-1], columns=['date', 'open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)
        mask = (df['date'] >= self.start_time)
        df=df.loc[mask].set_index('date')
        df.columns=['Open', 'High', 'Low', 'Close', 'Volume']
        df.rename_axis(None, inplace=True)
        self.df_60sec_bar = self.df_60sec_bar.join(df, lsuffix="DROP").filter(regex="^(?!.*DROP)")
        print(self.df_60sec_bar.head(5))
        print(self.df_60sec_bar.tail(5))
        self.midx_5sec = len(idx_5sec) - 1            # max idx
        self.midx_60sec = len(idx_60sec) - 1        # max idx

    def on_tick(self, k):
        """
        Essentially it does two things:
        1. Aggregate 5sec and 15 sec bars. This is more efficient than subscribing to IB real time bars
            * avoid transmitting a series of bars
            * avoid memory allocaiton of a series of bars
        2. Implement df.dropna().mean() or talib.SMA(df.dropna(), n).iloc[-1] in a more efficient way
            * avoid dropna empty bars for less traded symbols.
            * avoid averaging loop
        """
        super().on_tick(k)     # extra mtm calc

        if k.tick_type != TickType.TRADE:        # only trace trade bars
            return

        if k.timestamp < self.bar_start_time:     # bar_start_time < start_time
            return

        if k.timestamp > self.end_time:          # flat and shutdown
            if self.current_pos != 0:
                o = OrderEvent()
                o.full_symbol = self.symbols[0]
                o.order_type = OrderType.MARKET
                o.order_size = -self.current_pos
                _logger.info(f'EOD flat position, current size {self.current_pos}, order size {o.order_size}')
                self.current_pos = 0
                self.place_order(o)
            return
        
        # #--- 5sec bar ---#
        # while (self.eidx_5sec < self.midx_5sec) and (self.df_5sec_bar.index[self.eidx_5sec+1] < k.timestamp):
        #     self.eidx_5sec += 1
        
        # if self.df_5sec_bar.Open[self.eidx_5sec] == 0.0:       # new bar
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 0] = k.price      # O
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 1] = k.price      # H
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 2] = k.price      # L
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 3] = k.price      # C
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 4] = k.size       # V
        #     self.nbars_5sec += 1
        #     _logger.info(f'New 5sec bar {self.df_5sec_bar.index[self.eidx_5sec]} | {k.timestamp}')
            
        #     if self.nbars_5sec <= self.lookback_5sec:         # not enough bars
        #         self.sma_5sec += k.price/self.lookback_5sec
        #     else:        # enough bars
        #         while self.df_5sec_bar.Close[self.sidx_5sec] == 0.0:
        #             self.sidx_5sec += 1
        #         self.sma_5sec = self.sma_5sec + (k.price - self.df_5sec_bar.Close[self.sidx_5sec]) /self.lookback_5sec
        #         self.sidx_5sec += 1
        # else:  # same bar
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 1] = max(self.df_5sec_bar.High[self.eidx_5sec], k.price)
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 2] = min(self.df_5sec_bar.Low[self.eidx_5sec], k.price)
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 3] = k.price
        #     self.df_5sec_bar.iloc[self.eidx_5sec, 4] = k.size + self.df_5sec_bar.Volume[self.eidx_5sec]
        #     _logger.info(f'existing 5sec bar {self.df_5sec_bar.index[self.eidx_5sec]} | {k.timestamp}')

        #--- 60sec bar ---#
        while (self.eidx_60sec < self.midx_60sec) and (self.df_60sec_bar.index[self.eidx_60sec+1] < k.timestamp):
            self.eidx_60sec += 1
        
        if self.df_60sec_bar.Open[self.eidx_60sec] == 0.0:       # new bar
            self.df_60sec_bar.iloc[self.eidx_60sec, 0] = k.price      # O
            self.df_60sec_bar.iloc[self.eidx_60sec, 1] = k.price      # H
            self.df_60sec_bar.iloc[self.eidx_60sec, 2] = k.price      # L
            self.df_60sec_bar.iloc[self.eidx_60sec, 3] = k.price      # C
            self.df_60sec_bar.iloc[self.eidx_60sec, 4] = k.size       # V
            self.nbars_60sec += 1
            _logger.info(f'New 60sec bar {self.df_60sec_bar.index[self.eidx_60sec]} | {k.timestamp}')
            
            if self.nbars_60sec <= self.lookback_60sec:         # not enough bars
                self.sma_60sec += k.price/self.lookback_60sec
            else:        # enough bars
                while self.df_60sec_bar.Close[self.sidx_60sec] == 0.0:
                    self.sidx_60sec += 1
                self.sma_60sec = self.sma_60sec + (k.price - self.df_60sec_bar.Close[self.sidx_60sec]) /self.lookback_60sec
                self.sidx_60sec += 1

            #--- on 60sec bar ---#
            if (self.nbars_60sec >= self.lookback_60sec) and (k.timestamp > self.start_time):
                # self.dual_time_frame_rule(k.timestamp)
                _logger.info(f'SuperTrendStrategy Calling Rule, { self.nbars_60sec }')
                # print(self.df_60sec_bar.iloc[self.eidx_60sec])
                df = self.df_60sec_bar[self.df_60sec_bar[['Open','Close']].ne(0).any(1)]
                self.supertrend(df,7,3,1)
                self.supertrend(df,10,1,2)
                self.supertrend(df,11,2,3)
                print(df.loc[:, df.columns.isin(['Close', 'Volume','tr','atr1','upperband1','in_uptrend1','atr2','in_uptrend2','atr3','in_uptrend3'])].tail(5))
                # print (df.tail(5))
            else:
                _logger.info(f'SuperTrendStrategy wait for enough bars, { self.nbars_60sec } / { self.lookback_60sec }')
        else:  # same bar
            self.df_60sec_bar.iloc[self.eidx_60sec, 1] = max(self.df_60sec_bar.High[self.eidx_60sec], k.price)
            self.df_60sec_bar.iloc[self.eidx_60sec, 2] = min(self.df_60sec_bar.Low[self.eidx_60sec], k.price)
            self.df_60sec_bar.iloc[self.eidx_60sec, 3] = k.price
            self.df_60sec_bar.iloc[self.eidx_60sec, 4] = k.size + self.df_60sec_bar.Volume[self.eidx_60sec]
            _logger.info(f'Existing 60sec bar {self.df_60sec_bar.index[self.eidx_60sec]} | {k.timestamp}')

    def tr(self,df):
        df['previous_close'] = df['Close'].shift(1)
        df['high-low'] = abs(df['High'] - df['Low'])
        df['high-pc'] = abs(df['High'] - df['previous_close'])
        df['low-pc'] = abs(df['Low'] - df['previous_close'])

        tr = df[['high-low', 'high-pc', 'low-pc']].max(axis=1)
        return tr


    def atr(self,df, period):
        df['tr'] = self.tr(df)
        atr = df['tr'].rolling(period).mean()
        return atr

    def supertrend(self,df, period=7, atr_multiplier=3,idx=1):
        idx=str(idx)
        hl2 = (df['High'] + df['Low']) / 2
        df['atr'+idx] = self.atr(df, period)
        df['upperband'+idx] = hl2 + (atr_multiplier * df['atr'+idx])
        df['lowerband'+idx] = hl2 - (atr_multiplier * df['atr'+idx])
        df['in_uptrend'+idx] = True

        for current in range(1, len(df.index)):
            previous = current - 1

            if df['Close'][current] > df['upperband'+idx][previous]:
                df['in_uptrend'+idx][current] = True
            elif df['Close'][current] < df['lowerband'+idx][previous]:
                df['in_uptrend'+idx][current] = False
            else:
                df['in_uptrend'+idx][current] = df['in_uptrend'+idx][previous]

                if df['in_uptrend'+idx][current] and df['lowerband'+idx][current] < df['lowerband'+idx][previous]:
                    df['lowerband'+idx][current] = df['lowerband'+idx][previous]

                if not df['in_uptrend'+idx][current] and df['upperband'+idx][current] > df['upperband'+idx][previous]:
                    df['upperband'+idx][current] = df['upperband'+idx][previous]
            
        return df

    def dual_time_frame_rule(self, t):
        if self.sma_5sec > self.sma_60sec:
            if self.current_pos <= 0:
                o = OrderEvent()
                o.full_symbol = self.symbols[0]
                o.order_type = OrderType.MARKET
                o.order_size = 1 - self.current_pos
                _logger.info(f'DualTimeFrameStrategy long order placed, on tick time {t}, current size {self.current_pos}, order size {o.order_size}, ma_fast {self.sma_5sec}, ma_slow {self.sma_60sec}')
                self.current_pos = 1
                self.place_order(o)
            else:
                _logger.info(f'DualTimeFrameStrategy keeps long, on tick time {t}, current size {self.current_pos}, ma_fast {self.sma_5sec}, ma_slow {self.sma_60sec}')
        elif self.sma_5sec < self.sma_60sec:
            if self.current_pos >= 0:
                o = OrderEvent()
                o.full_symbol = self.symbols[0]
                o.order_type = OrderType.MARKET
                o.order_size = -1 - self.current_pos
                _logger.info(f'DualTimeFrameStrategy short order placed, on tick time {t}, current size {self.current_pos}, order size {o.order_size}, ma_fast {self.sma_5sec}, ma_slow {self.sma_60sec}')
                self.current_pos = -1
                self.place_order(o)
            else:
                _logger.info(f'DualTimeFrameStrategy keeps short, on tick time {t}, current size {self.current_pos}, ma_fast {self.sma_5sec}, ma_slow {self.sma_60sec}')
