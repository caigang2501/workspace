import backtrader as bt
from datetime import datetime
import quandl


my_api_key = 'tMv2k5ytCEegPn6iGF9X'

# 移动平均线
class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=7), bt.ind.SMA(period=20)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

# 移动平均线
class SMAStrategy(bt.Strategy):
    params = dict(sma_period=10)

    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.p.sma_period)
        self.crossover = bt.indicators.CrossOver(self.data.close, self.sma)

    def next(self):
        if self.crossover > 0:
            self.buy()

        if self.crossover < 0:
            self.sell()

# 双均线策略
class DualMovingAverage(bt.Strategy):
    params = dict(
        pfast=10,
        pslow=30
    )

    def __init__(self):
        self.fastma = bt.indicators.SMA(period=self.p.pfast)
        self.slowma = bt.indicators.SMA(period=self.p.pslow)
        self.crossover = bt.indicators.CrossOver(self.fastma, self.slowma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

# 布林带策略
class BollingerBands(bt.Strategy):
    params = dict(
        period=20,
        devfactor=2.0
    )

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.devfactor)
        self.crossover = bt.indicators.CrossOver(self.data.close, self.bollinger.lines.mid)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()


if __name__ == '__main__':
    
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(BollingerBands)
    
    # 获取 Quandl 数据
    data = bt.feeds.Quandl(
        dataname='AAPL',
        fromdate=datetime(2011, 1, 1),
        todate=datetime(2016, 1, 1),
        access_key= my_api_key
    )
    
    # 添加数据
    cerebro.adddata(data)
    
    # 设置起始资金
    cerebro.broker.setcash(100000.0)
    
    # 每次交易时买入的股票数量
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    
    # 设置交易佣金为 0.1%
    cerebro.broker.setcommission(commission=0)

    cerebro.addwriter(bt.WriterFile, csv=True, out='backtest_result.csv')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    results = cerebro.run()
    print('Sharpe Ratio:', 
          results[0].analyzers.sharpe_ratio.get_analysis())