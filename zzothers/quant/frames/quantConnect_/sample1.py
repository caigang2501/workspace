
from datetime import timedelta
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Indicators import SimpleMovingAverage
from QuantConnect.Data.Market import TradeBar

class FourHourAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)  # 设置回测开始日期
        self.SetEndDate(2021, 1, 1)    # 设置回测结束日期
        self.SetCash(10000)            # 设置初始资金

        # 添加 SPY 股票数据，设置为 1 小时分辨率
        self.symbol = self.AddEquity("SPY", Resolution.Hour).Symbol

        # 使用 4 小时 K 线，创建一个 4 小时 K 线合并器
        four_hour_consolidator = TradeBarConsolidator(timedelta(hours=4))
        four_hour_consolidator.DataConsolidated += self.FourHourBarHandler
        self.SubscriptionManager.AddConsolidator(self.symbol, four_hour_consolidator)

        # 创建一个简单移动平均指标
        self.sma = SimpleMovingAverage(14)

    def FourHourBarHandler(self, sender, bar):
        # 在 4 小时 K 线上执行策略逻辑
        if self.sma.IsReady and bar.Close > self.sma:
            # 如果收盘价上穿简单移动平均线，发出买入订单
            self.SetHoldings(self.symbol, 1.0)
        elif self.sma.IsReady and bar.Close < self.sma:
            # 如果收盘价下穿简单移动平均线，发出卖出订单
            self.Liquidate()

    def OnData(self, slice):
        # 在每个数据切片上调用
        pass

