import backtrader as bt

class MyStrategy(bt.Strategy):
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # 等待订单执行

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入: {order.executed.price}')

            elif order.issell():
                self.log(f'卖出: {order.executed.price}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单被取消/保证金不足/拒绝')

        # 将订单重置为 None，以便下一次交易
        self.order = None
    
    def next(self):
        if self.data.close[0] > self.data.close[-1]:
            # 买入逻辑
            self.buy()

        elif self.data.close[0] < self.data.close[-1]:
            # 卖出逻辑
            self.sell()


cerebro = bt.Cerebro()

# 添加策略
cerebro.MyStrategy(MyStrategy)

cerebro.broker = bt.brokers.IBBroker(host='127.0.0.1', port=7496, clientId=100)

cerebro.run()