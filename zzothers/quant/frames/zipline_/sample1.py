from zipline.api import order, record, symbol
from zipline.algorithm import TradingAlgorithm
import pandas as pd
import numpy as np

# 定义策略
def initialize(context):
    context.asset = symbol('AAPL')
    context.short_window = 40
    context.long_window = 100

# 每个交易日调用一次
def handle_data(context, data):
    # 获取历史价格数据
    prices = data.history(context.asset, fields='price', bar_count=context.long_window, frequency='1d')

    # 计算短期和长期移动平均线
    short_mavg = np.mean(prices[-context.short_window:])
    long_mavg = np.mean(prices)

    # 执行交易策略
    if short_mavg > long_mavg:
        # 短期均线上穿长期均线，执行买入操作
        order(context.asset, 100)
    elif short_mavg < long_mavg:
        # 短期均线下穿长期均线，执行卖出操作
        order(context.asset, -100)

    # 记录策略状态
    record(price=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)

# 运行策略
start_date = pd.Timestamp('2010-01-01', tz='UTC')
end_date = pd.Timestamp('2021-01-01', tz='UTC')
result = TradingAlgorithm(initialize=initialize, handle_data=handle_data).run(start=start_date, end=end_date)

# 可视化结果
result[['portfolio_value', 'price', 'short_mavg', 'long_mavg']].plot(title='Moving Average Crossover Strategy')
