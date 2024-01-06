import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# my_api_key = 'tMv2k5ytCEegPn6iGF9X'
my_api_key = 'JA9JqQJmTe75z8Ev1zPK'

# 假设您已经创建了名为 'data' 的 Quandl 数据源
data = bt.feeds.Quandl(
    dataname='AAPL',
    fromdate=datetime(2010, 1, 1),
    todate=datetime(2020, 1, 1),
    access_key= my_api_key
)

# AAPL AMD MSFT GOOGL
# 道琼斯指数（Dow Jones Industrial Average）的代码为 "^DJI"。
# 纳斯达克指数（NASDAQ Composite Index）的代码为 "^IXIC"。

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.run()
close_prices = np.array([data.lines[0].get(size=len(data))])

fs = 1000  # 采样频率
t = np.arange(0, 1, 1/fs)
signal = 5 * np.sin(2 * np.pi * 50 * t) + 3 * np.sin(2 * np.pi * 120 * t)

print(type(close_prices),close_prices)
print(type(data))
# 提取收盘价数据
close_prices = data.close.get(size=len(data))

# 计算傅里叶变换
# fft_result = np.fft.fft(close_prices)
# frequencies = np.fft.fftfreq(len(close_prices), 1/data.bars[0].freq)  # 获取频率轴

# # 将一半的频域系数设置为零，进行低通滤波（去掉高频部分）
# fft_result[abs(frequencies) > 0.01] = 0

# # 变回时域
# ifft_result = np.fft.ifft(fft_result)

# # 绘制原始曲线
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(data.datetime.datetime(), close_prices)
# plt.title('Original Close Prices')
# plt.xlabel('Time')
# plt.ylabel('Close Price')

# # 绘制变回的时域曲线
# plt.subplot(2, 1, 2)
# plt.plot(data.datetime.datetime(), ifft_result.real)  # 取实部，虚部通常为数值误差
# plt.title('Close Prices Reconstructed from Frequency Domain')
# plt.xlabel('Time')
# plt.ylabel('Close Price')

# plt.tight_layout()
# plt.show()


