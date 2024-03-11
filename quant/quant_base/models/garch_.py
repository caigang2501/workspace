import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
data_size = 100
returns = np.random.normal(0, 1, data_size)

# 计算收益率和波动率
returns = returns[1:]
volatility = np.sqrt(np.cumsum(returns**2))

# 构造 GARCH 模型
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_result = garch_model.fit(disp='off')

# 预测未来波动率
forecast_horizon = 10
forecast = garch_result.forecast(start=len(returns), horizon=forecast_horizon)

# 可视化结果
plt.plot(volatility, label='True Volatility')
plt.plot(range(len(returns), len(returns) + forecast_horizon), np.sqrt(forecast.variance.values[-1, :]), label='GARCH Forecast')
plt.title('GARCH Volatility Forecast')
plt.legend()
plt.show()
