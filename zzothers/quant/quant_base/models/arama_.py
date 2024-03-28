import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 生成示例数据
data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
df = pd.DataFrame(data, columns=['Price'])

# 拟合 ARIMA 模型
model = ARIMA(df['Price'], order=(1, 1, 1))  # 选择合适的阶数
result = model.fit()

# 预测未来数据点
forecast = result.get_forecast(steps=3)
print(forecast.predicted_mean)
