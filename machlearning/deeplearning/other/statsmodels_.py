import statsmodels.api as sm

data = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 创建 ARIMA 模型，这里以 ARIMA(1, 1, 1) 为例
order = (1, 1, 1)
model = sm.tsa.ARIMA(data, order=order)

# 拟合模型
result = model.fit()

# 打印模型的统计信息
print(result.summary())

# 获取预测值和置信区间
forecast = result.forecast(steps=3)  # 预测未来 3 个时间步
print(f"Forecast: {forecast}")
