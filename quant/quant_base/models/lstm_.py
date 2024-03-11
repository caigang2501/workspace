import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 生成示例数据
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 将数据转换为适用于 LSTM 的格式
X = data[:-1].reshape((len(data)-1, 1, 1))
y = data[1:]

# 训练模型
model.fit(X, y, epochs=100, batch_size=1, verbose=2)

# 预测未来数据点
future_data = np.array([100])
future_data = future_data.reshape((len(future_data), 1, 1))
prediction = model.predict(future_data, verbose=0)
print(prediction)
