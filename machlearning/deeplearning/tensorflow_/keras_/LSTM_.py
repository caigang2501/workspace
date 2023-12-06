from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# 生成一些示例数据（假设是时间序列数据）
data_dim = 16
timesteps = 8
num_samples = 1000

x_train = np.random.random((num_samples, timesteps, data_dim))
y_train = np.random.randint(2, size=(num_samples,))
print(x_train.shape,'\n',x_train[0],'\n',y_train.shape,y_train[0])
# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, data_dim)))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32)
