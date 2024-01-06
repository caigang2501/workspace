import numpy as np
import tensorflow as tf
import keras
from keras import layers
# 生成示例数据
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

# 构建 Transformer 模型
def transformer_model():
    inputs = layers.Input(shape=(1, 1))
    x = layers.TransformerEncoder(num_layers=2, d_model=64, num_heads=4, dropout=0.3)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="transformer_model")
    model.compile(optimizer='adam', loss='mse')
    return model

# 将数据转换为适用于 Transformer 的格式
X = data[:-1].reshape((len(data)-1, 1, 1))
y = data[1:]

# 构建、训练和预测 Transformer 模型
model = transformer_model()
model.fit(X, y, epochs=100, batch_size=1, verbose=2)

# 预测未来数据点
future_data = np.array([100])
future_data = future_data.reshape((len(future_data), 1, 1))
prediction = model.predict(future_data, verbose=0)
print(prediction)
