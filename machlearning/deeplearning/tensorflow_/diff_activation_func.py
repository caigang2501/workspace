import tensorflow as tf

import tensorflow as tf

# 方法一：
def diffaf_model():
    input_size = 22
    output_size = 1
    inputs = tf.keras.Input(shape=(input_size))
    # 定义层，并在该层内部使用不同的激活函数
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    y = tf.keras.layers.Dense(64, activation='sigmoid')(inputs)
    # 合并两个分支
    merged = tf.keras.layers.Concatenate()([x, y])
    # 添加其他层...
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(merged)
    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model 

# 方法二：
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation1, activation2):
        super(MyLayer, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units[1], activation=activation1)
        self.dense2 = tf.keras.layers.Dense(units[2], activation=activation2)

    def call(self, inputs):
        x1 = self.dense1(inputs)
        x2 = self.dense2(inputs)
        return tf.concat([x1, x2], axis=-1)

model = tf.keras.Sequential([
    MyLayer(units=(22,42), activation1='relu', activation2='sigmoid'),
    # 添加其他层...
    tf.keras.layers.Dense(3, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型概述
model.summary()



