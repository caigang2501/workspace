import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_simple_object_detection_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # 卷积部分
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten
    x = layers.Flatten()(x)

    # 全连接层
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # 输出层
    outputs = layers.Dense(num_classes + 4, activation='sigmoid')(x)  # 4表示目标边界框的坐标信息

    model = keras.Model(inputs=inputs, outputs=outputs, name='object_detection_model')
    return model

model = build_simple_object_detection_model(input_shape=(256, 256, 3), num_classes=20)  # 输入图像大小和类别数需要根据实际情况调整
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
