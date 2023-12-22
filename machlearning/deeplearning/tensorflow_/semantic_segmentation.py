import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_unet(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # 编码器部分
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # ...

    # 解码器部分
    up3 = layers.UpSampling2D(size=(2, 2))(conv3)
    up3 = layers.Conv2D(128, 2, activation='relu', padding='same')(up3)
    merge3 = layers.concatenate([conv2, up3], axis=3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    # ...

    # 输出层
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv10)

    model = keras.Model(inputs=inputs, outputs=outputs, name='unet')
    return model


model = build_unet(input_shape=(256, 256, 3), num_classes=21)  # 输入图像大小和类别数需要根据实际情况调整
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))


