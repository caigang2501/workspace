import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

# 构建生成对抗网络 (GAN) 模型
def build_gan(img_shape):
    generator = build_generator(img_shape)
    discriminator = build_discriminator(img_shape)

    discriminator.trainable = False

    gan_input = keras.Input(shape=img_shape)
    generated_img = generator(gan_input)
    gan_output = discriminator(generated_img)

    gan = keras.models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

# 构建生成器模型
def build_generator(img_shape):
    model = keras.Sequential()
    model.add(layers.Dense(256, input_dim=np.prod(img_shape), activation='relu'))
    model.add(layers.Reshape((8, 8, 4), input_shape=(256,)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid'))
    return model

# 构建判别器模型
def build_discriminator(img_shape):
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=img_shape, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# 加载图像数据集
def load_dataset():
    # 这里可以替换为你自己的图像数据集加载逻辑
    # 这里假设图像已经被处理成相同的尺寸
    # 返回一个 NumPy 数组，每个元素是一个图像
    pass

# 训练 GAN 模型
def train_gan(gan, generator, discriminator, dataset, img_shape, epochs=10000, batch_size=64):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        idx = np.random.randint(0, dataset.shape[0], half_batch)
        imgs = dataset[idx]

        noise = np.random.normal(0, 1, (half_batch, np.prod(img_shape)))
        gen_imgs = generator.predict(noise)

        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, np.prod(img_shape)))
        valid = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# 主函数
def main():
    img_shape = (64, 64, 1)  # 替换为你的图像尺寸
    dataset = load_dataset()

    generator = build_generator(img_shape)
    discriminator = build_discriminator(img_shape)
    gan = build_gan(img_shape)

    train_gan(gan, generator, discriminator, dataset, img_shape)

    # 生成并显示补全后的图像
    noise = np.random.normal(0, 1, (1, np.prod(img_shape)))
    completed_img = generator.predict(noise).reshape(img_shape)

    plt.imshow(completed_img.squeeze(), cmap='gray')
    plt.title('Completed Image')
    plt.show()

if __name__ == '__main__':
    main()
