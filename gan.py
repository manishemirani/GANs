from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
z_dim = 100  # noise vector size


# define generator network
def build_gen(image_shape, z_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(img_rows * img_cols * channels, activation="tanh"))
    model.add(Reshape(image_shape))
    return model


# define discriminator network
def build_dis(image_shape):
    model = Sequential()
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation="sigmoid"))
    return model


# define GAN network
def build_gan(gen, dis):
    model = Sequential()
    model.add(gen)
    model.add(dis)
    return model


dis_v = build_dis(img_shape)
dis_v.compile(loss="binary_crossentropy",
              optimizer=Adam(),
              metrics=["acc"])
gen_v = build_gen(img_shape, z_dim)
dis_v.trainable = False  # discriminator weights are stable(every network in GAN has its loss function)
gan_v = build_gan(gen_v, dis_v)
gan_v.compile(loss="binary_crossentropy",
              optimizer=Adam())


def train(iterations, batch_size, interval):
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train / 127.5 - 1.0  # [-1, 1]
    x_train = np.expand_dims(x_train, axis=3)

    real = np.ones((batch_size, 1))  # real labels
    fake = np.zeros((batch_size, 1))  # fake labels

    for iteration in range(iterations):
        ids = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[ids]

        z = np.random.normal(0, 1, (batch_size, z_dim))  # create noise vector (128, 100)
        gen_imgs = gen_v.predict(z)  # images from generator network

        fake_loss = dis_v.train_on_batch(gen_imgs, fake)
        real_loss = dis_v.train_on_batch(imgs, real)

        loss, accuracy = np.add(real_loss, fake_loss) / 2

        z = np.random.normal(0, 1, (batch_size, z_dim))
        gloss = gan_v.train_on_batch(z, real)

        if (iteration + 1) % interval == 0:
            print("%d [D loss: %f , acc: %.2f] [G loss: %f]" %
                  (iteration + 1, loss, 100.0 * accuracy, gloss))
            show_images(gen_v)

def show_images(gen):
    z = np.random.normal(0, 1, (16, 100))
    gen_imgs = gen.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(4, 4, figsize=(4, 4), sharey=True, sharex=True)

    cnt = 0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    fig.show()


train(10000, 128, 1000)
