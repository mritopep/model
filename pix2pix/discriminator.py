from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU


class Discriminator:
    def __new__(cls, image_shape):
        init = RandomNormal(stddev=0.02)

        in_src_image = Input(shape=image_shape)
        in_target_image = Input(shape=image_shape)
        merged = Concatenate()([in_src_image, in_target_image])

        d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(
            merged
        )
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
        patch_out = Activation("sigmoid")(d)

        model = Model([in_src_image, in_target_image], patch_out)

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
        return model
