from keras.optimizers import Adam
from keras.models import Model
from keras.models import Input
from keras.layers import BatchNormalization


class P2PGAN:
    
    def __new__(cls, g_model, d_model, image_shape):

        for layer in d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        in_src = Input(shape=image_shape)
        gen_out = g_model(in_src)
        dis_out = d_model([in_src, gen_out])

        model = Model(in_src, [dis_out, gen_out])

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(
            loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1, 100]
        )
        return model

