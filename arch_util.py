import keras
import numpy as np
import scipy.stats as st
from keras.layers import Layer
import keras.backend as K


def gaussian_filter_block(input_layer,
                          kernel_size=3,
                          strides=(1, 1),
                          dilation_rate=(1, 1),
                          padding="same",
                          activation=None,
                          trainable=False,
                          use_bias=False):
    """
    Build a gaussian filter block
    :return:
    """

    def _gaussian_kernel(kernlen=[21, 21], nsig=[3, 3]):
        """
        Returns a 2D Gaussian kernel array
        """
        assert len(nsig) == 2
        assert len(kernlen) == 2
        kern1d = []
        for i in range(2):
            interval = (2 * nsig[i] + 1.) / (kernlen[i])
            x = np.linspace(-nsig[i] - interval / 2., nsig[i] + interval / 2.,
                            kernlen[i] + 1)
            kern1d.append(np.diff(st.norm.cdf(x)))

        kernel_raw = np.sqrt(np.outer(kern1d[0], kern1d[1]))
        # divide by sum so they all add up to 1
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    # Initialise to set kernel to required value
    def kernel_init(shape, dtype):
        kernel = np.zeros(shape)
        kernel[:, :, 0, 0] = _gaussian_kernel([shape[0], shape[1]])
        return kernel

    return keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=1,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        trainable=trainable,
        depthwise_initializer=kernel_init,
        kernel_initializer=kernel_init)(input_layer)


class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(
            input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(
            input_shape[1], 1), initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W)+self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x*at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()
