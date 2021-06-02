import keras
import numpy as np
import scipy.stats as st
from keras.layers import Layer
import keras.backend as K
from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate
from tensorflow.keras.layers import Layer,InputLayer
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

def gaussian_filter_block(input_layer,
                          kernel_size=3,
                          strides=(1, 1),
                          dilation_rate=(1, 1),
                          padding="same",
                          activation=None,
                          trainable=True,
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


# class attention(Layer):
#     def __init__(self, **kwargs):
#         super(attention, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(
#             input_shape[-1], 1), initializer="normal")
#         self.b = self.add_weight(name="att_bias", shape=(
#             input_shape[1], 1), initializer="zeros")
#         super(attention, self).build(input_shape)

#     def call(self, x):
#         et = K.squeeze(K.tanh(K.dot(x, self.W)+self.b), axis=-1)
#         at = K.softmax(et)
#         at = K.expand_dims(at, axis=-1)
#         output = x*at
#         return K.sum(output, axis=1)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[-1])

#     def get_config(self):
#         return super(attention, self).get_config()

# def get_config(self):
#         return super(attention, self).get_config()
    
# def gaussian_glimpse(img_tensor, transform_params):
#     # parse arguments
#     h, w = (img_tensor.shape[1],img_tensor.shape[2])
#     H, W = img_tensor.shape.as_list()[1:3]
#     split_ax = transform_params.shape.ndims -1
#     uy, sy, dy, ux, sx, dx = tf.split(transform_params, 6, split_ax)
#     # create Gaussian masks, one for each axis
#     Ay = gaussian_mask(uy, sy, dy, h, H)
#     Ax = gaussian_mask(ux, sx, dx, w, W)
#     # extract glimpse
#     glimpse = tf.matmul(tf.matmul(Ay, img_tensor, adjoint_a=True), Ax)
#     return glimpse

# tx = tf.placeholder(tf.float32, x.shape, 'image')
# tu = tf.placeholder(tf.float32, [1], 'u')
# ts = tf.placeholder(tf.float32, [1], 's')
# td = tf.placeholder(tf.float32, [1], 'd')

# gaussian_att_params = tf.concat([tu, ts, td, tu, ts, td], -1)
# gaussian_glimpse_expr = gaussian_glimpse(tx, gaussian_att_params, )

# def gaussian_mask(u, s, d, R, C):
#     """
#     :param u: tf.Tensor, centre of the first Gaussian.
#     :param s: tf.Tensor, standard deviation of Gaussians.
#     :param d: tf.Tensor, shift between Gaussian centres.
#     :param R: int, number of rows in the mask, there is one Gaussian per row.
#     :param C: int, number of columns in the mask.
#     """
#     # indices to create centres
#     R = tf.to_float(tf.reshape(tf.range(R), (1, 1, R)))
#     C = tf.to_float(tf.reshape(tf.range(C), (1, C, 1)))
#     centres = u[np.newaxis, :, np.newaxis] + R * d
#     column_centres = C - centres
#     mask = tf.exp(-.5 * tf.square(column_centres / s))
#     # we add eps for numerical stability
#     normalised_mask = mask / (tf.reduce_sum(mask, 1, keep_dims=True) + 1e-8)
#     return normalised_mask

class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, inputs):
        hidden_states = inputs
        hidden_size = hidden_states.shape
        flattened = Flatten()(hidden_states)
        score_first_part = Dense(np.prod(hidden_size), use_bias=False, name='attention_score_vec')(flattened)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(flattened)
        score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = Dot(axes=[1, 1], name='context_vector')([flattened, attention_weights])
        pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
        attention_vector = Dense(np.prod(hidden_size), use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        _output = Reshape(hidden_size)(attention_vector)
        return _output

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate
# from tensorflow.keras.layers import Layer


# class Attention(Layer):

#     def __init__(self, units=128, **kwargs):
#         self.units = units
#         super().__init__(**kwargs)

#     def __call__(self, inputs):
#         """
#         Many-to-one attention mechanism for Keras.
#         @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
#         @return: 2D tensor with shape (batch_size, 128)
#         @author: felixhao28, philipperemy.
#         """
#         hidden_states = inputs
#         hidden_size = int(hidden_states.shape[2])
#         # Inside dense layer
#         #              hidden_states            dot               W            =>           score_first_part
#         # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
#         # W is the trainable weight matrix of attention Luong's multiplicative style score
#         score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
#         #            score_first_part           dot        last_hidden_state     => attention_weights
#         # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
#         h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
#         score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
#         attention_weights = Activation('softmax', name='attention_weight')(score)
#         # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
#         context_vector = Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
#         pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
#         attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
#         return attention_vector

#     def get_config(self):
#         return {'units': self.units}

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)