from __future__ import print_function, division
import scipy

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os


class Pix2Pix():
    def __init__(self, dataset_name, generator=None, discriminator=None, combined=None):
       
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        
        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

       
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

       
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        
        if discriminator:
            self.discriminator = discriminator
        else:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='mse',
                                       optimizer=optimizer,
                                       metrics=['accuracy'])

        
        if generator:
            self.generator = generator
        else:
            self.generator = self.build_generator()

        if combined:
            self.combined = combined
        else:
            
            img_A = Input(shape=self.img_shape)
            img_B = Input(shape=self.img_shape)

            
            fake_A = self.generator(img_B)

            
            self.discriminator.trainable = False

            
            valid = self.discriminator([fake_A, img_B])

            self.combined = Model(
                inputs=[img_A, img_B], outputs=[valid, fake_A])
            self.combined.compile(loss=['mse', 'mae'],
                                  loss_weights=[1, 100],
                                  optimizer=optimizer)

    def build_generator(self):
        

        def conv2d(layer_input, filters, f_size=4, bn=True):
            
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1,
                       padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        
        d0 = Input(shape=self.img_shape)

        
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)


        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):

            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1,  include_val=True, step_print=100):

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        sp = 1
        gen_loss = 0
        disc_loss_1 = 0
        disc_loss_2 = 0

        for epoch in range(epochs):
            el = 0
            c = 0

            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size, include_val)):

                fake_A = self.generator.predict(imgs_B)

                
                d_loss_real = self.discriminator.train_on_batch(
                    [imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch(
                    [fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                
                g_loss = self.combined.train_on_batch(
                    [imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time

                el += g_loss[0]
                c += 1

                gen_loss += g_loss[0]
                disc_loss_2 += d_loss[1]
                disc_loss_1 += d_loss[0]

                if sp % step_print == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                          batch_i, self.data_loader.n_batches,
                                                                                                          disc_loss_1 / step_print, 100 *
                                                                                                          disc_loss_2 / step_print,
                                                                                                          gen_loss / step_print,
                                                                                                          elapsed_time))
                    gen_loss = 0
                    disc_loss_2 = 0
                    disc_loss_1 = 0

                sp += 1
              

            print("Avg. epoch loss :", "%.4f" % (el/c))

    def sample_images(self, samples = 3, is_test = True):
        
        r, c = 3, samples

        imgs_A, imgs_B = self.data_loader.load_data(
            batch_size=samples, is_testing=is_test)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
  
