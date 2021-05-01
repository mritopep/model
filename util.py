from numpy import load, zeros, ones
from numpy.random import randint
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
import cv2


def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']

    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = randint(0, trainA.shape[0], n_samples)

    X1, X2 = trainA[ix], trainB[ix]

    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)

    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def pad_2d(data, r, c):
    res = np.zeros((r, c))
    m, n = data.shape

    res[(r-m)//2:(r-m)//2+m, (c-n)//2:(c-n)//2+n] = data
    return res


def crop_2d(data, r, c):
    m, n = data.shape

    return data[(m-r)//2:(m-r)//2+r, (n-c)//2:(n-c)//2+c]


def display_image(image_z, npa):
    plt.imshow(npa[image_z], cmap=plt.cm.Greys_r)
    plt.axis('off')

    plt.show()


def post_process(data, thresholding=False):

    if thresholding:
        img_a = 1 + data[0]
        img_b = 127.5*img_a
        ret, opt2 = cv2.cv2.threshold(img_b, 145, 255, cv2.THRESH_BINARY)

        data[0] = opt2

    return data


def pre_process_mri(image, gamma_correction=False):

    if gamma_correction:
        gamma = 0.15
        invGamma = 1.0 / gamma

        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        image = cv2.LUT(image, table).astype(np.uint8)

    return image


def calc_avg_ssim(fake_images, out_images):

    diff_res = np.zeros(fake_images.shape)
    avg_score = 0
    for i in range(len(fake_images)):
        (score, diff) = ssim(
            fake_images[i], out_images[i], full=True, multichannel=True)
        diff_res[i] = (diff * 255).astype("uint8")
        avg_score += score

    return (avg_score / len(fake_images), diff_res)
