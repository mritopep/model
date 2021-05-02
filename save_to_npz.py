# med2image, opencv should be installed in the python environment
# assuming the system has Linux OS

from os import listdir
from os import system as run
import numpy as np
from numpy import savez_compressed
import cv2


def pre_process_mri(image, gamma_correction=False):

    if gamma_correction:
        gamma = 0.15
        invGamma = 1.0 / gamma

        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        image = cv2.LUT(image, table).astype(np.uint8)

    return image


# file name should be having .nii
def save_to_npz(file, des = ""):

    run("mkdir images")
    
    run("med2image -i " + file + " -o /images/slice")
    
    folder = "images/"
    files = sorted(listdir(folder))
    list = []

    for i in range(len(files)):

        data = cv2.imread(folder + files[i])
        data = pre_process_mri(data, gamma_correction=True)

        list.append(data)

    list = np.asarray(list)
    
    if des != "" : des += "/"
    
    filename = des + 'images.npz'
    savez_compressed(filename, list)
    
    run("rm -rf images")

    return filename


