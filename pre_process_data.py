from os import listdir, path
from os import system as run

import cv2
from med2img.med2img import convert
import numpy as np


def pad_2d(data, r, c):
    res = np.zeros((r, c))
    m, n = data.shape
    res[(r-m)//2:(r-m)//2+m, (c-n)//2:(c-n)//2+n] = data
    return res


def crop_2d(data, r, c):
    m, n = data.shape
    return data[(m-r)//2:(m-r)//2+r, (n-c)//2:(n-c)//2+c]


def gamma_correction(image, gamma=0.15):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table).astype(np.uint8)
    return image


def post_process(mr_folder, pet_folder, target, percentage=8):

    l = listdir(mr_folder)
    v = []
    ps = []
    g = 1
    f = 0
    l.sort()

    for i in l:

        mr_data = pad_2d(gamma_correction(
            cv2.imread(mr_folder + "/" + i, 0)), 256, 256)

        pet_data = pad_2d(cv2.imread(pet_folder + "/" + i, 0), 256, 256)

        a = mr_data.shape[0] * mr_data.shape[1]
        x = len(mr_data[mr_data.min() == mr_data])/a
        y = len(pet_data[pet_data.min() == pet_data])/a

        v.append(abs(x-y)*100)

        data = np.concatenate((pet_data, mr_data), axis=1)

        if abs(x-y)*100 <= percentage:
            cv2.imwrite(path.join(target, str(g)) + '.png', data)
        else:
            f += 1
            ps.append(i)

        g += 1

    print(f, "poor samples")
    return v, l, ps


def pre_process(group, data_folder, img_folder, target_folder, filter_percentage=8):
    subjects = listdir(data_folder)

    g = 0
    for subject in subjects:
        mr_file = path.join(data_folder, subject, "mri.nii")
        pet_file = path.join(data_folder, subject, "pet.nii")
        t1 = path.join("A", str(g))
        t2 = path.join("B", str(g))
        g += 1

        convert(mr_file, path.join(img_folder, t1))
        convert(pet_file, path.join(img_folder, t2))

    print(len(listdir(path.join(img_folder, "A"))),
          "unfiltered images and", g, "subjects for group :", group)

    target = path.join(target_folder, group)

    l, names, ps = post_process(path.join(img_folder, "A"), path.join(
        img_folder, "B"), target, percentage=filter_percentage)


if __name__ == "__main__":
    pre_process("train", "train_data", "train_img", "img_data")

    pre_process("test", "test_data", "test_img", "img_data")
