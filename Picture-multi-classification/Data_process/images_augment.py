import os
import cv2
import random
from matplotlib import pyplot as plt


def cropImg(img):
    bottom = random.randint(img.shape[0]-30, img.shape[0])
    left = random.randint(0, 30)
    right = random.randint(img.shape[1]-30, img.shape[1])
    top = random.randint(0, 30)
    img_crop = img[top:bottom, left:right]
    img_resize = cv2.resize(img_crop, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # B, G, R = cv2.split(img_resize)
    # img_resize_rgb = cv2.merge((R, G, B))
    # plt.imshow(img_resize_rgb)
    return img_resize


def rotateImg(img):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), random.randint(0,15), 1) # center, angle, scale
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # B, G, R = cv2.split(img_rotate)
    # img_rotate_rgb = cv2.merge((R, G, B))
    # plt.imshow(img_rotate_rgb)
    return img_rotate


def read_directory(directory_name, animal_name):
    k = 310
    for filename in os.listdir(directory_name):
        print(directory_name + "/" + filename)
        img = cv2.imread(directory_name + "/" + filename, 1)
        if img is None:
            continue
        B, G, R = cv2.split(img)
        img_rgb = cv2.merge((R, G, B))
        plt.imshow(img_rgb)
        img_new = cropImg(rotateImg(img))
        # B, G, R = cv2.split(img_new)
        # img_new_rgb = cv2.merge((R, G, B))
        # plt.imshow(img_new_rgb)
        cv2.imwrite(directory_name + '/' + animal_name + str(k) + '.jpg', img_new)
        k += 1


read_directory('./train/chickens', 'chickens')
read_directory('./train/rabbits', 'rabbits')