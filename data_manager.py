import re
import os
import numpy as np
import cv2
from config import *
from random import shuffle
import tensorflow as tf
from my_func import rotate
from skimage import exposure
from config import IMAGE_MODE




class DataManager(object):
    def __init__(self, imageList, param, shuffle=True):
        """
        """
        self.shuffle = shuffle
        self.__Param = param
        self.image_list = imageList
        self.data_size = len(imageList)
        self.data_dir = param["data_dir"]
        self.epochs_num = param["epochs_num"]
        if param["mode"] == "visualization" or param["mode"] == "testing":
            self.batch_size = param["batch_size_inference"]
        else:
            self.batch_size = param["batch_size"]
        self.number_batch = int(np.floor(len(self.image_list) / self.batch_size))
        self.next_batch = self.get_next()

    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.string))
        dataset = dataset.repeat(self.epochs_num)
        # if self.shuffle:
        #     dataset = dataset.shuffle(1)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        rand_index = np.arange(len(self.image_list))
        np.random.shuffle(rand_index)
        for index in range(len(self.image_list)):
            image_path = self.image_list[rand_index[index]]
            number = int(image_path.split('\\')[-1].split('-')[0])

            image = self.read_data(image_path)

            image = image / 255

            label = self.scalar2onehot(number)

            if self.__Param["mode"] == "train_decision" or self.__Param["mode"] == "train_segmentation":
                aug_random = np.random.uniform()
                if aug_random > 0.7:
                    pass

                    # adjust_gamma
                    if np.random.uniform() > 0.7:
                        expo = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                        image = exposure.adjust_gamma(image, expo)

                    # # flip
                    # if np.random.uniform() > 0.7:
                    #     aug_seed = np.random.randint(-1, 2)
                    #     image = cv2.flip(image, aug_seed)
                    #
                    #
                    # # rotate
                    # if np.random.uniform() > 0.7:
                    #     angle = np.random.randint(-5, 5)
                    #     image = rotate(image, angle)

                    # GaussianBlur
                    if np.random.uniform() > 0.7:
                        image = cv2.GaussianBlur(image, (5, 5), 0)

                    # # shift
                    # if np.random.uniform() > 0.7:
                    #     dx = np.random.randint(-5, 5)  # width*5%
                    #     dy = np.random.randint(-5, 5)  # Height*10%
                    #     rows, cols = image.shape[:2]
                    #     dx = cols * dx * 0.01
                    #     dy = rows * dy * 0.01
                    #     M = np.float32([[1, 0, dx], [0, 1, dy]])  # (x,y) -> (dx,dy)
                    #     image = cv2.warpAffine(image, M, (cols, rows))
            image = image[:, :, np.newaxis]

            yield image, label, image_path

    def read_data(self, image_path):

        img = cv2.imread(image_path, 0)  # /255.#read the gray image
        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        return img

    def scalar2onehot(self, number):
        label = np.zeros((200,))
        label[number-1] = 1

        return label

    def ImageBinarization(self,img, threshold=1):
        img = np.array(img)
        image = np.where(img > threshold, 1, 0)
        return image

    def rotate(self, image, angle, center=None, scale=1.0):  # 1
        (h, w) = image.shape[:2]  # 2
        if center is None:  # 3
            center = (w // 2, h // 2)  # 4

        M = cv2.getRotationMatrix2D(center, angle, scale)  # 5

        rotated = cv2.warpAffine(image, M, (w, h))  # 6
        return rotated  # 7


