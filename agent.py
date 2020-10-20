import tensorflow as tf
import numpy as np
# import cv2
import os
from PIL import Image
# import time
from data_manager import DataManager
from model import Model
from config import IMAGE_SIZE, TRAIN_MODE_IN_TRAIN, TRAIN_MODE_IN_TEST, TRAIN_MODE_IN_VALID, IMAGE_MODE, TEST_RATIO
import utils
# from datetime import datetime
from tqdm import tqdm
import pathlib
from timeit import default_timer as timer
from iouEval import iouEval
from matplotlib import pyplot as plt


class Agent(object):
    def __init__(self, param):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.__Param = param
        self.init_datasets()  # 初始化数据管理器
        self.model = Model(self.sess, self.__Param)  # 建立模型
        self.logger = utils.get_logger(param["Log_dir"])

    def run(self):
        if self.__Param["mode"] == "train_decision":
            self.train_decision()
        elif self.__Param["mode"] == "savePb":
            self.savePb()
        elif self.__Param["mode"] == "testing":
            self.test()
        else:
            print("got a unexpected mode ,please set the mode  'training', 'testing' or 'savePb' ")

    def init_datasets(self):
        if self.__Param["mode"] != "savePb":
            self.image_list_train = self.listData_train(self.__Param["data_dir"])
            self.DataManager_train = DataManager(self.image_list_train, self.__Param)
            self.DataManager_valid = DataManager(self.image_list_train, self.__Param, shuffle=False)

    def train_decision(self):

        with self.sess.as_default():
            self.logger.info('start training decision net')

            print('Start training for {} epoches, {} steps per epoch'.format(self.__Param["epochs_num"],
                                                                             self.DataManager_train.number_batch))
            best_loss = 10000
            for i in range(self.model.step, self.__Param["epochs_num"] + self.model.step):
                print('Epoch {}:'.format(i))
                with tqdm(total=self.DataManager_train.number_batch) as pbar:
                    # epoch start
                    iter_loss = 0.0
                    num_step = 0.0
                    accuracy = 0.0
                    guo = 0
                    lou = 0
                    for batch in range(self.DataManager_train.number_batch):
                        # batch start
                        img_batch, label_batch, _ = self.sess.run(self.DataManager_train.next_batch)

                        loss_value_batch = 0

                        _, loss_value_batch, decision_out = self.sess.run([self.model.optimize_decision,
                                                                           self.model.decision_loss,
                                                                           self.model.decison_out],
                                                                           feed_dict={self.model.Image: img_batch,
                                                                                      self.model.label: label_batch,
                                                                                      self.model.is_training_seg: TRAIN_MODE_IN_TRAIN,
                                                                                      self.model.is_training_dec: TRAIN_MODE_IN_TRAIN})

                        iter_loss += loss_value_batch
                        pbar.update(1)

                    false_count = 0
                    for batch in range(self.DataManager_valid.number_batch):
                        # batch start
                        img_batch, label_batch, paths = self.sess.run(self.DataManager_train.next_batch)

                        decision_out = self.sess.run([self.model.decison_out],
                                                       feed_dict={self.model.Image: img_batch,
                                                                  self.model.label: label_batch,
                                                                  self.model.is_training_seg: TRAIN_MODE_IN_VALID,
                                                                  self.model.is_training_dec: TRAIN_MODE_IN_VALID})

                        decision_out = np.array(decision_out)
                        lab = np.argmax(label_batch, axis=1)
                        for i in range(len(decision_out)):
                            if decision_out[0][i] != lab[i]:
                                false_count += 1
                                print(paths[i])


                pbar.clear()
                pbar.close()

                iter_loss /= self.DataManager_train.number_batch
                # print('epoch:[{}] ,train_mode, loss: {}' .format(self.model.step, iter_loss))
                print('epoch:[{}] ,train_mode, loss: {}, false_count: {}' .format(self.model.step, iter_loss, false_count))
                # 验证
                self.model.step += 1
                # 保存模型
                if i % self.__Param["save_frequency"] == 0 or i == self.__Param["epochs_num"] + self.model.step - 1:
                    # if val_loss < best_loss:
                    #     best_loss = val_loss
                    #     print('reduce loss to {}, saving model at epoch:{}'.format(val_loss, i))
                    self.model.save()

    def savePb(self):
        self.model.save_PbModel()

    def test(self):
        with self.sess.as_default():
            count = 0
            DataManager = self.DataManager_train
            for batch in range(DataManager.number_batch):
                img_batch, label_batch, image_paths = self.sess.run(DataManager.next_batch)

                decision_out = self.sess.run([self.model.decison_out],
                                             feed_dict={self.model.Image: img_batch,
                                                        self.model.label: label_batch})
                decision_out = np.array(decision_out)

                if np.argmax(label_batch) != decision_out[0]:
                    count += 1
                    print(image_paths)
                    print(np.argmax(label_batch))
                    print(decision_out[0])
                    print('----------------------------------')
            print(count)

    def listData_train(self, data_dir, test_ratio=TEST_RATIO):

        train_image_root = pathlib.Path(data_dir)
        train_image_paths = list(train_image_root.glob('*'))
        train_image_paths = [str(path) for path in train_image_paths]
        return train_image_paths




