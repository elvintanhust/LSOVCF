# -*- coding:utf-8 -*-
import os
import time
from loss_func import (EELoss, min_prob, gauss_dim, gauss_dim_half, gauss)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import train_data
import numpy as np
import logging
import matplotlib
matplotlib.use('SVG')


RELEASE_MODE = True


class BaseTrainTest(object):

    def __init__(self,
                 deep_model,
                 root_dir,
                 folder_ext="",
                 load_model=False,
                 batch_size=100,
                 start_epoch=1,
                 epochs=200,
                 learning_rate=1e-3,
                 distribute_dim=None,
                 explore_ratio=0.5,
                 replay_buffer=None,
                 data=None,
                 class_num=10,
                 ):
        self.deep_model = deep_model
        self.loss_func = EELoss(replay_buffer, class_num=class_num, probability_dim=distribute_dim)
        self.replay_buffer = replay_buffer
        self.load_model = load_model
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.distribute_dim = distribute_dim
        self.data = data
        self.class_num = class_num
        self.explore_ratio = explore_ratio
        self.workers = 4
        
        folder_name = folder_ext
        self.unique_str = str(int(time.time()))
        self.model_dir = os.path.join(root_dir, folder_name)
        if not os.path.exists(self.model_dir) and RELEASE_MODE:
            os.makedirs(self.model_dir)
        self.default_model_params = os.path.join(root_dir, "default.pkl")
        
        self.logger = self.init_logger(folder_name)
        self.log("load_model:" + str(load_model))
        self.log("root_dir:" + str(root_dir))
        self.log("batch_size:" + str(batch_size))
        self.log("start_epoch:" + str(start_epoch))
        self.log("epochs:" + str(epochs))
        self.log("learning_rate:" + str(learning_rate))
        self.log("loss_func:" + str(self.loss_func))
        self.log("deep_model:" + str(self.deep_model))
        self.log("distribute_dim:" + str(self.distribute_dim))
        self.log("explore_ratio:" + str(self.explore_ratio))
        self.log("class_num:" + str(self.class_num))
        self.log("buffer param:" + str(self.replay_buffer.CAPACITY) + "--" + str(self.replay_buffer.PULL_CNT))

    def init_logger(self, name):
        fmt = "%(asctime)-15s %(levelname)s %(message)s"
        date_fmt = "%a %d %b %Y %H:%M:%S"
        formatter = logging.Formatter(fmt, date_fmt)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if RELEASE_MODE:
            logger_path = os.path.join(self.model_dir, "log.txt")
            fh = logging.FileHandler(logger_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def log(self, msg):
        self.logger.info(msg)

    def save_model(self, state_dict, path):
        if RELEASE_MODE:
            torch.save(state_dict, path)

    def cal_kl_divergence(self, y_distribute, x_distribute):
        x_distribute = np.where(x_distribute < min_prob, min_prob, x_distribute)
        y_distribute = np.where(y_distribute < min_prob, min_prob, y_distribute)
        ret = np.sum(np.multiply(y_distribute, np.log(np.divide(y_distribute, x_distribute))))
        return ret

    def training_data_statistics(self, outs, labels):
        json_to_save = dict()
        train_data_distribute = []
        class_distribute_expectation = np.zeros(shape=self.class_num)
        class_distribute = np.zeros(shape=(self.class_num, self.distribute_dim))
        class_cnt = np.zeros(shape=self.class_num)
        distribute_to_cluster = dict()
        for i in range(len(labels)):
            out = outs[i]
            label = labels[i]
            info_dict = dict()
            info_dict["label"] = label
            info_dict["distribute"] = out
            expectation = np.sum(np.multiply(out, range(self.distribute_dim)))
            info_dict["expectation"] = expectation
            train_data_distribute.append(info_dict)

            class_distribute[label] += out
            class_distribute_expectation[label] += expectation
            class_cnt[label] += 1

            if label not in distribute_to_cluster:
                distribute_to_cluster[label] = []
            distribute_to_cluster[label].append(out)

        distribute_expectation = []
        for i in range(self.class_num):
            expectation = class_distribute_expectation[i] / (class_cnt[i] + 0.0)
            distribute_expectation.append(expectation)
            class_distribute[i] = class_distribute[i] / (class_cnt[i] + 0.0)
        distribute_expectation = np.asarray(distribute_expectation)
        return class_distribute, distribute_expectation

    def find_belong_label_kl(self, class_distribute, distribute):
        if class_distribute is None:
            return -1
        min_kl_diver = 999999
        min_kl_diver_ind = 0
        for i in range(self.class_num):
            kl_diver = self.cal_kl_divergence(class_distribute[i], distribute)
            if kl_diver < min_kl_diver:
                min_kl_diver = kl_diver
                min_kl_diver_ind = i
        return min_kl_diver_ind


class TrainTest(BaseTrainTest):
    def __init__(self, *args, **kwargs):
        super(TrainTest, self).__init__(*args, **kwargs)
        self.deep_model = self.deep_model.cuda()

        self.deep_model = nn.DataParallel(self.deep_model)
        self.optimizer = torch.optim.SGD(self.deep_model.parameters(), lr=self.learning_rate, momentum=0.9,
                                         weight_decay=1e-4)
        if self.load_model and os.path.exists(self.default_model_params):
            self.deep_model.load_state_dict(torch.load(self.default_model_params))

    def fill_buffer(self):
        train_load = DataLoader(
            train_data.TrainData(self.data,
                                 need_pre_process=True),
            batch_size=self.batch_size,
            shuffle=True, num_workers=self.workers, drop_last=True)
        while not self.replay_buffer.is_full():
            for i, data in enumerate(train_load, 0):
                img, label, _ = data
                img, label = img.cuda(), label.cuda()
                self.optimizer.zero_grad()
                outs = self.deep_model(img)
                self.loss_func(outs, label)
            print("fill buffer %d%%" % self.replay_buffer.fill_ratio())

    def train(self):
        self.fill_buffer()
        train_load = DataLoader(
            train_data.TrainData(self.data,
                                 need_pre_process=True),
            batch_size=self.batch_size,
            shuffle=True, num_workers=self.workers, drop_last=True)
        validation_load = DataLoader(
            train_data.ValidationData(self.data), batch_size=self.batch_size,
            shuffle=False, num_workers=self.workers)

        self.log("Start Training!!!")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.log("=" * 30)
            explore_ratio, lr = self.cal_explore_ratio_and_lr(epoch)
            self.loss_func.explore_ratio = explore_ratio
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            self.log(
                "[training epoch %d/%d][learning_rate %f][explore_ratio %f]" % (
                    epoch, self.epochs, self.optimizer.param_groups[0]["lr"], self.loss_func.explore_ratio))
            self.train_epoch(train_load, epoch)
            self.validation_epoch(validation_load, epoch)
            if epoch % 50 == 0:
                self.test_epoch()

        self.log("Finish Training!!!")
        self.save_model(self.deep_model.state_dict(), os.path.join(self.model_dir, "final_%s.pkl" % self.unique_str))

    def cal_explore_ratio_and_lr(self, epoch):
        if epoch < 100:
            lr = self.learning_rate
        elif epoch < 150:
            lr = self.learning_rate * 0.1
        else:
            lr = self.learning_rate * 0.01
        return self.explore_ratio, lr

    def test(self, model_params=None):
        if model_params is None:
            model_params = self.default_model_params

        if os.path.exists(model_params):
            self.deep_model.load_state_dict(torch.load(model_params))
            self.test_epoch()

    def train_epoch(self, train_data_load, epoch):
        self.deep_model.train()

        train_loss_kl_clu = 0
        iter_cnt = 0
        for i, data in enumerate(train_data_load, 0):
            img, label, _ = data
            img, label = img.cuda(), label.cuda()

            self.optimizer.zero_grad()
            outs = self.deep_model(img)

            loss_kl_clu = self.loss_func(outs, label)
            loss_kl_clu.backward()
            loss_kl_clu_val = loss_kl_clu.item()

            self.optimizer.step()
            train_loss_kl_clu += loss_kl_clu_val

            iter_cnt += 1
        self.log(
            "[training accumulation epoch:%d][loss %.4f]" % (epoch, train_loss_kl_clu / iter_cnt))

        torch.cuda.empty_cache()

    def validation_epoch(self, validation_data_load, epoch):
        buffer_distribute = self.get_buffer_distribute()

        self.deep_model.eval()
        iter_cnt = 0
        buffer_ok = 0
        val_loss_kl_clu = 0
        with torch.no_grad():
            for i, data in enumerate(validation_data_load):
                img, label, _ = data
                img = img.cuda()

                outs = self.deep_model(img)
                loss_kl_clu = self.loss_func(outs, label)
                loss_kl_clu_val = loss_kl_clu.item()
                val_loss_kl_clu += loss_kl_clu_val

                outs = outs.cpu().detach().numpy()
                label = label.numpy()

                for ind in range(len(label)):
                    buffer_predict = self.find_belong_label_kl(buffer_distribute, outs[ind])
                    if buffer_predict == label[ind]:
                        buffer_ok += 1
                iter_cnt += 1
            accuracy = buffer_ok * 100.0 / (len(validation_data_load.dataset) + 0.0)
            self.log("[validation epoch:%d][ok_cnt %d][accuracy %.4f][loss %.4f]" % (
                epoch, buffer_ok, accuracy, val_loss_kl_clu / iter_cnt))
            torch.cuda.empty_cache()

    def test_epoch(self):
        test_data_load = DataLoader(
            train_data.TestData(self.data), batch_size=10,
            shuffle=False, num_workers=self.workers)
        class_distribute = self.get_train_distribute()
        with torch.no_grad():
            ok = 0
            for i, data in enumerate(test_data_load):
                img, label, _ = data
                img = img.cuda()
                outs = self.deep_model(img)
                outs = outs.cpu().detach().numpy()
                label = label.numpy()

                for ind in range(len(label)):
                    predict = self.find_belong_label_kl(class_distribute, outs[ind])
                    if predict == label[ind]:
                        ok += 1
            accuracy = ok * 100.0 / (len(test_data_load.dataset))
            self.log("[test][accuracy %.2f%%]" % accuracy)
            torch.cuda.empty_cache()

    def get_buffer_distribute(self):
        buffer_expectation = self.replay_buffer.pull_all()
        if buffer_expectation is None:
            buffer_distribute = None
        else:
            buffer_distribute = np.zeros(
                shape=(self.class_num, self.distribute_dim), dtype=np.float32)
            for i in range(self.class_num):
                if not np.isnan(buffer_expectation[i]):
                    expectation_i = int(buffer_expectation[i])
                else:
                    expectation_i = int(self.distribute_dim / 2)
                expectation_i = self.distribute_dim - gauss_dim if expectation_i + gauss_dim_half > self.distribute_dim - 2 else expectation_i - gauss_dim_half
                expectation_i = 0 if expectation_i - gauss_dim_half < 0 else expectation_i - gauss_dim_half
                for j in range(gauss_dim):
                    buffer_distribute[i][j + expectation_i] = gauss[j]

        return buffer_distribute

    def get_train_distribute(self):
        with torch.no_grad():
            train_data_load = DataLoader(
                train_data.TrainData(self.data,
                                     need_pre_process=False),
                batch_size=self.batch_size,
                shuffle=False, num_workers=self.workers, drop_last=True)
            self.deep_model.eval()
            out_list, label_list = None, None
            for i, data in enumerate(train_data_load):
                img, label, _ = data
                img = img.cuda()
                outs = self.deep_model(img)
                outs = outs.cpu().detach().numpy()
                label = label.numpy()

                if out_list is None:
                    out_list = outs
                else:
                    out_list = np.concatenate((out_list, outs), axis=0)
                if label_list is None:
                    label_list = label
                else:
                    label_list = np.concatenate((label_list, label), axis=0)

            class_distribute, distribute_expectation = self.training_data_statistics(out_list, label_list)
            torch.cuda.empty_cache()
        return class_distribute
