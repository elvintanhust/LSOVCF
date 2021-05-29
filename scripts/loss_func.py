# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import random

gauss = [1.33830625e-04, 4.43186162e-03, 5.39911274e-02, 2.41971446e-01,
         3.98943469e-01, 2.41971446e-01, 5.39911274e-02, 4.43186162e-03,
         1.33830625e-04]
gauss_dim = len(gauss)
gauss_dim_half = int(gauss_dim / 2)
min_prob = 1e-8
distance_threshold = int(1.5 * gauss_dim)


def cal_kl_divergence(y_distribute, x_distribute):
    return torch.mul(y_distribute, torch.log(torch.div(y_distribute, x_distribute))).sum(dim=-1)


def disperse_expectation(class_num, outs_expectation, probability_dim):
    sort_arr = np.zeros(shape=(class_num, 2))
    for i in range(class_num):
        sort_arr[i, 0] = i
        sort_arr[i, 1] = outs_expectation[i]
    sort_arr = sort_arr[np.lexsort([sort_arr[:, 1]]), :]

    if class_num % 2 == 0:
        mid_label = int(class_num / 2.0)
        for i in range(mid_label):
            if sort_arr[mid_label + i, 1] - sort_arr[mid_label + i - 1, 1] < distance_threshold:
                sort_arr[mid_label + i, 1] = sort_arr[mid_label + i - 1, 1] + distance_threshold

            if sort_arr[mid_label - i, 1] - sort_arr[mid_label - i - 1, 1] < distance_threshold:
                sort_arr[mid_label - i - 1, 1] = sort_arr[mid_label - i, 1] - distance_threshold
    else:
        mid_label = int(class_num / 2.0)
        for i in range(mid_label):
            if sort_arr[mid_label + i + 1, 1] - sort_arr[mid_label + i, 1] < distance_threshold:
                sort_arr[mid_label + i + 1, 1] = sort_arr[mid_label + i, 1] + distance_threshold

            if sort_arr[mid_label - i, 1] - sort_arr[mid_label - i - 1, 1] < distance_threshold:
                sort_arr[mid_label - i - 1, 1] = sort_arr[mid_label - i, 1] - distance_threshold

    for i in range(class_num):
        sort_arr[i, 1] = int(sort_arr[i, 1])

    if sort_arr[0, 1] < 0:
        offset = abs(sort_arr[0, 1]) + gauss_dim_half
        for i in range(class_num):
            sort_arr[i, 1] = sort_arr[i, 1] + offset

    if sort_arr[-1, 1] >= probability_dim:
        offset = sort_arr[-1, 1] - probability_dim + gauss_dim_half
        for i in range(class_num):
            sort_arr[i, 1] = sort_arr[i, 1] - offset

    if sort_arr[0, 1] < 0:
        offset = abs(sort_arr[0, 1]) + gauss_dim_half
        for i in range(class_num):
            sort_arr[i, 1] = sort_arr[i, 1] + offset
            if i + 1 < class_num and sort_arr[i + 1, 1] - sort_arr[i, 1] > distance_threshold:
                break
            if i + 1 >= class_num:
                break
            offset = distance_threshold - (sort_arr[i + 1, 1] - sort_arr[i, 1])

    for i in range(class_num):
        outs_expectation[int(sort_arr[i, 0])] = sort_arr[i, 1]
    return outs_expectation


def np_cal_kl_divergence(y_distribute, x_distribute):
    x_distribute = np.where(x_distribute < min_prob, min_prob, x_distribute)
    y_distribute = np.where(y_distribute < min_prob, min_prob, y_distribute)
    ret = np.sum(np.multiply(y_distribute, np.log(np.divide(y_distribute, x_distribute))))
    return ret


def find_belong_label(class_distribute, distribute):
    min_kl_dis = 999999
    belong_ind = 0
    for ib in range(len(class_distribute)):
        if np.sum(class_distribute[ib]) < 0.5:
            continue
        kl_dis = np_cal_kl_divergence(class_distribute[ib], distribute)
        if kl_dis < min_kl_dis:
            min_kl_dis = kl_dis
            belong_ind = ib
    return belong_ind


class ReplayBuffer(object):
    def __init__(self, capacity, pull_cnt, class_num, distribute_dim=512):
        self.CAPACITY = capacity
        self.PULL_CNT = pull_cnt
        self.CLASS_NUM = class_num
        self.DISTRIBUTE_DIM = distribute_dim
        self.EXPECTATION_POOL = []

    def is_full(self):
        return len(self.EXPECTATION_POOL) >= self.CAPACITY

    def fill_ratio(self):
        return int(len(self.EXPECTATION_POOL) * 100.0 / self.CAPACITY)

    def clear(self):
        self.EXPECTATION_POOL = []

    def push(self, class_expectation, accuracy):
        if len(self.EXPECTATION_POOL) < self.CAPACITY:
            self.EXPECTATION_POOL.insert(0, [class_expectation, accuracy])
        else:
            self.EXPECTATION_POOL.pop()
            self.EXPECTATION_POOL.insert(0, [class_expectation, accuracy])

    def pull(self):
        if len(self.EXPECTATION_POOL) < self.PULL_CNT:
            return None
        samples = random.sample(self.EXPECTATION_POOL, self.PULL_CNT)
        class_expectation = np.zeros(shape=self.CLASS_NUM, dtype=np.float32)
        total_weight = 0.0
        for class_expectation_sample, accuracy in samples:
            class_expectation[:] = class_expectation[:] + np.multiply(class_expectation_sample, accuracy)
            total_weight += accuracy
        class_expectation = class_expectation / (total_weight + 0.0)
        class_expectation = disperse_expectation(self.CLASS_NUM, class_expectation, self.DISTRIBUTE_DIM)
        return class_expectation

    def pull_all(self):
        if len(self.EXPECTATION_POOL) < self.PULL_CNT:
            return None
        class_expectation = np.zeros(shape=self.CLASS_NUM, dtype=np.float32)
        total_weight = 0.0
        for class_expectation_sample, accuracy in self.EXPECTATION_POOL:
            class_expectation[:] = class_expectation[:] + np.multiply(class_expectation_sample, accuracy)
            total_weight += accuracy
        class_expectation = class_expectation / (total_weight + 0.0)
        class_expectation = disperse_expectation(self.CLASS_NUM, class_expectation, self.DISTRIBUTE_DIM)
        return class_expectation


class EELoss(nn.Module):
    def __init__(self, replay_buffer=None, class_num=10, probability_dim=256):
        super().__init__()
        self.explore_ratio = 1.0
        self.replay_buffer = replay_buffer
        self.class_num = class_num
        self.probability_dim = probability_dim

    def expectation_to_distribute(self, expectations):
        distribute_arr = np.zeros(shape=(self.class_num, self.probability_dim), dtype=np.float32)
        for i in range(self.class_num):
            if not np.isnan(expectations[i]):
                expectation_i = int(expectations[i])
            else:
                expectation_i = int(self.probability_dim / 2)
            expectation_i = self.probability_dim - gauss_dim if expectation_i + gauss_dim_half > self.probability_dim - 2 else expectation_i - gauss_dim_half
            expectation_i = 0 if expectation_i - gauss_dim_half < 0 else expectation_i - gauss_dim_half
            for j in range(gauss_dim):
                distribute_arr[i][j + expectation_i] = gauss[j]
        return distribute_arr

    def cal_class_expectation(self, outs, labels, batch_size):
        sequence_arr = np.arange(self.probability_dim)
        distribute_arr = np.zeros(shape=(self.class_num, self.probability_dim), dtype=np.float32)
        class_sample_cnt = np.zeros(self.class_num)
        for i in range(batch_size):
            cur_label = labels[i]
            distribute_arr[cur_label, :] += outs[i]
            class_sample_cnt[cur_label] += 1

        outs_expectation = np.zeros(self.class_num)
        for i in range(self.class_num):
            distribute_arr[i, :] = distribute_arr[i, :] / (class_sample_cnt[i] + 0.0000001)
            distribute_arr[i, :] = np.multiply(distribute_arr[i, :], sequence_arr)
            outs_expectation[i] = int(np.sum(distribute_arr[i]))
        outs_expectation = disperse_expectation(self.class_num, outs_expectation, self.probability_dim)

        distribute_arr = self.expectation_to_distribute(outs_expectation)

        return distribute_arr, outs_expectation

    def update_pool(self, np_outs, np_labels, outs_shape):
        explore_response_arr, outs_expectation = self.cal_class_expectation(
            np_outs, np_labels, outs_shape[0])

        ok_cnt = 0
        for out, label in zip(np_outs, np_labels):
            if label == find_belong_label(explore_response_arr, out):
                ok_cnt += 1
        accuracy = ok_cnt / (len(np_outs) + 0.0)

        self.replay_buffer.push(outs_expectation, accuracy)
        return explore_response_arr

    def forward(self, outs, labels):
        outs = torch.where(outs.cpu() > 0, outs.cpu(), torch.tensor(0.0))
        np_labels = labels.cpu().detach().numpy()
        np_outs = outs.detach().numpy()
        outs_shape = np_outs.shape

        explore_response_arr = self.update_pool(np_outs, np_labels, outs_shape)
        experience_expectation = self.replay_buffer.pull()
        experience_response_arr = None

        if experience_expectation is not None and self.explore_ratio > 0:
            experience_response_arr = self.expectation_to_distribute(experience_expectation)

        explore_response = np.zeros(shape=outs_shape, dtype=np.float32)
        for i in range(outs_shape[0]):
            explore_response[i, :] = explore_response_arr[np_labels[i]]
        explore_response = torch.from_numpy(explore_response)

        x_distribute = torch.where(outs > min_prob, outs, torch.tensor(min_prob))
        explore_response = torch.where(explore_response > min_prob, explore_response, torch.tensor(min_prob))
        loss_explore = cal_kl_divergence(explore_response, x_distribute)
        loss_explore = torch.mean(loss_explore)
        loss_explore = torch.mul(loss_explore, torch.tensor(self.explore_ratio))

        if experience_response_arr is not None and self.explore_ratio > 0:
            experience_response = np.zeros(shape=outs_shape, dtype=np.float32)
            for i in range(outs_shape[0]):
                experience_response[i, :] = experience_response_arr[np_labels[i]]
            experience_response = torch.from_numpy(experience_response)

            experience_response = torch.where(experience_response > min_prob, experience_response, torch.tensor(min_prob))
            loss_experience = cal_kl_divergence(experience_response, x_distribute)
            loss_experience = torch.mean(loss_experience)
            loss_experience = torch.mul(loss_experience, torch.tensor(1.0 - self.explore_ratio))
            return torch.add(loss_explore, loss_experience)
        else:
            return loss_explore


