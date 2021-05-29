# -*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
import pickle
import torchvision
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import check_integrity, download_url
import sys

IMG_SIZE = 96


class CifarData(object):

    def __init__(self, folder_path, data_name="cifar10"):
        self.data_name = data_name
        self.folder_path = folder_path
        self.data_meta = {}

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.check_file()

        self.train_data = []
        self.validation_data = []
        self.test_data = []

        if data_name == "cifar10":
            self.load_cifar10()
        elif data_name == "cifar100":
            self.load_cifar100()

    def check_file(self):
        def _check_integrity(file_list, base_folder):
            root = self.folder_path
            for fentry in (file_list):
                filename, md5 = fentry[0], fentry[1]
                fpath = os.path.join(root, base_folder, filename)
                if not check_integrity(fpath, md5):
                    return False
            return True

        import tarfile
        # download cifar10
        base_folder = 'cifar-10-batches-py'
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
        train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]

        test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]
        meta = {
            'filename': 'batches.meta',
            'key': 'label_names',
            'md5': '5ff9c542aee3614f3951f8cda6e48888',
        }

        if not _check_integrity(train_list + test_list, base_folder):
            download_url(url, self.folder_path, filename=filename, md5=tgz_md5)

            with tarfile.open(os.path.join(self.folder_path, filename), "r:gz") as tar:
                tar.extractall(path=self.folder_path)

        # download cifar100
        base_folder = 'cifar-100-python'
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = "cifar-100-python.tar.gz"
        tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
        train_list = [
            ['train', '16019d7e3df5f24257cddd939b257f8d'],
        ]

        test_list = [
            ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
        ]
        meta = {
            'filename': 'meta',
            'key': 'fine_label_names',
            'md5': '7973b15100ade9c7d40fb424638fde48',
        }
        if not _check_integrity(train_list + test_list, base_folder):
            download_url(url, self.folder_path, filename=filename, md5=tgz_md5)
            with tarfile.open(os.path.join(self.folder_path, filename), "r:gz") as tar:
                tar.extractall(path=self.folder_path)

    def load_cifar10(self):
        path = os.path.join(self.folder_path, "cifar-10-batches-py/batches.meta")
        with open(path, 'rb') as fo:
            self.data_meta = pickle.load(fo, encoding='bytes')
            print(self.data_meta)

        total_data = []
        total_label = []

        def load_cifar_batch(name):
            path = os.path.join(self.folder_path, "cifar-10-batches-py", name)
            with open(path, 'rb') as fo:
                batch_data = pickle.load(fo, encoding='bytes')
            if not batch_data:
                print("Error load data!  path:", path)
                sys.exit(0)

            x = batch_data[b'data'] 
            y = batch_data[b'labels'] 
            x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
            y = np.array(y)
            total_data.append(x)
            total_label.append(y)

        load_cifar_batch("data_batch_1")
        load_cifar_batch("data_batch_2")
        load_cifar_batch("data_batch_3")
        load_cifar_batch("data_batch_4")
        load_cifar_batch("data_batch_5")
        load_cifar_batch("test_batch")
        total_data = np.concatenate(total_data)  # [ndarray, ndarray] 合并为一个ndarray
        total_label = np.concatenate(total_label)

        train_data, train_label = total_data[:45000], total_label[:45000]
        validation_data, validation_label = total_data[45000:50000], total_label[45000:50000]
        test_data, test_label = total_data[50000:], total_label[50000:]

        self.train_data = [train_data, train_label, train_label]
        self.validation_data = [validation_data, validation_label, validation_label]
        self.test_data = [test_data, test_label, test_label]

    def load_cifar100(self):
        path = os.path.join(self.folder_path, "cifar-100-python/meta")
        with open(path, 'rb') as fo:
            self.data_meta = pickle.load(fo, encoding='bytes')
            print(self.data_meta)

        path = os.path.join(self.folder_path, "cifar-100-python/train")
        with open(path, 'rb') as fo:
            train_data = pickle.load(fo, encoding='bytes')
        x = train_data[b'data']
        y1 = train_data[b'fine_labels']
        y2 = train_data[b'coarse_labels']
        temp_train_data = x.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        temp_train_label1 = np.array(y1)
        temp_train_label2 = np.array(y2)

        path = os.path.join(self.folder_path, "cifar-100-python/test")
        with open(path, 'rb') as fo:
            test_data = pickle.load(fo, encoding='bytes')
        x = test_data[b'data']
        y1 = test_data[b'fine_labels']
        y2 = test_data[b'coarse_labels']
        temp_test_data = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        temp_test_label1 = np.array(y1)
        temp_test_label2 = np.array(y2)

        self.fine_2_coarse = np.zeros(100, dtype=np.int)
        for i, j in zip(temp_test_label1, temp_test_label2):
            self.fine_2_coarse[int(i)] = int(j)

        train_data, train_label, coarse_train_label = temp_train_data[:45000], temp_train_label1[:45000], temp_train_label2[:45000]
        validation_data, validation_label, coarse_validation_label = temp_train_data[45000:], temp_train_label1[45000:], temp_train_label2[45000:]
        test_data, test_label, coarse_test_label = temp_test_data[:], temp_test_label1[:], temp_test_label2[:]

        self.train_data = [train_data, train_label, coarse_train_label]
        self.validation_data = [validation_data, validation_label, coarse_validation_label]
        self.test_data = [test_data, test_label, coarse_train_label]

    def get_train_data(self):
        return self.train_data[0], self.train_data[1], self.train_data[2]

    def get_validation_data(self):
        return self.validation_data[0], self.validation_data[1], self.validation_data[2]

    def get_test_data(self):
        return self.test_data[0], self.test_data[1], self.test_data[2]


class Cutout(object):
    def __init__(self, hole_size):
        self.hole_size = hole_size

    def __call__(self, img):
        y = np.random.randint(IMG_SIZE)
        x = np.random.randint(IMG_SIZE)

        half_size = self.hole_size // 2

        x1 = np.clip(x - half_size, 0, IMG_SIZE)
        x2 = np.clip(x + half_size, 0, IMG_SIZE)
        y1 = np.clip(y - half_size, 0, IMG_SIZE)
        y2 = np.clip(y + half_size, 0, IMG_SIZE)

        img_np = np.array(img)

        img_np[y1:y2, x1:x2] = 0
        img = Image.fromarray(img_np.astype('uint8')).convert('RGB')
        return img


class TrainData(Dataset):
    def __init__(self, data_pool, need_pre_process=True):
        self.data = data_pool.get_train_data()
        if need_pre_process:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE), 2),
                torchvision.transforms.RandomCrop(IMG_SIZE, padding=12),
                Cutout(10),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE), 2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.data[0][index]))
        label = torch.from_numpy(np.array(self.data[1][index])).long()
        coarse_label = torch.from_numpy(np.array(self.data[1][index])).long()
        return image, label, coarse_label

    def __len__(self):
        return len(self.data[1])


class ValidationData(Dataset):
    def __init__(self, data_pool, *args, **kwargs):
        self.data = data_pool.get_validation_data()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE), 2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.data[0][index]))
        label = torch.from_numpy(np.array(self.data[1][index])).long()
        coarse_label = torch.from_numpy(np.array(self.data[1][index])).long()
        return image, label, coarse_label

    def __len__(self):
        return len(self.data[1])


class TestData(Dataset):
    def __init__(self, data_pool, *args, **kwargs):
        self.data = data_pool.get_test_data()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE), 2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.data[0][index]))
        label = torch.from_numpy(np.array(self.data[1][index])).long()
        coarse_label = torch.from_numpy(np.array(self.data[1][index])).long()
        return image, label, coarse_label

    def __len__(self):
        return len(self.data[1])
