# -*- coding:utf-8 -*-
import sys
import os
import argparse


def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--distribute-dim", type=int, default=256,
                        help="output space dim")
    parser.add_argument("-e", "--explore-ratio", type=float, default=0.3,
                        help="explore ratio")
    parser.add_argument("-g", "--gpu-id", type=str, default="0",
                        choices=["0", "1", "2", "3"],
                        help="GPU ID")
    return parser.parse_args()


def main():
    params = parse_parameters()
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
    distribute_dim = params.distribute_dim
    explore_ratio = params.explore_ratio

    lr = 0.1
    epochs = 200

    model_dir = os.path.join(sys.path[0], "../model/")
    data_dir = os.path.join(sys.path[0], "../data/")
    data = train_data.CifarData(data_dir, data_name="cifar10")
    class_num = 10
    replay_buffer = ReplayBuffer(capacity=1500, pull_cnt=500, class_num=class_num, distribute_dim=distribute_dim)
    batch_size = class_num * 10

    model = resnet_model.resnet18(num_classes=distribute_dim, use_softmax=True)
    train_test = TrainTest(model, model_dir, folder_ext="EE_Loss",
                           distribute_dim=distribute_dim, explore_ratio=explore_ratio,
                           data=data, batch_size=batch_size, replay_buffer=replay_buffer,
                           epochs=epochs, learning_rate=lr, class_num=class_num)
    train_test.train()


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(sys.path[0], '')))
    sys.path.append(os.path.abspath(os.path.join(sys.path[0], '', "..")))
    sys.path.append(sys.path[0])
    from loss_func import ReplayBuffer
    from train_test import TrainTest
    import train_data as train_data
    import resnet_model as resnet_model

    main()

