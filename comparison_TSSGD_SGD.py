import os
import torch
import json
import numpy as np
from data_loader import DataLoader
from scaler import Scaler
from quantile_loss import QuantileLoss
from lstm import LSTM
from seed_torch import seed_torch
from ptssgd import PTSSGD
from htssgd import HTSSGD
from sgd import SGD

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
configs = json.load(open('config.json', 'r'))

# establish scaler
scaler = Scaler(
    path=os.path.join(configs['data']['path']['task1_train']),
    cols=configs['data']['columns'],
    start=configs['data']['task1_train_start'])

# read train datasets, reshape it and supervise it
train_datasets = DataLoader(path=os.path.join(configs['data']['path']['task1_train']),
                            start=configs['data']['task1_train_start'],
                            cols=configs['data']['columns'],
                            time_flag=configs['data']['time_flag']['task1_train'],
                            scaler=scaler)
# define loss function prepare and data
loss_func = QuantileLoss(configs['model']['quantiles'])

x, y = train_datasets.get_train(configs=configs,
                                device=device,
                                mode_flag=configs['data']['mode']['train'])


def PTSSGD_online(learning_rate, w, af):
    # PTSSGD online model
    seed_torch()
    model_online = LSTM(configs=configs,
                        device=device,
                        loss_function=loss_func,
                        scaler=scaler,
                        save_path="lr=" + str(lr) + "_PTSSGD.pt").to(device)
    optimizer_PTSSGD = PTSSGD(model_online.parameters(), lr=learning_rate, window_size=w, a=af)
    PTSSGD_losses, PTSSGD_times = model_online.train_online_ptssgd(x, y, optimizer_PTSSGD)

    np.savetxt('PTSSGD_online_lr=' + str(learning_rate) + '_w=' + str(w) + '.txt', PTSSGD_losses)
    np.savetxt('PTSSGD_online_lr=' + str(learning_rate) + '_w=' + str(w) + '_times.txt', PTSSGD_times)


def SGD_online(learning_rate):
    # SGD online model
    seed_torch()
    model_SGD = LSTM(configs=configs,
                     device=device,
                     loss_function=loss_func,
                     scaler=scaler,
                     save_path="lr=" + str(lr) + "_SGD_online.pt").to(device)
    optimizer_SGD = SGD(model_SGD.parameters(), lr=learning_rate)
    SGD_losses, SGD_times = model_SGD.train_online_sgd(x, y, optimizer_SGD)

    np.savetxt('SGD_online_lr=' + str(learning_rate) + '.txt', SGD_losses)
    np.savetxt('SGD_online_lr=' + str(learning_rate) + '_times.txt', SGD_times)


def SGD_offline(learning_rate):
    # SGD offline model
    seed_torch()
    model_SGD = LSTM(configs=configs,
                     device=device,
                     loss_function=loss_func,
                     scaler=scaler,
                     save_path="lr=" + str(lr) + "_SGD_offline.pt").to(device)
    optimizer_SGD = SGD(model_SGD.parameters(), lr=learning_rate)
    SGD_losses, SGD_times = model_SGD.train_offline_sgd(x, y, optimizer_SGD)

    np.savetxt('SGD_offline_lr=' + str(learning_rate) + '.txt', SGD_losses)
    np.savetxt('SGD_offline_lr=' + str(learning_rate) + '_times.txt', SGD_times)


def HTSSGD_online(learning_rate, w):
    # HTSSGD online model
    seed_torch()
    model_HTSSGD = LSTM(configs=configs,
                        device=device,
                        loss_function=loss_func,
                        scaler=scaler,
                        save_path="lr=" + str(lr) + "_HTSSGD.pt").to(device)
    optimizer_HTSSGD = HTSSGD(model_HTSSGD.parameters(), lr=learning_rate)
    HTSSGD_losses, HTSSGD_times = model_HTSSGD.train_online_htssgd(x, y, w, optimizer_HTSSGD)

    np.savetxt('HTSSGD_online_lr=' + str(learning_rate) + '_w=' + str(w) + '.txt', HTSSGD_losses)
    np.savetxt('HTSSGD_online_lr=' + str(learning_rate) + '_w=' + str(w) + '_times.txt', HTSSGD_times)


if __name__ == '__main__':
    learning_rate = [0.1, 0.3, 0.5, 0.7, 0.9, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    window_size = [10, 20, 50, 100, 150, 200]
    a = 0.99

    for lr in learning_rate:
        SGD_online(lr)

    for lr in learning_rate:
        for w in window_size:
            PTSSGD_online(lr, w, a)

    for lr in learning_rate:
        for w in window_size:
            HTSSGD_online(lr, w)

    for lr in learning_rate:
        SGD_offline(lr)
