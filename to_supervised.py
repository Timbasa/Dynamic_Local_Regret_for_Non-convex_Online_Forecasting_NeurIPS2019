import numpy as np


# transform the datasets to features and labels for RNN model
def to_supervised(configs, train, mode_flag):
    n_input = configs['model']['input_layer']
    n_out = configs['model']['output_size']
    train_x, train_y = list(), list()
    in_start = 0
    for _ in range(len(train)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if mode_flag == configs['data']['mode']['validation']:
            out_end = in_end + n_out * 28
        if out_end <= len(train):
            train_x.append(train[in_start: in_end, :])
            if mode_flag == configs['data']['mode']['validation']:
                train_y.append(train[in_end: out_end, :])
            else:
                train_y.append(train[in_end: out_end, 0])
            if mode_flag == configs['data']['mode']['test']:
                in_start += 24
            elif mode_flag == configs['data']['mode']['train']:
                in_start += 3
            elif mode_flag == configs['data']['mode']['validation']:
                in_start += 3
        else:
            break
    return np.asarray(train_x), np.asarray(train_y)
