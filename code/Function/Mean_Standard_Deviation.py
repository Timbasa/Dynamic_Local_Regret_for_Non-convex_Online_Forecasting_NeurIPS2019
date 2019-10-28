import numpy as np


# calculate the mean and std of 4 models  with different learning rates when window size is 200
def mean_std():
    learning_rate = [1, 3, 5, 9]
    w = 200
    for lr in learning_rate:
        SGD_online_losses = np.loadtxt('SGD_online_lr=' + str(lr) + '.txt')
        SGD_offline_losses = np.loadtxt('SGD_offline_lr=' + str(lr) + '.txt')
        DTSSGD_online_losses = np.loadtxt('DTSSGD_online_lr=' + str(lr) + '_w=' + str(w) + '.txt')
        STSSGD_online_losses = np.loadtxt('STSSGD_online_lr=' + str(lr) + '_w=' + str(w) + '.txt')

        SGD_online_mean = np.mean(SGD_online_losses[50:])
        SGD_offline_mean = np.mean(SGD_offline_losses[50:])
        DTSSGD_mean = np.mean(DTSSGD_online_losses[50:])
        STSSGD_mean = np.mean(STSSGD_online_losses[50:])

        SGD_online_std = np.std(SGD_online_losses)
        SGD_offline_std = np.std(SGD_offline_losses)
        DTSSGD_std = np.std(DTSSGD_online_losses)
        STSSGD_std = np.std(STSSGD_online_losses)

        print('learning rate: {}, window size: {}, alpha: 0.99'.format(lr, w))
        print('SGD online mean: {}, standard deviation: {}'.format(SGD_online_mean, SGD_online_std))
        print('SGD offline mean: {}, standard deviation: {}'.format(SGD_offline_mean, SGD_offline_std))
        print('DTS-SGD online mean: {}, standard deviation: {}'.format(DTSSGD_mean, DTSSGD_std))
        print('STS-SGD online mean: {}, standard deviation: {}'.format(STSSGD_mean, STSSGD_std))
        print('')
