import numpy as np


# calculate the mean and std of 4 models  with different learning rates when window size is 200
def mean_std():
    learning_rate = [1, 3, 5, 9]
    w = 200
    for lr in learning_rate:
        SGD_online_losses = np.loadtxt('SGD_online_lr=' + str(lr) + '.txt')
        SGD_offline_losses = np.loadtxt('SGD_offline_lr=' + str(lr) + '.txt')
        PTSSGD_online_losses = np.loadtxt('PTSSGD_online_lr=' + str(lr) + '_w=' + str(w) + '.txt')
        HTSSGD_online_losses = np.loadtxt('HTSSGD_online_lr=' + str(lr) + '_w=' + str(w) + '.txt')

        SGD_online_mean = np.mean(SGD_online_losses[50:])
        SGD_offline_mean = np.mean(SGD_offline_losses[50:])
        PTSSGD_mean = np.mean(PTSSGD_online_losses[50:])
        HTSSGD_mean = np.mean(HTSSGD_online_losses[50:])

        SGD_online_std = np.std(SGD_online_losses)
        SGD_offline_std = np.std(SGD_offline_losses)
        PTSSGD_std = np.std(PTSSGD_online_losses)
        HTSSGD_std = np.std(HTSSGD_online_losses)

        print('learning rate: {}, window size: {}, alpha: 0.99'.format(lr, w))
        print('SGD online mean: {}, standard deviation: {}'.format(SGD_online_mean, SGD_online_std))
        print('SGD offline mean: {}, standard deviation: {}'.format(SGD_offline_mean, SGD_offline_std))
        print('PTS-SGD online mean: {}, standard deviation: {}'.format(PTSSGD_mean, PTSSGD_std))
        print('HTS-SGD online mean: {}, standard deviation: {}'.format(HTSSGD_mean, HTSSGD_std))
        print('')
