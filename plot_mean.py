import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

window_size = 200

def plot_sum():
    learning_rate_1 = [0.1, 0.3, 0.5, 0.7, 0.9, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    # learning_rate_2 = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 3]
    # x_axis = [str(math.log(x)) for x in learning_rate]
    SGD_online_sum = []
    PTSSGD_sum = []
    HTSSGD_sum = []
    SGD_offline_sum = []
    for lr in learning_rate_1:
        SGD_online_losses = np.loadtxt('SGD_online_lr=' + str(lr) + '.txt')
        SGD_online_sum.append(np.sum(SGD_online_losses[50:]))
    for lr in learning_rate_1:
        PTSSGD_online_losses = np.loadtxt('PTSSGD_online_lr=' + str(lr) + '_w=' + str(window_size) + '.txt')
        PTSSGD_sum.append(np.sum(PTSSGD_online_losses[50:]))
    for lr in learning_rate_1:
        HTSSGD_online_losses = np.loadtxt('HTSSGD_online_lr=' + str(lr) + '_w=' + str(window_size) + '.txt')
        HTSSGD_sum.append(np.sum(HTSSGD_online_losses[50:]))
    for lr in learning_rate_1:
        SGD_offline_losses = np.loadtxt('SGD_offline_lr=' + str(lr) + '.txt')
        SGD_offline_sum.append(np.sum(SGD_offline_losses[50:]))

    plt.figure()
    plt.plot(learning_rate_1, SGD_online_sum, 'r-+', label='SGD online', lw=2)
    plt.plot(learning_rate_1, PTSSGD_sum, 'b-+', label='PTS-SGD', lw=2)
    plt.plot(learning_rate_1, HTSSGD_sum, 'g-+', label='HTS-SGD', lw=2)
    plt.plot(learning_rate_1, SGD_offline_sum, 'g-+', label='SGD offline', lw=2)
    plt.xlabel(r'$\eta$', fontsize=25)
    plt.ylabel(r'Cumulative $QL_{grand}$', fontsize=25)
    plt.legend(loc='best', fontsize=16)
    plt.grid(True, which='both')
    plt.yticks((16000, 18000, 20000, 22000, 24000), ('16k', '18k', '20k', '22k', '24k'))
    for label in plt.gca().get_xticklabels():
        label.set_visible(False)
    for label in plt.gca().get_xticklabels()[::3]:
        label.set_visible(True)
    plt.xscale('log')
    plt.tick_params(labelsize=17)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_sum()

