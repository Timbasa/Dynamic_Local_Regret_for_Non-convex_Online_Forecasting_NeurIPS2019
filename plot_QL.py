import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

learning_rate = [0.8, 1, 3, 5, 7, 9]
window_size = [20, 50, 100, 150, 200]
ymin, ymax = 7, 60

def plot_SGD_offline():
    # SGD_online_benchmark_losses_08 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_benchmark_lr=' + str(0.8) + '.txt')
    SGD_online_benchmark_losses_1 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_benchmark_lr=' + str(1) + '.txt')
    SGD_online_benchmark_losses_3 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_benchmark_lr=' + str(3) + '.txt')
    SGD_online_benchmark_losses_5 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_benchmark_lr=' + str(5) + '.txt')
    SGD_online_benchmark_losses_7 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_benchmark_lr=' + str(7) + '.txt')
    SGD_online_benchmark_losses_9 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_benchmark_lr=' + str(9) + '.txt')
    # SGD_online_benchmark_losses_11 = np.loadtxt(
    #     'Output/4_28_2019/SGD_online_benchmark_large_lr/SGD_online_benchmark_lr=' + str(11) + '.txt')
    # SGD_online_benchmark_losses_13 = np.loadtxt(
    #     'Output/4_28_2019/SGD_online_benchmark_large_lr/SGD_online_benchmark_lr=' + str(13) + '.txt')
    # SGD_online_benchmark_losses_15 = np.loadtxt(
    #     'Output/4_28_2019/SGD_online_benchmark_large_lr/SGD_online_benchmark_lr=' + str(15) + '.txt')
    # SGD_online_benchmark_losses_17 = np.loadtxt(
    #     'Output/4_28_2019/SGD_online_benchmark_large_lr/SGD_online_benchmark_lr=' + str(17) + '.txt')
    # SGD_online_benchmark_losses_19 = np.loadtxt(
    #     'Output/4_28_2019/SGD_online_benchmark_large_lr/SGD_online_benchmark_lr=' + str(19) + '.txt')
    # SGD_online_benchmark_losses_21 = np.loadtxt(
    #     'Output/4_28_2019/SGD_online_benchmark_large_lr/SGD_online_benchmark_lr=' + str(21) + '.txt')
    # SGD_online_benchmark_losses_23 = np.loadtxt(
    #     'Output/4_28_2019/SGD_online_benchmark_large_lr/SGD_online_benchmark_lr=' + str(23) + '.txt')
    # SGD_online_benchmark_losses_25 = np.loadtxt(
    #     'Output/4_28_2019/SGD_online_benchmark_large_lr/SGD_online_benchmark_lr=' + str(25) + '.txt')

    plt.figure()
    # plt.plot(SGD_online_benchmark_losses_08, color='black', linestyle='-', label='lr = 0.8', lw=1.5)
    plt.plot(SGD_online_benchmark_losses_1, color='r', linestyle='-', label=r'$\eta$ = 1', lw=1.5)
    plt.plot(SGD_online_benchmark_losses_3, color='b', linestyle='-', label=r'$\eta$ = 3', lw=1.5)
    plt.plot(SGD_online_benchmark_losses_5, color='y', linestyle='-', label=r'$\eta$ = 5', lw=1.5)
    plt.plot(SGD_online_benchmark_losses_7, color='g', linestyle='-', label=r'$\eta$ = 7', lw=1.5)
    plt.plot(SGD_online_benchmark_losses_9, color='m', linestyle='-', label=r'$\eta$= 9', lw=1.5)
    # plt.plot(SGD_online_benchmark_losses_11, color='olive', linestyle='-', label='lr = 11', lw=1.5)
    # plt.plot(SGD_online_benchmark_losses_13, color='silver', linestyle='-', label='lr = 13', lw=1.5)
    # plt.plot(SGD_online_benchmark_losses_15, color='darkred', linestyle='-', label='lr = 15', lw=1.5)
    # plt.plot(SGD_online_benchmark_losses_17, color='black', linestyle='-', label=r'$\eta$ = 17', lw=3)
    # plt.plot(SGD_online_benchmark_losses_19, color='yellowgreen', linestyle='-', label='lr = 19', lw=1.5)
    # plt.plot(SGD_online_benchmark_losses_21, color='darkblue', linestyle='-', label='lr = 21', lw=1.5)
    # plt.plot(SGD_online_benchmark_losses_23, color='fuchsia', linestyle='-', label='lr = 23', lw=1.5)
    # plt.plot(SGD_online_benchmark_losses_25, color='green', linestyle='-', label=r'$\eta$ = 25', lw=3)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('New observed datasets', fontsize=25)
    plt.ylabel(r'$QL_{grand}$', fontsize=25)
    plt.legend(loc='upper right', fontsize=16)
    # plt.legend(loc=2, fontsize=14, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.tick_params(labelsize=17)
    plt.tight_layout()
    plt.ylim(ymin, ymax)
    plt_save = 'SGD_offline_with_different_lr.png'
    plt.savefig(plt_save)
    plt.show()


def plot_SGD_online():
    # SGD_online_losses_08 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_lr=' + str(0.8) + '.txt')
    SGD_online_losses_1 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_lr=' + str(1) + '.txt')
    SGD_online_losses_3 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_lr=' + str(3) + '.txt')
    SGD_online_losses_5 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_lr=' + str(5) + '.txt')
    SGD_online_losses_7 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_lr=' + str(7) + '.txt')
    SGD_online_losses_9 = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_lr=' + str(9) + '.txt')

    plt.figure()
    # plt.plot(SGD_online_losses_08, 'k-', label=r'$\eta$ = 0.8', lw=1.5)
    plt.plot(SGD_online_losses_1, 'r-', label=r'$\eta$ = 1', lw=1.5)
    plt.plot(SGD_online_losses_3, 'b-', label=r'$\eta$ = 3', lw=1.5)
    plt.plot(SGD_online_losses_5, 'y-', label=r'$\eta$ = 5', lw=1.5)
    plt.plot(SGD_online_losses_7, 'g-', label=r'$\eta$ = 7', lw=1.5)
    plt.plot(SGD_online_losses_9, 'm-', label=r'$\eta$ = 9', lw=1.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('New observed datasets', fontsize=25)
    plt.ylabel(r'$QL_{grand}$', fontsize=25)
    plt.legend(loc='upper right', fontsize=16)
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.tick_params(labelsize=17)
    plt.tight_layout()
    # plt.ylim(ymin, ymax)
    plt_save = 'SGD_online_with_different_lr.png'

    plt.savefig(plt_save)
    plt.show()


def plot_HTSSGD_online():
    window_size = [20, 100, 150, 200]
    for w in window_size:
        # HSGD_online_losses_08 = np.loadtxt('Output/4_16_2019/updated_TSSGD/HSGD_online_lr=' + str(0.8) + '_w=' + str(w) + '.txt')
        HSGD_online_losses_1 = np.loadtxt('Output/4_16_2019/updated_TSSGD/HSGD_online_lr=' + str(1) + '_w=' + str(w) + '.txt')
        HSGD_online_losses_3 = np.loadtxt('Output/4_16_2019/updated_TSSGD/HSGD_online_lr=' + str(3) + '_w=' + str(w) + '.txt')
        HSGD_online_losses_5 = np.loadtxt('Output/4_16_2019/updated_TSSGD/HSGD_online_lr=' + str(5) + '_w=' + str(w) + '.txt')
        HSGD_online_losses_7 = np.loadtxt('Output/4_16_2019/updated_TSSGD/HSGD_online_lr=' + str(7) + '_w=' + str(w) + '.txt')
        HSGD_online_losses_9 = np.loadtxt('Output/4_16_2019/updated_TSSGD/HSGD_online_lr=' + str(9) + '_w=' + str(w) + '.txt')

        plt.figure()
        # plt.plot(HSGD_online_losses_08, 'k-', label=r'$\eta$ = 0.8', lw=1.5)
        plt.plot(HSGD_online_losses_1, 'r-', label=r'$\eta$ = 1', lw=1.5)
        plt.plot(HSGD_online_losses_3, 'b-', label=r'$\eta$ = 3', lw=1.5)
        plt.plot(HSGD_online_losses_5, 'y-', label=r'$\eta$ = 5', lw=1.5)
        plt.plot(HSGD_online_losses_7, 'g-', label=r'$\eta$ = 7', lw=1.5)
        plt.plot(HSGD_online_losses_9, 'm-', label=r'$\eta$ = 9', lw=1.5)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('New observed datasets', fontsize=25)
        plt.ylabel(r'$QL_{grand}$', fontsize=25)
        # plt.legend(loc='upper right', fontsize=16)
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.tick_params(labelsize=17)
        plt.tight_layout()
        plt.ylim(ymin, ymax)
        plt_save = 'HTSSGD_online_with_different_lr_and_w_=_' + str(w) + '.png'

        plt.savefig(plt_save)
        plt.show()


def plot_PTSSGD_online():
    window_size = [20, 100, 150, 200]
    for w in window_size:
        # TSSGD_online_losses_08 = np.loadtxt('Output/4_16_2019/updated_TSSGD/TSSGD_online_lr=' + str(0.8) + '_w=' + str(w) + '.txt')
        TSSGD_online_losses_1 = np.loadtxt('Output/4_16_2019/updated_TSSGD/TSSGD_online_lr=' + str(1) + '_w=' + str(w) + '.txt')
        TSSGD_online_losses_3 = np.loadtxt('Output/4_16_2019/updated_TSSGD/TSSGD_online_lr=' + str(3) + '_w=' + str(w) + '.txt')
        TSSGD_online_losses_5 = np.loadtxt('Output/4_16_2019/updated_TSSGD/TSSGD_online_lr=' + str(5) + '_w=' + str(w) + '.txt')
        TSSGD_online_losses_7 = np.loadtxt('Output/4_16_2019/updated_TSSGD/TSSGD_online_lr=' + str(7) + '_w=' + str(w) + '.txt')
        TSSGD_online_losses_9 = np.loadtxt('Output/4_16_2019/updated_TSSGD/TSSGD_online_lr=' + str(9) + '_w=' + str(w) + '.txt')

        plt.figure()
        # plt.plot(TSSGD_online_losses_08, 'k-', label='lr = 0.8', lw=1.5)
        plt.plot(TSSGD_online_losses_1, 'r-', label=r'$\eta$ = 1', lw=1.5)
        plt.plot(TSSGD_online_losses_3, 'b-', label=r'$\eta$ = 3', lw=1.5)
        plt.plot(TSSGD_online_losses_5, 'y-', label=r'$\eta$ = 5', lw=1.5)
        plt.plot(TSSGD_online_losses_7, 'g-', label=r'$\eta$ = 7', lw=1.5)
        plt.plot(TSSGD_online_losses_9, 'm-', label=r'$\eta$ = 9', lw=1.5)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('New observed datasets', fontsize=25)
        plt.ylabel(r'$QL_{grand}$', fontsize=25)
        # plt.legend(loc='upper right', fontsize=16)
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.tick_params(labelsize=17)
        plt.tight_layout()
        plt.ylim(ymin, ymax)
        plt_save = 'PTSSGD_online_with_different_lr_and_w_=_' + str(w) + '.png'
        plt.savefig(plt_save)
        plt.show()


def comparison_QL():
    learning_rate = [1, 3, 5, 9]
    # window_size = [200]
    w = 200
    for lr in learning_rate:
        # for w in window_size:
        SGD_online_losses = np.loadtxt('Output/SGD_online_lr=' + str(lr) + '.txt')
        # SGD_online_50_losses = np.loadtxt('Output/SGD_online_lr=' + str(lr) + '_momentum=0.5.txt')
        # SGD_online_90_losses = np.loadtxt('Output/SGD_online_lr=' + str(lr) + '_momentum=0.9.txt')
        SGD_online_99_losses = np.loadtxt('Output/SGD_online_lr=' + str(lr) + '_momentum=0.99.txt')

        SGD_offline_losses = np.loadtxt('Output/SGD_offline_lr=' + str(lr) + '.txt')
        # SGD_offline_50_losses = np.loadtxt('Output/SGD_offline_lr=' + str(lr) + '_momentum=0.5.txt')
        # SGD_offline_90_losses = np.loadtxt('Output/SGD_offline_lr=' + str(lr) + '_momentum=0.9.txt')
        SGD_offline_99_losses = np.loadtxt('Output/SGD_offline_lr=' + str(lr) + '_momentum=0.99.txt')

        TSSGD_online_losses = np.loadtxt('Output/PTSSGD_online_lr=' + str(lr) + '_w=' + str(w) + '.txt')
        # HSGD_online_losses = np.loadtxt('Output/HSGD_online_lr=' + str(lr) + '_w=' + str(w) + '.txt')

        plt.figure()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.plot(SGD_online_losses, 'k-', label='SGD online', lw=1.5)
        # plt.plot(SGD_online_50_losses, 'g-', label='SGD online m=0.5', lw=1.5)
        # plt.plot(SGD_online_90_losses, 'r-', label='SGD online m=0.9', lw=1.5)
        plt.plot(SGD_online_99_losses, 'y-', label='SGD online m=0.99', lw=1.5)

        plt.plot(SGD_offline_losses, 'g-', label='SGD offline', lw=1.5)
        # plt.plot(SGD_offline_50_losses, 'g-', label='SGD offline m=0.5', lw=1.5)
        # plt.plot(SGD_offline_90_losses, 'r-', label='SGD offline m=0.9', lw=1.5)
        plt.plot(SGD_offline_99_losses, 'r-', label='SGD offline m=0.99', lw=1.5)

        plt.plot(TSSGD_online_losses, 'b-', label='PTS-SGD online', lw=1.5)
        # plt.plot(HSGD_online_losses, 'y-', label='HTS-SGD online', lw=1.5)
        plt.xlabel('New observed datasets', fontsize=25)
        plt.ylabel(r'$QL_{grand}$', fontsize=25)
        plt.legend(loc='upper right', fontsize=16)
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.tick_params(labelsize=17)
        plt.tight_layout()
        plt.ylim(ymin, ymax)
        # plt.savefig('SGD_TSSGD_HSGD_lr=' + str(lr) + '_w=' + str(w) + '.png')
        plt.show()


def comparison_times():
    learning_rate = [9]
    window_size = [20, 50, 200]
    for lr in learning_rate:
        for w in window_size:
            SGD_online_times = []
            SGD_online_benchmark_times = []
            TSSGD_online_times = []
            HSGD_online_times = []

            SGD_online_time = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_lr=' + str(lr) + '_times.txt')
            SGD_online_benchmark_time = np.loadtxt('Output/4_16_2019/updated_TSSGD/SGD_online_benchmark_lr=' + str(lr) + '_times.txt')
            TSSGD_online_time = np.loadtxt('Output/4_16_2019/updated_TSSGD/TSSGD_online_lr=' + str(lr) + '_w=' + str(w) + '_times.txt')
            HSGD_online_time = np.loadtxt('Output/4_16_2019/updated_TSSGD/HSGD_online_lr=' + str(lr) + '_w=' + str(w) + '_times.txt')

            for i in range(len(SGD_online_time)):
                SGD_online_times.append(sum(SGD_online_time[:i + 1]))
                SGD_online_benchmark_times.append(sum(SGD_online_benchmark_time[:i + 1]))
                TSSGD_online_times.append(sum(TSSGD_online_time[:i + 1]))
                HSGD_online_times.append(sum(HSGD_online_time[:i + 1]))

            plt.figure()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(SGD_online_times, 'k-', label='SGD online', lw=3)
            plt.plot(SGD_online_benchmark_times, 'g-', label='SGD offline', lw=3)
            plt.plot(TSSGD_online_times, 'r-', label='PTS-SGD online', lw=3)
            plt.plot(HSGD_online_times, 'y-', label='HTS-SGD online', lw=3)
            plt.xlabel('New observed datasets', fontsize=25)
            plt.ylabel('GPU seconds', fontsize=25)
            # plt.legend(loc='best', fontsize=16)
            plt.grid(True, which='both')
            plt.tick_params(labelsize=17)
            plt.tight_layout()
            plt.ylim(0, 80)
            plt.savefig('SGD_TSSGD_HSGD_lr=' + str(lr) + '_w=' + str(w) + '_times.png')
            # plt.show()


def comparison_PTS_SGD_V1():
    learning_rate = [7]
    window_size = [200]

    for lr in learning_rate:
        for w in window_size:
            # for a in af:
            plt.figure()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            PTS_SGD_online_losses = np.loadtxt(
                'Output/5_6_2019/TSSGD/TSSGD_online_lr=' + str(lr) + '_w=' + str(w) + '_a=0.99.txt')
            plt.plot(PTS_SGD_online_losses, color='red', linestyle='-', label='PTS-SGD', lw=1.5)
            PTS_SGD_online_v1_losses = np.loadtxt(
                'Output/5_6_2019/TSSGD/TSSGD_online_v1_lr=' + str(lr) + '_w=' + str(w) + '_a=0.99.txt')
            plt.plot(PTS_SGD_online_v1_losses, color='green', linestyle='-', label='PTS-SGD_v1', lw=1.5)
            plt.xlabel('New observed datasets', fontsize=25)
            plt.ylabel(r'$QL_{grand}$', fontsize=25)
            plt.legend(loc='upper right', fontsize=14)
            plt.yscale('log')
            plt.grid(True, which='both')
            plt.tick_params(labelsize=12)
            plt.tight_layout()
            plt.ylim(ymin, ymax)
            plt.savefig('Comparison between PTS-SGD and PTS-SGD_v1.png')
            plt.show()



if __name__ == '__main__':
    # plot_SGD_offline()
    # plot_SGD_online()
    # plot_HTSSGD_online()
    # plot_PTSSGD_online()
    comparison_QL()
    # comparison_times()
    # comparison_best()
    # larger_lr()
    # comparison_different_alpha()
    # comparison_PTS_SGD_V1()
    # plot_99_SGD_offline()
    # plot_99_SGD_online()
    # plot_99_PTS_SGD_online()
    # plot_99_PTS_SGD_v1_online()

