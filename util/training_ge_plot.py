import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    args = len(sys.argv)
    for i in range(1, args - 1, 3):
        experiment_value_batch = sys.argv[i]
        experiment_value_epochs = sys.argv[i + 1]
        experiment_value_training = sys.argv[i + 2]
        file_name = get_file_name(experiment_value_batch, experiment_value_epochs, experiment_value_training)

        file = open(file_name, "r")
        lines = file.readlines()
        lines = list(map(lambda x: float(x[0: len(x) - 3]), lines))
        plt.plot(lines, label="btc=" + experiment_value_batch + ", ep=" + experiment_value_epochs + ", "
                              + "tr=" + experiment_value_training)
    plt.xlabel("Trace (x10)")
    plt.ylabel("Guessing Entropy")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer x-axis ticks
    plt.xlim(0, 10)
    # plt.ylim(0, 40)
    plt.show()


def get_file_name(experiment_value_batch, experiment_value_epochs, experiment_value_training):
    return "results/experiment-training/ge_lg6_ng160_agelu_ld4_dd0.3_nd250_adelu_noadam_nol0.0002_bs" \
        + experiment_value_batch + "_ep" \
        + experiment_value_epochs + "_tr" \
        + experiment_value_training + ".txt"


if __name__ == '__main__':
    main()
