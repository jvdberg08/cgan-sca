import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    args = len(sys.argv)
    for i in range(1, args - 1, 2):
        experiment_value_gen = sys.argv[i]
        experiment_value_disc = sys.argv[i + 1]
        file_name = get_file_name(experiment_value_gen, experiment_value_disc)

        file = open(file_name, "r")
        lines = file.readlines()
        lines = list(map(lambda x: float(x[0: len(x) - 3]), lines))
        plt.plot(lines, label="ng=" + experiment_value_gen + ", nd=" + experiment_value_disc)
    plt.xlabel("Trace (x10)")
    plt.ylabel("Guessing Entropy")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer x-axis ticks
    plt.xlim(0, 25)
    # plt.ylim(0, 40)
    plt.show()



def get_file_name(experiment_value_gen, experiment_value_disc):
    return "results/experiment-nodes/ge_lg6_ng" + experiment_value_gen \
        + "_agelu_ld4_dd0.3_nd" + experiment_value_disc \
        + "_adelu_noadam_nol0.0002_bs400_ep10_tr200000.txt"


if __name__ == '__main__':
    main()
