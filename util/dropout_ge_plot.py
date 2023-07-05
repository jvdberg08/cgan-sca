import sys
import matplotlib.pyplot as plt


def main():
    for experiment_value in sys.argv[1:]:
        file_name = get_file_name(experiment_value)

        file = open(file_name, "r")
        lines = file.readlines()
        lines = list(map(lambda x: float(x[0: len(x) - 3]), lines))
        plt.plot(lines, label="d=" + experiment_value)
    plt.xlabel("Trace (x10)")
    plt.ylabel("Guessing Entropy")
    plt.xlim(0, 10)
    plt.legend()
    plt.show()


def get_file_name(experiment_value):
    return "results/experiment-dropout/ge_lg6_ng160_agelu_ld4_dd" + experiment_value + "_nd250_adelu_noadam_nol0" \
                                                                                       ".0002_bs400_ep10_tr200000.txt"


if __name__ == '__main__':
    main()
