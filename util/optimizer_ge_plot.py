import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    args = len(sys.argv)
    for i in range(1, args - 1, 2):
        experiment_value_optimizer = sys.argv[i]
        experiment_value_learning = sys.argv[i + 1]
        file_name = get_file_name(experiment_value_optimizer, experiment_value_learning)

        file = open(file_name, "r")
        lines = file.readlines()
        lines = list(map(lambda x: float(x[0: len(x) - 3]), lines))
        plt.plot(lines, label="o=" + experiment_value_optimizer + ", l=" + experiment_value_learning)
    plt.xlabel("Trace (x10)")
    plt.ylabel("Guessing Entropy")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer x-axis ticks
    plt.xlim(0, 20)
    # plt.ylim(0, 40)
    plt.show()


def get_parameter_value(file, parameter):
    index = file.index(parameter) + len(parameter)
    value = ''
    while file[index] != '_' and file[index:] != '.txt':
        value += file[index]
        index += 1
    return value


def get_file_name(experiment_value_optimizer, experiment_value_learning):
    return "results/experiment-optimizer/ge_lg6_ng160_agelu_ld4_dd0.3_nd250_adelu_no" + experiment_value_optimizer \
        + "_nol" + experiment_value_learning + "_bs400_ep10_tr200000.txt"


if __name__ == '__main__':
    main()
