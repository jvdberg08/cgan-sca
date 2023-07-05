import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    args = len(sys.argv)
    for i in range(1, args - 1, 6):
        layers_gen = sys.argv[i]
        nodes_gen = sys.argv[i + 1]
        layers_disc = sys.argv[i + 2]
        dropout_disc = sys.argv[i + 3]
        nodes_disc = sys.argv[i + 4]
        index = sys.argv[i + 5]
        file_name = get_file_name(
            layers_gen,
            nodes_gen,
            layers_disc,
            dropout_disc,
            nodes_disc
        )

        file = open(file_name, "r")
        lines = file.readlines()
        lines = list(map(lambda x: float(x[0: len(x) - 3]), lines))
        plt.plot(lines, label="CGAN_" + index + "(" + dropout_disc + ")")
    plt.xlabel("Trace (x10)")
    plt.ylabel("Guessing Entropy")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer x-axis ticks
    # plt.xlim(0, 0)
    # plt.ylim(0, 40)
    plt.show()


def get_file_name(layers_gen, nodes_gen, layers_disc, dropout_disc, nodes_disc, ):
    return "results/experiment-dropout2/ge_lg" + layers_gen \
        + "_ng" + nodes_gen \
        + "_agelu_ld" + layers_disc \
        + "_dd" + dropout_disc \
        + "_nd" + nodes_disc \
        + "_adelu_noadam_nol0.0002_bs400_ep10_tr200000.txt"


if __name__ == '__main__':
    main()
