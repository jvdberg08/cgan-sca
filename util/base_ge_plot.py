import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    file_name = "results/experiment-base/ge_lg0_ng0_agelu_ld0_dd0.0_nd0_adelu_noadam_nol0.0_bs0_ep0_tr0.txt"

    file = open(file_name, "r")
    lines = file.readlines()
    lines = list(map(lambda x: float(x[0: len(x) - 3]), lines))
    plt.plot(lines, label="MLP")
    plt.xlabel("Trace (x10)")
    plt.ylabel("Guessing Entropy")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer x-axis ticks
    # plt.xlim(0, 180)
    # plt.ylim(0, 20)
    plt.show()


if __name__ == '__main__':
    main()
