import matplotlib.pyplot as plt
import numpy as np


def plot_coeffs(z_list, redshift_coeffs):
    for log_scale in range(2):
        plt.figure(2)

        sqrtn = int(np.ceil(np.sqrt(len(redshift_coeffs[0]))))

        plt.figure(figsize=(4 * sqrtn, 3 * sqrtn))

        for j in range(len(redshift_coeffs[0])):
            plt.subplot(sqrtn, sqrtn, j + 1)
            plt.plot(z_list, redshift_coeffs[:, j], ".", alpha=0.1, color="b")
            plt.title("Coeff %i" % j)
            if log_scale:
                plt.xscale("log")
        plt.savefig("redshift_coeffs" + "_log" * log_scale + ".pdf", bbox_inches="tight")
        plt.close()
