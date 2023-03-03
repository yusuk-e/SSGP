# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

import matplotlib.pyplot as plt
import seaborn as sns

dpi=100
sns.set()
sns.set_style("whitegrid", {'grid.linestyle': '--'})
sns.set_context("paper", 1.5, {"lines.linewidth": 1.5})
sns.set_palette("deep")

def plot(file, x, ys, xlabel, ylabel, legend):
    colors = ['blue','orange','green']
    fig = plt.figure(figsize=(8, 4), facecolor='white', dpi=dpi)
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1, frameon=True)
        ax.plot(x, ys[i], c=colors[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(legend[i])
        ax.yaxis.offsetText.set_fontsize(16)
        plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    plt.tight_layout()
    plt.savefig(file, format='pdf')
    plt.close()
