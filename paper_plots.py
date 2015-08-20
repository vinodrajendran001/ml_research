import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy
from scipy.optimize import curve_fit
from scipy.special import expit as sigmoid


def plot_bond_encoding(theta=None, start=1.0, end=6.0, slope=20., segments=5, show=False):
    if theta is None:
        theta = numpy.linspace(start, end, segments)
    distance = numpy.linspace(0, 8, 250)
    value = sigmoid(slope*numpy.subtract.outer(theta, distance)).T
    plt.plot(distance, value)
    plt.yticks(numpy.linspace(0, 1.1, 4))
    plt.xlabel("Bond Length ($\AA$)")
    plt.ylabel("Weight")
    if show:
        plt.show()


def plot_bond_threshold(theta=None, start=1.0, end=6.0, segments=5, show=False):
    if theta is None:
        theta = numpy.linspace(start, end, segments)
    distance = numpy.linspace(0, 8, 1000)
    value1 = numpy.greater.outer(theta, distance).astype(int).T
    theta2 = numpy.array([0] + theta[:-1].tolist())
    value2 = numpy.less.outer(theta2, distance).astype(int).T
    plt.plot(distance, value1 & value2)
    plt.yticks(numpy.linspace(0, 1.1, 4))
    plt.xlabel("Bond Length ($\AA$)")
    plt.ylabel("Weight")
    if show:
        plt.show()


def both_bond_methods():
    plt.figure(1)
    plt.subplot(211)
    plot_bond_threshold()

    plt.subplot(212)
    plot_bond_encoding()

    plt.show()


def plot_num_sample_data(num, train_mean, train_std, test_mean, test_std):
    plt.errorbar(num, train_mean, yerr=train_std, label="Train")
    lin_xvals = numpy.linspace(min(num), max(num), 200)

    labels = ["Test", "Train"]
    fs = [
        lambda x, a, b: a * x ** b,
        lambda x, a, b: a * numpy.log(x) + b,
    ]
    means = [test_mean, train_mean]
    stds = [test_std, train_std]

    for label, f, mean, std in zip(labels, fs, means, stds):
        plt.errorbar(num, mean, yerr=std, label=label)
        params, _ = curve_fit(f, num, mean)
        lin_yvals = f(lin_xvals, *params)
        plt.plot(lin_xvals, lin_yvals, '-', label="%s Fit" % label)

    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Absolute Error (kcal/mol)")
    plt.legend(loc="best")
    plt.show()


def plot_combination_order(data, labels):
    ticks = ["atom", "bond", "angle", "dihedral", "trihedral"]
    xdata = range(1, len(data[0]) + 1)

    for label, ydata in zip(labels, data):
        # print ydata, xdata
        plt.plot(xdata, ydata, label=label)

    plt.legend(loc="best")
    ax = plt.gca()
    ax.set_xticks(xdata)
    ax.set_xticklabels(ticks)
    plt.xlabel("Order Terms")
    plt.ylabel("Mean Absolute Error (kcal/mol)")
    plt.show()


def plot_order_errors(data, labels):
    ind = numpy.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind + width, data, width)
    ax.set_ylabel('Mean Absolute Error (kcal/mol)')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(labels)
    plt.show()


def plot_linear_kernel_pairs(pairs):
    xdata, ydata = zip(*pairs)
    maxval = max(max(xdata), max(ydata))

    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, '.')
    ax.plot([0, maxval], [0, maxval], '--')
    plt.xlabel("Linear Ridge Mean Absolute Error (kcal/mol)")
    plt.ylabel("Kernel Ridge Mean Absolute Error (kcal/mol)")

    # 15 is the zoom, loc is nuts
    axins = zoomed_inset_axes(ax, 15, loc=5)
    axins.plot(xdata, ydata, '.')
    axins.plot([0, maxval], [0, maxval], '--')

    # sub region of the original image
    axins.set_xlim(1, 6)
    axins.set_ylim(1, 6)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.draw()
    plt.show()


def plot_connectivity_distance(data, labels):
    xdata = range(1,len(data[0]) + 1)

    for label, y in zip(labels, data):
        plt.plot(xdata, y, label=label)

    plt.legend(loc="best")
    ax = plt.gca()
    ax.set_xticks(xdata)
    plt.xlabel("Connection Length")
    plt.ylabel("Error (kcal/mol)")
    plt.show()


if __name__ == "__main__":
    import json

    with open("results/plot_data.json") as f:
        all_data = json.load(f)

    both_bond_methods()

    data = all_data["num_sample"]
    plot_num_sample_data(data["nums"], data["train_mean"], data["train_std"], data["test_mean"], data["test_std"])

    data = all_data["combination_order"]
    plot_combination_order(data.values(), data.keys())

    data = all_data["order_errors"]
    plot_order_errors(data["values"], data["labels"])

    plot_linear_kernel_pairs(data["linear_kernel_pairs"])

    data = all_data["connectivity_distance"]
    plot_connectivity_distance(data.values(), data.keys())
