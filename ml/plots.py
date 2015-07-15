from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
from scipy.optimize import curve_fit
from scipy.interpolate import griddata


def get_histogram_plot(property_name, values, units, title=""):
    plt.hist(values, bins=50, normed=False, color="g", alpha=0.75, histtype='stepfilled')
    plt.title(title)
    plt.xlabel("%s (%s)" % (property_name, units))
    plt.ylabel("Count")
    plt.show()


def get_multi_histogram_plot(property_names, values_list, title="", normalize=True):
    plt.title(title)
    for name, values in zip(property_names, values_list):
        if normalize:
            temp = values - values.min()
            values = temp / temp.max()
        plt.hist(values, bins=50, normed=False, alpha=1./len(values_list), histtype='stepfilled', label=name)
    plt.legend(loc="best")
    plt.xlabel("Values (unitless)")
    plt.ylabel("Count")
    plt.show()


def get_line_plot(xvals, yvals, title="", fit=None):
    plt.plot(xvals, yvals, '-')
    if fit is not None:
        lin_xvals = numpy.linspace(numpy.min(xvals), numpy.max(xvals), 200)
        params, _ = curve_fit(fit, xvals, yvals)
        lin_yvals = fit(lin_xvals, *params)
        plt.plot(lin_xvals, lin_yvals, '-')
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title)
    plt.show()


def get_bar_plots(names, values, title=""):
    pass


def get_3d_suface(xs, ys, zs, method="linear"):
    A = numpy.array(zip(xs, ys))
    X = numpy.linspace(min(xs), max(xs), 100)
    Y = numpy.linspace(min(ys), max(ys), 100)
    X, Y = numpy.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs, ys, zs)
    Z = griddata(A, zs, (X, Y), method=method)

    ax.plot_surface(X, Y, Z, #rstride=1, cstride=1,# cmap=cm.coolwarm,
            linewidth=0, antialiased=False)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def get_3d_scatter(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def get_matrix_plot(mat, extent=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = ax.matshow(mat, extent=extent)
    fig.colorbar(m)
    plt.show()
