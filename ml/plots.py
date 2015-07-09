from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
from scipy.optimize import curve_fit
from scipy.interpolate import griddata


def get_histogram_plot(property_name, values, units, title=""):
    n, bins, patches = plt.hist(values, 50, normed=1, histtype='stepfilled')
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    plt.title(title)
    plt.xlabel("%s (%s)" % (property_name, units))
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

