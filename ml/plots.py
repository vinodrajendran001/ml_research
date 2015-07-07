import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

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