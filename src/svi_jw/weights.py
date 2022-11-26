from matplotlib import pyplot as plt
import numpy
from numpy import sqrt, pi, square, exp, linspace, std

from src.svi_jw.svi_func import svi


def norm(x, mu, sigma):
    sigma*=.3
    C1 = sqrt(2*pi)
    term1 = 1/(sigma*C1)
    num = -square(x-mu)
    denom = 2*square(sigma)
    return term1 * exp(num/denom)


def make_weights(x, y):
    sigma = std(x)
    mu = x[numpy.where(y==y.min())]
    return norm(x, mu, sigma), mu


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    param = [0.001073661028253302, 0.32676198342955054, -0.07130173981948848, -0.3237675244847015, 0.9182850078924533]
    x=numpy.linspace(-1.5, 1.5)
    y = svi(x, *param)
    # x, y = make_points(-.0, .4, noise.L)
    w, mu = make_weights(x, y)

    plt.plot(x,w)
    plt.twinx()
    plt.axvline(x=mu, color="r")
    plt.scatter(x, y)

    plt.show()

