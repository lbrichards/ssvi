import json

import matplotlib.pyplot as plt
import numpy

from src.essvi.essvi_func import phi, _rho, ESSVI, ESSVI_p_rho
from src.svi_jw.svi_func import svi


def svi_to_essvi(svi_param):
    a, b, m, rho, sigma = svi_param
    fi = numpy.sqrt(1-rho**2)/sigma
    theta = 2*b*sigma/numpy.sqrt(1-rho**2)
    print(a, theta*(1-rho**2)/2)
    print(m, rho/fi)

    return theta, fi, rho


def essvi_to_svi(theta, essvi_param):
    fi = phi(theta, essvi_param)
    rho = _rho(theta, essvi_param)
    a = theta * (1 - rho ** 2) / 2
    b = theta * fi / 2
    m = - rho / fi
    sigma = numpy.sqrt(1 - rho ** 2) / fi

    return a, b, m, rho, sigma


def plot_essvi_to_svi(essvi_param, svi_params, theta):
    xx = numpy.linspace(-1.5, 1.5)
    w_essvi = ESSVI(xx, theta, essvi_param)
    w_svi = svi(xx, *svi_params)
    plt.plot(xx, w_essvi, 'r', lw=5, alpha=.6, label='essvi')
    plt.plot(xx, w_svi, 'g--', lw=5, alpha=.6, label='svi')
    plt.grid()
    plt.legend()
    plt.show()


def plot_svi_to_essvi(essvi_param, svi_params):
    xx = numpy.linspace(-1.5, 1.5)
    w_essvi = ESSVI_p_rho(xx, *essvi_param)
    w_svi = svi(xx, *svi_params)
    plt.plot(xx, w_essvi, 'r', lw=5, alpha=.6, label='essvi')
    plt.plot(xx, w_svi, 'g--', lw=5, alpha=.6, label='svi')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    svi_param = json.loads(json.load(open('svi.json')))
    essvi_param = json.load(open('essvi.json'))
    thetas = json.loads(json.load(open('theta.json')))

    for t, theta in thetas.items():
        # svi_params = essvi_to_svi(essvi_param)
        # plot_essvi_to_svi(essvi_param, svi_params, theta)
        plot_svi_to_essvi(svi_to_essvi(svi_param[t]), svi_param[t])
