import json

import numpy
from matplotlib import pyplot as plt
from path import Path

from src.data_utils import get_test_data
from src.essvi.essvi_func import ESSVI
from src.essvi.essvi_func import g as eg
from src.svi_jw.calibration import svi
from src.svi_jw.svi_func import g, omega_dbl_prime


def plot_compare(essvi_param, svi_param, thetas):
    # plt.plot(thetas.keys(), thetas.values())
    # plt.show()
    df = get_test_data(data_root=Path("../data"))
    xx = numpy.linspace(-1.5, 1.5)
    for t, theta in thetas.items():
        slice = df[df.t == float(t)]
        x, y = slice.k.values, slice.iv.values ** 2 * float(t)
        params = svi_param[str(t)]
        plt.subplot(211)
        plt.scatter(x, y, c='m', label='Market')
        plot_slice(essvi_param, params, t, theta, xx)
        plt.show()


def plot_slice(essvi_param, params, t, theta, xx):
    w1 = ESSVI(xx, theta, essvi_param)
    w2 = svi(xx, *params)
    egg = eg(xx, theta, essvi_param)
    gg = g(xx, params)
    plt.plot(xx, w1, 'g', label='ESSVI Model')
    plt.plot(xx, w2, 'r', label='SVI Model')
    plt.grid()
    plt.legend()
    plt.twinx()
    plt.plot(xx, w1 - w2)
    plt.title(f't = {t}')
    plt.subplot(212)
    plt.plot(xx, egg, 'g', label='g for ESSVI')
    plt.plot(xx, gg, 'r', label='g for SVI')
    plt.grid()
    plt.legend()


if __name__ == '__main__':
    svi_param = json.loads(json.load(open('svi.json')))
    essvi_param = json.load(open('essvi.json'))
    thetas = json.loads(json.load(open('theta.json')))
    plot_compare(essvi_param, svi_param, thetas)