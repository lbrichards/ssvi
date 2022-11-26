import json

import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d
from py_vollib.black import black
from py_vollib.black.implied_volatility import implied_volatility as biv
from pathlib import Path
from pylab import *
from src.data_utils import get_test_data, generate_slices_from_df
from src.essvi.essvi_func import surface_essvi, get_theta_at_k0_for_slice, ESSVI


def iv_bid_ask_for_slice(slice, t):
    iva = []
    ivb = []
    for i, row in slice.iterrows():
        if row.k < 0:
            ivb.append(biv(row.pbid * row.S, row.F_market, row.K, row.r, t, row.flag))
            iva.append(biv(row.pask * row.S, row.F_market, row.K, row.r, t, row.flag))
        else:
            ivb.append(biv(row.cbid * row.S, row.F_market, row.K, row.r, t, row.flag))
            iva.append(biv(row.cask * row.S, row.F_market, row.K, row.r, t, row.flag))

    return iva, ivb

def calibrate_essvi(df, rho_func=None, outlier=True, Plot=False):
    y = []
    thetas = {0: 0}
    for slice in generate_slices_from_df(df, outlier):
        thetas[slice.t.unique()[0]] = get_theta_at_k0_for_slice(slice)
        for w in slice.w:
            y.append(w)

    def objective(params):
        guess = surface_essvi(df, rho_func, params, outlier)
        return np.abs(guess - y).sum()

    Lambda, eta, a, b, c = 0.2, 0.4, .1, .2, .1

    sol = opt.minimize(
        fun=objective,
        x0=numpy.array([Lambda, eta, a, b, c]),
        method="SLSQP",
        tol=1e-8
    )

    print(sol)
    # return sol.x
    for slice in generate_slices_from_df(df, outlier):
        theta = get_theta_at_k0_for_slice(slice)
        t = slice.t.unique()[0]
        if Plot:
            prices = []
            ivs = []
            ivb = []
            iva = []
            for i, row in slice.iterrows():
                w = ESSVI(row.k, theta, sol.x)
                iv = np.sqrt(w / t)
                ivs.append(iv)
                prices.append(black(row.flag, row.F_market, row.K, t, row.r, iv))
                if row.k < 0:
                    ivb.append(biv(row.pbid * row.S, row.F_market, row.K, row.r, t, row.flag))
                    iva.append(biv(row.pask * row.S, row.F_market, row.K, row.r, t, row.flag))
                else:
                    ivb.append(biv(row.cbid * row.S, row.F_market, row.K, row.r, t, row.flag))
                    iva.append(biv(row.cask * row.S, row.F_market, row.K, row.r, t, row.flag))

            plt.plot(slice.k, ivs, 'g', label='Model iv')
            plt.plot(slice.k, ivb, 'b', label='Market iv bid', alpha=.6)
            plt.plot(slice.k, iva, 'r', label='Market iv ask', alpha=.6)
            plt.grid()
            plt.legend()
            plt.xlim(-2, 2)
            plt.ylim(0, 3)
            plt.title(f't = {t}')
            plt.show()

    # plot_surface(sol, thetas)
    return sol.x


def plot_surface(sol, thetas):
    xx, TT = np.linspace(-3., 3., 20), np.linspace(.01, 1, 20)
    f = interp1d(list(thetas.keys()), list(thetas.values()), kind='quadratic', fill_value="extrapolate")
    g = lambda t: max(f(t), .1)
    wss = []
    for t in TT:
        theta = g(t)
        ws = ESSVI(xx, theta, sol.x)
        wss.append(ws)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection='3d')
    xxx, TTT = np.meshgrid(xx, TT)
    wss = np.array(wss)
    iv = numpy.sqrt(wss / TTT)
    ax.plot_surface(
        xxx, TTT, iv,
        cmap=plt.cm.viridis, rstride=1, cstride=1, linewidth=0)
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("essvi")
    ax.set_title("ESSVI")
    plt.show()


if __name__ == '__main__':
    thetas = {}
    df = get_test_data(data_root=Path("../data"))
    df['w'] = (df.iv ** 2) * df.t
    for slice in generate_slices_from_df(df, outlier=True):
        theta = get_theta_at_k0_for_slice(slice)
        thetas[slice.t.unique()[0]] = theta
    res = calibrate_essvi(df, Plot=True)
    json.dump(list(res), open('essvi.json', 'w'))
    json.dump(json.dumps(thetas), open('theta.json', 'w'))
