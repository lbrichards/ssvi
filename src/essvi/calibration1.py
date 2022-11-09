import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d
from py_vollib.black import black
from pathlib import Path
from pylab import *
from src.data_utils import get_test_data, generate_slices_from_df


def phi(theta, params):
    Lambda, eta, a, b, c = params
    return eta / pow(theta, Lambda)


def _rho(theta, params):
    Lammda, eta, a, b, c = params
    return a * np.exp(-b * theta) + c


def get_theta_at_k0_for_slice(slice):
    idx = numpy.argmin(abs(slice.k))
    k1, iv1 = slice.k.values[idx], slice.iv.values[idx]
    if k1 < 0:
        k2, iv2 = slice.k.values[idx + 1], slice.iv.values[idx + 1]
    else:
        k2, iv2 = k1, iv1
        k1, iv1 = slice.k.values[idx - 1], slice.iv.values[idx - 1]

    # generate line between two points
    # y intersect become iv for k=0
    iv_k0 = (k1 * iv2 - k2 * iv1) / (k1 - k2)  # b

    return slice.t.unique()[0] * iv_k0 ** 2


def essvi(df, rho_func, params):
    ws = []
    for slice in generate_slices_from_df(df):
        theta = get_theta_at_k0_for_slice(slice)
        for x in slice.k:
            p = phi(theta, params)
            if rho_func:
                rho = rho_func(theta)
            else:
                rho = _rho(theta, params)
            ws.append(0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho)))

    return numpy.array(ws)


def price_obj(df, params):
    prices = []
    for slice in generate_slices_from_df(df):
        theta = get_theta_at_k0_for_slice(slice)
        t = slice.t.unique()[0]
        for i, row in slice.iterrows():
            x = row.k
            p = phi(theta, params)
            rho = _rho(theta, params)
            w =0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))
            iv = np.sqrt(w / t)
            prices.append(black(row.flag,row.F_market, row.K, t, row.r, iv))


    return numpy.array(prices)


def calibrate_essvi(df, rho_func = None):
    y = []
    thetas = {0:0}
    for slice in generate_slices_from_df(df):
        thetas[slice.t.unique()[0]] = get_theta_at_k0_for_slice(slice)
        for w in slice.w:
            y.append(w)

    def objective(params):
        guess = essvi(df, rho_func, params)
        return abs(guess - y).sum()

    Lambda, eta, a, b, c = 0.2, 0.4, .1, .2, .1

    sol = opt.minimize(
        fun=objective,
        x0=numpy.array([Lambda, eta, a, b, c]),
        method="SLSQP",
        tol=.0000001
    )

    print(sol)

    # for slice in generate_slices_from_df(df):
    #     theta = get_theta_at_k0_for_slice(slice)
    #     prices =[]
    #     t = slice.t.unique()[0]
    #
    #     for i, row in slice.iterrows():
    #         w = ESSVI(row.k, theta, sol.x)
    #         iv = np.sqrt(w / t)
    #         prices.append(black(row.flag,row.F_market, row.K, t, row.r, iv))
    #
    #     plt.plot(slice.k, slice.mid- np.array(prices), 'r', label='market w')
    #     # plt.plot(slice.k, prices, 'g', label='calibrate w')
    #     plt.grid()
    #     plt.legend()
    #     # plt.ylim(0, 1.5)
    #     plt.title(f't = {t}')
    #     plt.show()
    #
    plot_surface(sol, thetas)


def plot_surface(sol, thetas):
    xx, TT = np.linspace(-3., 3., 20), np.linspace(.01, 10, 20)
    f = interp1d(list(thetas.keys()), list(thetas.values()), kind='quadratic', fill_value="extrapolate")
    g = lambda t: max(f(t), .1)
    wss = []
    for t in TT:
        theta = g(t)
        ws = []
        for k in xx:
            ws.append(ESSVI(k, theta, sol.x))
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
    ax.set_title("eSSVI")
    plt.show()


def ESSVI(x, theta, params):
    p = phi(theta, params)
    rho = _rho(theta, params)
    return 0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))


if __name__ == '__main__':
    thetas = {}
    df = get_test_data(data_root=Path("../data"))
    df['w'] = (df.iv ** 2) * df.t
    # for slice in generate_slices_from_df(df):
    #     theta = get_theta_at_k0_for_slice(slice)
    calibrate_essvi(df)


# compare with mcd calibration
