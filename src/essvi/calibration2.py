import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d
from py_vollib.black import black

from src.essvi.calibration1 import calibrate_essvi


def phi(theta, params):
    Lambda, eta, p = params
    return eta / pow(theta, Lambda)


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


def ssvi(slice, params):
    ws = []
    theta = get_theta_at_k0_for_slice(slice)
    for x in slice.k:
        p = phi(theta, params)
        rho = params[2]
        ws.append(0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho)))

    return numpy.array(ws)


def calibrate_ssvi(slice):
    y = []
    for w in slice.w:
        y.append(w)

    def objective(params):
        guess = ssvi(slice, params)
        return abs(guess - y).sum()

    Lambda, eta, p = 0.32, 0.94, .09

    sol = opt.minimize(
        fun=objective,
        x0=numpy.array([Lambda, eta, p]),
        method="SLSQP",
        tol=.0000001
    )

    print(sol)

    theta = get_theta_at_k0_for_slice(slice)
    prices =[]
    t = slice.t.unique()[0]
    for i, row in slice.iterrows():
        w = ESSVI(row.k, theta, sol.x)
        iv = np.sqrt(w / t)
        prices.append(black(row.flag,row.F_market, row.K, t, row.r, iv))

    Plot = False
    if Plot:
        plt.plot(slice.k, (slice.mid - np.array(prices))/slice.F, 'r', label='market w')
        # plt.plot(slice.k, prices, 'g', label='calibrate w')
        plt.grid()
        plt.legend()
        # plt.ylim(0, 1.5)
        plt.title(f't = {t}')
        plt.show()

    return sol


def ESSVI(x, theta, params):
    p = phi(theta, params)
    rho = params[2]
    return 0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))


if __name__ == '__main__':
    from pathlib import Path
    from pylab import *
    from src.data_utils import get_test_data, generate_slices_from_df

    thetas = {}
    df = get_test_data(data_root=Path("../data"))
    df['w'] = (df.iv ** 2) * df.t
    sols = []
    tt = []
    for slice in generate_slices_from_df(df):
        tt.append(get_theta_at_k0_for_slice(slice))
        theta = get_theta_at_k0_for_slice(slice)
        sols.append(calibrate_ssvi(slice).x[2])

    sols = np.array(sols)
    rho_func = interp1d(tt, sols, kind='linear')
    calibrate_essvi(df, rho_func)

    # t = np.linspace(min(tt), max(tt))
    # plt.plot(t, rho_func(t), 'r')
    # plt.scatter(tt, sols)
    # plt.show()
