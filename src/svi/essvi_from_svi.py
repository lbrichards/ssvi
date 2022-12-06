import json

import numpy
import pandas
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from src.data_utils import get_test_data, generate_slices_from_df
import scipy.optimize as opt
from path import Path

from src.essvi.calibration import plot_surface, iv_bid_ask_for_slice
from src.essvi.essvi_func import phi, _rho, get_theta_at_k0_for_slice, ESSVI, ESSVI_theta_p_rho
from src.svi.ssvi_from_svi import error_count_essvi
from src.svi.svi import error_count
from src.svi_jw.svi_func import svi


def svi_to_essvi(svi_param):
    a, b, m, rho, sigma = svi_param
    fi = numpy.sqrt(1-rho**2)/sigma
    theta = 2*b*sigma/numpy.sqrt(1-rho**2)

    return theta, fi, rho


def calibrate_phi_func_parameters(thetas, phis):
    phi_values = numpy.array(list(phis.values()))
    theta_values = numpy.array(list(thetas.values()))

    def objective(params):
        fi = phi(theta_values, params)
        return numpy.square(phi_values - fi).sum()

    sol = opt.minimize(
        fun=objective,
        x0=numpy.array([.1, 0.1]),
        method="SLSQP",
        tol=1e-8
    )

    return sol.x


def calibrate_rho_func_parameters(thetas, rhos):
    rho_values = numpy.array(list(rhos.values()))
    theta_values = numpy.array(list(thetas.values()))

    def objective(params):
        rho = _rho(theta_values, params)
        return numpy.square(rho_values - rho).sum()

    sol = opt.minimize(
        fun=objective,
        x0=numpy.array([.1, 0.1, .2]),
        method="SLSQP",
        tol=1e-8
    )

    return sol.x


def Plot():
    global t
    print(thetas, phis, rhos)
    print(Lambda, eta, a, b, c)
    tf = [float(i) for i in thetas.keys()]
    t = [f'{float(i):0.2f}' for i in thetas.keys()]
    theta_values = numpy.array(list(thetas.values()))
    phi_values = numpy.array(list(phis.values()))
    rho_values = numpy.array(list(rhos.values()))
    f = interp1d(tf, theta_values, kind='quadratic', fill_value="extrapolate")


    plt.plot(t, phi(theta_values, (Lambda, eta)), label='phi from Model')
    plt.plot(t, phi_values, label='phi from svi')
    plt.grid()
    plt.legend()
    plt.show()
    plt.plot(t, _rho(theta_values, (a, b, c)), label='rho from Model')
    plt.plot(t, rho_values, label='rho from svi')
    plt.grid()
    plt.legend()
    plt.show()
    plt.plot(t, f(t), label='rho from Model')
    plt.plot(t, theta_values, label='theta from svi')
    plt.grid()
    plt.legend()
    plt.show()


def plot_slices(thetas, svi_params, essvi_params, calibrated = False):
    df_origin = get_test_data(data_root=Path("../data"))
    df_origin['w'] = (df_origin.iv ** 2) * df_origin.t
    xx = numpy.linspace(-1.5, 1.5)
    for t, theta in thetas.items():
        t = float(t)
        slice_origin = df_origin[df_origin.t == float(t)]
        iva, ivb = iv_bid_ask_for_slice(slice_origin, float(t))

        x, y = slice_origin.k.values, slice_origin.iv.values ** 2 * float(t)
        params = svi_params[str(t)]
        if calibrated:
            w2 = ESSVI(xx, theta, essvi_params)
        else:
            essvi_params = (theta, phis[str(t)], rhos[str(t)])
            w2 = ESSVI_theta_p_rho(xx, *essvi_params)
        w3 = svi(xx, *params)

        plt.figure(figsize=[10, 8])
        plt.scatter(x, y, c='m', label='Market')
        for k, i1, i2 in zip(x, ivb, iva):
            plt.plot([k, k], [i1 ** 2 * t, i2 ** 2 * t], 'y.-', lw=2)
        plt.plot(xx, w2, 'r', label='ESSVI using SVI Model', lw=2, alpha=.5)
        plt.plot(xx, w3, 'b', label='SVI Model', lw=2, alpha=.5)
        plt.title(f't: {t}\n'
                  'inlier essvi accuracy: '
                  f'{100 - 100 * numpy.count_nonzero(error_count_essvi(slice_origin, theta, essvi_params)) / len(slice_origin):0.2f}%'
                  '\ninlier svi accuracy: '
                  f'{100 - 100 * numpy.count_nonzero(error_count(slice_origin, params)) / len(slice_origin):0.2f}%'
                  )
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':

    df_origin = get_test_data(data_root=Path("../data"))
    df_origin['w'] = (df_origin.iv ** 2) * df_origin.t

    svi_params = json.loads(json.load(open('svi.json')))
    thetas = {}
    phis = {}
    rhos = {}
    for t, params in svi_params.items():
        thetas[t], phis[t], rhos[t] = svi_to_essvi(params)

    Lambda, eta = calibrate_phi_func_parameters(thetas, phis)
    a, b, c = calibrate_rho_func_parameters(thetas, rhos)
    _Plot = False
    if _Plot:
        Plot()

    essvi_params = Lambda, eta, a, b, c
    plot_surface(essvi_params, thetas)
    plot_slices(thetas, svi_params, essvi_params)