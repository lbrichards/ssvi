import json

import numpy
from pylab import *
from src.data_utils import get_test_data, generate_slices_from_df
'''
phi function

Arbitrage-free SVI volatility surfaces
Jim Gatheral, Antoine Jacquiery
March 22, 2013
'''

def phi(theta, params):
    if len(params) == 2:
        Lambda, eta = params
    else:
        Lambda, eta, a, b, c = params
    return eta / (pow(theta, Lambda)*pow(1+theta, 1-Lambda))

def _rho(theta, params):
    if len(params) == 3:
        a, b, c = params
    else:
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


def surface_essvi(df, rho_func, params, outlier):
    ws = []
    for slice in generate_slices_from_df(df, outlier):
        theta = get_theta_at_k0_for_slice(slice)
        for x in slice.k:
            p = phi(theta, params)
            if rho_func:
                rho = rho_func(theta)
            else:
                rho = _rho(theta, params)
            ws.append(0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho)))

    return numpy.array(ws)


def ESSVI(x, theta, params):
    p = phi(theta, params)
    rho = _rho(theta, params)
    return 0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))


def ESSVI_theta_p_rho(x, theta, p, rho):
    return 0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))


def ESSVI_theta_si_rho(x, theta, si, rho):
    return 0.5 * (theta + rho * si * x +
                  np.sqrt(
                      np.square(si * x + theta*rho) +
                      theta**2 * (1. - rho **2)
                    )
                  )


def ESSVI1(x, theta, params):
    ## First derivative with respect to x
    p = phi(theta, params)
    rho = _rho(theta, params)
    return 0.5 * theta * p * (p * x + rho * np.sqrt(p * p * x * x + 2. * p * rho * x + 1.) + rho) / np.sqrt(
        p * p * x * x + 2. * p * rho * x + 1.)


def ESSVI2(x, theta, params):
    ## Second derivative with respect to x
    p = phi(theta, params)
    rho = _rho(theta, params)
    return 0.5 * theta * p * p * (1. - rho * rho) / (
                (p * p * x * x + 2. * p * rho * x + 1.) * np.sqrt(p * p * x * x + 2. * p * rho * x + 1.))


def g(x, theta, params):
    w = ESSVI(x, theta, params)
    w1 = ESSVI1(x, theta, params)
    w2 = ESSVI2(x, theta, params)
    return (1. - 0.5 * x * w1 / w) * (1. - 0.5 * x * w1 / w) - 0.25 * w1 * w1 * (0.25 + 1. / w) + 0.5 * w2



