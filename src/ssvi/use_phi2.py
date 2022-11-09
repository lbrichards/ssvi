"""
From Jack Jacquier
https://github.com/JackJacquier/SSVI/blob/master/SSVILocalVol.ipynb
"""

import numpy as np

def phi(theta, params):
    gamma, eta, sigma, rho = params
    return eta / pow(theta, gamma)


def SSVI(x, t, params):
    gamma, eta, sigma, rho = params
    theta = sigma * sigma * t
    p = phi(theta, params)
    return 0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))


def SSVI1(x, t, params):
    ## First derivative with respect to x
    gamma, eta, sigma, rho = params
    theta = sigma * sigma * t
    p = phi(theta, params)
    return 0.5 * theta * p * (p * x + rho * np.sqrt(p * p * x * x + 2. * p * rho * x + 1.) + rho) / np.sqrt(
        p * p * x * x + 2. * p * rho * x + 1.)


def SSVI2(x, t, params):
    ## Second derivative with respect to x
    gamma, eta, sigma, rho = params
    theta = sigma * sigma * t
    p = phi(theta, params)
    return 0.5 * theta * p * p * (1. - rho * rho) / (
                (p * p * x * x + 2. * p * rho * x + 1.) * np.sqrt(p * p * x * x + 2. * p * rho * x + 1.))


def SSVIt(x, t, params):
    ## First derivative with respect to t, by central difference
    eps = 0.0001
    return (SSVI(x, t + eps, params) - SSVI(x, t - eps, params)) / (2. * eps)


def g(x, t, params):
    w = SSVI(x, t, params)
    w1 = SSVI1(x, t, params)
    w2 = SSVI2(x, t, params)
    return (1. - 0.5 * x * w1 / w) * (1. - 0.5 * x * w1 / w) - 0.25 * w1 * w1 * (0.25 + 1. / w) + 0.5 * w2


def dminus(x, t, params):
    vsqrt = np.sqrt(SSVI(x, t, params))
    return -x / vsqrt - 0.5 * vsqrt


def densitySSVI(x, t, params):
    dm = dminus(x, t, params)
    return g(x, t, params) * np.exp(-0.5 * dm * dm) / np.sqrt(2. * np.pi * SSVI(x, t, params))


def SSVI_LocalVarg(x, t, params):
    ## Compute the equivalent SSVI local variance
    return SSVIt(x, t, params) / g(x, t, params)
