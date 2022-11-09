import numpy as np


def phi(theta, params):
    gamma, eta, sigma, a, b, c = params
    return eta / pow(theta, gamma)


def _rho(theta, params):
    gamma, eta, sigma, a, b, c = params
    return a*np.exp(-b*theta)+c


def ESSVI(x, theta, params):
    p = phi(theta, params)
    rho = _rho(theta, params)
    return 0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))

