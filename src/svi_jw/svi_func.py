from numpy import sqrt, square
import numpy


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


def svi(x, a, b, m, rho, sigma):
    return a + b * (rho * (x - m) + sqrt(square(x - m) + square(sigma)))


def omega_prime(x, params):
    a, b, m, rho, sigma = params
    return b * (rho + (-m + x) / sqrt(sigma ** 2 + (-m + x) ** 2))


def omega_dbl_prime(x, params):
    a, b, m, rho, sigma = params
    return b * ((-m + x) * (m - x) / (sigma ** 2 +
                                      (-m + x) ** 2) ** (3 / 2) +
                1 / sqrt(sigma ** 2 + (-m + x) ** 2))


def g(x, params):
    k = x
    a, b, m, rho, sigma = params
    omega = svi(x, a, b, m, rho, sigma)

    # avoid zero division
    omega[omega == 0] = 1e-12

    term1 = (1 - k * omega_prime(k, params) / 2 * omega) ** 2
    term2 = ((omega_prime(k, params) ** 2) / 4) * (1 / omega + 1 / 4)
    term3 = omega_dbl_prime(k, params) / 2
    return term1 - term2 + term3


if __name__ == '__main__':
    from sympy import diff, sin, exp, sqrt
    from sympy.abc import x, a, b, rho, m, sigma

    expr = a + b * (rho * (x - m) + sqrt(square(x - m) + square(sigma)))
    d1=diff(expr, x)
    print(d1)
    print(diff(d1, x))