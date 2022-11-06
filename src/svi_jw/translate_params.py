import numpy
from numpy import sqrt, sign, square

"""
Taken from Gatheral & Jaquier, 2013
'Arbitrage-free SVI volatility surfaces'
"""


# ---------------------------------------------
# mapping of SVI raw params to SVI-JW params
# Eq. 3.5 page 7


def v(t, a, b, rho, m, sigma):
    return (a + b * (-rho * m + sqrt(square(m) + square(sigma)))) / t


def phi(t, v_t, a, b, rho, m, sigma):
    w_t = w(v_t, t)
    return 1 / sqrt(w_t) * b / 2 * (-m / sqrt(square(m) + square(sigma)) + rho)


def p(t, v_t, a, b, rho, m, sigma):
    w_t = v_t * t
    return 1 / sqrt(w_t) * b * (1 - rho)


def c(t, v_t, a, b, rho, m, sigma):
    w_t = w(v_t, t)
    return 1 / sqrt(w_t) * b * (1 + rho)


def v_tilde(t, v_t, a, b, rho, m, sigma):
    return (a + b * sigma * sqrt(1 - square(rho))) / t


def w(v_t, t):
    return v_t * t


def jw_params_from_raw_params(t, a, b, rho, m, sigma):
    v_t = v(t, a, b, rho, m, sigma)
    phi_t = phi(t, v_t, a, b, rho, m, sigma)
    p_t = p(t, v_t, a, b, rho, m, sigma)
    c_t = c(t, v_t, a, b, rho, m, sigma)
    v_tilde_t = v_tilde(t, v_t, a, b, rho, m, sigma)
    return [v_t, phi_t, p_t, c_t, v_tilde_t]


# ---------------------------------------------
# mapping of SVI-JW params to SVI raw params
# Lemma 3.2 page 8

def calc_beta(t, v_t, phi_t, p_t, c_t, v_tilde_t):
    w_t = v_t * t
    b = calc_b(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    rho = calc_rho(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    return rho - (2 * phi_t * sqrt(w_t)) / b


def calc_alpha(t, v_t, phi_t, p_t, c_t, v_tilde_t):
    beta = calc_beta(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    return sign(beta) * sqrt(1 / square(beta) - 1)


def calc_b(t, v_t, phi_t, p_t, c_t, v_tilde_t):
    w_t = w(v_t, t)
    return sqrt(w_t) / 2 * (c_t + p_t)


def calc_rho(t, v_t, phi_t, p_t, c_t, v_tilde_t):
    w_t = w(v_t, t)
    b = calc_b(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    return 1 - p_t * sqrt(w_t) / b


def calc_a(t, v_t, phi_t, p_t, c_t, v_tilde_t):
    b = calc_b(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    rho = calc_rho(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    m = calc_m(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    sigma = calc_sigma(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    return v_tilde_t * t - b * sigma * sqrt(1 - square(rho))


def calc_m(t, v_t, phi_t, p_t, c_t, v_tilde_t):
    alpha = calc_alpha(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    beta = calc_beta(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    rho = calc_rho(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    b = calc_b(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    num = (v_t - v_tilde_t) * t
    denom = b * (-rho + sign(alpha) * sqrt(1 + square(alpha)) - alpha * sqrt(1 - square(rho)))
    return num / denom


def calc_sigma(t, v_t, phi_t, p_t, c_t, v_tilde_t):
    m = calc_m(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    if m == 0:
        a = calc_a(t, v_t, phi_t, p_t, c_t, v_tilde_t)
        b = calc_b(t, v_t, phi_t, p_t, c_t, v_tilde_t)
        return (v_t - a) / b
    else:
        alpha = calc_alpha(t, v_t, phi_t, p_t, c_t, v_tilde_t)
        return alpha * m


def raw_params_from_jw_params(t, v_t, phi_t, p_t, c_t, v_tilde_t):
    a = calc_a(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    b = calc_b(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    rho = calc_rho(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    m = calc_m(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    sigma = calc_sigma(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    return [a, b, rho, m, sigma]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.use("TkAgg")
    from src.svi.raw import total_implied_variance as svi

    x = numpy.linspace(-1.5, 1.5)
    # Axel Vogt exxample, Example 3.1, Eq. 3.8, p 10
    Vogt_params = a, b, m, rho, sigma = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153)
    raw_params = numpy.array([a, b, rho, m, sigma])
    t = 1  # same page
    jw_params = numpy.array(jw_params_from_raw_params(t, *raw_params))
    # from Example 5.1, p 20
    jw_correct = numpy.array(
        (0.01742625, -0.1752111, 0.6997381, 1.316798, 0.0116249))
    assert numpy.allclose(jw_correct, jw_params, atol=1e-6)
    raw_params2 = numpy.array(raw_params_from_jw_params(t, *jw_params))
    assert numpy.allclose(raw_params, raw_params2, atol=1e-6)
    y1 = svi(x, *raw_params)
    y2 = svi(x, *raw_params2)
    plt.title("Conversion of SVI raw -> SVI-JW -> SVI raw params")
    plt.plot(x, y1, 'k', lw=3, label="SVI with original raw params")
    plt.plot(x, y2, 'y--', lw=3, label="SVI with raw params recovered from jw equivalent")
    plt.legend()
    plt.tight_layout()
    plt.show()
