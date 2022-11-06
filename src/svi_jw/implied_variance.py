from translate_params import raw_params_from_jw_params, jw_params_from_raw_params
from src.svi.raw import total_implied_variance
import numpy

@numpy.vectorize
def implied_variance(k, t, v_t, phi_t, p_t, c_t, v_tilde_t):
    """
    From page 6:

    ``The SVI-Jump-Wings (SVI-JW) parameterization of the implied
    variance v (rather than the implied total variance w)''

    :param k: log-moneyness ln(K/F)
    :param t: maturity in years
    :param v_t: vt gives the ATM variance
    :param phi_t: gives the ATM skew
    :param p_t: gives the slope of the left (put) wing
    :param c_t: gives the slope of the right (call) wing
    :param v_tilde_t: is the minimum implied variance
    :return: implied variance v
    """
    raw_params =  raw_params_from_jw_params(t, v_t, phi_t, p_t, c_t, v_tilde_t)
    implied_variance = total_implied_variance(k, *raw_params)/t
    return implied_variance

def demo_svi_raw_equivalence(t = 1):
    """
    Note:  This plot is t-invariant
    because we always use t to map the Vogt params to their
    equivalent JW params.
    """
    from src.svi.raw import total_implied_variance
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")
    # Axel Vogt exxample, Example 3.1, Eq. 3.8, p 10
    Vogt_params = a, b, m, rho, sigma = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153)
    raw_params = numpy.array([a, b, rho, m, sigma])
    if t == 1:
        # from Example 5.1, p 20
        jw_params = numpy.array(
            (0.01742625, -0.1752111, 0.6997381, 1.316798, 0.0116249))
    else:
        jw_params = jw_params_from_raw_params(t, *raw_params)
    x = numpy.linspace(-1.5, 1.5)
    total_variance_1 = total_implied_variance(x, *raw_params)
    v = implied_variance(x, t, *jw_params)
    total_variance_2 = v * t
    plt.title(f"""Illustrating equivalence of SVI Raw and SVI-JW
    Case $t={t}$""")
    plt.plot(x, total_variance_1, 'k', lw=3, label="SVI from raw params")
    plt.plot(x, total_variance_2, 'y--', lw=3, label="SVI-JW from equivalent params")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    demo_svi_raw_equivalence(t=5)
