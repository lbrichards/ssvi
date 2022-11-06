from numpy import sqrt, vectorize

@vectorize
def total_implied_variance(k, a, b, rho, m, sigma):
    """
    Gatheral 2012 pg. 10

    :param k: log moneyness ln(K/F)
    :param a: prop. to general level of variance, a vertical translation of the smile
    :param b: prop. to slope of both wings, tightening the smile
    :param rho: prop to the counter-clockwise rotation of the smile
    :param m: translates the smile rightward
    :param sigma: inv. prop to ATM curvature of smile
    :return: len(k) 1d array of the total implied variance "w" per strike k
    """

    return a+b*(rho*(k-m)+sqrt((k-m)**2+sigma**2))

