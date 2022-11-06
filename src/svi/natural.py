from numpy import sqrt, vectorize, square


@vectorize
def w(k, delta, mu, rho, omega, zeta):
    """
    Gatheral 2012 pg. 12, a "natural generalization of the time oo Heston smile"

    :return: len(k) 1d array of total implied variance per strike
    """
    assert omega>=0
    assert abs(rho)<1
    assert zeta>0

    return delta+(omega/2)*(
            1+zeta*rho*(k-mu)+sqrt(
                square(
                    zeta*(k-mu)+rho
                )+(1-rho**2)
            )
        )

