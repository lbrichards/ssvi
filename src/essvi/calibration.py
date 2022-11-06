import numpy
"""
Refer to ``Robust calibration and arbitrage-free interpolation of SSVI slices''
by Corbetta, Cohort, Laachir, and Martini, (2019)



"""

def get_kstar_and_theta_star_for_slice(slice):
    """
    Section 2.1, page 2

    Note that theta has a dual meaning in the text.  Sometimes it means t as
    in maturity, as in this case:

     *     *
    θ = w(k, θ)

    and the RHS of:
             *     *
        θ = θ ρθφk

    Taking θ on the RHS as t avoids the confusion of defining θ in terms of itself.
    The new definition of θ becomes:

             *       *
        θ = θ  - ρtφk

    """
    idx = numpy.argmin(abs(slice.k))
    k_star = slice.k.values[idx]
    theta_star = slice.t.unique()[0]* slice.iv.values[idx] **2
    return k_star, theta_star



if __name__ == '__main__':
    from pathlib import Path
    from pylab import *
    from src.data_utils import get_test_data, generate_slices
    slice = list(
        generate_slices(data_root=Path("../data")))[0]
    plt.scatter(slice.k, slice.iv)
    k_star, theta_star = get_kstar_and_theta_star_for_slice(slice)
    plt.scatter([k_star],[theta_star], s=50, color="g")
    plt.title(f"t = {slice.t.unique()[0]:0.3f}\n"
        f"k* = {k_star:0.3f}")
    plt.axvline(k_star, color='r')
    plt.show()