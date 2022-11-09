import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy import stats

"""
Refer to ``The Calibrated SSVI Method - Implied Volatility Surface Construction''
by ADAM ÖHMAN, (2019)
"""


def get_theta_at_k0_for_slice(slice):
    idx = numpy.argmin(abs(slice.k))
    k1, iv1 = slice.k.values[idx], slice.iv.values[idx]
    if k1<0:
        k2, iv2 = slice.k.values[idx+1], slice.iv.values[idx+1]
    else:
        k2, iv2 = k1, iv1
        k1, iv1 = slice.k.values[idx-1], slice.iv.values[idx-1]

    iv_k0 = (k1 * iv2 - k2 * iv1) / (k1 - k2) # for k=0 the b is iv

    return slice.t.unique()[0] * iv_k0 ** 2


if __name__ == '__main__':
    from pathlib import Path
    from pylab import *
    from src.data_utils import get_test_data, generate_slices
    Plot = False
    thetas = {}
    for slice in generate_slices(data_root=Path("../data")):
        theta = get_theta_at_k0_for_slice(slice)
        thetas[slice.t.unique()[0]] = theta
        if Plot:
            plt.scatter(slice.k, slice.iv)
            plt.scatter([0], [theta], s=50, color="r")
            plt.title(f"t = {slice.t.unique()[0]:0.3f}\n")
            plt.axvline(0, color='r')
            plt.show()

    print(thetas)
    b = .1
    Theta = np.array(list(thetas.keys()))
    X = np.exp(-b*Theta)
    Y = list(thetas.values())
    # params = np.polyfit(X, Y, 1)
    params = stats.linregress(X, Y)
    a = params[0]
    c = params[1]
    print(a,b,c)
    plt.scatter(X, Y)
    yfit = [c + a * xi for xi in X]
    plt.plot(X, yfit, 'r')

    plt.show()
