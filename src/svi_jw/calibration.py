import numpy
import scipy.optimize as opt

from src.svi_jw.svi_func import svi, g
from src.svi_jw.translate_params import jw_params_from_raw_params
from matplotlib import pyplot as plt

from src.svi_jw.weights import make_weights


def calibrate_svi(slice, i=None, ps=True):
    t = slice.t.unique()[0]
    x, y = slice.k.values, slice.iv.values**2*t

    bnd = [
        [0.001, max(y)],
        [0.001, 1],
        [x.min() * 2, x.max() * 2],
        [-1, 1],
        [0, 1]
    ]

    def constraint1(params):
        a, b, m, rho, sigma = params
        return a + b * sigma * numpy.sqrt(1 - rho ** 2)

    def constraint2(params):
        return g(x, params)

    def constraint3(params):
        a, b, rho, m, sigma = params
        idx_mid = numpy.where(y == y.min())
        x_at_ymin = x[idx_mid].mean()
        y_model = svi(x, a, b, rho, m, sigma)
        p10 = numpy.percentile(y_model, 0.1)
        return p10 - y.min()

    def constraint4(params):
        # positive second derivative
        a, b, rho, m, sigma = params
        xtest = numpy.linspace(-2, 2)
        y = svi(xtest, a, b, rho, m, sigma)
        return numpy.gradient(numpy.gradient(y, edge_order=2), edge_order=2).sum()

    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    con3 = {'type': 'ineq', 'fun': constraint3}
    con4 = {'type': 'ineq', 'fun': constraint4}

    def objective(params):
        a, b, m, rho, sigma = params
        guess = svi(x, a, b, m, rho, sigma)
        if (guess<0).any():
            return 1
        w, mu = make_weights(x, y)
        return (numpy.square(guess - y)*w).sum()

    sol = opt.minimize(
        fun=objective,
        bounds=bnd,
        constraints=[con1, con2, con4],
        # constraints=[con1, con2, con3, con4],
        x0=numpy.array([0.5 * y.min(), .1, -.5, 0.1, 1]),
        # method="SLSQP",
        method="trust-constr",
        tol=1e-8
    )

    # print(sol)
    jw_params = jw_params_from_raw_params(t, *sol.x)
    xx = numpy.linspace(-1.5,1.5)
    y_out = svi(xx, *sol.x)

    plt.scatter(x, y, c='r', label=f"input data")
    plt.plot(xx, y_out, 'g', label=f"Model")
    plt.title(slice.t.unique()[0])
    plt.legend()
    plt.grid(True)
    return sol.x


if __name__ == '__main__':
    from pathlib import Path
    from pylab import *
    from src.data_utils import get_test_data, generate_slices_from_df

    thetas = {}
    df = get_test_data(data_root=Path("../data"))
    raw_params = {}
    tt = []
    for slice in generate_slices_from_df(df, outlier=False):
        raw_params[slice.t.unique()[0]]=list(calibrate_svi(slice))
        plt.show()

    # json.dump(json.dumps(raw_params), open('svi.json', 'w'))



