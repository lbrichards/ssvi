import json

# import matplotlib.pyplot as plt
import numpy
import scipy.optimize as opt
from pylab import *
from src.essvi.essvi_func import ESSVI, ESSVI_theta_p_rho, phi, _rho
from src.svi_jw.svi_func import svi


def svi2ssvi(svi_param, theta):
    a, b, m, rho, sigma = svi_param
    x = numpy.linspace(-1.5, 1.5, 20)
    y = svi(x, *svi_param)

    bnd = [
        [0.001, 10]]

    def objective(params):
        guess = ESSVI_theta_p_rho(x, *params)
        return np.abs(guess - y).sum()

    def constraint1(params):
        theta, p, r = params
        return p - numpy.sqrt(1-rho**2)/sigma

    def constraint2(params):
        theta, p, r = params
        return theta - (2*b*sigma/numpy.sqrt(1-rho**2))

    def constraint4(params):
        theta, p, r = params
        return theta - (2*a/(1-rho**2))

    def constraint3(params):
        theta, p, r = params
        return p + rho/m

    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    con3 = {'type': 'ineq', 'fun': constraint3}
    con4 = {'type': 'ineq', 'fun': constraint4}

    theta, p, rho = 0.4, .1, .1

    sol = opt.minimize(
        fun=objective,
        bounds=bnd,
        x0=numpy.array([theta, p, rho]),
        constraints=[con1, con4],
        method="trust-constr",
        tol=1e-12
    )

    print(sol)
    y1 = ESSVI_theta_p_rho(x, *sol.x)
    plt.plot(x,y, label='svi')
    plt.plot(x,y1, label='ssvi')
    return sol.x


def svi2essvi(svi_param, theta):
    x = numpy.linspace(-1.5, 1.5)
    y = svi(x, *svi_param)

    def objective(params):
        guess = ESSVI(x, theta, params)
        return np.abs(guess - y).sum()

    Lambda, eta, a, b, c = 0.2, 0.4, .1, .2, .1

    sol = opt.minimize(
        fun=objective,
        x0=numpy.array([Lambda, eta, a, b, c]),
        method="SLSQP",
        tol=1e-8
    )

    print(sol.x)
    y1 = ESSVI(x, theta, sol.x)
    plt.plot(x, y1, label='essvi')

    return sol.x


if __name__ == '__main__':
    svi_params = json.loads(json.load(open('..\svi_jw\svi.json')))
    essvi_params = json.load(open('..\svi_jw\essvi.json'))
    thetas = json.loads(json.load(open('..\svi_jw\\theta.json')))

    ps=[]
    p1s=[]
    rs=[]
    r1s=[]
    for t, theta in thetas.items():
        if float(t)<.6:
            continue
        svi_param = svi_params[str(t)]
        theta, p, r = svi2ssvi(svi_param, theta)
        ps.append(p)
        rs.append(r)
        # params = svi2essvi(svi_param, theta)
        # p1s.append(phi(theta, params))
        # r1s.append(_rho(theta, params))
        plt.legend()
        plt.grid()
        plt.show()

    # plt.plot(ps, 'r--', lw=2)
    # plt.plot(p1s, 'g', lw=2, alpha=.5)
    # plt.show()
    # plt.plot(rs, 'r--', lw=2)
    # plt.plot(r1s, 'g', lw=2, alpha=.5)
    # plt.show()