import json

import matplotlib.pyplot as plt
import numpy
from scipy.interpolate import interp1d
from scipy.stats import linregress

from src.essvi.calibration import plot3d
from src.essvi.essvi_func import phi, _rho, ESSVI
from src.svi_jw.mapping_svi_ssvi import svi_to_essvi
from src.svi_jw.svi_func import svi
import scipy.optimize as opt


def calibrate_phi_func_parameters(theta_values, phi_values):
    def objective(params):
        fi = phi(theta_values, params)
        return numpy.square(phi_values - fi).sum()
    bnd = [
        [.001, .999],
        [0.001, 1000]
    ]

    sol = opt.minimize(
        fun=objective,
        bounds=bnd,
        x0=numpy.array([.1, 0.1]),
        method="SLSQP",
        tol=1e-8
    )
    Plot = False
    if Plot:
        th = numpy.linspace(.002, .5)
        plt.plot(theta_values, phi_values, 'o', label='Market')
        plt.plot(th, phi(th, sol.x), label='Model')
        plt.legend()
        plt.grid()
        plt.title('phi')
        plt.show()
    print(f'Lambda , eta: {sol.x}')
    return sol.x


def calibrate_rho_func_parameters(theta_values, rho_values):
    def objective(params):
        rho = _rho(theta_values, params)
        return numpy.square(rho_values - rho).sum()

    sol = opt.minimize(
        fun=objective,
        x0=numpy.array([.1, 0.1, .2]),
        method="SLSQP",
        tol=1e-8
    )
    print(sol.x)
    Plot = False
    if Plot:
        plt.plot(theta_values, rho_values, label='Market')
        plt.plot(theta_values, _rho(theta_values, sol.x), label='Model')
        plt.legend()
        plt.grid()
        plt.title('rho')
        plt.show()
    return sol.x


class SsviPhiRho:
    def __init__(self, svi_params):
        self.svi_params = svi_params
        self.ts = [float(t) for t in svi_params.keys()]
        self.tmin = min(self.ts)
        self.tmax = max(self.ts)
        self._essvi_funcs()

    def _essvi_funcs(self):
        thetas = []
        thetas1 = []

        phis = []
        rhos = []
        for t, param in self.svi_params.items():
            theta, fi, rho = svi_to_essvi(param)
            theta_from_svi = svi(0, *param)
            thetas.append(theta)
            thetas1.append(theta_from_svi)
            phis.append(fi)
            rhos.append(rho)


        thetas = numpy.array(thetas)
        phis = numpy.array(phis)
        rhos = numpy.array(rhos)
        self.theta_func = interp1d(self.ts, thetas, kind='linear', fill_value="extrapolate")
        xx= numpy.linspace(.01, 1)
        # plt.plot(xx, self.theta_func(xx))
        # plt.scatter(self.ts, thetas, color='r', s=80)
        # plt.plot(self.ts, thetas1, 'go')
        # plt.show()

        Lambda, eta = calibrate_phi_func_parameters(thetas, phis)
        a, b, c = calibrate_rho_func_parameters(thetas, rhos)
        self.essvi_params = (Lambda, eta, a, b, c)

    def ssvi(self, x, t):
        theta = self.theta_func(t)
        return ESSVI(x, theta, self.essvi_params)


def Plot3d():
    global t
    svi_params = json.loads(json.load(open('svi1.json')))
    c_ssvi = SsviPhiRho(svi_params)
    xx = numpy.linspace(-1.5, 1.5)
    tt = numpy.linspace(.01, 1)
    wss = []
    for t in tt:  # svi_params.keys():
        wss.append(c_ssvi.ssvi(xx, float(t)))
    plot3d(xx, tt, wss)


def test_extra_interp(n):
    svi_params = json.loads(json.load(open('svi1.json')))
    s = svi_params.copy()
    ts = list(s.keys())
    n = n if n >= 0 else len(ts) + n
    s = {ts[i]: s[ts[i]] for i in range(len(ts)) if i != n}
    c_ssvi = SsviPhiRho(s)
    c_ssvi1 = SsviPhiRho(svi_params)
    x = [round(float(x),2) for x in svi_params.keys()]
    # plt.plot(x, c_ssvi1.theta_func(x))
    # plt.plot(x, c_ssvi.theta_func(x))
    # plt.show()
    xx = numpy.linspace(-1.5, 1.5)
    for t in svi_params.keys():
        ws = c_ssvi.ssvi(xx, float(t))
        plt.plot(xx, ws, label='ssvi')
        plt.plot(xx, svi(xx, *svi_params[t]), 'r--', label='svi')
        plt.title(f't:{t}')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    test_extra_interp(0)
    # Plot3d()
