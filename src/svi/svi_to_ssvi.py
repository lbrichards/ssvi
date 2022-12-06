import json

import matplotlib.pyplot as plt
import numpy
from scipy.interpolate import interp1d

from src.essvi.calibration import plot3d
from src.essvi.essvi_func import ESSVI_theta_p_rho
from src.svi_jw.mapping_svi_ssvi import svi_to_essvi
from src.svi_jw.svi_func import svi


class Ssvi:
    def __init__(self, svi_params):
        self.svi_params = svi_params
        self.ts = [float(t) for t in svi_params.keys()]
        self.tmin = min(self.ts)
        self.tmax = max(self.ts)
        self._essvi_funcs()

    def _essvi_funcs(self):
        thetas = []
        phis = []
        rhos = []
        for t, param in self.svi_params.items():
            theta, fi, rho = svi_to_essvi(param)
            thetas.append(theta)
            phis.append(fi)
            rhos.append(rho)

        thetas = numpy.array(thetas)
        phis = numpy.array(phis)
        rhos = numpy.array(rhos)
        self.theta_func = interp1d(self.ts, thetas, kind='cubic')
        self.si_func = interp1d(self.ts, thetas * phis, kind='cubic')
        self.fi_func = interp1d(self.ts, phis, kind='cubic')
        self.si_rho_func = interp1d(self.ts, rhos*thetas * phis, kind='cubic')
        # self.rho_func = interp1d(self.ts, rhos, kind='cubic')

    def rho_func(self, t):
        return self.si_rho_func(t)/self.si_func(t)

    def ssvi_fi(self, x, t):
        theta = self.theta_func(t)
        rho = self.rho_func(t)
        p = self.fi_func(t)
        return 0.5 * theta * (1. + rho * p * x + numpy.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))

    def ssvi(self, x, t):
        theta, si, rho = self.ssvi_param_for_maturity(t)
        print(t, theta, si, rho)
        return 0.5 * (theta + rho * si * x +
                      numpy.sqrt(
                          numpy.square(si * x + theta * rho) +
                          theta ** 2 * (1. - rho ** 2)
                      )
                      )

    def ssvi_param_for_maturity(self, t):
        if t <= self.tmin:
            return self.min_params(t)
        elif t >= self.tmax:
            return self.max_params(t)
        else:
            return self.theta_func(t), self.si_func(t), self.rho_func(t)

    def min_params(self, t):
        T1 = self.tmin
        Lambda = t / T1
        return Lambda * self.theta_func(T1), Lambda * self.si_func(T1), self.rho_func(T1)

    def max_params(self, t):
        TN = self.tmax
        TN_1 = self.ts[-2]
        thetaN = self.theta_func(TN)
        thetaN_1 = self.theta_func(TN_1)
        c = (thetaN - thetaN_1) / (TN - TN_1)
        return thetaN + c * (t - TN), self.si_func(TN), self.rho_func(TN)


def Plot3d():
    global t
    svi_params = json.loads(json.load(open('svi.json')))
    c_ssvi = Ssvi(svi_params)
    xx = numpy.linspace(-1.5, 1.5)
    tt = numpy.linspace(.01, 1)
    wss = []
    for t in tt:  # svi_params.keys():

        wss.append(c_ssvi.ssvi(xx, float(t)))
    plot3d(xx, tt, wss)


def test_extra_interp(n):
    svi_params = json.loads(json.load(open('svi.json')))
    s = svi_params.copy()
    ts = list(s.keys())
    n = n if n >= 0 else len(ts) + n
    s = {ts[i]: s[ts[i]] for i in range(len(ts)) if i != n}
    c_ssvi = Ssvi(s)
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
    test_extra_interp(2)
    # Plot3d()
