"""
We know it is possible to create a function f such that:

f(X_ssvi) = X_svi for a specific time slice

Goal: find an inverse function g(X_svi) = X_ssvi such that the answer X_svi is
unique, and recovers the original input to f.  In other words,

g(f(X_ssvi))== X_ssvi, and
f(g(X_svi))== X_svi
"""

import numpy
from matplotlib import pyplot as plt
from src.svi.raw import total_implied_variance
from src.ssvi.use_phi2 import SSVI

k = numpy.linspace(-1, 1)
t = 0.6385749302384577
a, b, rho, m, sigma  = [
    0.001073661028253302,
    0.32676198342955054,
    -0.07130173981948848,
    -0.3237675244847015,
    0.9182850078924533
]

omega0 = total_implied_variance(
    k, a, b, rho, m, sigma
)

plt.plot(k, omega0)
plt.show()


def f(ssvi_param):
    fi, rho, theta = ssvi_param
    a = theta * (1 - rho ** 2) / 2
    b = theta * fi / 2
    m = - rho / fi
    sigma = numpy.sqrt(1 - rho ** 2) / fi
    return a, b, m, rho, sigma

# SSVI(k, t, X_ssvi) # gamma, eta, sigma, rho








theta = 0.2968023710554979
# lambda, eta, a, b, c
Lambda, eta, a, b, c = [
    0.5571929268231374,
    0.9054684816164823,
    -0.4400804083680912,
    2.577453986220404,
    0.009431946667673415
]

fi = eta / (pow(theta, Lambda)*pow(1+theta, 1-Lambda))
rho = a * numpy.exp(-b * theta) + c
X_ssvi = [fi, rho, theta]


# a, b, m, rho, sigma


print(X_ssvi)