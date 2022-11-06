import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from src.ssvi.use_phi2 import *
USE_LOCAL=False


sigma, gamma, eta, rho = 0.2, 0.4, 0.2, -0.4
xx, TT = np.linspace(-3., 3., 20), np.linspace(0.1, 2., 20)

print("Consistency check to avoid static arbitrage: ", (gamma - 0.25*(1.+np.abs(rho))>0.))
params = gamma, eta, sigma, rho

localVarianceSSVI = [[SSVI_LocalVarg(x, t, params) for x in xx] for t in TT]

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(projection='3d')
xxx, TTT = np.meshgrid(xx, TT)
localVarianceSSVI = np.array(localVarianceSSVI)
ax.plot_surface(
    xxx, TTT, localVarianceSSVI,
    cmap=plt.cm.viridis, rstride=1, cstride=1, linewidth=0)
ax.set_xlabel("Log-moneyness")
ax.set_ylabel("Maturity")
ax.set_zlabel("Local variance")
ax.set_title("SSVI local variance")
plt.show()