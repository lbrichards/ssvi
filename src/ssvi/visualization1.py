import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
sigma, gamma, rho = 0.2, 0.8, -0.7
xx, TT = np.linspace(-1., 1., 50), np.linspace(0.001, 5., 50)
from src.ssvi.use_phi1 import *
USE_LOCAL=False


print("Consistency check to avoid static arbitrage: ", (gamma - 0.25*(1.+np.abs(rho))>0.))
params = gamma, sigma, rho

localVarianceSSVI = [[SSVI_LocalVarg(x, t, params) for x in xx] for t in TT]
impliedVarianceSSVI = [[SSVI(x, t, params) for x in xx] for t in TT]

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(projection='3d')
xxx, TTT = np.meshgrid(xx, TT)
localVarianceSSVI = np.array(localVarianceSSVI)
impliedVarianceSSVI = np.array(impliedVarianceSSVI)
if USE_LOCAL:
    ax.plot_surface(xxx, TTT, localVarianceSSVI, cmap=plt.cm.viridis)
else:
    ax.plot_surface(xxx, TTT, impliedVarianceSSVI, cmap=plt.cm.viridis)
ax.set_xlabel("Log-moneyness")
ax.set_ylabel("Maturity")

if USE_LOCAL:
    ax.set_title("SSVI local variance")
    ax.set_zlabel("Local variance")
else:
    ax.set_title("SSVI variance")
    ax.set_zlabel("variance")

plt.tight_layout()
plt.show()