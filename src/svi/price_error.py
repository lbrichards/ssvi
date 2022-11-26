import json

import matplotlib.pyplot as plt
import numpy
from path import Path

from src.data_utils import get_test_data
from src.essvi.essvi_func import ESSVI
from src.svi_jw.svi_func import svi
from py_vollib.black import black


@numpy.vectorize
def vect_black(F, k, t, r, sigma):
    K = numpy.exp(k) * F
    if k<0:
        return black('p', F, K, t, r, sigma)
    else:
        return black('c', F, K, t, r, sigma)


def plot_market(df):
    S = df.S.unique()
    fp = True
    for i, row in df.iterrows():
        if row.k<0:
            if fp:
                plt.plot([row.k,row.k], [row.pbid*S, row.pask*S], 'y.-', lw=2, label='market price')
                fp=False
            else:
                plt.plot([row.k,row.k], [row.pbid*S, row.pask*S], 'y.-', lw=2)
        else:
            plt.plot([row.k,row.k], [row.cbid*S, row.cask*S], 'y.-', lw=2)


if __name__ == '__main__':
    svi_params = json.loads(json.load(open('..\svi_jw\svi.json')))
    df_origin = get_test_data(data_root=Path("../data"))
    print(df_origin.keys())

    essvi_params = json.load(open('..\svi_jw\essvi.json'))
    thetas = json.loads(json.load(open('..\svi_jw\\theta.json')))
    x = numpy.linspace(-1.5, 1.5)
    essvi_param = essvi_params
    for t, param in svi_params.items():
        theta = thetas[t]
        t = float(t)
        w1 = ESSVI(x, theta, essvi_param)
        ivs1 = numpy.sqrt(w1/t)
        w = svi(x, *param)
        ivs = numpy.sqrt(w/t)
        df = df_origin[df_origin.t == t]
        F = df.F_market.unique()
        r = df.r.unique()
        p = vect_black(F, x, t, r, ivs)
        p1 = vect_black(F, x, t, r, ivs1)
        plot_market(df)
        plt.plot(x, p, 'r', alpha=.4, label='svi model price', lw=2)
        plt.plot(x, p1, 'b', alpha=.4, label='essvi model price', lw=2)
        plt.ylabel('price')
        plt.xlabel('k')
        plt.title(f't:{t}')
        plt.grid()
        plt.legend()
        plt.show()
