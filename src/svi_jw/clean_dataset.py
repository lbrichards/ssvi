import json

import numpy
import pandas
from matplotlib import pyplot as plt
from src.data_utils import get_test_data, generate_slices_from_df
from src.essvi.calibration import get_theta_at_k0_for_slice, ESSVI, calibrate_essvi, iv_bid_ask_for_slice
from path import Path

from src.essvi.essvi_func import ESSVI_p_rho
from src.svi_jw.calibration import svi
from src.svi_jw.compare_svi_and_essvi import plot_compare
from src.svi_jw.mapping_svi_ssvi import svi_to_essvi

df_origin = get_test_data(data_root=Path("../data"))


svi_param = json.loads(json.load(open('svi.json')))
thetas = {}

df = get_test_data(data_root=Path("../data"))
df['w'] = (df.iv ** 2) * df.t
xx = numpy.linspace(-1.5, 1.5)
dfs = []
for t, params in svi_param.items():
    t = float(t)
    data = []
    w = svi(xx, *params)
    for i, x in enumerate(xx):
        iv = numpy.sqrt(w/t)
        data.append({
            't': t,
            'k': x,
            'iv': iv[i],
            'w': w[i],
            # 'flag': 'p' if x<0 else 'c'
        })
    dfs.append(pandas.DataFrame(data))

df = pandas.concat(dfs)
df.index = range(len(df))
essvi_param1 = json.load(open('essvi.json'))
essvi_param = calibrate_essvi(df, outlier=False)
xx = numpy.linspace(-1.5, 1.5)
for slice in generate_slices_from_df(df, outlier=False):
    t = float(slice.t.unique()[0])
    slice_origin = df_origin[df_origin.t == float(t)]
    iva, ivb = iv_bid_ask_for_slice(slice_origin, t)
    theta = get_theta_at_k0_for_slice(slice)
    thetas[t] = theta

    x, y = slice_origin.k.values, slice_origin.iv.values ** 2 * float(t)
    params = svi_param[str(t)]
    # essvi_param = svi_to_essvi(params)
    # w2 = ESSVI_p_rho(xx, *essvi_param)
    w1 = ESSVI(xx, theta, essvi_param1)
    w2 = ESSVI(xx, theta, essvi_param)
    w3 = svi(xx, *params)

    plt.figure(figsize=[10,8])
    # plt.scatter(x, y, c='m', label='Market')
    for k, i1, i2 in zip(x, ivb, iva):
        plt.plot([k, k], [i1**2*t, i2**2*t], 'y.-', lw=2)
    plt.plot(xx, w1, 'g', label='ESSVI Model', lw=2, alpha=.5)
    plt.plot(xx, w2, 'r', label='ESSVI using SVI Model', lw=2, alpha=.5)
    plt.plot(xx, w3, 'b', label='SVI Model', lw=2, alpha=.5)
    plt.title(f't: {t}')
    plt.grid()
    plt.legend()
    plt.show()

# plot_compare(essvi_param, svi_param, thetas)