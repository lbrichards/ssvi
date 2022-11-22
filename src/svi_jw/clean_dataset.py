import json

import numpy
import pandas
from matplotlib import pyplot as plt
from src.data_utils import get_test_data, generate_slices_from_df
from src.essvi.calibration import get_theta_at_k0_for_slice, ESSVI, calibrate_essvi
from path import Path

from src.svi_jw.calibration import svi
from src.svi_jw.compare_svi_and_essvi import plot_compare

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
essvi_param = calibrate_essvi(df, outlier=False)
for slice in generate_slices_from_df(df, outlier=False):
    theta = get_theta_at_k0_for_slice(slice)
    thetas[slice.t.unique()[0]] = theta


plot_compare(essvi_param, svi_param, thetas)