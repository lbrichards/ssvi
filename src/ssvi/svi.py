import numpy
import scipy.optimize as opt

from src.essvi.calibration import iv_bid_ask_for_slice
from src.svi_jw.calibration import calibrate_svi
from src.svi_jw.svi_func import svi, g
from src.svi_jw.translate_params import jw_params_from_raw_params


def error_count(slice, svi_param):
    t = slice.t.unique()[0]
    x = slice.k.values
    iva, ivb = iv_bid_ask_for_slice(slice, t)
    w = svi(x, *svi_param)
    iv = numpy.sqrt(w/t)
    plt.plot(x, w, 'b')
    for k, i1, i2 in zip(x, ivb, iva):
        plt.plot([k, k], [i1**2*t, i2**2*t], 'y.-', lw=2)

    width = x.max()-x.min()
    x_mid = x[argmin(ivb)]
    dist = numpy.abs(x_mid-x)
    aa = numpy.where(dist < width / 3)
    iva = numpy.array(iva)[aa]
    ivb = numpy.array(ivb)[aa]
    iv = numpy.array(iv)[aa]

    return [1 for i,a,b in zip(iv, iva, ivb) if i>a or i<b]


if __name__ == '__main__':
    from pathlib import Path
    from pylab import *
    from src.data_utils import get_test_data, generate_slices_from_df

    thetas = {}
    df = get_test_data(data_root=Path("../data"))
    raw_params = {}
    tt = []
    for slice in list(generate_slices_from_df(df, outlier=True)):
        t = float(slice.t.unique()[0])
        if t<5./365:
            continue
        raw_params[slice.t.unique()[0]] = list(calibrate_svi(slice))
        plt.title(f't: {t}\n'
                  'inlier accuracy: '
                  f'{100-100*numpy.count_nonzero(error_count(slice, raw_params[t]))/len(slice):0.2f}%'
                  )
        plt.show()

    # json.dump(json.dumps(raw_params), open('svi.json', 'w'))



