import numpy as np
import json
import numpy
import pandas
from pathlib import Path
from scipy.interpolate import interp1d
from py_vollib.black.implied_volatility import implied_volatility as biv

import pytz
from dateutil.parser import parse
import datetime
from scipy.spatial import ConvexHull


def get_test_data(data_root, currency='btc'):
    data_frames = []
    fnames = data_root.glob(f"{currency}*.csv")
    for f in fnames:
        data_frames.append(pandas.read_csv(f))
    df = pandas.concat(data_frames)
    df.index = range(len(df))
    del df["Unnamed: 0"]
    df["maturity_ts"] = df.maturity.apply(maturity_timestamp_from_string)
    df["snapshot_ts"]= datetime.datetime.fromtimestamp(df.ts.values[0], tz=pytz.UTC)
    df["t"] = df.apply(lambda row: calc_t(row.snapshot_ts, row.maturity_ts), axis=1)
    Ffunc = futures_curve(data_root, currency)
    df["F_market"] = df.t.apply(Ffunc)
    df["r"] = numpy.log(df.F_market/df.S)/df.t
    df["k"] = numpy.log(df.K/df.F_market)
    df["mid"] = df.apply(lambda row: (row.cbid+row.cask)/2 if row.k>0 else \
        (row.pbid + row.pask)/2, axis=1)
    df["flag"] = df.k.apply(lambda val: "p" if val<0 else "c")
    df.mid*=df.S
    df = df[~numpy.isnan(df.mid)]
    df["iv"] = df.apply(lambda row: biv1(row), axis=1)
    return df


def biv1(row):
    try:
        return biv(row.mid, row.F_market, row.K, row.r, row.t, row.flag)
    except:
        return None


def generate_slices_from_df(df, outlier):
    unique_t =sorted(df.t.unique())
    for t in unique_t:
        test = df[df.t == t]
        if outlier:
            test = remove_outlier_from_slice(test)
        if len(test)>6:
            yield test


def generate_slices(data_root):
    df = get_test_data(data_root)
    unique_t =sorted(df.t.unique())
    for t in unique_t:
        test = df[df.t == t]
        if len(test)>10:
            yield test


def maturity_timestamp_from_string(s):
    d = parse(s).date()
    return datetime.datetime.combine(d,datetime.time(8), tzinfo=pytz.UTC)



def get_snapshot_datetime_and_spot(data_root):
    fname = list(data_root.glob("*.csv"))[0]
    df = pandas.read_csv(fname)
    dt = datetime.datetime.fromtimestamp(df.ts.values[0], tz=pytz.UTC)
    S = df.S.values[0]
    return dt, S


def date_from_futures_code(code):
    return datetime.datetime.combine(
        parse(code[3:]).date(),datetime.time(8), tzinfo=pytz.UTC)

def calc_t(snapshot_timestamp, maturity_timestamp):
    td = maturity_timestamp - snapshot_timestamp
    return td.days / 365 + td.seconds / (3600 * 24 * 365)


def futures_curve(data_root, currency):
    jsonfile = list(data_root.glob(f"{currency}*.json"))[0]
    with open(jsonfile, "r") as f:
        data = json.load(f)
    data = {date_from_futures_code(k):v for k, v in data.items()}
    sdt, S = get_snapshot_datetime_and_spot(data_root)
    data = {calc_t(sdt, k):v for k,v in data.items()}
    data = {k:v for k,v in data.items() if k>0}
    data[0]=S
    idx = numpy.argsort(list(data.keys()))
    t = numpy.array(list(data.keys()))[idx]
    F = numpy.array(list(data.values()))[idx]
    interpf = interp1d(t,F,kind="quadratic")
    return interpf


def remove_outlier_from_slice(slice):
    x, y1 = slice.k, slice.iv
    viewer_position = [(0, -1)]
    outlier_idx1, _ = detect_outliers(x, y1, viewer_position)
    idx = [x for x in range(len(slice)) if x not in outlier_idx1]
    return slice.iloc[idx]


def visible_points(hull):
    xypairs = set()
    for visible_facet in hull.simplices[hull.good]:
        facet = hull.points[visible_facet]
        for x,y in facet:
            xypairs.add((x,y))
    xypairs = sorted(list(xypairs), key=lambda t:t[0])
    return np.array(list(xypairs))


def symmetrical_difference_2d_arrays(a,b, ignore):

    A = arr2d_to_set_of_tuples(a)
    B = arr2d_to_set_of_tuples(b)
    SD = A.symmetric_difference(B)
    tuple_to_ignore = ignore[0]
    if tuple_to_ignore in SD:
        SD.remove(tuple_to_ignore)
    if len(SD)==0:
        return np.array([])
    return np.array(list(SD))


def arr2d_to_set_of_tuples(arr):
    return set([tuple(pt) for pt in arr])


def detect_outliers(x, y, viewer_position):
    pts_in = np.vstack((x, y)).T
    pts_in= np.concatenate((pts_in, np.array(viewer_position)))
    n = len(pts_in)
    hull = ConvexHull(points=pts_in,
                          qhull_options=f'QG{n-1}')
    visible = visible_points(hull)
    SD = symmetrical_difference_2d_arrays(pts_in, visible, ignore=viewer_position)
    outlier_idx = np.array(list(sorted([np.where(p==pts_in)[0][0] for p in SD])), dtype=int)
    return outlier_idx, visible


if __name__ == '__main__':
    get_test_data(data_root=Path("data"))
    exit()
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")

    interpf  = futures_curve(data_root=Path("data"))
    t = numpy.linspace(0,interpf.x.max())
    y = interpf(t)
    plt.plot(t, y)
    plt.grid(True)
    plt.show()


    # print(get_snapshot_datetime())
