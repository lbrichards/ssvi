import json
import numpy
import pandas
from pathlib import Path
from scipy.interpolate import interp1d
from py_vollib.black.implied_volatility import implied_volatility as biv

import pytz
from dateutil.parser import parse
import datetime

def get_test_data(data_root):
    data_frames = []
    fnames = data_root.glob("*.csv")
    for f in fnames:
        data_frames.append(pandas.read_csv(f))
    df = pandas.concat(data_frames)
    df.index = range(len(df))
    del df["Unnamed: 0"]
    df["maturity_ts"] = df.maturity.apply(maturity_timestamp_from_string)
    df["snapshot_ts"]= datetime.datetime.fromtimestamp(df.ts.values[0], tz=pytz.UTC)
    df["t"] = df.apply(lambda row: calc_t(row.snapshot_ts, row.maturity_ts), axis=1)
    Ffunc = futures_curve(data_root)
    df["F_market"] = df.t.apply(Ffunc)
    df["r"] = numpy.log(df.F_market/df.S)/df.t
    df["k"] = numpy.log(df.K/df.F_market)
    df["mid"] = df.apply(lambda row: (row.cbid+row.cask)/2 if row.k>0 else \
        (row.pbid + row.pask)/2, axis=1)
    df["flag"] = df.k.apply(lambda val: "p" if val<0 else "c")
    df.mid*=df.S
    df = df[~numpy.isnan(df.mid)]
    df["iv"] = df.apply(lambda row: biv(row.mid, row.F_market, row.K, row.r, row.t,row.flag), axis=1)
    return df

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


def futures_curve(data_root):
    jsonfile = list(data_root.glob("*.json"))[0]
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




