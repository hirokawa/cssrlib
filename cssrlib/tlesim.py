"""
module for TLE processing
"""

from ephem import readtle
import numpy as np
from cssrlib.gnss import id2sat


def loadname(file):
    """ load satellite list from file """
    satlist = {}
    with open(file, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            v = line.split()
            if len(v) >= 2:
                satlist[v[1]] = v[0]
    return satlist


def loadTLE(tle_file, satlst=None):
    """ load TLE from file """
    with open(tle_file, 'r') as f:
        satlist = []
        for l1 in f:
            l2 = f.readline()
            l3 = f.readline()
            norad_id = l2[2:8]
            if satlst is not None:
                if norad_id in satlst:
                    name = satlst[norad_id]
                else:
                    continue
            else:
                name = l1
            sat = readtle(name, l2, l3)
            satlist.append(sat)

    return satlist


def tleorb(sat, dates, obs=None):
    """ calculate orbit based on TLE """
    nsat = len(sat)
    nd = len(dates)
    lat = np.zeros((nd, nsat))
    lon = np.zeros((nd, nsat))
    el = np.zeros((nd, nsat))
    az = np.zeros((nd, nsat))
    sats = np.zeros(nsat, dtype=int)
    for k, sv in enumerate(sat):
        sat_ = id2sat(sv.name)
        if sat_ <= 0:
            continue
        sats[k] = sat_
        for i, t in enumerate(dates):
            sv.compute(t)
            lat[i, k] = sv.sublat
            lon[i, k] = sv.sublong
            if obs is not None:
                obs.date = t
                sv.compute(obs)
                el[i, k] = sv.alt
                az[i, k] = sv.az

    return lat, lon, az, el, sats
