"""
Test script for peph module
"""
import numpy as np

from cssrlib.peph import atxdec, biasdec, peph
from cssrlib.peph import searchpcv, antModelRx, antModelTx
from cssrlib.rinex import rnxdec
from cssrlib.gnss import Nav
from cssrlib.gnss import epoch2time, time2epoch, time2doy, timeadd
from cssrlib.gnss import sat2id, id2sat, sys2char
from cssrlib.gnss import rSigRnx
from cssrlib.gnss import pos2ecef, enu2xyz

ep = [2021, 9, 22, 12, 0, 0]
time = epoch2time(ep)
doy = int(time2doy(time))

bdir = '../../../../cssrlib-data/data/'
orbfile = bdir+"COD0MGXFIN_{:4d}{:03d}0000_01D_05M_ORB.SP3".format(ep[0], doy)
clkfile = bdir+"COD0MGXFIN_{:4d}{:03d}0000_01D_30S_CLK.CLK".format(ep[0], doy)
dcbfile = bdir+"COD0MGXFIN_{:4d}{:03d}0000_01D_01D_OSB.BIA".format(ep[0], doy)
atxfile = bdir+"igs14.atx"

sat = id2sat("G01")
sig = rSigRnx("GC1C")

if True:

    print("Test SP3 and Clock-RINEX module")
    print()

    rnx = rnxdec()
    nav = Nav()
    sp = peph()

    nav = sp.parse_sp3(orbfile, nav)
    sp.write_sp3('test_peph.sp3', nav)

    nav = rnx.decode_clk(clkfile, nav)

    n = 10
    rs = np.zeros((1, 6))
    dts = np.zeros((1, 2))
    for k in range(n):

        t = timeadd(time, 30*k)
        ep = time2epoch(t)

        rs[0, :], dts[0, :], var = sp.peph2pos(t, sat, nav)

        print("{:4d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}  {:s}  "
              "{:15.4f} {:15.4f} {:15.4f} {:15.6f}"
              .format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], sat2id(sat),
                      rs[0, 0], rs[0, 1], rs[0, 2], dts[0, 0]*1e6))

    print()

if False:

    print("Test ANTEX module")
    print()

    atx = atxdec()
    atx.readpcv(atxfile)

    for prn in ("G01", "R03", "C22", "J02", "E02"):

        sat = id2sat(prn)

        # Retrieve satellite antennas
        #
        pcv = searchpcv(atx.pcvs, sat, time)
        if pcv is None:
            print("ERROR: no PCV data for {}".format(sat2id(sat)))
            continue

        print("{}".format(sat2id(pcv.sat)))
        for sig, off in pcv.off.items():
            print("  {} {:3s} PCO [mm] X {:7.4f} Y {:7.4f} Z {:7.4f} \n"
                  "        PCV [mm] {}"
                  .format(sys2char(sig.sys), sig.str(),
                          off[0], off[1], off[2],
                          " ".join(["{:6.2f}".format(v) for v in pcv.var[sig]])))
        print()

    # Retrieve station antennas
    #
    antr = "{:16s}{:4s}".format("JAVRINGANT_DM", "SCIS")
    antb = "{:16s}{:4s}".format("TRM59800.80", "NONE")

    for ant in (antb, antr):

        pcv = searchpcv(atx.pcvr, ant, time)
        if pcv is None:
            print("ERROR: no PCV data for {}".format(ant))
            continue

        print("{:20s}".format(pcv.type))
        for sig, off in pcv.off.items():
            print("  {} {:3s} PCO [mm] E {:7.4f} N {:7.4f} U {:7.4f} \n"
                  "        PCV [mm] {}"
                  .format(sys2char(sig.sys), sig.str(),
                          off[0], off[1], off[2],
                          " ".join(["{:6.2f}".format(v) for v in pcv.var[sig]])))
        print()

    print("Test antenna modules")
    print()

    nav = Nav()

    # Store PCV/PCO information for satellites and receiver
    #
    nav.sat_ant = atx.pcvs
    nav.rcv_ant = searchpcv(atx.pcvr, antr, time)
    nav.rcv_ant_b = searchpcv(atx.pcvr, antb, time)

    # Satellite antenna model
    #

    # Receiver position
    #
    lat = np.deg2rad(45)
    lon = np.deg2rad(11)
    hgt = 0

    rr = pos2ecef(np.array([lat, lon, hgt]))

    # LOS vector on local ENU frame
    #
    az = np.deg2rad(45)
    el = np.deg2rad(45)
    e = np.array([np.sin(az)*np.cos(el),
                  np.cos(az)*np.cos(el),
                  np.sin(el)])

    # Convert from ENU to ECEF
    #
    A = enu2xyz(rr)
    e = A@e

    sat = id2sat("E02")
    sigs = [rSigRnx("EC1C"), rSigRnx("EC5Q")]
    dant = antModelRx(nav, rr, e, sigs)

    txt = "{:20s}  (El {:5.2f} deg)  ".format(antr, np.rad2deg(el))
    for i, sig in enumerate(sigs):
        txt += " {} {:6.3f} m".format(sig, dant[i])
    print(txt)

    # Satellite antenna model
    #

    # LOS vector in local antenna frame
    #
    az = np.deg2rad(0)
    el = np.deg2rad(0)
    e = np.array([np.sin(az)*np.cos(el),
                  np.cos(az)*np.cos(el),
                  np.sin(el)])

    # Satellite position
    #
    rs = rr + e*20180e3

    sat = id2sat("E02")
    sigs = [rSigRnx("EC1C"), rSigRnx("EC5Q")]
    dant = antModelTx(nav, e, sigs, sat, time, rs)

    txt = "{:20s}  (El {:5.2f} deg)  ".format(sat2id(sat), np.rad2deg(el))
    for i, sig in enumerate(sigs):
        txt += " {} {:6.3f} m".format(sig, dant[i])
    print(txt)

    print()

if False:

    print("Test Bias-SINEX module")
    print()

    bd = biasdec()
    bd.parse(dcbfile)

    sat = id2sat("G03")
    sig = rSigRnx("GC1W")

    bias, std, = bd.getosb(sat, time, sig)
    assert bias == 7.6934
    assert std == 0.0

    print("{:s} {:s} {:8.5f} {:6.4f}"
          .format(sat2id(sat), sig.str(), bias, std))

    sig = rSigRnx("GL1W")
    bias, std, = bd.getosb(sat, time, sig)
    assert bias == 0.00038
    assert std == 0.0

    print("{:s} {:s} {:8.5f} {:6.4f}"
          .format(sat2id(sat), sig.str(), bias, std))
    print()
