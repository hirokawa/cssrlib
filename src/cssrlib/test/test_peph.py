"""
Test script for peph module
"""
import numpy as np
from os.path import expanduser
from cssrlib.peph import atxdec, biasdec, peph
from cssrlib.rinex import rnxdec
from cssrlib.gnss import Nav
from cssrlib.gnss import epoch2time, time2epoch, timeadd
from cssrlib.gnss import sat2id, id2sat, sys2char
from cssrlib.gnss import uGNSS, uTYP, uSIG, rSigRnx


bdir = expanduser('~/GNSS_DAT/')
atxfile = bdir+"IGS/ANTEX/igs14.atx"
orbfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_15M_ORB.SP3"
clkfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_30S_CLK.CLK"
dcbfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_01D_OSB.BIA"

time = epoch2time([2021, 3, 19, 12, 0, 0])
sat = id2sat("G01")
sig = rSigRnx(uGNSS.GPS, uTYP.C, uSIG.L1C)

if False:

    rnx = rnxdec()
    nav = Nav()
    sp = peph()

    nav = sp.parse_sp3(orbfile, nav)
    nav = rnx.decode_clk(clkfile, nav)

    n = 10
    rs = np.zeros((1, 6))
    dts = np.zeros((1, 2))
    for k in range(n):
        t = timeadd(time, 30*k)
        rs[0, :], dts[0, :], var = sp.peph2pos(t, sat, nav)

        ep = time2epoch(t)
        print("{:4d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}  {:s}  "
              "{:15.4f} {:15.4f} {:15.4f} {:15.6f}"
              .format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], sat2id(sat),
                      rs[0, 0], rs[0, 1], rs[0, 2], dts[0, 0]*1e6))

if True:

    nav = Nav()
    sp = peph()

    nav = sp.parse_sp3(orbfile, nav)

    atx = atxdec()
    atx.readpcv(atxfile)

    antr = "{:16s}{:4s}".format("JAVRINGANT_DM", "SCIS")
    antb = "{:16s}{:4s}".format("TRM59800.80", "NONE")

    for ant in (antr, antb):

        pcv = atx.searchpcvr(ant, time)
        if pcv is None:
            print("ERROR: no PCV data for {}".format(ant))

        print("{:20s}".format(pcv.type))
        for sig, off in pcv.off.items():
            print("  {} {:3s} PCO  [m] E {:7.4f} N {:7.4f} U {:7.4f} \n"
                  "        PCV [mm] {}"
                  .format(sys2char(sig.sys), sig.str(),
                          off[0], off[1], off[2],
                          " ".join(["{:6.2f}".format(v) for v in pcv.var[sig]])))
        print()

    """
    rs, dts, var = sp.peph2pos(time, sat, nav)
    off = atx.satantoff(time, rs[0:3], sat, sig)

    ep = time2epoch(time)
    print("{:4d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}  {:s}  {:s} {:7.4f}"
          .format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], sat2id(sat),
                  sig.str(), off))
    """

if False:

    bd = biasdec()
    bd.parse(dcbfile)

    sat = id2sat("G03")
    sig = rSigRnx(uGNSS.GPS, uTYP.C, uSIG.L1W)

    bias, std, = bd.getosb(sat, time, sig)
    assert bias == 7.6934
    assert std == 0.0

    print("{:s} {:s} {:8.5f} {:6.4f}"
          .format(sat2id(sat), sig.str(), bias, std))

    sig = rSigRnx(uGNSS.GPS, uTYP.L, uSIG.L1W)
    bias, std, = bd.getosb(sat, time, sig)
    assert bias == 0.00038
    assert std == 0.0

    print("{:s} {:s} {:8.5f} {:6.4f}"
          .format(sat2id(sat), sig.str(), bias, std))
