"""
Test script for peph module
"""
import numpy as np
from os.path import expanduser
from cssrlib.peph import biasdec, peph, satantoff, readpcv, searchpcv
from cssrlib.rinex import rnxdec
from cssrlib.gnss import Nav
from cssrlib.gnss import epoch2time, time2epoch, timeadd, time2str
from cssrlib.gnss import sat2id, id2sat
from cssrlib.gnss import uGNSS, uTYP, uSIG, rSigRnx


bdir = expanduser('~/GNSS_DAT/')
atxfile = bdir+"IGS/ANTEX/igs14.atx"
orbfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_15M_ORB.SP3"
clkfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_30S_CLK.CLK"
dcbfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_01D_OSB.BIA"

time = epoch2time([2021, 3, 19, 12, 0, 0])
sat = id2sat("G01")

if False:

    rnx = rnxdec()
    nav = Nav()
    sp = peph()

    nav = sp.parse_sp3(orbfile, nav)
    nav = rnx.decode_clk(clkfile, nav)
    nav.pcvs, nav.pcvr = readpcv(atxfile)

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

if False:

    nav = Nav()
    sp = peph()

    nav = sp.parse_sp3(orbfile, nav)
    nav.pcvs, nav.pcvr = readpcv(atxfile)

    """
    for pcv in nav.pcvs:
        print(sat2id(pcv.sat), pcv.type)
    """

    pcv = searchpcv(sat, time, nav.pcvs)
    print(pcv.sat, pcv.type)

    name = "TRM57971.00     NONE"
    pcv = searchpcv(name, time, nav.pcvr)
    print(pcv.sat, pcv.type)

    rs, dts, var = sp.peph2pos(time, sat, nav)
    off = satantoff(time, rs[0:3], sat, nav)
    print(off)

    """
    ep = time2epoch(time)
    print("{:4d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}  {:s}  "
          "{:15.4f} {:15.4f} {:15.4f}"
          .format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], sat2id(sat),
                  off[0], off[1], off[2]))
    """

if True:

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
