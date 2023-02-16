"""
Test script for peph module
"""
import numpy as np
from os.path import expanduser
from cssrlib.peph import biasdec, peph, satantoff, readpcv
from cssrlib.rinex import rnxdec
from cssrlib.gnss import Nav, epoch2time, time2epoch, timeadd, rSIG


bdir = expanduser('~/GNSS_DAT/')
orbfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_15M_ORB.SP3"
clkfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_30S_CLK.CLK"
atxfile = bdir+"IGS/ANTEX/igs14.atx"
dcbfile = bdir+"COD0IGSRAP/2021/COD0IGSRAP_20210780000_01D_01D_OSB.BIA"
#dcbfile = bdir+"CAS0MGXRAP/2021/CAS0MGXRAP_20210780000_01D_01D_DCB.BSX"
#dcbfile = bdir+"CAS0MGXRAP/2021/CAS0MGXRAP_20210780000_01D_01D_OSB.BIA"

time = epoch2time([2021, 3, 19, 0, 0, 0])
sat = 3

rnx = rnxdec()
nav = Nav()

if False:

    sp = peph()
    nav = sp.parse_sp3(orbfile, nav)
    nav = rnx.decode_clk(clkfile, nav)
    nav.pcvs = readpcv(atxfile)

    n = 10
    rs = np.zeros((1, 6))
    dts = np.zeros((1, 2))
    for k in range(n):
        t = timeadd(time, 30*k)
        rs[0, :], dts[0, :], var = sp.peph2pos(t, sat, nav)

        ep = time2epoch(t)
        print("{:4d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}  {:02d}  "
              "{:15.4f} {:15.4f} {:15.4f} {:15.6f}"
              .format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], sat,
                      rs[0, 0], rs[0, 1], rs[0, 2], dts[0, 0]*1e6))

if False:

    rs, dts, var = sp.peph2pos(time, sat, nav)
    off = satantoff(time, rs[0:3], sat, nav)

if True:

    bd = biasdec()
    bd.parse(dcbfile)

    sig = "C1W"
    bias, std, = bd.getosb(sat, time, sig)
    assert bias == 7.6934
    assert std == 0.0
    print("{:02d} {:s} {:8.5f} {:6.4f}".format(sat, sig, bias, std))

    sig = "L1W"
    bias, std, = bd.getosb(sat, time, sig)
    assert bias == 0.00038
    assert std == 0.0
    print("{:02d} {:s} {:8.5f} {:6.4f}".format(sat, sig, bias, std))
