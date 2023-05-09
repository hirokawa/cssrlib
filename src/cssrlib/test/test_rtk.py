"""
 static test for RTK
"""

import matplotlib.pyplot as plt
import numpy as np
import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.rtk import rtkinit, relpos
from cssrlib.peph import searchpcv, atxdec
from os.path import expanduser

bdir = '../data/'
navfile = bdir+'SEPT078M.21P'
obsfile = bdir+'SEPT078M1.21O'
basefile = bdir+'3034078M1.21O'

xyz_ref = [-3962108.673,   3381309.574,   3668678.638]
pos_ref = gn.ecef2pos(xyz_ref)

sigs_b = [rn.rSigRnx("GC1C"), rn.rSigRnx("EC1X"),
          rn.rSigRnx("GC2W"), rn.rSigRnx("EC5X"),
          rn.rSigRnx("GL1C"), rn.rSigRnx("EL1X"),
          rn.rSigRnx("GL2W"), rn.rSigRnx("EL5X")]

sigs = [rn.rSigRnx("GC1C"), rn.rSigRnx("EC1C"),
        rn.rSigRnx("GC2W"), rn.rSigRnx("EC5Q"),
        rn.rSigRnx("GL1C"), rn.rSigRnx("EL1C"),
        rn.rSigRnx("GL2W"), rn.rSigRnx("EL5Q")]

# rover
dec = rn.rnxdec()
dec.setSignals(sigs)

nav = gn.Nav()
dec.decode_nav(navfile, nav)

# base
decb = rn.rnxdec()
decb.setSignals(sigs_b)

decb.decode_obsh(basefile)
dec.decode_obsh(obsfile)

nep = 60
#nep = 45

time = gn.epoch2time([2021, 3, 19, 12, 0, 0])

if True:

    bdir = expanduser('~/GNSS_DAT/')
    atxfile = bdir+"IGS/ANTEX/igs14.atx"
    
    atx = atxdec()
    atx.readpcv(atxfile)
    
    antr = "{:16s}{:4s}".format("JAVRINGANT_DM", "SCIS")
    antb = "{:16s}{:4s}".format("TRM59800.80", "NONE")

    # Store PCV/PCO information for satellites and receiver
    #
    nav.sat_ant = atx.pcvs
    nav.rcv_ant = searchpcv(atx.pcvr, antr, time)
    nav.rcv_ant_b = searchpcv(atx.pcvr, antb, time)
    nav.sig_tab = dec.sig_tab
    nav.sig_tab_b = decb.sig_tab


# GSI 3034 fujisawa
nav.rb = [-3959400.631, 3385704.533, 3667523.111]
t = np.zeros(nep)
enu = np.zeros((nep, 3))
smode = np.zeros(nep, dtype=int)

rtkinit(nav, dec.pos)
rr = dec.pos
for ne in range(nep):
    obs, obsb = rn.sync_obs(dec, decb)
    if ne == 0:
        t0 = nav.t = obs.t
    relpos(nav, obs, obsb)
    t[ne] = gn.timediff(nav.t, t0)
    sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]
    enu[ne, :] = gn.ecef2enu(pos_ref, sol-xyz_ref)
    smode[ne] = nav.smode

dec.fobs.close()
decb.fobs.close()

fig_type = 1
ylim = 0.2

if fig_type == 1:
    plt.plot(t, enu)
    plt.xticks(np.arange(0, nep+1, step=30))
    plt.ylabel('position error [m]')
    plt.xlabel('time[s]')
    plt.legend(['east', 'north', 'up'])
    plt.grid()
    plt.axis([0, ne, -ylim, ylim])
    plt.show()
else:
    plt.plot(enu[:, 0], enu[:, 1])
    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    plt.grid()
    plt.axis([-ylim, ylim, -ylim, ylim])
    plt.show()
