"""
 static test for RTK
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.rtk import rtkinit, relpos
from cssrlib.gnss import rSigRnx
from cssrlib.peph import atxdec, searchpcv

bdir = '../data/'
navfile = bdir+'SEPT078M.21P'
obsfile = bdir+'SEPT078M1.21O'
basefile = bdir+'3034078M1.21O'
atxfile = bdir+"test.atx"

xyz_ref = [-3962108.673,   3381309.574,   3668678.638]
pos_ref = gn.ecef2pos(xyz_ref)

# Define signals to be processed
#
sigs = [rSigRnx("GC1C"), rSigRnx("GC2W"),
        rSigRnx("EC1C"), rSigRnx("EC5Q"),
        rSigRnx("GL1C"), rSigRnx("GL2W"),
        rSigRnx("EL1C"), rSigRnx("EL5Q")]

sigsb = [rSigRnx("GC1C"), rSigRnx("GC2W"),
         rSigRnx("EC1X"), rSigRnx("EC5X"),
         rSigRnx("GL1C"), rSigRnx("GL2W"),
         rSigRnx("EL1X"), rSigRnx("EL5X")]

# rover
#
dec = rn.rnxdec()
dec.setSignals(sigs)

nav = gn.Nav()
dec.decode_nav(navfile, nav)

# base
#
decb = rn.rnxdec()
decb.setSignals(sigsb)

decb.decode_obsh(basefile)
dec.decode_obsh(obsfile)


# Set rover and base antenna
#
dec.ant = "{:16s}{:4s}".format("JAVRINGANT_DM", "SCIS")
decb.ant = "{:16s}{:4s}".format("TRM59800.80", "NONE")

atx = atxdec()
atx.readpcv(atxfile)

# Set antenna PCO/PCV data
#
nav.rcv_ant = searchpcv(atx.pcvr, dec.ant,  dec.ts)
nav.rcv_ant_b = searchpcv(atx.pcvr, decb.ant, decb.ts)

if nav.rcv_ant is None:
    print("ERROR: no rover antenna found!")
    sys.exit(1)

if nav.rcv_ant_b is None:
    print("ERROR: no base antenna found!")
    sys.exit(1)

nep = 60

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
