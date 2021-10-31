import matplotlib.pyplot as plt
import numpy as np
import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.rtk import rtkinit, relpos

navfile = '../data/SEPT2650.21P'
obsfile = '../data/SEPT265G.21O'
basefile = '../data/3034265G.21O'

xyz_ref = gn.pos2ecef([35.342058098, 139.521986657, 47.5515], True)

# rover
dec = rn.rnxdec()
nav = gn.Nav()
dec.decode_nav(navfile, nav)

# base
decb = rn.rnxdec()
decb.decode_obsh(basefile)
dec.decode_obsh(obsfile)

nep = 360
# GSI 3034 fujisawa
nav.rb = [-3959400.631, 3385704.533, 3667523.111]
t = np.zeros(nep)
enu = np.zeros((nep, 3))
smode = np.zeros(nep, dtype=int)
rr0 = [-3961951.1326752,  3381198.11019757,  3668916.0417232]  # from pntpos
pos_ref = gn.ecef2pos(xyz_ref)

if True:
    rtkinit(nav, rr0)
    rr = rr0
    for ne in range(nep):
        obs, obsb = rn.sync_obs(dec, decb)
        if ne == 0:
            t0 = nav.t = obs.t
        relpos(nav, obs, obsb)
        t[ne] = gn.timediff(nav.t, t0)
        sol = nav.x[0:3]
        enu[ne, :] = gn.ecef2enu(pos_ref, sol-xyz_ref)
        smode[ne] = nav.smode

    dec.fobs.close()
    decb.fobs.close()

if True:
    idx4 = np.where(smode == 4)[0]
    idx5 = np.where(smode == 5)[0]
    idx1 = np.where(smode == 1)[0]

    lbl_t = ['east [m]', 'north [m]', 'up [m]']
    fig = plt.figure(figsize=(6, 10))

    for k in range(3):
        plt.subplot(3, 1, k+1)
        plt.plot(t, enu[:, k], '-', color='gray')
        # plt.plot(t[idx1], enu[idx1, k], 'm.', label='stdpos')
        plt.plot(t[idx5], enu[idx5, k], 'g.', markersize=1, label='float')
        plt.plot(t[idx4], enu[idx4, k], 'b.', markersize=1, label='fix')
        plt.xticks(np.arange(0, nep+1, step=30))
        plt.ylabel(lbl_t[k])
        plt.xlabel('time[s]')
        if k == 0:
            plt.legend()
        plt.grid()
    plt.show()

    plt.plot(enu[:, 0], enu[:, 1], '-', color='gray')
    # plt.plot(enu[idx1, 0], enu[idx1, 1], 'm.', markersize=1, label='stdpos')
    plt.plot(enu[idx5, 0], enu[idx5, 1], 'g.', markersize=1, label='float')
    plt.plot(enu[idx4, 0], enu[idx4, 1], 'b.', markersize=1, label='fix')

    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()
