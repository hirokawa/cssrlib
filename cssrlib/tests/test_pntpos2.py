import matplotlib.pyplot as plt
import numpy as np
from cssrlib.rinex import rnxdec
from cssrlib.gnss import ecef2pos, timediff, dops, ecef2enu, pos2ecef, xyz2enu
from cssrlib.pntpos import stdinit, pntpos

rr0 = xyz_ref = pos2ecef([35.342058098, 139.521986657, 47.5515], True)
pos_ref = ecef2pos(xyz_ref)
E = xyz2enu(pos_ref)

navfile = '../data/SEPT2650.21P'
obsfile = '../data/SEPT265G.21O'

dec = rnxdec()
nav = stdinit()
nav = dec.decode_nav(navfile, nav)
nep = 360
t = np.zeros(nep)
enu = np.zeros((nep, 3))
sol = np.zeros((nep, nav.nx))
dop = np.zeros((nep, 4))
nsat = np.zeros(nep, dtype=int)
if dec.decode_obsh(obsfile) >= 0:
    nav.x[0:3] = rr0
    for ne in range(nep):
        obs = dec.decode_obs()
        if ne == 0:
            t0 = obs.t
        if ne >= 57:
            ne
        t[ne] = timediff(obs.t, t0)
        nav, az, el = pntpos(obs, nav)
        sol[ne, :] = nav.x
        dop[ne, :] = dops(az, el)
        enu[ne, :] = ecef2enu(pos_ref, sol[ne, 0:3]-xyz_ref)
        nsat[ne] = len(el)
    dec.fobs.close()


if True:
    lbl_t = ['east [m]', 'north [m]', 'up [m]']
    fig = plt.figure(figsize=(6, 10))

    for k in range(3):
        plt.subplot(3, 1, k+1)
        plt.plot(t, enu[:, k])
        plt.ylabel(lbl_t[k])
        plt.xlabel('time[s]')
        plt.grid()
    plt.show()

    venu = sol[:, 3:6]@E.T

    plt.figure()
    plt.plot(t, venu)
    plt.ylabel('velocity [m/s]')
    plt.xlabel('time[s]')
    plt.legend(['east', 'north', 'up'])
    plt.grid()
    plt.axis([0, nep, -10, 10])
    plt.show()

    sol[0, 7] = np.nan
    plt.figure()
    plt.subplot(211)
    plt.plot(t, sol[:, 6]-sol[0, 6])
    plt.ylabel('clock bias [m]')
    plt.grid()
    plt.subplot(212)
    plt.plot(t, sol[:, 7])
    plt.ylabel('clock drift [m/s]')
    plt.xlabel('time[s]')
    plt.grid()
    plt.show()

if True:
    plt.figure()
    plt.plot(enu[:, 0], enu[:, 1], '-', color='gray')
    plt.plot(enu[:, 0], enu[:, 1], 'm.', markersize=1)
    plt.xlabel('easting[m]')
    plt.ylabel('northing[m]')
    plt.grid()
    plt.axis('equal')
    plt.show()

    plt.figure()
    plt.plot(t, dop[:, 1:])
    plt.legend(['pdop', 'hdop', 'vdop'])
    plt.grid()
    plt.axis([0, nep, 0, 3])
    plt.xlabel('time[s]')
    plt.show()
