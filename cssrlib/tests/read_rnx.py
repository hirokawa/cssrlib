import numpy as np
from cssrlib.gnss import Nav, ecef2pos, geodist, satazel, timediff
from cssrlib.ephemeris import findeph, eph2pos
from cssrlib.plot import skyplot, plot_elv
from cssrlib.rinex import rnxdec

navfile = '../data/SEPT078M.21P'
obsfile = '../data/SEPT078M.21O'

dec = rnxdec()
nav = Nav()
nav = dec.decode_nav(navfile, nav)

nep = 900
elv = np.ones((nep, dec.MAXSAT))*np.nan
azm = np.ones((nep, dec.MAXSAT))*np.nan
mag = np.zeros((nep, dec.MAXSAT), dtype=int)
t = np.zeros(nep)*np.nan
if dec.decode_obsh(obsfile) >= 0:
    rr = dec.pos
    pos = ecef2pos(rr)
    for ne in range(nep):
        obs = dec.decode_obs()
        if ne == 0:
            t0 = obs.t
        t[ne] = timediff(obs.t, t0)
        for k, sat in enumerate(obs.sat):
            eph = findeph(nav.eph, obs.t, sat)
            rs, dts = eph2pos(obs.t, eph)
            r, e = geodist(rs, rr)
            azm[ne, sat-1], elv[ne, sat-1] = satazel(pos, e)
            # mag[ne,sat-1]=obs.mag[k,0]

    dec.fobs.close()

plot_elv(t, elv)
skyplot(azm, elv)
