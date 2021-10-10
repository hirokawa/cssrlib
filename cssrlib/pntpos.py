""" module to calculate standard positioning """
import numpy as np
from cssrlib.gnss import rCST, ecef2pos, geodist, satazel, ionmodel,\
    tropmodel, Nav, tropmapf, kfupdate
from cssrlib.ephemeris import findeph, satposs


def varerr(nav, el):
    """ variation of measurement """
    s_el = np.sin(el)
    if s_el <= 0.0:
        return 0.0
    a = nav.err[1]
    b = nav.err[2]
    return a**2+(b/s_el)**2


def stdinit():
    """ initialize standard positioning """
    nav = Nav()
    nav.na = 6
    nav.nx = 8
    nav.x = np.zeros(nav.nx)
    sig_p0 = 100.0*np.ones(3)
    sig_v0 = 10.0*np.ones(3)
    nav.P = np.diag(np.hstack((sig_p0**2, sig_v0**2, 100, 100)))
    dt = 1
    nav.Phi = np.eye(nav.nx)
    nav.Phi[0:3, 3:6] = dt*np.eye(3)
    nav.Phi[6, 7] = dt
    nav.elmin = np.deg2rad(10)
    sq = 1e-2
    nav.Q = np.diag([0, 0, 0, sq, sq, sq, 0, 1e-6])
    nav.err = [0, 0.3, 0.3]
    return nav


def rescode(obs, nav, rs, dts, svh, x):
    """ calculate code residuals """
    n = obs.sat.shape[0]
    rr = x[0:3]
    dtr = x[nav.na]
    pos = ecef2pos(rr)
    v = np.zeros(n)
    H = np.zeros((n, nav.nx))
    azv = np.zeros(n)
    elv = np.zeros(n)
    nv = 0
    for i in range(n):
        if np.linalg.norm(rs[i, :]) < rCST.RE_WGS84 or svh[i] > 0:
            continue
        r, e = geodist(rs[i, :], rr)
        az, el = satazel(pos, e)
        if el < nav.elmin:
            continue
        eph = findeph(nav.eph, obs.t, obs.sat[i])
        if obs.P[i, 0] == 0:
            continue
        P = obs.P[i, 0]-eph.tgd*rCST.CLIGHT
        dion = ionmodel(obs.t, pos, az, el, nav.ion)
        trop_hs, trop_wet, _ = tropmodel(obs.t, pos, el)
        mapfh, mapfw = tropmapf(obs.t, pos, el)
        dtrp = mapfh*trop_hs+mapfw*trop_wet
        v[nv] = P-(r+dtr-rCST.CLIGHT*dts[i]+dion+dtrp)
        H[nv, 0:3] = -e
        H[nv, nav.na] = 1
        azv[nv] = az
        elv[nv] = el
        nv += 1
    v = v[0:nv]
    H = H[0:nv, :]
    azv = azv[0:nv]
    elv = elv[0:nv]
    return v, H, nv, azv, elv


def pntpos(obs, nav):
    """ calculate point positioning """
    rs, _, dts, svh = satposs(obs, nav)
    x = nav.x.copy()
    P = nav.P.copy()
    x = nav.Phi@x
    P = nav.Phi@P@nav.Phi.T+nav.Q
    v, H, _, az, el = rescode(obs, nav, rs, dts, svh, x)
    if abs(np.mean(v)) > 100:
        x[nav.na] = np.mean(v)
        v -= x[nav.na]
    n = len(v)
    R = np.zeros((n, n))
    for k in range(n):
        R[k, k] = varerr(nav, el[k])
    nav.x, nav.P = kfupdate(x, P, H, v, R)
    return nav, az, el
