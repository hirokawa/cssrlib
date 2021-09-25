# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
import cssrlib.gnss as gn
from cssrlib.gnss import rCST, sat2prn, ecef2pos, geodist, satazel, ionmodel, \
    tropmodel, timediff, antmodel, uGNSS
from cssrlib.ephemeris import findeph, satposs
from cssrlib.cssrlib import sSigGPS, sSigGAL, sSigQZS, sCType
from cssrlib.ppp import tidedisp, shapiro, windupcorr
from cssrlib.rtk import IB, ddres, kfupdate, resamb_lambda, valpos, holdamb

MAXITR = 10
ELMIN = 10
NX = 4


def logmon(nav, t, sat, cs, iu=None):
    week, tow = gn.time2gpst(t)
    if iu is None:
        cpc = cs.cpc
        prc = cs.prc
        osr = cs.osr
    else:
        cpc = cs.cpc[iu, :]
        prc = cs.prc[iu, :]
        osr = cs.osr[iu, :]
    if nav.loglevel >= 2:
        n = cpc.shape
        for i in range(n[0]):
            if cpc[i, 0] == 0 and cpc[i, 1] == 0:
                continue
            nav.fout.write("%6d\t%3d\t%8.3f\t%8.3f\t%8.3f\t%8.3f\t%2d\t"
                           % (tow, sat[i], cpc[i, 0], cpc[i, 1],
                              prc[i, 0], prc[i, 1], cs.iodssr))
            nav.fout.write("%8.3f\t%8.3f\t%8.3f\t%8.3f\t%8.3f\t"
                           % (osr[i, 0], osr[i, 1], osr[i, 2],
                              osr[i, 3], osr[i, 4]))
            nav.fout.write("%8.3f\t%8.3f\t%8.3f\t%8.3f\n"
                           % (osr[i, 5], osr[i, 6], osr[i, 7], osr[i, 8]))
    return 0


def rtkinit(nav, pos0=np.zeros(3)):
    nav.nf = 2
    nav.pmode = 1  # 0:static, 1:kinematic

    nav.na = 3 if nav.pmode == 0 else 6
    nav.ratio = 0
    nav.thresar = [2]
    nav.nx = nav.na+gn.uGNSS.MAXSAT*nav.nf
    nav.x = np.zeros(nav.nx)
    nav.P = np.zeros((nav.nx, nav.nx))
    nav.xa = np.zeros(nav.na)
    nav.Pa = np.zeros((nav.na, nav.na))
    nav.nfix = nav.neb = 0
    nav.phw = np.zeros(gn.uGNSS.MAXSAT)

    # parameter for PPP-RTK
    nav.eratio = [50, 50]
    nav.err = [100, 0.00707, 0.00354]
    nav.sig_p0 = 30.0
    nav.sig_v0 = 10.0
    nav.sig_n0 = 30.0
    nav.sig_qp = 0.1
    nav.sig_qv = 0.01
    nav.tidecorr = True
    nav.armode = 1  # 1:contunous,2:instantaneous,3:fix-and-hold
    nav.gnss_t = [uGNSS.GPS, uGNSS.GAL, uGNSS.QZS]

    #
    nav.x[0:3] = pos0
    di = np.diag_indices(6)
    nav.P[di[0:3]] = nav.sig_p0**2
    nav.q = np.zeros(nav.nx)
    nav.q[0:3] = nav.sig_qp**2
    if nav.pmode >= 1:
        nav.P[di[3:6]] = nav.sig_v0**2
        nav.q[3:6] = nav.sig_qv**2
    # obs index
    i0 = {gn.uGNSS.GPS: 0, gn.uGNSS.GAL: 0, gn.uGNSS.QZS: 0}
    i1 = {gn.uGNSS.GPS: 1, gn.uGNSS.GAL: 2, gn.uGNSS.QZS: 1}
    freq0 = {gn.uGNSS.GPS: nav.freq[0], gn.uGNSS.GAL: nav.freq[0],
             gn.uGNSS.QZS: nav.freq[0]}
    freq1 = {gn.uGNSS.GPS: nav.freq[1], gn.uGNSS.GAL: nav.freq[2],
             gn.uGNSS.QZS: nav.freq[1]}
    nav.obs_idx = [i0, i1]
    nav.obs_freq = [freq0, freq1]
    nav.cs_sig_idx = {gn.uGNSS.GPS: [sSigGPS.L1C, sSigGPS.L2W],
                      gn.uGNSS.GAL: [sSigGAL.L1X, sSigGAL.L5X],
                      gn.uGNSS.QZS: [sSigQZS.L1C, sSigQZS.L2X]}

    nav.fout = None
    nav.logfile = 'log.txt'
    if nav.loglevel >= 2:
        nav.fout = open(nav.logfile, 'w')


def rescode(itr, obs, nav, rs, dts, svh, x):
    nv = 0
    n = obs.sat.shape[0]
    rr = x[0:3]
    dtr = x[3]
    pos = ecef2pos(rr)
    v = np.zeros(n)
    H = np.zeros((n, NX))
    az = np.zeros(n)
    el = np.zeros(n)
    for i in range(n):
        sys, prn = sat2prn(obs.sat[i])
        if np.linalg.norm(rs[i, :]) < rCST.RE_WGS84:
            continue
        r, e = geodist(rs[i, :], rr)
        az[i], el[i] = satazel(pos, e)
        if el[i] < np.deg2rad(ELMIN):
            continue
        eph = findeph(nav.eph, obs.t, obs.sat[i])
        P = obs.P[i, 0]-eph.tgd*rCST.CLIGHT
        dion = ionmodel(obs.t, pos, az[i], el[i], nav.ion)
        dtrp = tropmodel(obs.t, pos, el[i])
        v[nv] = P-(r+dtr-rCST.CLIGHT*dts[i]+dion+dtrp)
        H[nv, 0:3] = -e
        H[nv, 3] = 1
        nv += 1

    return v, H, nv, az, el


def udstate_ppp(nav, obs):
    tt = 1.0

    ns = len(obs.sat)
    sys = []
    sat = obs.sat
    for sat_i in obs.sat:
        sys_i, prn = gn.sat2prn(sat_i)
        sys.append(sys_i)

    # pos,vel
    na = nav.na
    if nav.pmode >= 1:
        F = np.eye(na)
        F[0:3, 3:6] = np.eye(3)*tt
        nav.x[0:3] += tt*nav.x[3:6]
        Px = nav.P[0:na, 0:na]
        Px = F.T@Px@F
        Px[np.diag_indices(nav.na)] += nav.q[0:nav.na]*tt
        nav.P[0:na, 0:na] = Px
    # bias
    for f in range(nav.nf):
        # cycle slip check by LLI
        for i in range(ns):
            if sys[i] not in nav.gnss_t:
                continue
            j = nav.obs_idx[f][sys[i]]
            if obs.lli[i, j] & 1 == 0:
                continue
            nav.x[IB(sat[i], f, nav.na)] = 0

        bias = np.zeros(ns)
        offset = 0
        na = 0
        for i in range(ns):
            if sys[i] not in nav.gnss_t:
                continue
            j = nav.obs_idx[f][sys[i]]
            freq = nav.obs_freq[f][sys[i]]
            cp = obs.L[i, j]
            pr = obs.P[i, j]
            if cp == 0.0 or pr == 0.0:
                continue
            bias[i] = cp-pr*freq/gn.rCST.CLIGHT
            amb = nav.x[IB(sat[i], f, nav.na)]
            if amb != 0.0:
                offset += bias[i]-amb
                na += 1
        # adjust phase-code coherency
        if na > 0:
            db = offset/na
            for i in range(gn.uGNSS.MAXSAT):
                if nav.x[IB(i+1, f, nav.na)] != 0.0:
                    nav.x[IB(i+1, f, nav.na)] += db
        # initialize ambiguity
        for i in range(ns):
            j = IB(sat[i], f, nav.na)
            if bias[i] == 0.0 or nav.x[j] != 0.0:
                continue
            nav.x[j] = bias[i]
            nav.P[j, j] = nav.sig_n0**2
    return 0


def zdres(nav, obs, rs, vs, dts, rr, cs):
    """ non-differencial residual """
    week, tow = gn.time2gpst(obs.t)
    _c = gn.rCST.CLIGHT
    nf = nav.nf
    n = len(obs.P)
    y = np.zeros((n, nf*2))
    el = np.zeros(n)
    e = np.zeros((n, 3))
    rr_ = rr.copy()
    if nav.tidecorr:
        pos = gn.ecef2pos(rr_)
        disp = tidedisp(gn.gpst2utc(obs.t), pos)
        rr_ += disp
    pos = gn.ecef2pos(rr_)

    inet = cs.find_grid_index(pos)
    dlat, dlon = cs.get_dpos(pos)
    trph, trpw = cs.get_trop(dlat, dlon)

    trop_hs, trop_wet, z = tropmodel(obs.t, pos)
    trop_hs0, trop_wet0, z = tropmodel(obs.t, [pos[0], pos[1], 0])
    r_hs = trop_hs/trop_hs0
    r_wet = trop_wet/trop_wet0

    stec = cs.get_stec(dlat, dlon)

    cs.cpc = np.zeros((n, nf))
    cs.prc = np.zeros((n, nf))
    cs.osr = np.zeros((n, 2*nf+5))

    for i in range(n):
        sat = obs.sat[i]
        sys, prn = gn.sat2prn(sat)
        if sys not in nav.gnss_t or sat in nav.excl_sat:
            continue
        if sat not in cs.lc[inet].sat_n:
            continue
        idx_n = np.where(cs.sat_n == sat)[0][0]  # global
        idx_l = np.where(cs.lc[inet].sat_n == sat)[0][0]  # local
        kidx = [-1]*nav.nf
        nsig = 0
        for k, sig in enumerate(cs.sig_n[idx_n]):
            for f in range(nav.nf):
                if sig == nav.cs_sig_idx[sys][f]:
                    kidx[f] = k
                    nsig += 1
        if nsig < nav.nf:
            continue
        # check for measurement consistency
        if True:
            flg_m = True
            for f in range(nav.nf):
                k = nav.obs_idx[f][sys]
                if obs.P[i, k] == 0.0 or obs.L[i, k] == 0.0 \
                        or obs.lli[i, k] == 1:
                    flg_m = False
            if obs.S[i, 0] < nav.cnr_min:
                flg_m = False
            if flg_m is False:
                continue
        else:
            if obs.P[i, 0] == 0.0 or obs.L[i, 0] == 0.0:
                continue

        r, e[i, :] = gn.geodist(rs[i, :], rr_)
        az, el[i] = gn.satazel(pos, e[i, :])
        if el[i] < nav.elmin:
            continue

        freq = np.zeros(nav.nf)
        lam = np.zeros(nav.nf)
        iono = np.zeros(nav.nf)
        for f in range(nav.nf):
            freq[f] = nav.obs_freq[f][sys]
            lam[f] = gn.rCST.CLIGHT/freq[f]
            iono[f] = 40.3e16/(freq[f]*freq[f])*stec[idx_l]
        iono_ = 40.3e16/(freq[0]*freq[1])*stec[idx_l]

        # global/local signal bias
        cbias = np.zeros(nav.nf)
        pbias = np.zeros(nav.nf)

        if cs.lc[0].cbias is not None:
            cbias += cs.lc[0].cbias[idx_n][kidx]
        if cs.lc[0].pbias is not None:
            pbias += cs.lc[0].pbias[idx_n][kidx]
        if cs.lc[inet].cbias is not None:
            cbias += cs.lc[inet].cbias[idx_l][kidx]
        if cs.lc[inet].pbias is not None:
            pbias += cs.lc[inet].pbias[idx_l][kidx]
            t1 = timediff(obs.t, cs.lc[0].t0[sCType.ORBIT])
            t2 = timediff(obs.t, cs.lc[inet].t0[sCType.PBIAS])
            if t1 >= 0 and t1 < 30 and t2 >= 30:
                pbias += nav.dsis[sat]*0

        # relativity effect
        relatv = shapiro(rs[i, :], rr_)

        # tropospheric delay
        mapfh, mapfw = gn.tropmapf(obs.t, pos, el[i])
        trop = mapfh*trph*r_hs+mapfw*trpw*r_wet
        # phase wind-up effect
        nav.phw[sat-1] = windupcorr(obs.t, rs[i, :], vs[i, :], rr_,
                                    nav.phw[sat-1])
        phw = lam*nav.phw[sat-1]
        antr = antmodel(nav, el[i], nav.nf)
        # range correction
        prc_c = trop+relatv+antr
        # prc_c += nav.dorb[sat]-nav.dclk[sat]
        cs.prc[i, :] = prc_c+iono+cbias
        cs.cpc[i, :] = prc_c-iono+pbias+phw
        cs.osr[i, :] = [pbias[0], pbias[1], cbias[0], cbias[1], trop,
                        iono_, relatv, nav.dorb[sat], nav.dclk[sat]]
        r += -_c*dts[i]

        for f in range(nf):
            k = nav.obs_idx[f][sys]
            y[i, f] = obs.L[i, k]*lam[f]-(r+cs.cpc[i, f])
            y[i, f+nf] = obs.P[i, k]-(r+cs.prc[i, f])

    return y, e, el


def relpos(nav, obs, cs):
    rs, vs, dts, svh = satposs(obs, nav, cs)
    # Kalman filter time propagation
    udstate_ppp(nav, obs)
    xa = np.zeros(nav.nx)
    xp = nav.x.copy()

    # non-differencial residual for rover
    yu, eu, elu = zdres(nav, obs, rs, vs, dts, xp[0:3], cs)

    iu = np.where(elu >= nav.elmin)[0]
    sat = obs.sat[iu]
    y = yu[iu, :]
    e = eu[iu, :]
    el = elu[iu]
    nav.sat = sat
    nav.y = y

    logmon(nav, obs.t, sat, cs, iu)

    ny = y.shape[0]
    if ny < 6:
        nav.P[np.diag_indices(3)] = 1.0
        nav.smode = 5
        return -1

    # DD residual
    v, H, R = ddres(nav, xp, y, e, sat, el)
    Pp = nav.P.copy()
    # Kalman filter measurement update
    xp, Pp = kfupdate(xp, Pp, H, v, R)

    if True:
        # non-differencial residual for rover after measurement update
        yu, eu, elu = zdres(nav, obs, rs, vs, dts, xp[0:3], cs)
        y = yu[iu, :]
        e = eu[iu, :]
        ny = y.shape[0]
        nav.y2 = y
        if ny < 6:
            return -1
        # reisdual for float solution
        v, H, R = ddres(nav, xp, y, e, sat, el)
        if valpos(nav, v, R):
            nav.x = xp
            nav.P = Pp

    nb, xa = resamb_lambda(nav, sat)
    nav.smode = 5  # float
    if nb > 0:
        yu, eu, elu = zdres(nav, obs, rs, vs, dts, xa[0:3], cs)
        y = yu[iu, :]
        e = eu[iu, :]
        v, H, R = ddres(nav, xa, y, e, sat, el)
        if valpos(nav, v, R):  # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
            if nav.armode == 3:  # fix and hold
                holdamb(nav, xa)  # hold fixed ambiguity
            # if rtk->sol.chisq<max_inno[4] (5)
            nav.smode = 4  # fix
    return 0
