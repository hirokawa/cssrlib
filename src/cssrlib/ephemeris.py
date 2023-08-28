"""
module for ephemeris processing
"""

import numpy as np
from cssrlib.gnss import uGNSS, rCST, sat2prn, timediff, timeadd, vnorm
from cssrlib.cssrlib import sCSSRTYPE as sc
from cssrlib.cssrlib import sCType

MAX_ITER_KEPLER = 30
RTOL_KEPLER = 1e-13


def findeph(nav, t, sat, iode=-1, mode=0):
    """ find ephemeric for sat """
    dt_p = 3600*4
    eph = None
    for eph_ in nav:
        if eph_.sat != sat:
            continue
        dt = timediff(t, eph_.toe)
        if (iode < 0 or eph_.iode == iode) and eph_.mode == mode and \
                abs(dt) < dt_p:
            eph = eph_
            break
    return eph


def dtadjust(t1, t2, tw=604800):
    """ calculate delta time considering week-rollover """
    dt = timediff(t1, t2)
    if dt > tw:
        dt -= tw
    elif dt < -tw:
        dt += tw
    return dt


def eph2pos(t, eph, flg_v=False):
    """ calculate satellite position based on ephemeris """
    sys, prn = sat2prn(eph.sat)
    if sys == uGNSS.GAL:
        mu = rCST.MU_GAL
        omge = rCST.OMGE_GAL
    elif sys == uGNSS.BDS:
        mu = rCST.MU_BDS
        omge = rCST.OMGE_BDS
    else:  # GPS,QZS
        mu = rCST.MU_GPS
        omge = rCST.OMGE
    dt = dtadjust(t, eph.toe)
    n0 = np.sqrt(mu/eph.A**3)
    dna = eph.deln
    Ak = eph.A
    if eph.mode > 0:
        dna += 0.5*dt*eph.delnd
        Ak += dt*eph.Adot
    n = n0+dna
    M = eph.M0+n*dt
    E = M
    for _ in range(10):
        Eold = E
        sE = np.sin(E)
        E = M+eph.e*sE
        if abs(Eold-E) < 1e-12:
            break
    cE = np.cos(E)
    dtc = dtadjust(t, eph.toc)
    dtrel = -2.0*np.sqrt(mu*eph.A)*eph.e*sE/rCST.CLIGHT**2
    dts = eph.af0+eph.af1*dtc+eph.af2*dtc**2 + dtrel

    nus = np.sqrt(1.0-eph.e**2)*sE
    nuc = cE-eph.e
    nue = 1.0-eph.e*cE

    nu = np.arctan2(nus, nuc)
    phi = nu+eph.omg
    h2 = np.array([np.cos(2.0*phi), np.sin(2.0*phi)])
    u = phi+np.array([eph.cuc, eph.cus])@h2
    r = Ak*nue+np.array([eph.crc, eph.crs])@h2
    h = np.array([np.cos(u), np.sin(u)])
    xo = r*h

    inc = eph.i0+eph.idot*dt+np.array([eph.cic, eph.cis])@h2
    si = np.sin(inc)
    ci = np.cos(inc)

    if sys == uGNSS.BDS and (prn <= 5 or prn >= 59):  # BDS GEO
        Omg = eph.OMG0+eph.OMGd*dt-omge*eph.toes
        sOmg = np.sin(Omg)
        cOmg = np.cos(Omg)
        p = np.array([cOmg, sOmg, 0])
        q = np.array([-ci*sOmg, ci*cOmg, si])
        rg = xo@np.array([p, q])
        so = np.sin(omge*dt)
        co = np.cos(omge*dt)
        Mo = np.array([[co, so*rCST.COS_5, so*rCST.SIN_5],
                       [-so, co*rCST.COS_5, co*rCST.SIN_5],
                       [0.0,   -rCST.SIN_5,    rCST.COS_5]])
        rs = Mo@rg
    else:
        Omg = eph.OMG0+eph.OMGd*dt-omge*(eph.toes+dt)
        sOmg = np.sin(Omg)
        cOmg = np.cos(Omg)
        p = np.array([cOmg, sOmg, 0])
        q = np.array([-ci*sOmg, ci*cOmg, si])
        rs = xo@np.array([p, q])

    if flg_v:  # satellite velocity
        qq = np.array([si*sOmg, -si*cOmg, ci])
        Ed = n/nue
        nud = np.sqrt(1.0-eph.e**2)/nue*Ed
        h2d = 2.0*nud*np.array([-h[1], h[0]])
        ud = nud+np.array([eph.cuc, eph.cus])@h2d
        rd = Ak*eph.e*sE*Ed+np.array([eph.crc, eph.crs])@h2d

        hd = np.array([-h[1], h[0]])
        xod = rd*h+(r*ud)*hd
        incd = eph.idot+np.array([eph.cic, eph.cis])@h2d
        omegd = eph.OMGd-omge

        pd = np.array([-p[1], p[0], 0])*omegd
        qd = np.array([-q[1], q[0], 0])*omegd+qq*incd

        vs = xo@np.array([pd, qd])+xod@np.array([p, q])
        return rs, vs, dts

    return rs, dts


def eph2clk(time, eph):
    """ calculate clock offset based on ephemeris """
    t = timediff(time, eph.toc)
    for _ in range(2):
        t -= eph.af0+eph.af1*t+eph.af2*t**2
    dts = eph.af0+eph.af1*t+eph.af2*t**2
    return dts


def satposs(obs, nav, cs=None, orb=None):
    """
    Calculate pos/vel/clk for observed satellites

    The satellite position, velocity and clock offset are computed at
    transmission epoch. The signal time-of-flight is computed from
    a pseudorange measurement corrected by the satellite clock offset,
    hence the observations are required at this stage. The satellite clock
    is already corrected for the relativistic effects. The satellite health
    indicator is extracted from the broadcast navigation message.

    Parameters
    ----------
    obs : Obs()
        contains GNSS measurements
    nav : Nav()
        contains coarse satellite orbit and clock offset information
    cs  : cssr_has()
        contains precise SSR corrections for satellite orbit and clock offset
    obs : peph()
        contains precise satellite orbit and clock offset information

    Returns
    -------
    rs  : np.array() of float
        satellite position in ECEF [m]
    vs  : np.array() of float
        satellite velocities in ECEF [m/s]
    dts : np.array() of float
        satellite clock offsets [s]
    svh : np.array() of int
        satellite health code [-]
    nsat : int
        number of effective satellite
    """

    n = obs.sat.shape[0]
    rs = np.zeros((n, 3))
    vs = np.zeros((n, 3))
    dts = np.zeros(n)
    svh = np.zeros(n, dtype=int)
    iode = -1
    nsat = 0

    for i in range(n):

        sat = obs.sat[i]
        sys, _ = sat2prn(sat)

        # Skip undesired constellations
        #
        if sys not in obs.sig.keys():
            continue

        pr = obs.P[i, 0]  # TODO: catch invalid observation!
        t = timeadd(obs.t, -pr/rCST.CLIGHT)

        if nav.ephopt == 4:

            rs_, dts_, _ = orb.peph2pos(t, sat, nav)
            if rs_ is None or dts_ is None:
                continue
            dt = dts_[0]

            eph = findeph(nav.eph, t, sat)
            if eph is None:
                svh[i] = 1
                continue
            svh[i] = eph.svh

        else:

            if cs is not None:

                if cs.iodssr_c[sCType.ORBIT] == cs.iodssr:
                    if sat not in cs.sat_n:
                        continue
                    idx = cs.sat_n.index(sat)
                else:
                    if cs.iodssr_c[sCType.ORBIT] == cs.iodssr_p:
                        if sat not in cs.sat_n_p:
                            continue
                        idx = cs.sat_n_p.index(sat)
                    else:
                        continue

                iode = cs.lc[0].iode[idx]
                dorb = cs.lc[0].dorb[idx, :]  # radial,along-track,cross-track

                if cs.cssrmode == sc.BDS_PPP:  # consistency check for IOD corr
                    if cs.lc[0].iodc[idx] == cs.lc[0].iodc_c[idx]:
                        dclk = cs.lc[0].dclk[idx]
                    else:
                        if cs.lc[0].iodc[idx] == cs.lc[0].iodc_c_p[idx]:
                            dclk = cs.lc[0].dclk_p[idx]
                        else:
                            continue

                else:

                    if cs.cssrmode == sc.GAL_HAS_SIS:  # HAS only

                        if cs.mask_id != cs.mask_id_clk:  # mask has changed
                            if sat not in cs.sat_n_p:
                                continue
                            idx = cs.sat_n_p.index(sat)

                    else:

                        if cs.iodssr_c[sCType.CLOCK] == cs.iodssr:
                            if sat not in cs.sat_n:
                                continue
                            idx = cs.sat_n.index(sat)
                        else:
                            if cs.iodssr_c[sCType.CLOCK] == cs.iodssr_p:
                                if sat not in cs.sat_n_p:
                                    continue
                                idx = cs.sat_n_p.index(sat)
                            else:
                                continue

                    dclk = cs.lc[0].dclk[idx]

                if np.isnan(dclk) or np.isnan(dorb@dorb):
                    continue

                mode = cs.nav_mode[sys]

            else:

                mode = 0

            eph = findeph(nav.eph, t, sat, iode, mode=mode)
            if eph is None:
                svh[i] = 1
                continue

            svh[i] = eph.svh
            dt = eph2clk(t, eph)

        t = timeadd(t, -dt)

        if nav.ephopt == 4:  # precise ephemeris

            rs_, dts_, _ = orb.peph2pos(t, sat, nav)
            rs[i, :] = rs_[0:3]
            vs[i, :] = rs_[3:6]
            dts[i] = dts_[0]
            nsat += 1

        else:

            rs[i, :], vs[i, :], dts[i] = eph2pos(t, eph, True)

            # Apply SSR correction
            #
            if cs is not None:

                if cs.cssrmode == sc.BDS_PPP:
                    er = vnorm(rs[i, :])
                    rc = np.cross(rs[i, :], vs[i, :])
                    ec = vnorm(rc)
                    ea = np.cross(ec, er)
                    A = np.array([er, ea, ec])
                else:
                    ea = vnorm(vs[i, :])
                    rc = np.cross(rs[i, :], vs[i, :])
                    ec = vnorm(rc)
                    er = np.cross(ea, ec)
                    A = np.array([er, ea, ec])

                dorb_e = dorb@A

                rs[i, :] -= dorb_e
                dts[i] += dclk/rCST.CLIGHT

                ers = vnorm(rs[i, :]-nav.x[0:3])
                dorb_ = -ers@dorb_e
                sis = dclk-dorb_
                if cs.lc[0].t0[1].time % 30 == 0 and \
                        timediff(cs.lc[0].t0[1], nav.time_p) > 0:
                    if abs(nav.sis[sat]) > 0:
                        nav.dsis[sat] = sis - nav.sis[sat]
                    nav.sis[sat] = sis

                nav.dorb[sat] = dorb_
                nav.dclk[sat] = dclk
                nsat += 1

    if cs is not None:
        nav.time_p = cs.lc[0].t0[1]

    return rs, vs, dts, svh, nsat
