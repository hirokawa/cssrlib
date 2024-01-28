"""
module for ephemeris processing
"""

from cssrlib.cssrlib import sCType
from cssrlib.cssrlib import sCSSRTYPE as sc
import numpy as np
from cssrlib.gnss import uGNSS, rCST, sat2prn, timediff, timeadd, vnorm
from cssrlib.gnss import gtime_t, Geph, Eph, Alm, prn2sat, gpst2time, \
    time2gpst, timeget, time2gst, time2bdt, gst2time, bdt2time, epoch2time
from datetime import datetime
import xml.etree.ElementTree as et

MAX_ITER_KEPLER = 30
RTOL_KEPLER = 1e-13

MAXDTOE_t = {uGNSS.GPS: 7201.0, uGNSS.GAL: 14400.0, uGNSS.QZS: 7201.0,
             uGNSS.BDS: 21601.0, uGNSS.IRN: 7201.0, uGNSS.GLO: 1800.0,
             uGNSS.SBS: 360.0}


def findeph(nav, t, sat, iode=-1, mode=0):
    """ find ephemeris for sat """
    sys, _ = sat2prn(sat)
    eph = None
    tmax = MAXDTOE_t[sys]
    tmin = tmax + 1.0
    for eph_ in nav:
        if eph_.sat != sat or (iode >= 0 and iode != eph_.iode):
            continue
        if eph_.mode != mode:
            continue
        dt = abs(timediff(t, eph_.toe))
        if dt > tmax or eph_.mode != mode:
            continue
        if iode >= 0:
            return eph_
        if dt <= tmin:
            eph = eph_
            tmin = dt

    return eph


def dtadjust(t1, t2, tw=604800):
    """ calculate delta time considering week-rollover """
    dt = timediff(t1, t2)
    if dt > tw:
        dt -= tw
    elif dt < -tw:
        dt += tw
    return dt


def deq(x, acc):
    xdot = np.zeros(6)

    r2 = x[0:3]@x[0:3]
    r3 = r2*np.sqrt(r2)
    omg2 = rCST.OMGE_GLO**2

    if r2 <= 0.0:
        return xdot

    a = 1.5*rCST.J2_GLO*rCST.MU_GLO*rCST.RE_GLO**2/r2/r3
    b = 5.0*x[2]**2/r2
    c = -rCST.MU_GLO/r3-a*(1.0-b)

    xdot[0:3] = x[3:6]
    xdot[3] = (c+omg2)*x[0]+2.0*rCST.OMGE_GLO*x[4]
    xdot[4] = (c+omg2)*x[1]-2.0*rCST.OMGE_GLO*x[3]
    xdot[5] = (c-2.0*a)*x[2]
    xdot[3:6] += acc
    return xdot


def glorbit(t, x, acc):
    k1 = deq(x, acc)
    w = x + k1*t/2.0
    k2 = deq(w, acc)
    w = x + k2*t/2.0
    k3 = deq(w, acc)
    w = x + k3*t
    k4 = deq(w, acc)
    x += (k1+2.0*k2+2.0*k3+k4)*t/6.0
    return x


def geph2pos(time: gtime_t, geph: Geph, flg_v=False, TSTEP=1.0):
    """ calculate GLONASS satellite position based on ephemeris """
    t = timediff(time, geph.toe)
    dts = -geph.taun+geph.gamn*t
    x = np.zeros(6)
    x[0:3] = geph.pos
    x[3:6] = geph.vel

    tt = -TSTEP if t < 0.0 else TSTEP

    while True:
        if np.fabs(t) <= 1e-9:
            break
        if np.fabs(t) < TSTEP:
            tt = t
        x = glorbit(tt, x, geph.acc)
        t -= tt

        rs = x[0:3]
        vs = x[3:6]

    if flg_v:
        return rs, vs, dts

    return rs, dts


def geph2clk(time: gtime_t, geph: Geph):
    """ calculate GLONASS satellite clock offset based on ephemeris """
    ts = timediff(time, geph.toe)
    t = ts
    for i in range(2):
        t = ts - (-geph.taun+geph.gamn*t)
    return -geph.taun + geph.gamn*t


def eph2pos(t: gtime_t, eph: Eph, flg_v=False):
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
            if rs_ is None or dts_ is None or np.isnan(dts_[0]):
                continue
            dt = dts_[0]

            if len(nav.eph) > 0:
                eph = findeph(nav.eph, t, sat)
                if eph is None:
                    svh[i] = 1
                    continue
                svh[i] = eph.svh
            else:
                svh[i] = 0

        else:

            if cs is not None:

                if cs.iodssr >= 0 and cs.iodssr_c[sCType.ORBIT] == cs.iodssr:
                    if sat not in cs.sat_n:
                        continue
                elif cs.iodssr_p >= 0 and \
                        cs.iodssr_c[sCType.ORBIT] == cs.iodssr_p:
                    if sat not in cs.sat_n_p:
                        continue
                else:
                    continue

                if sat not in cs.lc[0].iode.keys():
                    continue

                iode = cs.lc[0].iode[sat]
                dorb = cs.lc[0].dorb[sat]  # radial,along-track,cross-track

                if cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5):
                    dorb += cs.lc[0].dvel[sat] * \
                        (timediff(obs.t, cs.lc[0].t0[sat][sCType.ORBIT]))

                if cs.cssrmode == sc.BDS_PPP:  # consistency check for IOD corr

                    if cs.lc[0].iodc[sat] == cs.lc[0].iodc_c[sat]:
                        dclk = cs.lc[0].dclk[sat]
                    else:
                        if cs.lc[0].iodc[sat] == cs.lc[0].iodc_c_p[sat]:
                            dclk = cs.lc[0].dclk_p[sat]
                        else:
                            continue

                else:

                    if cs.cssrmode == sc.GAL_HAS_SIS:  # HAS only
                        if cs.mask_id != cs.mask_id_clk:  # mask has changed
                            if sat not in cs.sat_n_p:
                                continue
                    else:
                        if cs.iodssr_c[sCType.CLOCK] == cs.iodssr:
                            if sat not in cs.sat_n:
                                continue
                        else:
                            if cs.iodssr_c[sCType.CLOCK] == cs.iodssr_p:
                                if sat not in cs.sat_n_p:
                                    continue
                            else:
                                continue

                    dclk = cs.lc[0].dclk[sat]

                    if cs.lc[0].cstat & (1 << sCType.HCLOCK) and \
                            sat in cs.lc[0].hclk.keys() and \
                            not np.isnan(cs.lc[0].hclk[sat]):
                        dclk += cs.lc[0].hclk[sat]

                    if cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5):
                        dclk += cs.lc[0].ddft[sat] * \
                            (timediff(obs.t, cs.lc[0].t0[sat][sCType.CLOCK]))

                if np.isnan(dclk) or np.isnan(dorb@dorb):
                    continue

                mode = cs.nav_mode[sys]

            else:

                mode = 0

            if sys == uGNSS.GLO:
                geph = findeph(nav.geph, t, sat, iode, mode=mode)
                if geph is None:
                    svh[i] = 1
                    continue

                svh[i] = geph.svh
                dt = geph2clk(t, geph)

                if sat not in nav.glo_ch:
                    nav.glo_ch[sat] = geph.frq

            else:
                eph = findeph(nav.eph, t, sat, iode, mode=mode)
                if eph is None:
                    svh[i] = 1
                    continue

                svh[i] = eph.svh
                dt = eph2clk(t, eph)

        t = timeadd(t, -dt)

        if nav.ephopt == 4:  # precise ephemeris

            rs_, dts_, _ = orb.peph2pos(t, sat, nav)
            rs[i, :] = rs_[0: 3]
            vs[i, :] = rs_[3: 6]
            dts[i] = dts_[0]
            nsat += 1

        else:

            if sys == uGNSS.GLO:
                rs[i, :], vs[i, :], dts[i] = geph2pos(t, geph, True)
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

                if cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5):
                    dorb_e = dorb
                else:
                    dorb_e = dorb@A

                rs[i, :] -= dorb_e
                dts[i] += dclk/rCST.CLIGHT

                if cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5,
                                   sc.DGPS) and sys == uGNSS.GPS:
                    dts[i] -= eph.tgd

                ers = vnorm(rs[i, :]-nav.x[0: 3])
                dorb_ = -ers@dorb_e
                sis = dclk-dorb_
                if cs.lc[0].t0[sat][sCType.ORBIT].time % 30 == 0 and \
                        timediff(cs.lc[0].t0[sat][sCType.ORBIT], nav.time_p) > 0:
                    if abs(nav.sis[sat]) > 0:
                        nav.dsis[sat] = sis - nav.sis[sat]
                    nav.sis[sat] = sis

                nav.dorb[sat] = dorb_
                nav.dclk[sat] = dclk
            elif nav.smode == 1 and nav.nf == 1:  # standalone positioing
                dts[i] -= eph.tgd

            nsat += 1

    if cs is not None:
        if sat in cs.lc[0].t0 and sCType.ORBIT in cs.lc[0].t0[sat]:
            nav.time_p = cs.lc[0].t0[sat][sCType.ORBIT]

    return rs, vs, dts, svh, nsat


def loadXmlAlmanac(fname, sys=uGNSS.GAL):
    """ load Galileo Almanac in XML format:
      https://www.gsc-europa.eu/gsc-products/almanac
    """
    alm_t = []
    root = et.parse(fname).getroot()

    dstr = root.find("./header/GAL-header/issueDate").text
    d = datetime.fromisoformat(dstr)
    ep = [d.year, d.month, d.day, d.hour, d.minute, d.second]
    tref = epoch2time(ep)
    week_ref, tow_ref = time2gst(tref)
    week_ref = week_ref//4*4

    h = root.find('body').find('Almanacs')
    for sv in h.findall('svAlmanac'):
        prn = int(sv.find('SVID').text)

        sts_fnav = sv.find('svFNavSignalStatus')
        sts_E5a = int(sts_fnav.find('statusE5a').text)

        sts_inav = sv.find('svINavSignalStatus')
        sts_E5b = int(sts_inav.find('statusE5b').text)
        sts_E1B = int(sts_inav.find('statusE1B').text)

        alm_ = sv.find('almanac')
        sat = prn2sat(sys, prn)

        alm = Alm(sat)
        rA = float(alm_.find('aSqRoot').text) + np.sqrt(29600e3)
        alm.A = rA**2
        alm.e = float(alm_.find('ecc').text)
        deltai = float(alm_.find('deltai').text)*rCST.SC2RAD
        alm.i0 = 56.0*rCST.D2R + deltai
        alm.OMG0 = float(alm_.find('omega0').text)*rCST.SC2RAD
        alm.OMGd = float(alm_.find('omegaDot').text)*rCST.SC2RAD
        alm.omg = float(alm_.find('w').text)*rCST.SC2RAD
        alm.M0 = float(alm_.find('m0').text)*rCST.SC2RAD
        alm.af0 = float(alm_.find('af0').text)
        alm.af1 = float(alm_.find('af1').text)
        alm.ioda = float(alm_.find('iod').text)
        alm.toas = float(alm_.find('t0a').text)
        wna = float(alm_.find('wna').text)

        alm.toa = gst2time(week_ref + wna, alm.toas)
        alm.svh = (sts_E5a << 4) | (sts_E5b << 2) | (sts_E1B)

        alm_t.append(alm)

    return alm_t


def loadyuma(fname, sys=uGNSS.GPS):
    """ load Yuma almanac """
    alm_t = []
    if sys == uGNSS.GPS or sys == uGNSS.QZS:
        week_ref, _ = time2gpst(timeget())
    elif sys == uGNSS.GAL:
        week_ref, _ = time2gst(timeget())
    elif sys == uGNSS.BDS:
        week_ref, _ = time2bdt(timeget())
    else:
        return alm_t
    flg = False

    with open(fname, 'rt') as fh:
        for line in fh:

            v = line.split(':')
            if v[0][0] == '*':  # comment
                continue
            elif v[0] == 'ID':
                prn = int(v[1])
                sat = prn2sat(sys, prn)
                alm = Alm(sat)
                flg = True
            elif v[0] == 'Health':
                alm.svh = int(v[1])
            elif v[0] == 'Eccentricity':
                alm.e = float(v[1])
            elif v[0] == 'Time of Applicability(s)':
                alm.toas = float(v[1])
            elif v[0] == 'Orbital Inclination(rad)':
                alm.i0 = float(v[1])
            elif v[0] == 'Rate of Right Ascen(r/s)':
                alm.OMGd = float(v[1])
            elif v[0] == 'SQRT(A)  (m 1/2)':
                sqrtA = float(v[1])
                alm.A = sqrtA**2
            elif v[0] == 'Right Ascen at Week(rad)' or \
                    v[0] == 'Right Ascen at TOA(rad)':
                alm.OMG0 = float(v[1])
            elif v[0] == 'Argument of Perigee(rad)':
                alm.omg = float(v[1])
            elif v[0] == 'Mean Anom(rad)':
                alm.M0 = float(v[1])
            elif v[0] == 'Af0(s)':
                alm.af0 = float(v[1])
            elif v[0] == 'Af1(s/s)':
                alm.af1 = float(v[1])
            elif v[0] == 'week':
                alm.week = int(v[1])
                alm.week += week_ref//1023*1023
                if alm.week > week_ref:
                    alm.week -= 1023

                alm.sattype = 0
                if sys == uGNSS.GPS or sys == uGNSS.QZS:
                    alm.toa = gpst2time(alm.week, alm.toas)
                elif sys == uGNSS.GAL:
                    alm.toa = gst2time(alm.week, alm.toas)
                elif sys == uGNSS.BDS:
                    alm.toa = bdt2time(alm.week, alm.toas)

                if flg:
                    alm_t.append(alm)
                    flg = False

    return alm_t


def findalm(alm_t, t, sat, tmax=np.inf):
    """ find almanac for sat """
    sys, _ = sat2prn(sat)
    alm = None
    tmin = tmax + 1.0
    for alm_ in alm_t:
        if alm_.sat != sat:
            continue
        dt = abs(timediff(t, alm_.toa))
        if dt > tmax:
            continue
        if dt <= tmin:
            alm = alm_
            tmin = dt

    return alm


def alm2pos(t: gtime_t, alm: Alm):
    """ calculate satellite position based on ephemeris """
    sys, prn = sat2prn(alm.sat)
    if sys == uGNSS.GAL:
        mu = rCST.MU_GAL
        omge = rCST.OMGE_GAL
    elif sys == uGNSS.BDS:
        mu = rCST.MU_BDS
        omge = rCST.OMGE_BDS
    else:  # GPS,QZS
        mu = rCST.MU_GPS
        omge = rCST.OMGE
    dt = dtadjust(t, alm.toa)
    n0 = np.sqrt(mu/alm.A**3)
    M = alm.M0+n0*dt
    E = M
    for _ in range(10):
        Eold = E
        sE = np.sin(E)
        E = M+alm.e*sE
        if abs(Eold-E) < 1e-12:
            break
    cE = np.cos(E)
    u = np.arctan2(np.sqrt(1.0-alm.e**2)*sE, cE-alm.e)+alm.omg
    r = alm.A*(1.0-alm.e*cE)
    i = alm.i0
    Omg = alm.OMG0+(alm.OMGd-omge)*dt-omge*alm.toas
    x, y = r*np.cos(u), r*np.sin(u)
    cosO, sinO = np.cos(Omg), np.sin(Omg)
    cosi, sini = np.cos(i), np.sin(i)

    rs = np.array([x*cosO-y*cosi*sinO,
                   x*sinO+y*cosi*cosO,
                   y*sini])
    dts = alm.af0 + alm.af1*dt

    return rs, dts
