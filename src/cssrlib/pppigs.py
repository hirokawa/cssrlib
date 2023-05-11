"""
module for standard PPP positioning
"""
from collections import defaultdict
from copy import copy
from datetime import datetime
import numpy as np
import sys
import cssrlib.gnss as gn
from cssrlib.gnss import tropmodel, uGNSS, rCST, sat2id, sat2prn
from cssrlib.gnss import timeadd, time2epoch, time2str
from cssrlib.ppp import tidedisp, shapiro, windupcorr
from cssrlib.rtk import IB, ddres, resamb_lambda, valpos, holdamb, initx
from cssrlib.peph import antModelRx, antModelTx


def rtkinit(nav, pos0=np.zeros(3)):
    """ initialize variables for RTK """

    # Logging level
    nav.monlevel = 1

    # Number of frequencies
    #
    nav.nf = 2  # TODO: make obsolete if possible

    # Positioning mode
    # 0:static, 1:kinematic
    nav.pmode = 0

    # Number of tracking channels (1 per satellite)
    #
    nav.nChan = 20
    nav.satIdx = defaultdict(list)
    for n in range(nav.nChan):
        nav.satIdx[None].append(n)

    # State index
    #
    nav.idx_pos = -1
    nav.idx_vel = -1
    nav.idx_ztd = -1
    nav.idx_ion = -1

    # Position (and optional velocity) states
    #
    nav.na = 3 if nav.pmode == 0 else 6
    nav.nq = 3 if nav.pmode == 0 else 6

    # Zenith tropospheric delay state
    nav.na += 1
    nav.nq += 1

    # Slant ionospheric delay states
    nav.na += nav.nChan
    nav.nq += nav.nChan

    nav.ratio = 0
    nav.thresar = [2.0]

    # Index of position, velocity and ztd
    nav.idx_pos = 0
    nav.idx_vel = -1 if nav.pmode == 0 else 3
    nav.idx_ztd = 3 if nav.pmode == 0 else 6
    nav.idx_ion = 4 if nav.pmode == 0 else 7

    # state vector dimensions
    nav.nx = nav.na+gn.uGNSS.MAXSAT*nav.nf

    nav.x = np.zeros(nav.nx)
    nav.P = np.zeros((nav.nx, nav.nx))

    nav.xa = np.zeros(nav.na)
    nav.Pa = np.zeros((nav.na, nav.na))

    nav.nfix = nav.neb = 0
    nav.phw = np.zeros(gn.uGNSS.MAXSAT)
    nav.el = np.zeros(gn.uGNSS.MAXSAT)

    # parameter for PPP

    # observation noise parameters
    nav.eratio = [50, 50]
    nav.err = [0, 0.01, 0.005]/np.sqrt(2)

    # initial sigma for state covariance
    nav.sig_p0 = 30.0
    nav.sig_v0 = 1.0
    nav.sig_ztd0 = 0.010
    nav.sig_ion0 = 10.0
    nav.sig_n0 = 30.0

    # process noise sigma
    nav.sig_qp = 0.01
    nav.sig_qv = 1.0
    nav.sig_qion = 10.0  # [m/sqrt(s)]
    nav.sig_qztd = 0.1e-3/np.sqrt(30)  # [m/sqrt(s)]

    nav.tidecorr = True
    nav.armode = 3  # 1:continuous,2:instantaneous,3:fix-and-hold
    nav.elmaskar = np.deg2rad(20)  # elevation mask for AR
    #nav.gnss_t = [uGNSS.GPS, uGNSS.GAL]

    # Initial state vector
    nav.x[0:3] = pos0
    if nav.pmode >= 1:  # kinematic
        nav.x[3:6] = 0.0  # velocity

    # Diagonal elements of covariance matrix
    dP = np.diag(nav.P)
    dP.flags['WRITEABLE'] = True

    dP[0:3] = nav.sig_p0**2
    if nav.pmode >= 1:  # kinematic
        dP[3:6] = nav.sig_v0**2
    dP[nav.idx_ztd] = nav.sig_ztd0**2
    dP[nav.idx_ion:nav.idx_ion+nav.nChan] = nav.sig_ion0**2

    # Process noise
    nav.q = np.zeros(nav.nq)
    if nav.pmode >= 1:  # kinematic
        nav.q[0:3] = nav.sig_qp**2
        nav.q[3:6] = nav.sig_qv**2
    else:
        nav.q[0:3] = nav.sig_qp**2
    nav.q[nav.idx_ztd] = nav.sig_qztd**2
    nav.q[nav.idx_ion:nav.idx_ion+nav.nChan] = nav.sig_qion**2

    nav.fout = None
    nav.logfile = 'log.txt'
    if nav.loglevel >= 2:
        nav.fout = open(nav.logfile, 'w')
        nav.fout.write(
            "# tow\tsys\tprn\tpb1\tpb2\tcb1\tcb2\ttrop\tiono\tantr1\tantr2")
        nav.fout.write(
            "# \trelatv\twup1\twup2\tCPC1\tCPC2\tPRC1\tPRC2\tdorb\tdclk\n")


def sysidx(satlist, sys_ref):
    """ return index of satellites with sys=sys_ref """
    idx = []
    for k, sat in enumerate(satlist):
        sys, _ = gn.sat2prn(sat)
        if sys == sys_ref:
            idx.append(k)
    return idx


def varerr(nav, el, f):
    """ variation of measurement """
    s_el = np.sin(el)
    if s_el <= 0.0:
        return 0.0
    fact = nav.eratio[f-nav.nf] if f >= nav.nf else 1
    a = fact*nav.err[1]
    b = fact*nav.err[2]
    return 2.0*(a**2+(b/s_el)**2)


def udstate(nav, obs):
    """ time propagation of states and initialize """
    tt = gn.timediff(obs.t, nav.t)

    ns = len(obs.sat)
    sys = []
    sat = obs.sat
    for sat_i in obs.sat:
        sys_i, _ = gn.sat2prn(sat_i)
        sys.append(sys_i)

    # pos,vel,ztd,ion
    na = nav.na
    Phi = np.eye(na)
    dPhi = np.diag(Phi)
    dPhi.flags['WRITEABLE'] = True
    dPhi[na-nav.nChan:na] = 0.0
    if nav.pmode > 0:
        Phi[0:3, 3:6] = np.eye(3)*tt
    nav.x[0:na] = Phi@nav.x[0:na]
    nav.P[0:na, 0:na] = Phi@nav.P[0:na, 0:na]@Phi.T

    # Process noise
    dP = np.diag(nav.P)
    dP.flags['WRITEABLE'] = True
    dP[0:nav.nq] += nav.q[0:nav.nq]*tt

    # bias
    for f in range(nav.nf):
        # reset phase-bias if instantaneous AR or expire obs outage counter
        for i in range(gn.uGNSS.MAXSAT):
            sat_ = i+1
            nav.outc[i, f] += 1
            reset = (nav.outc[i, f] > nav.maxout)
            sys_i, _ = gn.sat2prn(sat_)
            if sys_i not in nav.gnss_t:
                continue
            j = IB(sat_, f, nav.na)
            if reset and nav.x[j] != 0.0:
                initx(nav, 0.0, 0.0, j)
                nav.outc[i, f] = 0

        # cycle slip check by LLI
        for i in range(ns):
            if sys[i] not in nav.gnss_t:
                continue
            j = nav.obs_idx[f][sys[i]]
            if obs.lli[i, j] & 1 == 0:
                continue
            initx(nav, 0.0, 0.0, IB(sat[i], f, nav.na))

        # Reset ambiguity if satellite is not available
        for i in range(ns):
            if sat[i] in nav.satIdx:
                continue
            initx(nav, 0.0, 0.0, IB(sat[i], f, nav.na))

        # bias
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
            if cp == 0.0 or pr == 0.0 or freq == 0.0:
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
            initx(nav, bias[i], nav.sig_n0**2, j)
    return 0


def zdres(nav, obs, bsx, rs, vs, dts, svh, rr):
    """ non-differential residual """
    _c = gn.rCST.CLIGHT
    nf = nav.nf
    n = len(obs.P)
    y = np.zeros((n, nf*2))
    el = np.zeros(n)
    e = np.zeros((n, 3))
    rr_ = rr.copy()

    # Tide corrections
    #
    if nav.tidecorr:
        pos = gn.ecef2pos(rr_)
        disp = tidedisp(gn.gpst2utc(obs.t), pos)
        rr_ += disp

    # Geodetic position
    #
    pos = gn.ecef2pos(rr_)

    # Tropospheric dry and wet delays at user position
    #
    trop_hs, trop_wet, _ = tropmodel(obs.t, pos)

    cpc = np.zeros((n, nf))
    prc = np.zeros((n, nf))

    for i in range(n):

        sat = obs.sat[i]
        sys, _ = gn.sat2prn(sat)

        if svh[i] > 0 or sys not in nav.gnss_t or sat in nav.excl_sat:
            continue

        # Check for measurement consistency
        #
        flg_m = True
        for f in range(nav.nf):
            k = nav.obs_idx[f][sys]
            if obs.P[i, k] == 0.0 or obs.L[i, k] == 0.0 or obs.lli[i, k] == 1:
                flg_m = False
        if obs.S[i, 0] < nav.cnr_min:
            flg_m = False
        if flg_m is False:
            continue

        r, e[i, :] = gn.geodist(rs[i, :], rr_)
        _, el[i] = gn.satazel(pos, e[i, :])
        if el[i] < nav.elmin:
            continue

        freq = np.zeros(nav.nf)
        lam = np.zeros(nav.nf)
        for f in range(nav.nf):
            freq[f] = nav.obs_freq[f][sys]
            lam[f] = gn.rCST.CLIGHT/freq[f]

        # Code and phase signal bias [ns]
        #
        cbias = np.zeros(nav.nf)
        pbias = np.zeros(nav.nf)

        if sys == uGNSS.GPS:
            cbias[0], _ = bsx.getosb(sat, obs.t, "C1C")
            cbias[1], _ = bsx.getosb(sat, obs.t, "C2W")
            pbias[0], _ = bsx.getosb(sat, obs.t, "L1C")
            pbias[1], _ = bsx.getosb(sat, obs.t, "L2W")
        elif sys == uGNSS.GAL:
            cbias[0], _ = bsx.getosb(sat, obs.t, "C1X")
            cbias[1], _ = bsx.getosb(sat, obs.t, "C5X")
            pbias[0], _ = bsx.getosb(sat, obs.t, "L1X")
            pbias[1], _ = bsx.getosb(sat, obs.t, "L5X")

        # relativity effect
        #
        relatv = shapiro(rs[i, :], rr_)

        # tropospheric delay mapping functions
        #
        mapfh, mapfw = gn.tropmapf(obs.t, pos, el[i])

        # tropospheric delay
        #
        trop = mapfh*trop_hs + mapfw*trop_wet

        # phase wind-up effect
        #
        nav.phw[sat-1] = windupcorr(obs.t, rs[i, :], vs[i, :], rr_,
                                    nav.phw[sat-1])
        phw = lam*nav.phw[sat-1]

        # Receiver antenna offset
        #
        antr = antmodel(nav, el[i], nav.nf)

        # range correction
        #
        prc_c = trop + relatv + antr

        prc[i, :] = prc_c + cbias*_c*1e-9
        cpc[i, :] = prc_c + pbias*_c*1e-9 + phw

        r += -_c*dts[i]

        for f in range(nf):
            k = nav.obs_idx[f][sys]
            y[i, f] = obs.L[i, k]*lam[f]-(r+cpc[i, f])
            y[i, f+nf] = obs.P[i, k]-(r+prc[i, f])

    return y, e, el


def ddcov(nb, n, Ri, Rj, nv):
    """ DD measurement error covariance """
    R = np.zeros((nv, nv))
    k = 0
    for b in range(n):
        for i in range(nb[b]):
            for j in range(nb[b]):
                R[k+i, k+j] = Ri[k+i]
                if i == j:
                    R[k+i, k+j] += Rj[k+i]
        k += nb[b]
    return R


def ddres(nav, t, x, y, e, sat, el, log=False):
    """
    SD/DD phase/code residuals

        nav :
        t   :
        x   :
        y   :
        e   :
        sat :
        el  :
    """
    _c = gn.rCST.CLIGHT
    nf = nav.nf
    ns = len(el)
    mode = 1 if len(y) == ns else 0  # 0:DD,1:SD
    nb = np.zeros(2*len(nav.gnss_t)*nf, dtype=int)
    Ri = np.zeros(ns*nf*2)
    Rj = np.zeros(ns*nf*2)
    nv = 0
    b = 0
    H = np.zeros((ns*nf*2, nav.nx))
    v = np.zeros(ns*nf*2)

    # Geodetic position
    pos = gn.ecef2pos(x[0:3])

    idx_f = [0, 1]
    for sys in nav.gnss_t:

        for f in range(nf):
            idx_f[f] = nav.obs_idx[f][sys]

        # Loop over twice the number of frequencies
        #   first for all carrier-phase observations
        #   second all pseudorange observations
        for f in range(0, nf*2):

            # Select carrier-phase frequency and iono frequency ratio
            freq = nav.freq[idx_f[f % 2]]
            if f < nf:  # carrier
                mu = -(nav.freq[idx_f[0]]/freq)**2
            else:  # code
                mu = +(nav.freq[idx_f[0]]/freq)**2

            # Select reference satellite
            idx = sysidx(sat, sys)

            if log and 1 == 0:
                for i in idx:
                    print(sys, sat[i])

            if len(idx) > 0:
                i = idx[np.argmax(el[idx])]

            # Loop over satellite
            for j in idx:

                # Skip reference satellite
                #
                if i == j:
                    continue

                # Skip invalid signals
                #
                if y[i, f] == 0.0 or y[j, f] == 0.0:
                    continue

                #  DD residual
                #
                if mode == 0:
                    if y[i+ns, f] == 0.0 or y[j+ns, f] == 0.0:
                        continue
                    v[nv] = (y[i, f]-y[i+ns, f])-(y[j, f]-y[j+ns, f])
                else:
                    v[nv] = y[i, f]-y[j, f]

                # SD line-of-sight vectors
                #
                H[nv, 0:3] = -e[i, :]+e[j, :]

                # SD troposphere
                #
                _, mapfwi = gn.tropmapf(t, pos, el[i])
                _, mapfwj = gn.tropmapf(t, pos, el[j])
                H[nv, nav.idx_ztd] = (mapfwi-mapfwj)
                v[nv] -= (mapfwi-mapfwj)*x[nav.idx_ztd]

                # SD ionosphere
                #
                idx_i = nav.satIdx[sat[i]] + nav.idx_ion
                idx_j = nav.satIdx[sat[j]] + nav.idx_ion
                H[nv, idx_i] = mu
                H[nv, idx_j] = mu
                v[nv] -= mu*(x[idx_i]-x[idx_j])

                if log:
                    print("{:%D %T} {:10.3f} {:10.3f} {:10.3f} "
                          .format(datetime.utcfromtimestamp(t.time),
                                  (mapfwi-mapfwj),
                                  nav.x[nav.idx_ztd],
                                  np.sqrt(nav.P[nav.idx_ztd, nav.idx_ztd])))

                # SD ambiguity
                #
                if f < nf:  # carrier-phase

                    idx_i = IB(sat[i], f, nav.na)
                    idx_j = IB(sat[j], f, nav.na)
                    lami = _c/freq
                    v[nv] -= lami*(x[idx_i]-x[idx_j])
                    H[nv, idx_i] = lami
                    H[nv, idx_j] = -lami
                    Ri[nv] = varerr(nav, el[i], f)  # measurement variance
                    Rj[nv] = varerr(nav, el[j], f)  # measurement variance
                    nav.vsat[sat[i]-1, f] = 1
                    nav.vsat[sat[j]-1, f] = 1

                else:  # pseudorange

                    Ri[nv] = varerr(nav, el[i], f)  # measurement variance
                    Rj[nv] = varerr(nav, el[j], f)  # measurement variance

                nb[b] += 1
                nv += 1

            b += 1

    v = np.resize(v, nv)
    H = np.resize(H, (nv, nav.nx))
    R = ddcov(nb, b, Ri, Rj, nv)

    return v, H, R


def kfupdate(x, P, H, v, R):
    """ Kalman filter measurement update """
    PHt = P@H.T
    S = H@PHt+R
    K = PHt@np.linalg.inv(S)
    x += K@v
    P = P - K@H@P

    return x, P, S


def satpreposs(obs, nav, orb):
    """ calculate pos/vel/clk for observed satellites  """
    n = obs.sat.shape[0]
    rs = np.zeros((n, 6))
    dts = np.zeros((n, 2))
    svh = np.zeros(n, dtype=int)

    for i in range(n):

        sat = obs.sat[i]
        sys, _ = sat2prn(sat)

        if sat < 1:
            continue
        if sys not in nav.gnss_t:
            continue

        pr = obs.P[i, 0]
        t = timeadd(obs.t, -pr/rCST.CLIGHT)

        rs[i, :], dts[i, :], _ = orb.peph2pos(t, sat, nav)

        t = timeadd(t, -dts[i, 0])
        rs[i, :], dts[i, :], _ = orb.peph2pos(t, sat, nav)

    return rs[:, 0:3], rs[:, 3:6], dts[:, 0], svh


def pppigspos(nav, obs, orb, bsx):
    """ PPP positioning with IGS files and conventions """

    # GNSS satellite positions, velocities and clock offsets
    #
    rs, vs, dts, svh = satpreposs(obs, nav, orb)

    # Kalman filter time propagation
    #
    udstate(nav, obs)

    xa = np.zeros(nav.nx)
    xp = nav.x.copy()

    # Non-differential residuals
    #
    yu, eu, elu = zdres(nav, obs, bsx, rs, vs, dts, svh, xp[0:3])

    # Select satellites above minimum elevation
    #
    iu = np.where(elu >= nav.elmin)[0]
    sat = obs.sat[iu]
    y = yu[iu, :]
    e = eu[iu, :]
    el = elu[iu]
    nav.sat = sat
    nav.el[sat-1] = el
    nav.y = y
    ns = len(sat)

    # Update satellite index map
    #
    # NOTE: currently iono and ambiguity parameters are not newly initialized
    #       on constellation changes! Must be added here.
    #
    # Remove missing satellites
    #
    satsIdx_ = copy(nav.satIdx)
    for s in satsIdx_.keys():
        # Skip void
        if s not in sat and s is not None:
            idx = nav.satIdx.pop(s)
            nav.satIdx[None].append(idx)
            print("Removing {} at {}".format(sat2id(s), idx))

    # Add new satellites
    #
    for s in sat:
        if s not in nav.satIdx.keys():
            idx = nav.satIdx.pop(None)
            if len(idx) > 0:
                nav.satIdx[s] = idx[0]
                nav.satIdx[None] = idx[1:]
                print("Adding   {} at {}".format(sat2id(s), idx[0]))
            else:
                print("ERROR: satellite index full!")
                sys.exit(1)

    """
    print()
    print("Satellite index")
    for s, n in nav.satIdx.items():
        print("{} : {}".format('---' if s is None else sat2id(s), n))
    print()
    """

    # Check if observations of at least 6 satellites are left over after editing
    #
    ny = y.shape[0]
    if ny < 6:
        nav.P[np.diag_indices(3)] = 1.0
        nav.smode = 5
        return -1

    # SD residuals
    #
    # NOTE: where are working on a reduced list of observations from here on
    #
    v, H, R = ddres(nav, obs.t, xp, y, e, sat, el)
    Pp = nav.P.copy()

    # Kalman filter measurement update
    #
    xp, Pp, _ = kfupdate(xp, Pp, H, v, R)

    # Non-differential residuals after measurement update
    #
    yu, eu, elu = zdres(nav, obs, bsx, rs, vs, dts, svh, xp[0:3])
    y = yu[iu, :]
    e = eu[iu, :]
    ny = y.shape[0]
    nav.y2 = y
    if ny < 6:
        return -1

    # Residuals for float solution
    #
    v, H, R = ddres(nav, obs, xp, y, e, sat, el)
    if valpos(nav, v, R):
        nav.x = xp
        nav.P = Pp
        nav.ns = 0
        for i in range(ns):
            j = sat[i]-1
            for f in range(nav.nf):
                if nav.vsat[j, f] == 0:
                    continue
                nav.lock[j, f] += 1
                nav.outc[j, f] = 0
                if f == 0:
                    nav.ns += 1
    else:
        nav.smode = 0

    nav.smode = 5  # 5: float ambiguities
    try:
        nb, xa = resamb_lambda(nav, sat)
    except:
        nb = 0
    if nb > 0:
        yu, eu, elu = zdres(nav, obs, bsx, rs, vs, dts, svh, xa[0:3])
        y = yu[iu, :]
        e = eu[iu, :]
        v, H, R = ddres(nav, obs, xa, y, e, sat, el)
        if valpos(nav, v, R):       # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
            if nav.armode == 3:     # fix and hold
                holdamb(nav, xa)    # hold fixed ambiguity
            nav.smode = 4           # fix

    # Store epoch for solution
    #
    nav.t = obs.t

    return 0
