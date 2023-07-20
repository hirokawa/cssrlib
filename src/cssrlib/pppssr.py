"""
module for standard PPP positioning
"""

import numpy as np

import cssrlib.gnss as gn
from cssrlib.ephemeris import satposs
from cssrlib.gnss import sat2id, sat2prn, rSigRnx, uTYP, uGNSS
from cssrlib.gnss import time2str
from cssrlib.ppp import tidedisp, shapiro, windupcorr
from cssrlib.peph import antModelRx, antModelTx
from cssrlib.rtk import IB, ddcov, resamb_lambda, valpos, holdamb, initx
from cssrlib.rtk import varerr
from cssrlib.cssrlib import ssig2rsig, sCType, sSigGPS


def IT(na):
    """ return index of zenith tropospheric delay estimate """
    return na-gn.uGNSS.MAXSAT-1


def II(s, na):
    """ return index of slant ionospheric delay estimate """
    return na-gn.uGNSS.MAXSAT+s-1


def ionoDelay(sig1, sig2, pr1, pr2):
    """
    Compute ionospheric delay based on dual-frequency pseudorange difference
    """
    f1_2 = sig1.frequency()**2
    f2_2 = sig2.frequency()**2
    return (pr1-pr2)/(1-f1_2/f2_2)


def rtkinit(nav, pos0=np.zeros(3), logfile=None):
    """ initialize variables for PPP """

    # Number of frequencies (actually signals!)
    #
    nav.nf = 2  # TODO: make obsolete if possible
    nav.ephopt = 2  # SSR-APC

    # Position (+ optional velocity), zenith tropo delay and slant ionospheric delay states
    #
    nav.na = (4 if nav.pmode == 0 else 7) + gn.uGNSS.MAXSAT
    nav.nq = (4 if nav.pmode == 0 else 7) + gn.uGNSS.MAXSAT

    nav.thresar = [2.0]

    # State vector dimensions (inlcuding slat iono delay and ambiguities)
    #
    nav.nx = nav.na+gn.uGNSS.MAXSAT*nav.nf

    nav.x = np.zeros(nav.nx)
    nav.P = np.zeros((nav.nx, nav.nx))

    nav.xa = np.zeros(nav.na)
    nav.Pa = np.zeros((nav.na, nav.na))

    nav.phw = np.zeros(gn.uGNSS.MAXSAT)
    nav.el = np.zeros(gn.uGNSS.MAXSAT)

    # Parameters for PPP
    #
    # Observation noise parameters
    #
    nav.eratio = [100, 100, 100]
    nav.err = [0, 0.003, 0.003]  # [m] sigma
    #nav.eratio = [50, 50]
    #nav.err = [0, 0.01, 0.005]/np.sqrt(2)

    # Initial sigma for state covariance
    #
    nav.sig_p0 = 100.0
    nav.sig_v0 = 1.0
    nav.sig_ztd0 = 0.25
    nav.sig_ion0 = 1.0
    nav.sig_n0 = 30.0

    # Process noise sigma
    #
    if nav.pmode == 0:
        nav.sig_qp = 100.0/np.sqrt(1)  # [m/sqrt(s)]
        nav.sig_qv = None
    else:
        nav.sig_qp = 0.01/np.sqrt(1)  # [m/sqrt(s)]
        nav.sig_qv = 1.0/np.sqrt(1)  # [m/s/sqrt(s)]
    nav.sig_qztd = 0.1/np.sqrt(3600)  # [m/sqrt(s)] -> 1 cm**2/h
    nav.sig_qion = 1.0  # [m]

    nav.tidecorr = True
    nav.armode = 0  # 0:float-ppp,1:continuous,2:instantaneous,3:fix-and-hold
    nav.elmaskar = np.deg2rad(20.0)  # elevation mask for AR
    nav.elmin = np.deg2rad(10.0)

    # Initial state vector
    #
    nav.x[0:3] = pos0
    if nav.pmode >= 1:  # kinematic
        nav.x[3:6] = 0.0  # velocity

    # Diagonal elements of covariance matrix
    #
    dP = np.diag(nav.P)
    dP.flags['WRITEABLE'] = True

    dP[0:3] = nav.sig_p0**2
    # Velocity
    if nav.pmode >= 1:  # kinematic
        dP[3:6] = nav.sig_v0**2
    # Tropo delay
    if nav.pmode >= 1:  # kinematic
        dP[6] = nav.sig_ztd0**2
    else:
        dP[3] = nav.sig_ztd0**2

    # Process noise
    #
    nav.q = np.zeros(nav.nq)
    nav.q[0:3] = nav.sig_qp**2
    # Velocity
    if nav.pmode >= 1:  # kinematic
        nav.q[3:6] = nav.sig_qv**2
    # Tropo delay
    if nav.pmode >= 1:  # kinematic
        nav.q[6] = nav.sig_qztd**2
    else:
        nav.q[3] = nav.sig_qztd**2
    # Iono delay
    if nav.pmode >= 1:  # kinematic
        nav.q[7:7+gn.uGNSS.MAXSAT] = nav.sig_qion**2
    else:
        nav.q[4:4+gn.uGNSS.MAXSAT] = nav.sig_qion**2

    # Logging level
    #
    nav.fout = None
    if logfile is None:
        nav.monlevel = 0
    else:
        nav.fout = open(logfile, 'w')


def sysidx(satlist, sys_ref):
    """ return index of satellites with sys=sys_ref """
    idx = []
    for k, sat in enumerate(satlist):
        sys, _ = gn.sat2prn(sat)
        if sys == sys_ref:
            idx.append(k)
    return idx


def udstate(nav, obs):
    """ time propagation of states and initialize """

    tt = gn.timediff(obs.t, nav.t)

    ns = len(obs.sat)
    sys = []
    sat = obs.sat
    for sat_i in obs.sat:
        sys_i, _ = gn.sat2prn(sat_i)
        sys.append(sys_i)

    # pos,vel,ztd,ion,amb
    #
    nx = nav.nx
    ni = nav.na-gn.uGNSS.MAXSAT
    Phi = np.eye(nx)
    Phi[ni:nav.na, ni:nav.na] = np.zeros((gn.uGNSS.MAXSAT, gn.uGNSS.MAXSAT))
    if nav.pmode > 0:
        nav.x[0:3] += nav.x[3:6]*tt
        Phi[0:3, 3:6] = np.eye(3)*tt
    nav.P[0:nx, 0:nx] = Phi@nav.P[0:nx, 0:nx]@Phi.T

    # Process noise
    #
    dP = np.diag(nav.P)
    dP.flags['WRITEABLE'] = True
    dP[0:nav.nq] += nav.q[0:nav.nq]*tt

    # Carrier-phase ambiguity
    #
    for f in range(nav.nf):

        # Reset phase-ambiguity if instantaneous AR or expire obs outage counter
        #
        for i in range(gn.uGNSS.MAXSAT):

            sat_ = i+1
            nav.outc[i, f] += 1
            reset = (nav.outc[i, f] > nav.maxout)
            sys_i, _ = gn.sat2prn(sat_)
            if sys_i not in obs.sig.keys():
                continue

            # Reset ambiguity estimate
            #
            j = IB(sat_, f, nav.na)
            if reset and nav.x[j] != 0.0:
                initx(nav, 0.0, 0.0, j)
                nav.outc[i, f] = 0

                if nav.monlevel > 0:
                    nav.fout.write("{}  {} - reset ambiguity  {}\n"
                                   .format(time2str(obs.t), sat2id(sat_),
                                           obs.sig[sys_i][uTYP.L][f]))

            # Reset slant ionospheric delay estimate
            #
            j = II(sat_, nav.na)
            if reset and nav.x[j] != 0.0:
                initx(nav, 0.0, 0.0, j)

                if nav.monlevel > 0:
                    nav.fout.write("{}  {} - reset ionosphere\n"
                                   .format(time2str(obs.t), sat2id(sat_)))

        # Cycle  slip check by LLI
        #
        for i in range(ns):
            if sys[i] not in obs.sig.keys():
                continue
            if obs.lli[i, f] & 1 == 0:
                continue
            initx(nav, 0.0, 0.0, IB(sat[i], f, nav.na))

        # Ambiguity
        #
        bias = np.zeros(ns)
        ion = np.zeros(ns)

        offset = 0
        na = 0
        for i in range(ns):

            if sys[i] not in obs.sig.keys():
                continue

            # Get dual-frequency pseudoranges for this constellation
            #
            sig1 = obs.sig[sys[i]][uTYP.C][0]
            sig2 = obs.sig[sys[i]][uTYP.C][1]

            pr1 = obs.P[i, 0]
            pr2 = obs.P[i, 1]

            # Skip zero observations
            #
            if pr1 == 0.0 or pr2 == 0.0:
                continue

            # Get iono delay at frequency of first signal
            #
            ion[i] = ionoDelay(sig1, sig2, pr1, pr2)

            # Get pseudorange and carrier-phase observation of signal f
            #
            sig = obs.sig[sys[i]][uTYP.L][f]
            lam = sig.wavelength()

            cp = obs.L[i, f]
            pr = obs.P[i, f]
            if cp == 0.0 or pr == 0.0 or lam is None:
                continue

            bias[i] = cp - pr/lam + 2.0 * \
                (sig1.frequency()**2/sig.frequency()**2)*ion[i]/lam

            amb = nav.x[IB(sat[i], f, nav.na)]
            if amb != 0.0:
                offset += bias[i] - amb
                na += 1

        # Adjust phase-code coherency
        #
        if na > 0:
            db = offset/na
            for i in range(gn.uGNSS.MAXSAT):
                if nav.x[IB(i+1, f, nav.na)] != 0.0:
                    nav.x[IB(i+1, f, nav.na)] += db

        # Initialize ambiguity
        #
        for i in range(ns):

            sys_i, _ = sat2prn(sat[i])

            j = IB(sat[i], f, nav.na)
            if bias[i] != 0.0 and nav.x[j] == 0.0:

                initx(nav, bias[i], nav.sig_n0**2, j)

                if nav.monlevel > 0:
                    sig = obs.sig[sys_i][uTYP.L][f]
                    nav.fout.write("{}  {} - init  ambiguity  {} {:12.3f}\n"
                                   .format(time2str(obs.t), sat2id(sat[i]), sig, bias[i]))

            j = II(sat[i], nav.na)
            if ion[i] != 0 and nav.x[j] == 0.0:

                initx(nav, ion[i], nav.sig_ion0**2, j)

                if nav.monlevel > 0:
                    nav.fout.write("{}  {} - init  ionosphere      {:12.3f}\n"
                                   .format(time2str(obs.t), sat2id(sat[i]), ion[i]))

    return 0


def zdres(nav, obs, cs, bsx, rs, vs, dts, svh, rr):
    """ non-differential residual """

    _c = gn.rCST.CLIGHT
    ns2m = _c*1e-9

    nf = nav.nf
    n = len(obs.P)
    y = np.zeros((n, nf*2))
    el = np.zeros(n)
    e = np.zeros((n, 3))
    rr_ = rr.copy()

    # Solid Earth tide corrections
    #
    # TODO: add solid earth tide displacements
    #
    if nav.tidecorr:
        pos = gn.ecef2pos(rr_)
        disp = tidedisp(gn.gpst2utc(obs.t), pos)
    else:
        disp = np.zeros(3)
    rr_ += disp

    # Geodetic position
    #
    pos = gn.ecef2pos(rr_)

    # Zenith tropospheric dry and wet delays at user position
    #
    trop_hs, trop_wet, _ = gn.tropmodel(obs.t, pos)

    cpc = np.zeros((n, nf))
    prc = np.zeros((n, nf))

    for i in range(n):

        sat = obs.sat[i]
        sys, _ = gn.sat2prn(sat)

        if svh[i] > 0 or sys not in obs.sig.keys() or sat in nav.excl_sat:
            continue

        # Check for measurement consistency
        #
        flg_m = True
        for f in range(nav.nf):

            if obs.P[i, f] == 0.0 or obs.L[i, f] == 0.0 or obs.lli[i, f] == 1:
                flg_m = False

        # Check C/N0 at first frequency
        #
        if obs.S[i, 0] < nav.cnr_min:
            flg_m = False

        if flg_m is False:
            continue

        # Geometric distance corrected for Earth rotation during flight time
        #
        r, e[i, :] = gn.geodist(rs[i, :], rr_)
        _, el[i] = gn.satazel(pos, e[i, :])
        if el[i] < nav.elmin:
            continue

        # Pseudorange and carrier-phase signals
        #
        sigsPR = obs.sig[sys][gn.uTYP.C]
        sigsCP = obs.sig[sys][gn.uTYP.L]

        # Wavelength
        lam = np.array([s.wavelength() for s in sigsCP])

        if nav.ephopt == 4:
            # Code and phase signal bias, converted from [ns] to [m]
            #
            cbias = np.array([bsx.getosb(sat, obs.t, s)*ns2m for s in sigsPR])
            pbias = np.array([bsx.getosb(sat, obs.t, s)*ns2m for s in sigsCP])
        else:  # from CSSR
            idx_n = np.where(cs.sat_n == sat)[0][0]
            kidx = [-1]*nav.nf
            nsig = 0
            for k, sig in enumerate(cs.sig_n[idx_n]):
                for f in range(nav.nf):
                    if cs.cssrmode == 1 and sys == uGNSS.GPS and sig == sSigGPS.L2P:
                        sig = sSigGPS.L2W  # work-around
                    if ssig2rsig(sys, uTYP.C, sig) == sigsPR[f]:
                        kidx[f] = k
                        nsig += 1
            if nsig < nav.nf:
                continue

            cbias = np.zeros(nav.nf)
            pbias = np.zeros(nav.nf)
            if cs.lc[0].cstat & (1 << sCType.CBIAS) == (1 << sCType.CBIAS):
                cbias = cs.lc[0].cbias[idx_n][kidx]
            if cs.lc[0].cstat & (1 << sCType.PBIAS) == (1 << sCType.PBIAS):
                pbias = cs.lc[0].pbias[idx_n][kidx]
                if cs.cssrmode == 1:  # for Gal HAS (cycle -> m)
                    pbias *= lam
            # For Galileo HAS, switch the sign of the biases
            if cs.cssrmode == 1:
                cbias *= -1
                pbias *= -1

        # Shapipo relativistic effect
        #
        relatv = shapiro(rs[i, :], rr_)

        # Tropospheric delay mapping functions
        #
        mapfh, mapfw = gn.tropmapf(obs.t, pos, el[i])

        # Tropospheric delay
        #
        trop = mapfh*trop_hs + mapfw*trop_wet

        # Phase wind-up effect
        #
        nav.phw[sat-1] = windupcorr(obs.t, rs[i, :], vs[i, :], rr_,
                                    nav.phw[sat-1], full=True)

        # cycle -> m
        phw = lam*nav.phw[sat-1]

        # Select APC reference signals
        #
        if cs.cssrmode == 1:
            if sys == uGNSS.GPS:
                sig0 = (rSigRnx("GC1W"), rSigRnx("GC2W"))
            elif sys == uGNSS.GAL:
                sig0 = (rSigRnx("EC1C"), rSigRnx("EC7Q"))
            else:
                sig0 = None
        else:
            sig0 = None

        # Receiver antenna offset
        #
        antrPR = antModelRx(nav, pos, e[i, :], sigsPR)
        antrCP = antModelRx(nav, pos, e[i, :], sigsCP)
        antsPR = antModelTx(nav, e[i, :], sigsPR, sat, obs.t, rs[i, :], sig0)
        antsCP = antModelTx(nav, e[i, :], sigsCP, sat, obs.t, rs[i, :], sig0)

        # Range correction
        #
        prc[i, :] = trop + antrPR + antsPR + cbias
        cpc[i, :] = trop + antrCP + antsCP + pbias + phw

        r += relatv - _c*dts[i]

        for f in range(nf):
            y[i, f] = obs.L[i, f]*lam[f]-(r+cpc[i, f])
            y[i, f+nf] = obs.P[i, f]-(r+prc[i, f])

    return y, e, el


def sdres(nav, obs, x, y, e, sat, el):
    """
    SD phase/code residuals

    Parameters
    ----------

    nav : Nav()
        Auxiliary data structure
    obs : Obs()
        Data structure with observations
    x   :
        State vector elements
    y   :
        Un-differenced corrected observations
    e   :
        Line-of-sight vectors
    sat : np.array of int
        List of satellites
    el  : np.array of float values
        Elevation angles

    Returns
    -------
    v   : np.array of float values
        Residuals of single-difference measurements
    H   : np.array of float values
        Jacobian matrix with partial derivatives of state variables
    R   : np.array of float values
        Covariance matrix of single-difference measurements
    """

    nf = nav.nf
    ns = len(el)

    nb = np.zeros(2*len(obs.sig.keys())*nf, dtype=int)

    Ri = np.zeros(ns*nf*2)
    Rj = np.zeros(ns*nf*2)

    nv = 0
    b = 0

    H = np.zeros((ns*nf*2, nav.nx))
    v = np.zeros(ns*nf*2)

    # Geodetic position
    #
    pos = gn.ecef2pos(x[0:3])

    # Loop over constellations
    #
    for sys in obs.sig.keys():

        # Slant ionospheric delay reference frequency
        #
        freq0 = obs.sig[sys][uTYP.L][0].frequency()

        # Loop over twice the number of frequencies
        #   first for all carrier-phase observations
        #   second all pseudorange observations
        #
        for f in range(0, nf*2):

            # Select carrier-phase frequency and iono frequency ratio
            #
            if f < nf:  # carrier
                sig = obs.sig[sys][uTYP.L][f]
                mu = -(freq0/sig.frequency())**2
            else:  # code
                sig = obs.sig[sys][uTYP.C][f % 2]
                mu = +(freq0/sig.frequency())**2

            # Select satellites from one constellation only
            #
            idx = sysidx(sat, sys)

            # Select reference satellite with highest elevation
            #
            if len(idx) > 0:
                i = idx[np.argmax(el[idx])]

                if nav.monlevel > 3:
                    nav.fout.write("{} prn0 {:3s}\n"
                                   .format(time2str(obs.t), sat2id(sat[i])))

            # Loop over satellites
            #
            for j in idx:

                # Skip reference satellite i
                #
                if i == j:
                    continue

                # Skip invalid signals
                #
                if y[i, f] == 0.0 or y[j, f] == 0.0:
                    continue

                #  Single-difference measurement
                #
                v[nv] = y[i, f] - y[j, f]

                # SD line-of-sight vectors
                #
                H[nv, 0:3] = -e[i, :] + e[j, :]

                # SD troposphere
                #
                _, mapfwi = gn.tropmapf(obs.t, pos, el[i])
                _, mapfwj = gn.tropmapf(obs.t, pos, el[j])

                idx_i = IT(nav.na)
                H[nv, idx_i] = mapfwi - mapfwj
                v[nv] -= (mapfwi - mapfwj)*x[idx_i]

                if nav.monlevel > 2:
                    nav.fout.write("{}         ztd ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f}\n"
                                   .format(time2str(obs.t),
                                           idx_i, idx_i,
                                           (mapfwi-mapfwj),
                                           x[IT(nav.na)],
                                           np.sqrt(nav.P[IT(nav.na), IT(nav.na)])))

                # SD ionosphere
                #
                idx_i = II(sat[i], nav.na)
                idx_j = II(sat[j], nav.na)
                H[nv, idx_i] = +mu
                H[nv, idx_j] = -mu
                v[nv] -= mu*(x[idx_i] - x[idx_j])

                if nav.monlevel > 2:
                    nav.fout.write("{} {}-{} ion ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}\n"
                                   .format(time2str(obs.t),
                                           sat2id(sat[i]), sat2id(sat[j]),
                                           idx_i, idx_j,
                                           mu,
                                           x[idx_i], x[idx_j],
                                           np.sqrt(nav.P[idx_i, idx_i]),
                                           np.sqrt(nav.P[idx_j, idx_j])))

                # SD ambiguity
                #
                if f < nf:  # carrier-phase

                    idx_i = IB(sat[i], f, nav.na)
                    idx_j = IB(sat[j], f, nav.na)

                    lami = sig.wavelength()

                    v[nv] -= lami*(x[idx_i]-x[idx_j])

                    H[nv, idx_i] = +lami
                    H[nv, idx_j] = -lami

                    Ri[nv] = varerr(nav, el[i], f)  # measurement variance
                    Rj[nv] = varerr(nav, el[j], f)  # measurement variance

                    nav.vsat[sat[i]-1, f] = 1
                    nav.vsat[sat[j]-1, f] = 1

                    if nav.monlevel > 2:
                        nav.fout.write("{} {}-{} amb ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}\n"
                                       .format(time2str(obs.t),
                                               sat2id(sat[i]), sat2id(sat[j]),
                                               idx_i, idx_j,
                                               lami,
                                               x[idx_i], x[idx_j],
                                               np.sqrt(nav.P[idx_i, idx_i]),
                                               np.sqrt(nav.P[idx_j, idx_j])))

                else:  # pseudorange

                    Ri[nv] = varerr(nav, el[i], f)  # measurement variance
                    Rj[nv] = varerr(nav, el[j], f)  # measurement variance

                if nav.monlevel > 1:
                    nav.fout.write("{} {}-{} ({:2d}) {} res {:10.3f} sig_i {:10.3f} sig_j {:10.3f}\n"
                                   .format(time2str(obs.t),
                                           sat2id(sat[i]), sat2id(sat[j]),
                                           nv, sig.str(),
                                           v[nv], np.sqrt(Ri[nv]), np.sqrt(Rj[nv])))

                nb[b] += 1
                nv += 1  # counter for single-difference observations

            b += 1  # counter for satellite

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


def ppppos(nav, obs, cs=None, orb=None, bsx=None):
    """
    PPP positioning
    """

    # Skip empty epochs
    #
    if len(obs.sat) == 0:
        return

    # GNSS satellite positions, velocities and clock offsets
    #
    rs, vs, dts, svh = satposs(obs, nav, cs, orb)

    # Kalman filter time propagation, initialization of ambiguities and iono
    #
    udstate(nav, obs)

    #xa = np.zeros(nav.nx)
    xp = nav.x.copy()

    # Non-differential residuals
    #
    yu, eu, elu = zdres(nav, obs, cs, bsx, rs, vs, dts, svh, xp[0:3])

    # Select satellites above minimum elevation
    #
    iu = np.where(elu >= nav.elmin)[0]
    sat = obs.sat[iu]
    y = yu[iu, :]
    e = eu[iu, :]
    el = elu[iu]

    # Store reduced satellite list
    #
    nav.sat = sat
    nav.el[sat-1] = el
    nav.y = y
    ns = len(sat)

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
    v, H, R = sdres(nav, obs, xp, y, e, sat, el)
    Pp = nav.P.copy()

    # Kalman filter measurement update
    #
    xp, Pp, _ = kfupdate(xp, Pp, H, v, R)

    # Non-differential residuals after measurement update
    #
    yu, eu, elu = zdres(nav, obs, cs, bsx, rs, vs, dts, svh, xp[0:3])
    y = yu[iu, :]
    e = eu[iu, :]
    ny = y.shape[0]
    if ny < 6:
        return -1

    # Residuals for float solution
    #
    v, H, R = sdres(nav, obs, xp, y, e, sat, el)
    if valpos(nav, v, R):
        nav.x = xp
        nav.P = Pp
        nav.ns = 0
        for i in range(ns):
            j = sat[i]-1
            for f in range(nav.nf):
                if nav.vsat[j, f] == 0:
                    continue
                nav.outc[j, f] = 0
                if f == 0:
                    nav.ns += 1
    else:
        nav.smode = 0

    nav.smode = 5  # 4: fixed ambiguities, 5: float ambiguities

    if nav.armode > 0:
        nb, xa = resamb_lambda(nav, sat)
        if nb > 0:
            yu, eu, elu = zdres(nav, obs, cs, bsx, rs, vs, dts, svh, xa[0:3])
            y = yu[iu, :]
            e = eu[iu, :]
            v, H, R = sdres(nav, obs, xa, y, e, sat, el)
            # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
            if valpos(nav, v, R):
                if nav.armode == 3:     # fix and hold
                    holdamb(nav, xa)    # hold fixed ambiguity
                nav.smode = 4           # fix

    # Store epoch for solution
    #
    nav.t = obs.t

    return 0
