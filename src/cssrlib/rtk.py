"""
module for RTK positioning

"""

import numpy as np

import cssrlib.gnss as gn
from cssrlib.ephemeris import satposs
from cssrlib.gnss import uTYP
from cssrlib.peph import antModelRx
from cssrlib.ppp import tidedisp
from cssrlib.mlambda import mlambda

VAR_HOLDAMB = 0.001


def rtkinit(nav, pos0=np.zeros(3), logfile=None):
    """ initalize RTK-GNSS parameters """
    nav.pmode = 1  # 0:static, 1:kinematic
    nav.monlevel = 1

    nav.na = 3 if nav.pmode == 0 else 6
    nav.nq = 3 if nav.pmode == 0 else 6

    nav.thresar = 2.0
    nav.nx = nav.na+gn.uGNSS.MAXSAT*nav.nf
    nav.x = np.zeros(nav.nx)
    nav.P = np.zeros((nav.nx, nav.nx))
    nav.xa = np.zeros(nav.na)
    nav.Pa = np.zeros((nav.na, nav.na))
    nav.el = np.zeros(gn.uGNSS.MAXSAT)

    # parameter for RTK
    nav.eratio = [50, 50]
    nav.err = [0, 0.01, 0.005]/np.sqrt(2)
    nav.sig_p0 = 30.0
    nav.sig_v0 = 1.0
    nav.sig_n0 = 30.0
    nav.sig_qp = 0.01
    nav.sig_qv = 1.0

    nav.armode = 1  # 1:continuous,2:instantaneous,3:fix-and-hold
    nav.elmaskar = np.deg2rad(20)  # elevation mask for AR
    nav.x[0:3] = pos0
    nav.x[3:6] = 0.0

    dP = np.diag(nav.P)
    dP.flags['WRITEABLE'] = True
    dP[0:3] = nav.sig_p0**2
    nav.q = np.zeros(nav.nq)
    if nav.pmode >= 1:  # kinematic
        dP[3:6] = nav.sig_v0**2
        nav.q[0:3] = nav.sig_qp**2
        nav.q[3:6] = nav.sig_qv**2
    else:
        nav.q[0:3] = nav.sig_qp**2

    # Logging level
    #
    nav.fout = None
    if logfile is None:
        nav.monlevel = 0
    else:
        nav.fout = open(logfile, 'w')


def zdres(nav, obs, rs, dts, svh, rr, rtype=1):
    """ non-differencial residual """
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
    for i in range(n):
        sys, _ = gn.sat2prn(obs.sat[i])
        if svh[i] > 0 or sys not in obs.sig.keys() or obs.sat[i] in nav.excl_sat:
            continue
        r, e[i, :] = gn.geodist(rs[i, :], rr_)
        _, el[i] = gn.satazel(pos, e[i, :])
        if el[i] < nav.elmin:
            continue
        r += -_c*dts[i]
        zhd, _, _ = gn.tropmodel(obs.t, pos, np.deg2rad(90.0), 0.0)
        mapfh, _ = gn.tropmapf(obs.t, pos, el[i])
        r += mapfh*zhd

        sigPR = obs.sig[sys][gn.uTYP.C]
        sigCP = obs.sig[sys][gn.uTYP.L]

        dantPR = antModelRx(nav, pos, e[i, :], sigPR, rtype)
        dantCP = antModelRx(nav, pos, e[i, :], sigCP, rtype)

        lam = [s.wavelength() for s in sigCP]

        for f in range(nf):
            if obs.L[i, f] == 0.0:
                y[i, f] = 0.0
            else:
                y[i, f] = obs.L[i, f]*lam[f]-r-dantCP[f]
            if obs.P[i, f] == 0.0:
                y[i, f+nf] = 0.0
            else:
                y[i, f+nf] = obs.P[i, f]-r-dantPR[f]
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


def sysidx(satlist, sys_ref):
    """ return index of satellites with sys=sys_ref """
    idx = []
    for k, sat in enumerate(satlist):
        sys, _ = gn.sat2prn(sat)
        if sys == sys_ref:
            idx.append(k)
    return idx


def IB(s, f, na=3):
    """ return index of phase ambguity """
    idx = na+gn.uGNSS.MAXSAT*f+s-1
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


def ddres(nav, obs, x, y, e, sat, el):
    """ DD phase/code residual """
    _c = gn.rCST.CLIGHT
    nf = nav.nf
    ns = len(el)
    mode = 1 if len(y) == ns else 0  # 0:DD,1:SD
    nb = np.zeros(2*len(obs.sig.keys())*nf, dtype=int)
    Ri = np.zeros(ns*nf*2)
    Rj = np.zeros(ns*nf*2)

    nv = 0
    b = 0
    H = np.zeros((ns*nf*2, nav.nx))
    v = np.zeros(ns*nf*2)

    for sys in obs.sig.keys():
        for f in range(0, nf*2):
            # reference satellite
            idx = sysidx(sat, sys)
            if len(idx) > 0:
                i = idx[np.argmax(el[idx])]
            for j in idx:
                if i == j:
                    continue
                if y[i, f] == 0.0 or y[j, f] == 0.0:
                    continue
                #  DD residual
                if mode == 0:
                    if y[i+ns, f] == 0.0 or y[j+ns, f] == 0.0:
                        continue
                    v[nv] = (y[i, f]-y[i+ns, f])-(y[j, f]-y[j+ns, f])
                else:
                    v[nv] = y[i, f]-y[j, f]
                H[nv, 0:3] = -e[i, :]+e[j, :]
                if f < nf:  # carrier
                    idx_i = IB(sat[i], f, nav.na)
                    idx_j = IB(sat[j], f, nav.na)
                    lami = obs.sig[sys][gn.uTYP.L][f].wavelength()
                    v[nv] -= lami*(x[idx_i]-x[idx_j])
                    H[nv, idx_i] = lami
                    H[nv, idx_j] = -lami
                    Ri[nv] = varerr(nav, el[i], f)
                    Rj[nv] = varerr(nav, el[j], f)
                    nav.vsat[sat[i]-1, f] = 1
                    nav.vsat[sat[j]-1, f] = 1
                else:
                    Ri[nv] = varerr(nav, el[i], f)
                    Rj[nv] = varerr(nav, el[j], f)
                nb[b] += 1
                nv += 1
            b += 1
    v = np.resize(v, nv)
    H = np.resize(H, (nv, nav.nx))
    R = ddcov(nb, b, Ri, Rj, nv)

    return v, H, R


def valpos(nav, v, R, thres=4.0):
    """ post-fit residual test """
    nv = len(v)
    fact = thres**2
    for i in range(nv):
        if v[i]**2 <= fact*R[i, i]:
            continue
        if nav.monlevel > 1:
            txt = "{:3d} is large: {:8.4f} ({:8.4f})".format(i, v[i], R[i, i])
            if nav.fout is None:
                print(txt)
            else:
                nav.fout.write(txt+"\n")
    return True


def ddidx(nav, sat):
    """ index for SD to DD transformation matrix D """
    nb = 0
    n = gn.uGNSS.MAXSAT
    na = nav.na
    ix = np.zeros((n, 2), dtype=int)
    nav.fix = np.zeros((n, nav.nf), dtype=int)
    for m in range(gn.uGNSS.GNSSMAX):
        k = na
        for f in range(nav.nf):
            for i in range(k, k+n):
                sat_i = i-k+1
                sys, _ = gn.sat2prn(sat_i)
                if (sys != m):
                    continue
                if sat_i not in sat or nav.x[i] == 0.0 \
                   or nav.vsat[sat_i-1, f] == 0:
                    continue
                if nav.el[sat_i-1] >= nav.elmaskar:
                    nav.fix[sat_i-1, f] = 2
                    break
                else:
                    nav.fix[sat_i-1, f] = 1
            for j in range(k, k+n):
                sat_j = j-k+1
                sys, _ = gn.sat2prn(sat_j)
                if (sys != m):
                    continue
                if i == j or sat_j not in sat or nav.x[j] == 0.0 \
                   or nav.vsat[sat_j-1, f] == 0:
                    continue
                if nav.el[sat_j-1] >= nav.elmaskar:
                    ix[nb, :] = [i, j]
                    nb += 1
                    nav.fix[sat_j-1, f] = 2
            k += n
    ix = np.resize(ix, (nb, 2))
    return ix


def restamb(nav, bias, nb):
    """ restore SD ambiguity """
    nv = 0
    xa = nav.x.copy()
    xa[0:nav.na] = nav.xa[0:nav.na]

    for m in range(gn.uGNSS.GNSSMAX):
        for f in range(nav.nf):
            n = 0
            index = []
            for i in range(gn.uGNSS.MAXSAT):
                sys, _ = gn.sat2prn(i+1)
                if sys != m or nav.fix[i, f] != 2:
                    continue
                index.append(IB(i+1, f, nav.na))
                n += 1
            if n < 2:
                continue
            xa[index[0]] = nav.x[index[0]]
            for i in range(1, n):
                xa[index[i]] = xa[index[0]]-bias[nv]
                nv += 1
    return xa


def resamb_lambda(nav, sat):
    """ resolve integer ambiguity using LAMBDA method """
    nx = nav.nx
    na = nav.na
    xa = np.zeros(na)
    ix = ddidx(nav, sat)
    nb = len(ix)
    if nb <= 0:
        print("no valid DD")
        return -1, -1

    # y=D*xc, Qb=D*Qc*D', Qab=Qac*D'
    y = nav.x[ix[:, 0]]-nav.x[ix[:, 1]]
    DP = nav.P[ix[:, 0], na:nx]-nav.P[ix[:, 1], na:nx]
    Qb = DP[:, ix[:, 0]-na]-DP[:, ix[:, 1]-na]
    Qab = nav.P[0:na, ix[:, 0]]-nav.P[0:na, ix[:, 1]]

    # MLAMBDA ILS
    b, s = mlambda(y, Qb)
    if s[0] <= 0.0 or s[1]/s[0] >= nav.thresar:
        nav.xa = nav.x[0:na].copy()
        nav.Pa = nav.P[0:na, 0:na].copy()
        bias = b[:, 0]
        y -= b[:, 0]
        K = Qab@np.linalg.inv(Qb)
        nav.xa -= K@y
        nav.Pa -= K@Qab.T

        # restore SD ambiguity
        xa = restamb(nav, bias, nb)
    else:
        nb = 0

    return nb, xa


def initx(nav, x0, v0, i):
    """ initialize x and P for index i """
    nav.x[i] = x0
    for j in range(nav.nx):
        nav.P[j, i] = nav.P[i, j] = v0 if i == j else 0


def kfupdate(x, P, H, v, R):
    """ Kalman filter measurement update """
    PHt = P@H.T
    S = H@PHt+R
    K = PHt@np.linalg.inv(S)
    x += K@v
    P = P - K@H@P

    return x, P, S


def udstate(nav, obs, obsb, iu, ir):
    """ states propagation for kalman filter """
    tt = gn.timediff(obs.t, nav.t)
    ns = len(iu)
    sys = []
    sat = obs.sat[iu]
    for sat_i in obs.sat[iu]:
        sys_i, _ = gn.sat2prn(sat_i)
        sys.append(sys_i)

    # pos,vel
    na = nav.na
    Phi = np.eye(nav.nx)
    if nav.na > 3:
        nav.x[0:3] += tt*nav.x[3:6]
        Phi[0:3, 3:6] = np.eye(3)*tt
    nav.P = Phi@nav.P@Phi.T
    dP = np.diag(nav.P)
    dP.flags['WRITEABLE'] = True
    dP[0:nav.nq] += nav.q[0:nav.nq]*tt

    # bias
    for f in range(nav.nf):
        # reset phase-bias if instantaneous AR or
        # expire obs outage counter
        for i in range(gn.uGNSS.MAXSAT):
            nav.outc[i, f] += 1
            reset = (nav.outc[i, f] > nav.maxout)
            sys_i, _ = gn.sat2prn(i+1)
            if sys_i not in obs.sig.keys():
                continue
            j = IB(i+1, f, nav.na)
            if reset and nav.x[j] != 0.0:
                initx(nav, 0.0, 0.0, j)
                nav.outc[i, f] = 0
        # cycle slip check by LLI
        for i in range(ns):
            if sys[i] not in obs.sig.keys():
                continue
            if obsb.lli[ir[i], f] & 1 == 0 and obs.lli[iu[i], f] & 1 == 0:
                continue
            initx(nav, 0.0, 0.0, IB(sat[i], f, nav.na))
        # bias
        bias = np.zeros(ns)
        offset = 0
        na = 0
        for i in range(ns):
            if sys[i] not in obs.sig.keys():
                continue
            lam = obs.sig[sys[i]][uTYP.C][f].wavelength()
            cp = obs.L[iu[i], f]-obsb.L[ir[i], f]
            pr = obs.P[iu[i], f]-obsb.P[ir[i], f]
            bias[i] = cp-pr/lam
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


def selsat(nav, obs, obsb, elb):
    """ select common satellite between rover and base station """
    # exclude satellite with missing observation and cycle slip for rover
    idx_u = []
    for k, _ in enumerate(obs.sat):
        if obs.P[k, 0] == 0.0 or obs.P[k, 1] == 0.0 or \
           obs.L[k, 0] == 0.0 or obs.L[k, 1] == 0.0 or \
           obs.lli[k, 0] > 0 or obs.lli[k, 1] > 0:
            continue
        idx_u.append(k)

    # exclude satellite with missing observation and cycle slip for base
    idx_r = []
    for k, _ in enumerate(obsb.sat):
        if obsb.P[k, 0] == 0.0 or obsb.P[k, 1] == 0.0 or \
           obsb.L[k, 0] == 0.0 or obsb.L[k, 1] == 0.0 or \
           obsb.lli[k, 0] > 0 or obsb.lli[k, 1] > 0 or \
           elb[k] < nav.elmin:
            continue
        idx_r.append(k)

    idx = np.intersect1d(obs.sat[idx_u], obsb.sat[idx_r], return_indices=True)
    k = len(idx[0])
    iu = np.array(idx_u)[idx[1]]
    ir = np.array(idx_r)[idx[2]]
    return k, iu, ir


def holdamb(nav, xa):
    """ hold integer ambiguity """
    nb = nav.nx-nav.na
    v = np.zeros(nb)
    H = np.zeros((nb, nav.nx))
    nv = 0
    for m in range(gn.uGNSS.GNSSMAX):
        for f in range(nav.nf):
            n = 0
            index = []
            for i in range(gn.uGNSS.MAXSAT):
                sys, _ = gn.sat2prn(i+1)
                if sys != m or nav.fix[i, f] != 2:
                    continue
                index.append(IB(i+1, f, nav.na))
                n += 1
                nav.fix[i, f] = 3  # hold
            # constraint to fixed ambiguity
            for i in range(1, n):
                v[nv] = (xa[index[0]]-xa[index[i]]) - \
                    (nav.x[index[0]]-nav.x[index[i]])
                H[nv, index[0]] = 1.0
                H[nv, index[i]] = -1.0
                nv += 1
    if nv > 0:
        R = np.eye(nv)*VAR_HOLDAMB
        # update states with constraints
        nav.x, nav.P, _ = kfupdate(nav.x, nav.P, H[0:nv, :], v[0:nv], R)
    return 0


def relpos(nav, obs, obsb):
    """ relative positioning for RTK-GNSS """
    nf = nav.nf
    if gn.timediff(obs.t, obsb.t) != 0:
        return -1

    rs, _, dts, svh, _ = satposs(obs, nav)
    rsb, _, dtsb, svhb, _ = satposs(obsb, nav)

    # non-differencial residual for base
    yr, er, elr = zdres(nav, obsb, rsb, dtsb, svhb, nav.rb, 0)
    ns, iu, ir = selsat(nav, obs, obsb, elr)
    y = np.zeros((ns*2, nf*2))
    e = np.zeros((ns*2, 3))

    if ns < 4:
        return -1

    y[ns:, :] = yr[ir, :]
    e[ns:, :] = er[ir, :]

    # Kalman filter time propagation
    udstate(nav, obs, obsb, iu, ir)

    #xa = np.zeros(nav.nx)
    xp = nav.x

    # non-differencial residual for rover
    yu, eu, el = zdres(nav, obs, rs, dts, svh, xp[0:3])

    y[:ns, :] = yu[iu, :]
    e[:ns, :] = eu[iu, :]
    el = el[iu]
    sat = obs.sat[iu]
    nav.el[sat-1] = el
    # DD residual
    v, H, R = ddres(nav, obs, xp, y, e, sat, el)
    Pp = nav.P

    # Kalman filter measurement update
    xp, Pp, _ = kfupdate(xp, Pp, H, v, R)

    # non-differencial residual for rover after measurement update
    yu, eu, _ = zdres(nav, obs, rs, dts, svh, xp[0:3])
    y[:ns, :] = yu[iu, :]
    e[:ns, :] = eu[iu, :]
    # residual for float solution
    v, H, R = ddres(nav, obs, xp, y, e, sat, el)
    if valpos(nav, v, R):
        nav.x = xp
        nav.P = Pp
    else:
        nav.smode = 0

    nb, xa = resamb_lambda(nav, sat)
    nav.smode = 5  # float
    if nb > 0:
        yu, eu, _ = zdres(nav, obs, rs, dts, svh, xa[0:3])
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        v, H, R = ddres(nav, obs, xa, y, e, sat, el)
        if valpos(nav, v, R):
            if nav.armode == 3:
                holdamb(nav, xa)
            nav.smode = 4  # fix
    nav.t = obs.t
    return 0
