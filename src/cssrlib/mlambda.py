"""
integer ambiguity resolution by LAMBDA

reference :
     [1] P.J.G.Teunissen, The least-square ambiguity decorrelation adjustment:
         a method for fast GPS ambiguity estimation, J.Geodesy, Vol.70, 65-82,
         1995
     [2] X.-W.Chang, X.Yang, T.Zhou, MLAMBDA: A modified LAMBDA method for
         integer least-squares estimation, J.Geodesy, Vol.79, 552-565, 2005

"""

import numpy as np
from scipy.stats import norm
from numpy.linalg import inv


def ldldecom(Q):
    n = len(Q)
    L = np.zeros((n, n))
    d = np.zeros(n)
    A = Q.copy()
    for i in range(n-1, -1, -1):
        d[i] = A[i, i]
        if d[i] <= 0.0:
            raise SystemExit("Qah should be positive definite.")
        L[i, :i+1] = A[i, :i+1]/np.sqrt(d[i])
        for j in range(i):
            A[j, :j+1] -= L[i, :j+1]*L[i, j]
        L[i, :i+1] /= L[i, i]

    return L, d


def reduction(L, d):
    n = len(d)
    Z = np.eye(n)
    j = n-2
    k = n-2
    while j >= 0:
        if j <= k:
            for i in range(j+1, n):
                mu = np.round(L[i, j])
                if mu != 0.0:
                    L[i:, j] -= mu*L[i:, i]
                    Z[:, j] -= mu*Z[:, i]
                # L,Z=gauss(L,Z,i,j)
        delta = d[j]+L[j+1, j]**2*d[j+1]
        if delta+1e-6 < d[j+1]:  # permutation
            eta = d[j]/delta
            lam = d[j+1]*L[j+1, j]/delta
            d[j] = eta*d[j+1]
            d[j+1] = delta
            L[j:j+2, :j] = np.array([[-L[j+1, j], 1], [eta, lam]])@L[j:j+2, :j]
            L[j+1, j] = lam
            # swap j,j+1 row
            tmp = L[j+2:, j+1].copy()
            L[j+2:, j+1] = L[j+2:, j].copy()
            L[j+2:, j] = tmp
            tmp = Z[:, j+1].copy()
            Z[:, j+1] = Z[:, j].copy()
            Z[:, j] = tmp

            k = j
            j = n-2
        else:
            j -= 1
    return L, d, Z


def sr_boost(d):
    """ Compute the bootstrapped success rate """
    Ps = np.prod(2*norm.cdf(0.5/np.sqrt(d))-1)
    return Ps


def msearch(L, d, zs, m=2):
    n = len(d)
    nn = 0
    imax = 0
    Chi2 = 1e18
    S = np.zeros((n, n))
    dist = np.zeros(n)
    zb = np.zeros(n)
    z = np.zeros(n)
    step = np.zeros(n)
    zn = np.zeros((n, m))
    s = np.zeros(m)
    k = n-1
    zb[-1] = zs[-1]
    z[-1] = round(zb[-1])
    y = zb[-1]-z[-1]
    step[-1] = np.sign(y)
    if step[-1] == 0:
        step[-1] = 1
    for _ in range(10000):
        newdist = dist[k]+y**2/d[k]
        if newdist < Chi2:
            if k != 0:
                k -= 1
                dist[k] = newdist
                S[k, :k+1] = S[k+1, :k+1]+(z[k+1]-zb[k+1])*L[k+1, :k+1]
                zb[k] = zs[k]+S[k, k]
                z[k] = round(zb[k])
                y = zb[k]-z[k]
                step[k] = np.sign(y)
                if step[k] == 0:
                    step[k] = 1
            else:
                if nn < m:
                    if nn == 0 or newdist > s[imax]:
                        imax = nn
                    zn[:, nn] = z
                    s[nn] = newdist
                    nn += 1
                else:
                    if newdist < s[imax]:
                        zn[:, imax] = z
                        s[imax] = newdist
                        imax = np.argmax(s)
                    Chi2 = s[imax]
                z[0] += step[0]
                y = zb[0]-z[0]
                step[0] = -step[0]-np.sign(step[0])
        else:
            if k == n-1:
                break
            k += 1
            z[k] += step[k]
            y = zb[k]-z[k]
            step[k] = -step[k]-np.sign(step[k])

    order = np.argsort(s)
    s = s[order]
    zn = zn[:, order]

    return zn, s


def parsearch(zhat, Qzhat, Z, L, d, P0=0.995, ncands=2):
    """ Partial Ambiguity Resolution """
    n = len(Qzhat)
    Ps = sr_boost(d)

    k = 0
    while Ps < P0 and k < (n-1):
        k += 1

        # Compute the bootstrapped success rate if the last n-k+1 ambiguities
        # would be fixed
        Ps = sr_boost(d[k:])

    if Ps > P0:

        zpar, sqnorm = msearch(L[k:, k:], d[k:], zhat[k:], ncands)

        Qzpar = Qzhat[k:, k:]
        Zpar = Z[:, k:]

        # First k-1 ambiguities are adjusted based on correlation with the
        # fixed ambiguities
        QP = Qzhat[:k, k:]@inv(Qzhat[k:, k:])

        zfix = np.zeros((k, ncands))
        for i in range(ncands):
            zfix[:, i] = zhat[:k] - QP@(zhat[k:]-zpar[:, i])

        zfix = np.concatenate((zfix, zpar), axis=0)
        nfix = n-k

    else:

        zpar = []
        Qzpar = []
        Zpar = []
        sqnorm = []
        Ps = np.nan
        zfix = zhat
        nfix = 0

    return zpar, sqnorm, Qzpar, Zpar, Ps, nfix, zfix


def mlambda(ahat, Qahat, ncands=2, armode=1, P0=0.995):
    """ modified LAMBDA method for ambiguity resolution """
    L, d = ldldecom(Qahat)
    L, d, Z = reduction(L, d)
    iZt = np.round(inv(Z.T))
    zhat = Z.T@ahat

    if armode == 1:
        zfix, s = msearch(L, d, zhat, ncands)
        Ps, nfix = np.nan, len(zhat)
    elif armode == 2:  # PAR
        Qzhat = Z.T@Qahat@Z
        zpar, s, Qzpar, Zpar, Ps, nfix, zfix = parsearch(zhat, Qzhat, Z, L, d,
                                                         P0, ncands)

    afix_ = iZt@zfix
    return afix_, s, nfix, Ps


if __name__ == '__main__':
    ncase = 1

    if ncase == 1:
        Qah = np.array([[6.2900, 5.9780, 0.5440], [
                       5.9780, 6.2920, 2.3400], [0.5440, 2.3400, 6.2880]])
        ah = np.array([5.45, 3.10, 2.97])
    elif ncase == 2:
        Qah = [[19068.8559508787,	-15783.9722820370,	-17334.2005875975,
                14411.9239749603,	10055.7170089359,	-14259.2952903872,
                14858.8484050976,	-12299.1993741839,	-13507.1694819930,
                11230.0704356810,	7835.62344938376,	-11111.1393808147],
               [-15783.9722820370,	59027.7038409815,	38142.6927531102,
                .717388024645,	-13830.0855960676,	27373.4263013019,
                -12299.1993747356,	45995.6129934030,	29721.5785731468,
                438.480887460148,	-10776.6902686912,	21329.9423774758],
               [-17334.2005875975,	38142.6927531102,	28177.5653893528,
                -7000.50220497045,	-11695.8674059306,	21886.1680630532,
                -13507.1694826246,	29721.5785738846,	21956.5440705992,
                -5454.93697674992,	-9113.66310734779,	17054.1567378091],
               [14411.9239749603,	562.717388024645,	-7000.50220497045,
                15605.5082283690,	5039.70281815470,	-9648.96530646004,
                11230.0704356773,	438.480887731461,	-5454.93697653627,
                12160.1358938811,	3927.04096307733,	-7518.67445855756],
               [10055.7170089359,	-13830.0855960676,	-11695.8674059306,
                5039.70281815470,	6820.77250679480,	-6880.24051213224,
               7835.62344947055,	-10776.6902682086,	-9113.66310687634,
               3927.04096320258,	5314.88728015545,	-5361.22656658847],
               [-14259.2952903872,	27373.4263013019,	21886.1680630532,
                -9648.96530646004,	-6880.24051213224,	23246.5489626945,
                -11111.1393809211,	21329.9423779274,	17054.1567375591,
                -7518.67445829957,	-5361.22656681708,	18114.1936088811],
               [14858.8484050976,	-12299.1993747356,	-13507.1694826246,
                11230.0704356773,	7835.62344947055,	-11111.1393809211,
               11578.3237340013,	-9583.79156943782,	-10525.0669778554,
               8750.70438611838,	6105.68076067050,	-8658.03053539344],
               [-12299.1993741839,	45995.6129934030,	29721.5785738846,
                438.480887731461,	-10776.6902682086,	21329.9423779274,
                -9583.79156943782,	35840.7376978353,	23159.6717654859,
                341.673569568934,	-8397.42083743563,	16620.7344703582],
               [-13507.1694819930,	29721.5785731468,	21956.5440705992,
                -5454.93697653627,	-9113.66310687634,	17054.1567375591,
                -10525.0669778554,	23159.6717654859,	17108.9956804894,
                -4250.60009053988,	-7101.55551676305,	13288.9534523001],
               [11230.0704356810,	438.480887460148,	-5454.93697674992,
                12160.1358938811,	3927.04096320258,	-7518.67445829957,
               8750.70438611838,	341.673569568934,	-4250.60009053988,
               9475.43086798586,	3060.03207008500,	-5858.70721928591],
               [7835.62344938376,	-10776.6902686912,	-9113.66310734779,
                3927.04096307733,	5314.88728015545,	-5361.22656681708,
               6105.68076067050,	-8397.42083743563,	-7101.55551676305,
               3060.03207008500,	4141.47090961885,	-4177.57899193454],
               [-11111.1393808147,	21329.9423774758,	17054.1567378091,
                -7518.67445855756,	-5361.22656658847,	18114.1936088811,
                -8658.03053539344,	16620.7344703582,	13288.9534523001,
                -5858.70721928591,	-4177.57899193454,	14114.9563601479]]
        Qah = np.array(Qah)
        ah = [-28490.8566886116, 65752.6299198198, 38830.3666554972,
              5003.70833517778, -29196.0699104593, -297.658932458787,
              -22201.0284440701, 51235.8374755528, 30257.7809603224,
              3899.40332138829, -22749.1853575113, -159.278779870217]
        ah = np.array(ah)

    afix, sqnorm = mlambda(ah, Qah)
