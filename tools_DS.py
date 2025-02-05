# Jose Benavides (jose.astroph@gmail.com)

# This file is part of the
#   MilaDS project (https://github.com/josegit88/MilaDS).
# Copyright (c) 2023, Jose Benavides
# License: MIT
#   Full Text: https://github.com/josegit88/MilaDS/blob/main/LICENSE

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy


# aperture velocity dispersion of NFW model at R=r_vir
def sapovervv(c, anis=None):
    if anis == None:
        anis = "isotropic"
        cfs = [0.344165, 0.312747, -0.336547, 0.0390675]

    if anis == "ML":
        cfs = [0.270654, 0.362998, -0.312684, 0.0161601]

    lc = np.log10(c)
    poly = cfs[0] + cfs[1] * lc + cfs[2] * (lc ** 2) + cfs[3] * (lc ** 3)
    c2nfwapx = 10.0 ** poly
    sapovervv = 1.0 / np.sqrt(c2nfwapx)
    return sapovervv


# general params of the cluster: V200, r200, M200
def nfw_params(zave, sigmaap, cin, auth_i=None):
    """
    halo data params

    Input:

    zave is the average redshift of the cluster
    sigmaap is the aperture velocity dispersion within r200
    cin is a guess for the NFW M(r) concentration

    r200 keyword if set, is used (in kpc)

    Output:

    Vvir[km/s], rvir[kpc],  Mvir[Msun]
    """

    grav = 43.0  # gravitational constant
    h0 = 0.7
    Omega0 = 0.3  # Omega_0
    OmegaLambda = 0.7  # Omega_Lambda
    hz = h0 * np.sqrt(Omega0 * ((1.0 + zave) ** 3) + OmegaLambda)
    delta = 200.0
    if auth_i is None:
        auth_i = "MDvdB200"

    anis = "ML"
    vvguess = sigmaap / sapovervv(cin, anis)
    rvguess = np.sqrt(2.0 / delta) / (0.001 * hz) * vvguess / 100.0  # in kpc

    mvguess = 1.0e11 * rvguess * (vvguess / 100.0) ** 2 / grav  # in Msun

    data_nfw = np.array([vvguess, rvguess, mvguess])

    return data_nfw


# -----------------------------------------------------------

# ************** sigma NFW profile by Biviano code: ********************
# concentration parameter for LCDM
# (Maccio, Dutton & van den Bosch 08, relaxed halos, WMAP5, Delta=200)
# (Maccio, Dutton & van den Bosch 08, relaxed halos, WMAP5, Delta=95)
def cofmvir(mvir, auth):
    if auth == "MDvdB200":
        # print("caso 1")
        ref = "MDvdB200"
        slope = -0.098
        norm = 6.76

    if auth == "MDvdB":
        # print("caso 2")
        ref = "MDvdB"
        slope = -0.094
        norm = 9.35

    cofmvir = norm * (mvir ** slope)
    return cofmvir


# aperture velocity dispersion of NFW model at R=r_vir
def sapovervv(c, anis=None):
    if anis == None:
        anis = "isotropic"
        cfs = [0.344165, 0.312747, -0.336547, 0.0390675]

    if anis == "ML":
        cfs = [0.270654, 0.362998, -0.312684, 0.0161601]

    lc = np.log10(c)
    poly = cfs[0] + cfs[1] * lc + cfs[2] * (lc ** 2) + cfs[3] * (lc ** 3)
    c2nfwapx = 10.0 ** poly
    sapovervv = 1.0 / np.sqrt(c2nfwapx)
    return sapovervv


# sigma_los approx for NFW with ML anisotropy with radius r_a= 1 a
def siglosnfwml1(x):
    """
    x: value to valuate projected velocity dispersion profile
    """
    lx = np.log10(x)
    cfs = [
        -0.14783,
        -0.110877,
        -0.135747,
        0.00194757,
        0.0231745,
        0.000631017,
        -0.00323355,
        -0.000637035,
    ]
    poly = (
        cfs[0]
        + cfs[1] * lx
        + cfs[2] * (lx ** 2)
        + cfs[3] * (lx ** 3)
        + cfs[4] * (lx ** 4)
        + cfs[5] * (lx ** 5)
        + cfs[6] * (lx ** 6)
        + cfs[7] * (lx ** 7)
    )
    siglosnfwml1 = 10.0 ** poly
    return siglosnfwml1


# NFW mass function
def massnfw(x, a, rvir):
    massnfw = (np.log(1.0 + x / a) - x / (x + a)) / (
        np.log(1.0 + rvir / a) - rvir / (rvir + a)
    )
    return massnfw


def nfwvdp_proj(zave, sigmaap, cin, rrv, r200=None, auth_i=None):
    """
    VDP from NFW M(r) and given beta(r)

    Input:

    zave is the average redshift of the cluster
    sigmaap is the aperture velocity dispersion within r200
    cin is a guess for the NFW M(r) concentration
    rrv are the radii in units of r200 where to compute the VDP

    r200 keyword if set, is used (in kpc)

    Output:

    vdp is the VDP in units of sigmaap
    """

    grav = 43.0  # gravitational constant
    h0 = 0.7
    Omega0 = 0.3  # Omega_0
    OmegaLambda = 0.7  # Omega_Lambda
    hz = h0 * np.sqrt(Omega0 * ((1.0 + zave) ** 3) + OmegaLambda)
    delta = 200.0
    if auth_i is None:
        auth_i = "MDvdB200"

    # guessing rvir using sigma_ap/v_vir as predicted
    # from NFW with concentration of c (=4 on 1st pass)

    anis = "ML"
    if r200 == None:
        vvguess = sigmaap / sapovervv(cin, anis)
        rvguess = np.sqrt(2.0 / delta) / (0.001 * hz) * vvguess / 100.0  # in kpc
        # print 'v200 and r200 evaluated: ', vvguess, rvguess

    if r200 != None:
        rvguess = r200
        vvguess = 10.0 * hz * 1.0e2 * r200 / 1.0e3
        # print 'v200 and r200 given: ', vvguess, rvguess

    mvguess = 1.0e11 * rvguess * (vvguess / 100.0) ** 2 / grav  # in Msun
    # print 'mass from v200 and r200: ', mvguess, ' at z=', zave

    # LCDM concentration

    c = cofmvir(mvguess / 1.0e12, auth=auth_i)
    mdelta = mvguess

    # print "concentration used = ",c

    rrv_i = rrv
    vdp_i = (
        siglosnfwml1(c * rrv_i)
        * np.sqrt(massnfw(1.0 / c, 1.0 / c, 1.0) * c)
        * vvguess
        / sigmaap
    )

    return vdp_i


# ----- sigma projected profile, equal number: ------
def sigma_proj_en(Rxy, Vz, Nbins=None, ddof=None):
    """
    projected velocity dispertion profile, in equal width bins
    """
    Nbins = 5 if Nbins is None else Nbins
    ddof = 0 if ddof is None else ddof

    data = np.column_stack((Rxy, Vz))

    data = data[data[:, 0].argsort()]

    proj_vals_eq_numb = []
    for jj in range(int(len(data) / Nbins), len(data), int(len(data) / Nbins)):
        data_in_bin = data[jj - int(len(data) / Nbins) : jj, :]

        mean_Rxy_j = np.mean(data_in_bin[:, 0])
        sigma_z_j = np.std(data_in_bin[:, 1], ddof=ddof)

        proj_vals_eq_numb.append((mean_Rxy_j, sigma_z_j))

    proj_vals_eq_numb = np.array(proj_vals_eq_numb)
    # 0:Rxy
    # 3:sigma_z
    # ------------------------------

    return proj_vals_eq_numb


# -------------------------------------------

# ------------ group size: as the average of mutual distance ----------------
def elements_below_diag_matriz_NN(matriz):
    elements = []
    for x in range(len(matriz[0])):
        for y in range(x):
            # print x,y   # Mostrar la celda que se sumara.
            elements.append(matriz[x, y])
    return np.array(elements)


def size_group_mutualdistance_2D(xy):
    """
    Estimate the size of a distribution of galaxies as the
    average of mutual distance. Fot this in necessary to
    introduce a 2D array with the values of xy positions
    """

    Ngal = len(xy)

    matriz_Rij = np.zeros((Ngal, Ngal))

    # calculo de las distancias mutuas:
    for ii in range(Ngal):
        # ii = 0
        dist_to_glx_i = np.sqrt(
            (xy[:, 0] - xy[ii, 0]) ** 2 + (xy[:, 1] - xy[ii, 1]) ** 2
        )
        matriz_Rij[ii, :] = dist_to_glx_i

    size_gr_2D = np.mean(elements_below_diag_matriz_NN(matriz_Rij))

    return size_gr_2D


# ---------------------------------------------------------------------------

# Local Regression (LOESS) estimation routine.
def loc_eval(x, b):
    """
    Evaluate `x` using locally-weighted regression parameters.
    Degree of polynomial used in loess is inferred from b. `x`
    is assumed to be a scalar.
    """
    loc_est = 0
    for i in enumerate(b):
        loc_est += i[1] * (x ** i[0])
    return loc_est


def loess(xvals, yvals, alpha, poly_degree=1, n=None, m=None):
    """
    Perform locally-weighted regression on xvals & yvals.
    Variables used inside `loess` function:

        n         => number of data points in xvals
        m         => nbr of LOESS evaluation points
        q         => number of data points used for each
                     locally-weighted regression
        v         => x-value locations for evaluating LOESS
        locsDF    => contains local regression details for each
                     location v
        evalDF    => contains actual LOESS output for each v
        X         => n-by-(poly_degree+1) design matrix
        W         => n-by-n diagonal weight matrix for each
                     local regression
        y         => yvals
        b         => local regression coefficient estimates.
                     b = `(X^T*W*X)^-1*X^T*W*y`. Note that `@`
                     replaces `np.dot` in recent numpy versions.
        local_est => response for local regression
    """
    # Sort dataset by xvals.
    all_data = sorted(zip(xvals, yvals), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)

    locsDF = pd.DataFrame(
        columns=[
            "loc",
            "x",
            "weights",
            "v",
            "y",
            "raw_dists",
            "scale_factor",
            "scaled_dists",
        ]
    )
    evalDF = pd.DataFrame(columns=["loc", "est", "b", "v", "g"])

    # n = len(xvals)
    # m = n + 1
    n = len(xvals) if n is None else n
    m = n + 1 if m is None else m
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = (max(xvals) - min(xvals)) / len(xvals)
    v_lb = max(0, min(xvals) - (0.5 * avg_interval))
    v_ub = max(xvals) + (0.5 * avg_interval)
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)

    # Generate design matrix based on poly_degree.
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T

    for i in v:

        iterpos = i[0]
        iterval = i[1]

        # Determine q-nearest xvals to iterval.
        iterdists = sorted(
            [(j, np.abs(j - iterval)) for j in xvals], key=lambda x: x[1]
        )

        _, raw_dists = zip(*iterdists)

        # Scale local observations by qth-nearest raw_dist.
        scale_fact = raw_dists[q - 1]
        scaled_dists = [(j[0], (j[1] / scale_fact)) for j in iterdists]
        weights = [
            (j[0], ((1 - np.abs(j[1] ** 3)) ** 3 if j[1] <= 1 else 0))
            for j in scaled_dists
        ]

        # Remove xvals from each tuple:
        _, weights = zip(*sorted(weights, key=lambda x: x[0]))
        _, raw_dists = zip(*sorted(iterdists, key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists, key=lambda x: x[0]))

        iterDF1 = pd.DataFrame(
            {
                "loc": iterpos,
                "x": xvals,
                "v": iterval,
                "weights": weights,
                "y": yvals,
                "raw_dists": raw_dists,
                "scale_fact": scale_fact,
                "scaled_dists": scaled_dists,
            }
        )

        locsDF = pd.concat([locsDF, iterDF1])
        W = np.diag(weights)
        y = yvals
        # b         = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
        b = np.matmul(
            np.linalg.inv(np.matmul(np.matmul(X.T, W), X)),
            np.matmul(np.matmul(X.T, W), y),
        )
        local_est = loc_eval(iterval, b)
        iterDF2 = pd.DataFrame(
            {"loc": [iterpos], "b": [b], "v": [iterval], "g": [local_est]}
        )

        evalDF = pd.concat([evalDF, iterDF2])

    # Reset indicies for returned DataFrames.
    locsDF.reset_index(inplace=True)
    locsDF.drop("index", axis=1, inplace=True)
    locsDF["est"] = 0
    evalDF["est"] = 0
    locsDF = locsDF[
        [
            "loc",
            "est",
            "v",
            "x",
            "y",
            "raw_dists",
            "scale_fact",
            "scaled_dists",
            "weights",
        ]
    ]

    # Reset index for evalDF.
    evalDF.reset_index(inplace=True)
    evalDF.drop("index", axis=1, inplace=True)
    evalDF = evalDF[["loc", "est", "v", "b", "g"]]

    return (locsDF, evalDF)
