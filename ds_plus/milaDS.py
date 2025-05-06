# Jose Benavides (jose.astroph@gmail.com)

# This file is part of the
#   MilaDS project (https://github.com/josegit88/MilaDS).
# Copyright (c) 2023, Jose Benavides
# License: MIT
#   Full Text: https://github.com/josegit88/MilaDS/blob/main/LICENSE

# -*- coding: utf-8 -*-
import numpy as np
import os
import scipy as sc
from scipy import stats
import random
from ds_plus import tools_DS

# import matplotlib.pyplot as plt
# plt.close()

# data_glx_in_cluster = np.genfromtxt("bullet_xyv.dat")
# 0:ID
# 1:X
# 2:Y
# 3:rest-frame Vel


def DSp_groups(
    Xcoor,
    Ycoor,
    Vlos,
    Zclus,
    cluster_name=None,
    nsims=None,
    Plim_P=None,
    Ng_jump=None,
    Ng_max=None,
    overlapping_groups=False,
    anti_fragmentation=True,
    compare_Method=None,
    auth_i=None,
    ddof=None,
):
    """
    Detection and characterization of substructures in galaxy clusters,
    implementing the DS+ method by Biviano+2017.

    Inputs:

    Xcoor, Ycoor: Coordinates (x,y) in physical kpc of the cluster galaxies. (1-D array per coordinate)
    Vlos: Line-of-sight velocities in km/s of the cluster galaxies. (1-D array)
    Zclus: redshift of the cluster. (float)
    cluster_name: String with the name of the cluster data. Is completely optional, by default is "cluster_proofs"
    nsims: integer with number of simulations, by default is 1000.
    Plim_P: minimun probability of DS+ selection, in percent, by deafult is 10 (10%)
    Ng_jump: integer with number of multiplicity of assigned groups, by default is 3
    Ng_max: integer with the maximun number of multipliplicity, by default is square(N_gal of the cluster)
    overlapping_groups: If False, then when a galaxy is assigned to a DS+ group, all other groups to which that galaxy belongs are discarded, by default is False
    anti_fragmentation: Combine one or more DS+ groups when the criteria of closeness and average group velocity are met, by defalut is True, and is necessary that overlapping_groups=False
    compare_Method: type of profile comparison, you can select between: "NFW", "fit", "loess", by default is "loess"
    auth_i: coefficients of model, by default is "MDvdB200"
    ddof: degrees of freedom, by default is 1

    Output:

    Return three arrays:
        - individual information of each galaxy per multiplicity in each DS+ group
        - information of assignment of DS+ groups, with no-overlaping galaxies
        - summary of DS+ identified groups

    Example brief running:

    milaDS.DSp_groups(Xcoor=data_sample[:,1], Ycoor=data_sample[:,2], Vlos=data_sample[:,3], Zclus=0.296, cluster_name="Cl1", nsims=100, Plim_P=10)

    """

    cluster_name = "cluster_proofs" if cluster_name is None else cluster_name
    nsims = 1000 if nsims is None else nsims
    Plim_P = 1 if Plim_P is None else Plim_P
    Ng_jump = 3 if Ng_jump is None else Ng_jump #minimum number of members in group
    Ng_max = int(np.sqrt(len(Xcoor))) if Ng_max is None else Ng_max #maximum number of members in a group
    compare_Method = "loess" if compare_Method is None else compare_Method
    ddof = 1 if ddof is None else ddof
    auth_i = "MDvdB200" if auth_i is None else auth_i

    ID_glx = range(len(Xcoor))
    data_glx_in_cluster = np.column_stack((ID_glx, Xcoor, Ycoor, Vlos))

    h_small = 0.6774
    G = (4.514e-39) * ((3.086e16) ** 2.0)  # G en unidades de km^2/s^2 *kpc/Msun
    H0 = h_small / 10.0  # H0 pasando de (km/s * 1/Mpc) a (km/s * 1/kpc)
    Omega_L0 = 0.73
    Omega_m0 = 0.27

    Plim = (
        1.0 * Plim_P / 100
    )  # arbitrarial value, if necesary to check how change the results with other values of Plim

    path_data_DS = "./data_DS_results/cluster_name_" + str(cluster_name) + "/"
    if not os.path.exists(path_data_DS):
        os.makedirs(path_data_DS)

    path_data_DS_results = (
        "./data_DS_results/cluster_name_"
        + str(cluster_name)
        + "/Plim_"
        + str(Plim_P)
        + "p/"
    )
    if not os.path.exists(path_data_DS_results):
        os.makedirs(path_data_DS_results)


    Vmean_clus = np.mean(data_glx_in_cluster[:, 3])
    sigma_1D_clus = np.std(data_glx_in_cluster[:, 3])
    V200_clus = np.sqrt(3.0) * sigma_1D_clus

    # Z_i = 0.296 # redshift
    Z_i = Zclus
    M200_clus_M1 = (
        (3.0 ** (3.0 / 2) * (sigma_1D_clus) ** 3.0 / G)
        * 1.0
        / (10.0 * H0 * (1 + Z_i) ** 1.5)
    )
    r200_clus_M1 = G * M200_clus_M1 / (V200_clus ** 2.0)

    conc_Biv = 6.76 * ((M200_clus_M1 / 1e12) ** (-0.098))
    params_cluster = tools_DS.nfw_params(
        zave=Zclus, sigmaap=sigma_1D_clus, cin=conc_Biv, auth_i="MDvdB200"
    )

    V200_clus = params_cluster[0]
    r200_clus = params_cluster[1]
    M200_clus = params_cluster[2]

    if compare_Method == "loess":
        # ----- equal number: ------
        proj_vals_eq_numb = tools_DS.sigma_proj_en(
            np.sqrt(data_glx_in_cluster[:, 1] ** 2 + data_glx_in_cluster[:, 2] ** 2),
            data_glx_in_cluster[:, 3],
            Nbins=5,
            ddof=1,
        )
        # 0:Rxy
        # 1:sigma_z

        regsDF, evalDF = tools_DS.loess(
            np.log10(proj_vals_eq_numb[:, 0]),
            np.log10(proj_vals_eq_numb[:, 1]),
            alpha=1.0,
            poly_degree=1,
            m=16,
        )
        l_x = evalDF["v"].values
        l_y = evalDF["g"].values

    if compare_Method == "fit":
        # fits:
        coef_fit_sigma = np.polyfit(10 ** l_x[:-1], 10 ** l_y[:-1], 3)
        r_min = 0.001
        r_max = max(proj_vals_eq_numb[:, 0])
        r_fit = np.linspace(r_min, r_max + 800, 1000)

        sigma_fit = (
            coef_fit_sigma[0] * (r_fit ** 3)
            + coef_fit_sigma[1] * (r_fit ** 2)
            + coef_fit_sigma[2] * (r_fit)
            + coef_fit_sigma[3]
        )
        func_sigma = (
            lambda Rpos: coef_fit_sigma[0] * (Rpos ** 3)
            + coef_fit_sigma[1] * (Rpos ** 2)
            + coef_fit_sigma[2] * (Rpos)
            + coef_fit_sigma[3]
        )
        # ------------------------------

    # ****************** ds - substructure program: **********************
    sh_id = data_glx_in_cluster[:, 0]
    Xpos = data_glx_in_cluster[:, 1]
    Ypos = data_glx_in_cluster[:, 2]
    Vel_LOS = data_glx_in_cluster[:, 3]

    Rij_to_center = np.sqrt(Xpos ** 2 + Ypos ** 2)

    data_glx_in_cluster = np.column_stack((sh_id, Xpos, Ypos, Vel_LOS, Rij_to_center))

    # Now, I will go to calculate the mean and dispersion velocities of the "linked" groups, following the multiplicity Ng(j) = j*3, j=1,2,... k
    # where Ng(k) > Nm^0.5, over each galaxy
    Ng_clus = len(data_glx_in_cluster)
    Ng_max = np.sqrt(len(data_glx_in_cluster))
    list_Ng = range(3, int(Ng_max), Ng_jump)

    data_DS_substructure = []

    for idx in range(0, len(data_glx_in_cluster)):

        sh_ID = data_glx_in_cluster[idx, 0].astype("int32")
        # print("idx:",idx+1, "sh ID:", sh_ID)

        for Ng_j in list_Ng:

            dist_to_glx_i = np.sqrt(
                (data_glx_in_cluster[:, 1] - data_glx_in_cluster[idx, 1]) ** 2.0
                + (data_glx_in_cluster[:, 2] - data_glx_in_cluster[idx, 2]) ** 2.0
            )

            # array with index and distances:
            data_idx_and_dist = np.hstack(
                (
                    (np.arange(Ng_clus)).reshape(Ng_clus, 1),
                    dist_to_glx_i.reshape(Ng_clus, 1),
                )
            )
            data_idx_and_dist_sort = data_idx_and_dist[
                data_idx_and_dist[:, 1].argsort()
            ]

            # size of the group:
            # size_group_i = data_idx_and_dist_sort[Ng_j - 1, 1]
            # size_group_i = tools_DS.size_group_mutualdistance_2D(data_glx_in_cluster[1:3, :])

            idx_interest = data_idx_and_dist_sort[0:Ng_j, 0].astype(
                "int32"
            )  # return the index for n-gals of interes, including the reference galaxy

            data_glx_interest = data_glx_in_cluster[
                idx_interest, :
            ]  # here we have the interest galaxies, now we go to calculate the rest!

            # size of the group:
            size_group_i = tools_DS.size_group_mutualdistance_2D(
                data_glx_interest[1:3, :]
            )

            av_dist = np.mean(data_glx_interest[:, 4])  # average distance to BCG
            dist_gr_r200units = av_dist / r200_clus
            # sigma_cl_at_GroupDist = V_cl_at_GroupDist/np.sqrt(3.) #sigma_LOS

            # using NFW profile:
            if compare_Method == "NFW":
                sigma_cl_at_GroupDist = sigma_1D_clus * tools_DS.nfwvdp_proj(
                    zave=Zclus,
                    sigmaap=sigma_1D_clus,
                    cin=conc_Biv,
                    rrv=dist_gr_r200units,
                    r200=None,
                    auth_i="MDvdB200",
                )

            # using smooth profile:
            if compare_Method == "fit":
                sigma_cl_at_GroupDist = func_sigma(av_dist)

            # simple interpolation
            if compare_Method == "loess":

                if av_dist < 10 ** l_x[0]:
                    R_mean_int = 10 ** l_x[0]  # 1st value
                    sigma_mean_int = 10 ** l_y[0]  # 1st value

                if av_dist > 10 ** l_x[-1]:
                    R_mean_int = 10 ** l_x[-1]  # 1st value
                    sigma_mean_int = 10 ** l_y[-1]  # 1st value

                if av_dist > 10 ** l_x[0] and av_dist < 10 ** l_x[-1]:
                    R_up = np.where(10 ** l_x >= av_dist)[0][0]
                    R_down = np.where(10 ** l_x < av_dist)[0][-1]
                    R_mean_int = np.mean(
                        [10 ** l_x[R_up], 10 ** l_x[R_down]]
                    )  # simple interpolation
                    sigma_mean_int = np.mean(
                        [10 ** l_y[R_up], 10 ** l_y[R_down]]
                    )  # simple interpolation

                sigma_cl_at_GroupDist = sigma_mean_int

            # === calculus for delta_v and deta_sigma with "real" galaxies: ====
            # t-Student and X^2 statics:
            t_Student = np.abs(
                sc.stats.t.ppf(0.16, (Ng_j - 1))
            )  # we take the abs because t.ppf return a negative value, can use sc.stats.t.ppf(1-0.16, (Ng_j-1) )
            X_square = sc.stats.chi2.ppf((1 - 0.16), (Ng_j - 1))

            V_mean_group = np.mean(data_glx_interest[:, 3])  # vel LOS
            sigma_group = np.std(data_glx_interest[:, 3], ddof=1)

            # delta v:
            delta_v = (
                Ng_j ** 0.5
                * np.abs(V_mean_group - Vmean_clus)
                / (t_Student * sigma_cl_at_GroupDist)
            )

            # delta sigma:
            delta_sigma = (sigma_cl_at_GroupDist - sigma_group) / (
                sigma_cl_at_GroupDist * (1.0 - np.sqrt((Ng_j - 1) / X_square))
            )
            # ==================================================================

            # === calc for delta_v and deta_sigma with "simulated" galaxies: ===
            N_sims = nsims
            data_sim_deltas = []
            for ss in range(N_sims):

                # we generate Ncl random numbers (with a normal distribution):
                random_vals = np.random.normal(
                    loc=0, scale=sigma_cl_at_GroupDist, size=Ng_clus
                )

                # Now, extract Ng_j random index:
                idx_random = random.sample(range(Ng_clus), Ng_j)

                # random velocities:
                Vel_gal_sim = random_vals[idx_random]

                V_mean_group_sim = np.mean(Vel_gal_sim)  # vel LOS
                sigma_group_sim = np.std(Vel_gal_sim, ddof=ddof)

                # delta v:
                delta_v_sim = (
                    Ng_j ** 0.5
                    * np.abs(V_mean_group_sim - Vmean_clus)
                    / (t_Student * sigma_cl_at_GroupDist)
                )

                # delta sigma:
                delta_sigma_sim = (sigma_cl_at_GroupDist - sigma_group_sim) / (
                    sigma_cl_at_GroupDist * (1.0 - np.sqrt((Ng_j - 1) / X_square))
                )

                data_sim_deltas.append(
                    (ss, sigma_group_sim, delta_v_sim, delta_sigma_sim)
                )

            data_sim_deltas = np.array(data_sim_deltas)
            # 0:n-sim
            # 1:sigma_group simulated
            # 2:delta_v simulated
            # 3:delta_sigma simulated

            sigma_mean_group_sim = np.mean(data_sim_deltas[:, 1])
            # ==================================================================

            # *********** P-values for delta_v and delta_sigma : ***************
            N_delta_v_above = len(
                np.where(data_sim_deltas[:, 2] > delta_v)[0]
            )  # number of times where delta_v_sim > delta_v_obs
            P_v = (1.0 * N_delta_v_above) / N_sims

            N_delta_sigma_above = len(
                np.where(data_sim_deltas[:, 3] > delta_sigma)[0]
            )  # same for sigma
            P_sigma = (1.0 * N_delta_sigma_above) / N_sims

            # print "P_v=", P_v, "P_sigma=",P_sigma
            # ******************************************************************

            # Minimun value between Pv and Ps:
            Pmin = min(P_v, P_sigma)

            data_DS_substructure.append(
                (
                    idx,
                    sh_ID,
                    Ng_j,
                    av_dist,
                    size_group_i,
                    sigma_group,
                    delta_v,
                    delta_sigma,
                    P_v,
                    P_sigma,
                    Pmin,
                )
            )
            # --------------------------------------------------------

    data_DS_substructure = np.array(data_DS_substructure)
    # 0:idx galaxies
    # 1:sh ID
    # 2:Ngal: multiplicity
    # 3:average Rij distance
    # 4:size of group
    # 5:sigma_LOS
    # 6:delta_v
    # 7:delta_sigma
    # 8:P_Vel
    # 9:P_sigma
    # 10:minimun value between Pv and Ps (ds)

    data_header = " idx_gal   gal_id   Ngal    Ri_avr(kpc)  size_i(kpc)   sigma_i(km/s)   delta_v    delta_sigma     P_v      P_sigma    Pmin(ds)"
    np.savetxt(
        path_data_DS
        + "data_DS_cluster_N_"
        + str(cluster_name)
        + "_"
        + str(nsims)
        + "sims.dat",
        data_DS_substructure,
        header=data_header,
        fmt=[
            "%7.f",
            "%9.f",
            "%7.f",
            "%13.2f",
            "%13.2f",
            "%13.2f",
            "%12.5f",
            "%12.5f",
            "%10.4f",
            "%10.4f",
            "%10.4f",
        ],
    )

    #print("end DS+ individual probabilities")

    # ****** Part 2 location program, using the nooverlaping principle: ********

    # first we need to generate a matrix with min(Pv,Ps), Ng_mult, idx_g, and then sort by the lowest probability:
    data_sort_by_Mult = np.zeros((1, 11))

    for Ng_j in list_Ng:
        data_DS_prob_Mult = data_DS_substructure[data_DS_substructure[:, 2] == Ng_j]
        data_sort_by_Mult = np.vstack((data_sort_by_Mult, data_DS_prob_Mult))

    data_sort_by_Mult = data_sort_by_Mult[1:, :]

    # now I need to sort information by ds minimum value:
    data_sort_by_ds = data_sort_by_Mult[data_sort_by_Mult[:, 10].argsort()]

    # We only consider galaxies with Pmin < Plim:
    data_below_Plim = data_sort_by_ds[data_sort_by_ds[:, 10] < Plim]

    allocation_data = np.zeros((Ng_clus, 12))
    # 0:idx galaxies
    # 1:sh ID
    # 2:Ngali: multiplicity proposed initially
    # 3:Ngalf: multiplicity after of assignation
    # 4:average Rij distance
    # 5:size of group
    # 6:sigma_LOS
    # 7:Pmin min(Pv, Ps)
    # 8:GrNr
    # 9-11:(X,Y) pos
    # 11:V_LOS

    # Iniial values:
    allocation_data[:, 0] = np.arange(Ng_clus)  # index
    allocation_data[:, 8] = -1  # initial fictitious GrNr
    allocation_data[:, 7] = -1  # initial fictitious Plim

    list_idx = []
    for nn in range(len(data_below_Plim)):
        idx_nn = data_below_Plim[nn, 0].astype("int32")
        if idx_nn not in list_idx:
            list_idx.append(idx_nn)

    count = 0  # counter for GrNr
    # for idx in list_idx[:3]:
    for idx in list_idx:

        # print "idx:", idx
        # ++++ condition for galaxies that were assigned: ++++
        if allocation_data[idx, 8] != -1:
            continue
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++

        data_gal_i_sort_by_ds = data_below_Plim[data_below_Plim[:, 0] == idx]
        Ngal_i = min(data_gal_i_sort_by_ds[:, 2].astype("int32"))
        row_Ngal_i = np.where(data_gal_i_sort_by_ds[:, 2] == Ngal_i)[0][0]

        # data group:
        shID_i = data_gal_i_sort_by_ds[row_Ngal_i, 1].astype("int32")  # "main" galaxy
        Rij_i = data_gal_i_sort_by_ds[row_Ngal_i, 3]
        size_i = data_gal_i_sort_by_ds[row_Ngal_i, 4]
        sigma_i = data_gal_i_sort_by_ds[row_Ngal_i, 5]
        Pmin_i = data_gal_i_sort_by_ds[row_Ngal_i, 10]
        GrNr_i = count
        # print("Pmin:", Pmin_i)

        # -------- now to obtain the informatio for the group members: --------

        # distances between galaxies:
        dist_to_glx_i = np.sqrt(
            (data_glx_in_cluster[:, 1] - data_glx_in_cluster[idx, 1]) ** 2.0
            + (data_glx_in_cluster[:, 2] - data_glx_in_cluster[idx, 2]) ** 2.0
        )

        # array with index and distances:
        data_idx_and_dist = np.hstack(
            (
                (np.arange(Ng_clus)).reshape(Ng_clus, 1),
                dist_to_glx_i.reshape(Ng_clus, 1),
            )
        )
        data_idx_and_dist_sort = data_idx_and_dist[data_idx_and_dist[:, 1].argsort()]

        # gals of interest:
        # size_group_i = data_idx_and_dist_sort[Ngal_i-1,1] #It is not necessary
        idx_interest = data_idx_and_dist_sort[0:Ngal_i, 0].astype(
            "int32"
        )  # return the index for n-gals of interes, including the reference galaxy
        data_glx_interest = data_glx_in_cluster[
            idx_interest, :
        ]  # here we have the interest galaxies, now we go to calculate the rest!

        # print len(idx_interest)

        # if the number or members is less to 3, not assing the galaxies to a group:
        if len(idx_interest) <= 2:
            #print(" few members")
            continue

        # ++++++++ overlapping condition: +++++++++
        if overlapping_groups == False:
            if np.sum(allocation_data[idx_interest, 8]) != -1 * len(idx_interest):
                continue
        if overlapping_groups == True:
            pass
        # +++++++++++++++++++++++++++++++++++++++++

        # print "Ngals:", Ngal_i
        for jj in idx_interest:
            # print "gal_j:", jj
            # ++++ again the condition for galaxies that were assigned: ++++
            if allocation_data[jj, 8] != -1:
                continue
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            allocation_data[jj, 0] = jj  # index
            allocation_data[jj, 1] = data_glx_in_cluster[jj, 0]  # shID
            allocation_data[jj, 2] = Ngal_i  # Ngal: multiplicity proposed initially
            allocation_data[jj, 4] = Rij_i  # average Rij distance
            allocation_data[jj, 5] = size_i  # 5:size of group
            allocation_data[jj, 6] = sigma_i  # 6:sigma_LOS
            # allocation_data[jj,8] = Pmin_i #7:Pmin min(Pv, Ps)
            allocation_data[jj, 8] = GrNr_i  # 8:GrNr
            allocation_data[jj, 9:11] = data_glx_in_cluster[jj, 1:3]  # (X,Y) pos in kpc
            allocation_data[jj, 11] = data_glx_in_cluster[jj, 3]  # vel LOS in km/s

            # Plim for individual galaxy:
            row_gid = np.where(
                data_DS_substructure[:, 1] == data_glx_in_cluster[jj, 0]
            )[0]
            info_gid = data_DS_substructure[row_gid, :]
            row_ngi = np.where(info_gid[:, 2] == Ngal_i)[0][0]
            Pmin_gid = info_gid[row_ngi, 10]
            allocation_data[jj, 7] = Pmin_gid
            # print "gal:",data_glx_in_cluster[jj,0], "Pmin_i:",Pmin_gid

        # ----
        count += 1

    data_gals_allocated_in_groups = allocation_data[allocation_data[:, 8] != -1]
    data_gals_allocated_without_groups = allocation_data[allocation_data[:, 8] == -1]

    for kk in range(len(data_gals_allocated_without_groups)):
        idx_kk = data_gals_allocated_without_groups[kk, 0].astype("int32")
        allocation_data[idx_kk, 1] = data_glx_in_cluster[idx_kk, 0]  # shID
        allocation_data[idx_kk, 9:11] = data_glx_in_cluster[
            idx_kk, 1:3
        ]  # (X,Y,Z) pos in kpc
        allocation_data[idx_kk, 11] = data_glx_in_cluster[
            idx_kk, 3
        ]  # (Vx,Vy,Vz) vels in km/s

    data_above_Plim = data_sort_by_ds[
        data_sort_by_ds[:, 10] >= Plim
    ]  # galaxies wihout group

    list_singles_reallocated = []
    for ss in range(len(data_above_Plim)):
        if data_above_Plim[ss, 0] in data_gals_allocated_in_groups[:, 0]:
            if data_above_Plim[ss, 0] not in list_singles_reallocated:
                list_singles_reallocated.append(data_above_Plim[ss, 0].astype("int32"))

    list_GrNr = []
    for gg in range(len(data_gals_allocated_in_groups)):
        GrNr_nn = data_gals_allocated_in_groups[gg, 8].astype("int32")
        if GrNr_nn not in list_GrNr:
            list_GrNr.append(GrNr_nn)

    # recount of final Ngal in GrNr:
    for gn in list_GrNr:
        rows_GrNr = np.where(allocation_data[:, 8] == gn)[0]
        Ngalf_gr = len(rows_GrNr)
        allocation_data[rows_GrNr, 3] = Ngalf_gr

    # =============== anti-fragmentation process: ====================
    if overlapping_groups == False and anti_fragmentation == True:
        list_Gr_DS = []
        pmin_Gr_DS = []
        for gg in range(len(allocation_data)):
            gr_X = allocation_data[gg, 8].astype("int32")
            data_gr_X = allocation_data[allocation_data[:, 8] == gr_X]
            pmin = min(data_gr_X[:, 7])
            Ng_X = data_gr_X[0, 3]
            if gr_X == -1:
                continue
            if gr_X not in list_Gr_DS:
                list_Gr_DS.append(gr_X)
                pmin_Gr_DS.append((pmin, Ng_X))

        list_Gr_DS = np.column_stack((list_Gr_DS, pmin_Gr_DS))
        list_Gr_DS = list_Gr_DS[list_Gr_DS[:, 1].argsort()]
        # -------------------------------

        merg_lost = []
        merg_add = []
        for ii in range(len(list_Gr_DS)):
            gr_i = list_Gr_DS[ii, 0].astype("int32")
            if gr_i in merg_lost:
                continue
            data_gr_i = allocation_data[allocation_data[:, 8] == gr_i]
            center_gri = [np.mean(data_gr_i[:, 9]), np.mean(data_gr_i[:, 10])]
            dist_to_cent_gr_i = np.sqrt(
                (data_gr_i[:, 9] - center_gri[0]) ** 2
                + (data_gr_i[:, 10] - center_gri[1]) ** 2
            )
            max_dist_i = max(dist_to_cent_gr_i)
            Vmean_gr_i = np.mean(data_gr_i[:, 11])
            V_to_gr_i = data_gr_i[:, 11] - Vmean_gr_i
            max_Vmean_gr_i = max(V_to_gr_i)

            centers_grs = []
            for jj in range(len(list_Gr_DS)):
                gr_j = list_Gr_DS[jj, 0].astype("int32")
                if gr_j == gr_i:
                    continue
                if gr_j in merg_lost:
                    continue
                data_gr_j = allocation_data[allocation_data[:, 8] == gr_j]
                center_grj = [np.mean(data_gr_j[:, 9]), np.mean(data_gr_j[:, 10])]
                dist_to_cent_gr_j = np.sqrt(
                    (data_gr_j[:, 9] - center_grj[0]) ** 2
                    + (data_gr_j[:, 10] - center_grj[1]) ** 2
                )
                max_dist_j = max(dist_to_cent_gr_j)
                Vmean_gr_j = np.mean(data_gr_j[:, 11])
                V_to_gr_j = data_gr_j[:, 11] - Vmean_gr_j
                max_Vmean_gr_j = max(np.abs(V_to_gr_j))

                d_ij = np.sqrt(
                    (center_gri[0] - center_grj[0]) ** 2
                    + (center_gri[1] - center_grj[1]) ** 2
                )

                # condition for keep the merge:
                if d_ij < max(max_dist_i, max_dist_j) and np.abs(
                    Vmean_gr_i - Vmean_gr_j
                ) < max(np.abs(max_Vmean_gr_i), np.abs(max_Vmean_gr_j)):
                    #print("merge between gr_i:", gr_i, "and gr_j:", gr_j)
                    merg_lost.append(gr_j)
                    merg_add.append(gr_i)

        merg_lost = np.column_stack((merg_add, merg_lost))

        for mm in range(len(merg_lost)):
            gr_add_i = merg_lost[mm, 0]
            gr_lost_i = merg_lost[mm, 1]
            row_gr_merg = np.where(allocation_data[:, 9] == gr_lost_i)[0]
            allocation_data[row_gr_merg, 9] = gr_add_i
    # ================================================================

    # save results:
    data_header_allocate = " idx_gal   shID  Ngali  Ngalf    Rij(kpc)  size(kpc)  sigm(km/s)  Pmin(ds)    GrNr    x(kpc)    y(kpc)   V_LOS(km/s)"
    np.savetxt(
        path_data_DS_results
        + "data_DS_groups_allocation_Plim_"
        + str(Plim_P)
        + "_"
        + str(nsims)
        + "sims_cluster_N_"
        + str(cluster_name)
        + ".dat",
        allocation_data,
        header=data_header_allocate,
        fmt=[
            "%7.f",
            "%8.f",
            "%6.f",
            "%6.f",
            "%11.2f",
            "%10.2f",
            "%10.2f",
            "%10.4f",
            "%7.f",
            "%10.2f",
            "%10.2f",
            "%10.2f",
        ],
    )

    # 0:idx galaxies
    # 1:sh ID
    # 2:Ngali: multiplicity proposed initially
    # 3:Ngalf: multiplicity after of assignation
    # 4:average Rij distance
    # 5:size of group
    # 6:sigma_LOS
    # 7:Pmin min(Pv, Ps)
    # 8:GrNr
    # 9-10:(X,Y) pos
    # 11:V_LOS

    #print("end DS+ groups location")
    # ************************************************************************

    # *********** part 3: summary of properties of DS+ groups: ***************
    gals_assigned_in_sub = allocation_data[allocation_data[:, 8] != -1]
    # gals_assigned_in_sub = gals_assigned_in_sub[gals_assigned_in_sub[:,3] >= 3]

    list_Gr = []
    for gg in range(len(gals_assigned_in_sub)):
        gr_i = gals_assigned_in_sub[gg, 8].astype("int32")
        if gr_i not in list_Gr:
            list_Gr.append(gr_i)

    summary_DS_grs = []
    for ii in range(len(list_Gr)):
        gi = list_Gr[ii]
        subgr_i = gals_assigned_in_sub[gals_assigned_in_sub[:, 8] == gi]
        Ng_gri = len(subgr_i)
        dist_gri = np.mean(subgr_i[:, 4])
        size_gri = np.mean(subgr_i[:, 5])
        sigma_gri = np.mean(subgr_i[:, 6])
        Vmean_gri = np.mean(subgr_i[:, 11])
        p_min = min(subgr_i[:, 7])
        p_avr = np.mean(subgr_i[:, 7])

        summary_DS_grs.append(
            (gi, Ng_gri, dist_gri, size_gri, sigma_gri, Vmean_gri, p_min, p_avr)
        )

    summary_DS_grs = np.array(summary_DS_grs)

    #if summary_DS_grs.shape == (0,):
        #print("groups not found")

    if len(summary_DS_grs) > 1:
        summary_DS_grs = summary_DS_grs[summary_DS_grs[:, 0].argsort()]
        data_header_summary = "  GrNr    Ngal    Rij(kpc)  size(kpc)  sigm(km/s)  Vmean(km/s) Pmin(ds)   Pmin_avr(ds)"
        np.savetxt(
            path_data_DS_results
            + "summary_data_DS_groups_"
            + str(Plim_P)
            + "_"
            + str(nsims)
            + "sims_cluster_N_"
            + str(cluster_name)
            + ".dat",
            summary_DS_grs,
            header=data_header_summary,
            fmt=[
                "%7.f",
                "%7.f",
                "%11.2f",
                "%10.2f",
                "%10.2f",
                "%10.2f",
                "%10.4f",
                "%10.4f",
            ],
        )

    #print("end summary DS+ groups")
    # ************************************************************************

    # ****** Returns: ********
    #print(len(summary_DS_grs), " DS+ groups detected")
    return data_DS_substructure, allocation_data, summary_DS_grs
    # ************************


# ---------------------------------------
# END PROGRAM
