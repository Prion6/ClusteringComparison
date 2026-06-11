from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
import numpy as np
import pandas as pd
from astropy.stats import biweight_location, biweight_scale

def gal_Mpc_coords(ra, dec, redshift_gal, x_center, y_center, H0=70, Om0=0.3):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    dist = cosmo.angular_diameter_distance(redshift_gal)

    # Convert the differences to astropy Quantities with degree units
    delta_ra = (ra - x_center) * u.deg
    delta_dec = (dec - y_center) * u.deg

    # Now the unit conversion to Mpc will use the correct 
    # angular-to-physical conversion factor
    x = (delta_ra * dist).to(u.Mpc, u.dimensionless_angles()).value
    y = (delta_dec * dist).to(u.Mpc, u.dimensionless_angles()).value

    return x, y

def gal_3D_Mpc_coords(
    ra,
    dec,
    redshift_gal,
    x_center,
    y_center,
    z_center,
    H0=70,
    Om0=0.3
):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    x,y = gal_Mpc_coords(ra,dec,redshift_gal,x_center,y_center,H0,Om0)

    # Line-of-sight comoving distance
    d_gal = cosmo.comoving_distance(redshift_gal)
    d_center = cosmo.comoving_distance(z_center)

    z = (d_gal - d_center).to(u.Mpc).value

    return x, y, z

def los_vel(redshift_gal, redshift_clus):

    c = const.c.value
    vel = (c*(redshift_gal - redshift_clus)/(1 + redshift_clus))

    return vel

def get_r_200(log_m200, redshift, H0=70, Om0=0.3):
    
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    # Convert mass from log(Msun) to Msun
    m200 = 10**log_m200 * u.Msun

    # Get critical density at redshift z
    rho_crit = cosmo.critical_density(redshift)  # in g/cm^3
    rho_crit = rho_crit.to(u.Msun / u.Mpc**3)  # convert units

    # Compute R200
    r200 = ((3 * m200) / (4 * np.pi * 200 * rho_crit))**(1/3)
    r200 = r200.to(u.Mpc).value

    return r200

def get_halo_radial_distances(
        cluster_df,
        noise_id = -1,
        ra_clus = 0.0, 
        dec_clus = 0.0, 
        z_clus = 0.0,  
        log_m200=-1.0, 
        ra_key="RA", 
        dec_key="DEC", 
        redshift="z_app", 
        log_m200_key="log(m_200)", 
        halo_ID_key = 'haloId'):

    if ra_clus == 0.0:
        ra_clus = biweight_location(cluster_df[ra_key].dropna().values)
    if dec_clus == 0.0:
        dec_clus = biweight_location(cluster_df[dec_key].dropna().values)
    if z_clus == 0.0:
        z_clus = biweight_location(cluster_df[redshift].dropna().values)

    if log_m200 == -1.0:
        log_m200 = cluster_df[log_m200_key].iloc[0]

    r200_val = get_r_200(log_m200, z_clus)

    real_halos = cluster_df[
        pd.to_numeric(cluster_df[halo_ID_key], errors="coerce") != noise_id
    ].copy()

    if real_halos.empty:
        return np.array([])

    halo_groups = (
        real_halos
        .groupby(halo_ID_key)
        .agg({
            ra_key: lambda x: biweight_location(x.dropna().astype(float).values),
            dec_key: lambda x: biweight_location(x.dropna().astype(float).values)
        })
        .sort_index()
    )

    r_norms = []
    for _, row in halo_groups.iterrows():
        x_mpc, y_mpc = gal_Mpc_coords(
            row['RA'], row['DEC'], z_clus, ra_clus, dec_clus
        )
        r_proj = np.sqrt(x_mpc**2 + y_mpc**2)
        r_norms.append(r_proj / r200_val)

    return np.array(r_norms)



