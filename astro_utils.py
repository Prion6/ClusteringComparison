from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
import numpy as np

def gal_Mpc_coords(ra, dec, redshift_gal, x_center, y_center, H0=70, Om0=0.3):
    
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    dist = cosmo.angular_diameter_distance(redshift_gal)

    ra = ra - x_center
    dec = dec - y_center

    x = (ra * dist).to(u.Mpc, u.dimensionless_angles()).value

    y = (dec * dist).to(u.Mpc, u.dimensionless_angles()).value

    return x, y

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