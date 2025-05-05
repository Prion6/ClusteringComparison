from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const

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