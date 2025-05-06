# Filename: cluster_kinematics.py
# Here we found a serie of scripts develop to estimate
# the kinemactic properties of the cluster

# - section dedicated to importing python modules
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import kstest
import scipy.stats as stats

# Section dedicated to importing the modules from astropy
from astropy.stats import biweight_scale
from astropy.stats import biweight_location
from astropy import constants as const
import astropy.units as u

# Section dedicated to importing the modules from CALSAGOS
from . import utils

# define speed of light in km/s
#c = 299792.458
speed_of_light = const.c
c_value = speed_of_light.value
c = c_value/1000.

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_escape_velocity(cluster_mass, cluster_radius):

    """ escape_velocity = cluster_kinematics.calc_escape_velocity(cluster_mass, cluster_radius)

	Function that estimates the escape velocity of a galaxy cluster
    using as input the m200 and r200 of the cluster

    This funcion was develop by D. Olave-Rojas (21/06/2016)

    The velocity escape is defined as v_e = sqrt(2GM/r)

	:param cluster_mass: cluster mass defined as m_200. This 
        parameter must be in M_sun
	:param cluster_radius: cluster radius defined as r_200.
        this parameter must be in Mpc

	:type cluster_mass: float
	:type cluster_radius: float

	:returns: The escape velocity of the cluster 
	:rtype: int, float

    .. note::
    
	The returned velocity is in km/s

	:Example:
    >>> import calsagos
    >>> calsagos.cluster_kinematics.calc_escape_velocity(1.279e15, 1.95)
    2375.2728725027764

	"""

    # -- defining the gravitational constant
    grav_const = const.G
    g = grav_const.value # in Kg m-3 s-2

    # -- defining the solar mass
    solar_mass = const.M_sun
    ms = solar_mass.value

    # -- converting the mass of the clusters in solar units to mass in kilograms
    mass_kg =  cluster_mass * ms # cluster mass in kg units

    # -- converting the r200 in Mpc to r200 in meters
    cluster_radius_pc = cluster_radius*(10**6.)
    cluster_radius_meter = cluster_radius_pc*(u.pc.to(u.m))

    # -- estimate of escape velocity
    esc_vel_mks = np.sqrt(2.* g * mass_kg * (cluster_radius_meter**(-1))) #escape velocity is in m s-1
    escape_velocity = esc_vel_mks/1000. # escape velocity is in km s-1
   
    # -- return output
    return escape_velocity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_escape_velocity_diaferio(cluster_mass, cluster_radius):

    """ escape_velocity = cluster_kinematics.calc_escape_velocity_diaferio(cluster_mass, cluster_radius)

	Function that estimates the escape velocity of a galaxy cluster
    using as input the m200 of the cluster

    The escape velocity, estimated for a cluster,
    using m200 and r200, is computed as Diaferio (1999)

    This funcion was develop by D. Olave-Rojas (21/06/2016)

	:param cluster_mass: cluster mass defined as m_200. This 
        parameter must be in M_sun
	:param cluster_radius: cluster radius defined as r_200.
        this parameter must be in Mpc

	:type cluster_mass: float
	:type cluster_radius: float

	:returns: The escape velocity of the cluster 
	:rtype: int, float

    .. note::

	The returned velocity is in km/s

	:Example:
    >>> from calsagos import cluster_kinematics
    >>> cluster_kinematics.calc_escape_velocity_diaferio(1.279e15, 1.949)
    2374.7018294642708

	"""

    # -- renaming constants
    K_1 = 92.7e-6
    
    # -- estimate of escape velocity
    escape_velocity =  K_1 * np.sqrt(cluster_mass/cluster_radius)  # escape velocity is in km s-1
   
    # -- return output
    return escape_velocity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_peculiar_velocity(redshift_array, cluster_redshift):

    """ calsagos.cluster_kinematics.calc_peculiar_velocity(redshift_array, cluster_redshift)

    This function was developed by P. Cerulo (28/11/2015)

	Function that estimates peculiar velocities from
    redshift (as in Harrison 1974)

	:param redshift_array: array with redshift of
        galaxies in the region of a cluster
    :param cluster_redshift: central redshift of
        a galaxy cluster

	:type redshift_array: array
    :type cluster_redshift: int, float

    :returns: peculiar velocity of galaxies
	:rtype: array

    .. note::

	The returned velocity is in km/s 

	"""

    # -- define output quantities
    dim = redshift_array.size

    peculiar_velocity = np.zeros(dim)
  
    # -- compute peculiar velocity
    for ii in range(dim):

        if redshift_array[ii] <= 0.0:

            peculiar_velocity[ii] = -99.9

        elif redshift_array[ii] > 0.0:
            
            peculiar_velocity[ii] = c * ((redshift_array[ii] - cluster_redshift)) / (1.0 + cluster_redshift)

    # -- END OF LOOP

    # -- return cluster velocity dispersion
    return peculiar_velocity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_cluster_velocity_dispersion(input_redshift_array, escape_velocity, starting_redshift):

    """ cluster_kinematics.calc_cluster_velocity_dispersion(input_redshift_array, escape_velocity, starting_redshift)

    This function was developed by P. Cerulo (28/11/2015)
    following Yahil & Vidal (1977)

    Funcion that estimates the velocity dispersion
    in a sample free of contaminants

    We recomend the user use this function
    when cluster members are selected by 
    using ISOMER

	:param input_redshift_array: array with redshift 
        of spectroscopic members of a cluster
    :param escape_velocity: escape velocity of 
        the cluster
    :param starting_redshift: central redshift 
        of the galaxy cluster

	:type input_redshift_array: array
    :type escape_velocity: int, float
    :type starting_redshift: int, float

    :returns: central redshift of the
        cluster and velocity dispersion
    :rtype: numpy array

    .. note::

	The returned velocity dispersion has km/s units

    calc_peculiar_velocity(redshift_array, cluster_redshift)[0] corresponds to the 
        cluster redshift
    calc_peculiar_velocity(redshift_array, cluster_redshift)[1] corresponds to the
        velocity dispersion of the cluster
        
	"""

    #-- removing all bad values in redshift array
    good_values = np.where( input_redshift_array > 0.0 )[0]
    redshift_array = input_redshift_array[good_values]

    #-- removing galaxies at more than 4000 km/s from the cluster initial redshift
    starting_peculiar_velocity = calc_peculiar_velocity(redshift_array, starting_redshift)
    starting_cluster_sample = np.where( (starting_peculiar_velocity > - escape_velocity) & (starting_peculiar_velocity < escape_velocity) )[0]

    #-- defining new set of cluster redshift including only galaxies with -v_esc < v_pec < +v_esc km/s
    cluster_redshift_array = redshift_array[starting_cluster_sample]

    #-- estimating cluster redshift for the new cluster sample
    cluster_redshift = biweight_location(cluster_redshift_array)

    
    while True:
        # -- estimating peculiar velocity and velocity dispersion for cleaned cluster sample
        peculiar_velocity = calc_peculiar_velocity(cluster_redshift_array, cluster_redshift)
        sigma = biweight_scale(peculiar_velocity)

        # -- removing all galaxies at more than 3 x sigma from the cluster redshift
        cluster_sample = np.where( (peculiar_velocity > -3*sigma) & (peculiar_velocity < 3*sigma) )[0]
        outliers = np.where( (peculiar_velocity <= -3*sigma) | (peculiar_velocity >= 3*sigma) )[0]

        # -- re-defining sample of redshifts and computing cluster redshift
        cluster_redshift_array = cluster_redshift_array[cluster_sample]
        cluster_redshift = biweight_location(cluster_redshift_array)

        # -- until there are no longer outliers
        if outliers.size == 0:
            break

    # -- return output array
    return np.array([cluster_redshift, sigma])

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_cluster_velocity_dispersion_error(input_redshift_array, escape_velocity, starting_redshift, n_bootstrap):
    
    """ cluster_kinematics.calc_cluster_velocity_dispersion_error(input_redshift_array, escape_velocity, starting_redshift, n_bootstrap)

    This function was developed by P. Cerulo (10/12/2015)

    Funcion that estimates the uncertainty on
    cluster velocity dispersion using a 
    boostrap technique. 
    
    We recomend the user use this function
    when cluster members are selected by 
    using ISOMER

	:param input_redshift_array: array with redshift 
        of spectroscopic members of a cluster
    :param escape_velocity: escape velocity of 
        the cluster
    :param starting_redshift: central redshift 
        of the galaxy cluster
    :param n_boostrap: number of bootstrap 
        simulations

	:type input_redshift_array: array
    :type escape_velocity: int, float
    :type starting_redshift: int, float
    :type n_bootstrap: int

    :returns: uncertainty on the cluster 
        velocity dispersion
    :rtype: array

    .. note::

	The returned uncertainty on the velocity 
        dispersion is in km/s
        
	"""
    #-- removing all bad values in redshift array
    good_values = np.where( input_redshift_array > 0.0 )[0]
    redshift_array = input_redshift_array[good_values]

    dim = redshift_array.size

    #-- defining arrays with output quantities and quantities useful for calculations
    bootstrap_sigma = np.zeros(n_bootstrap)


    print("starting boostrap estimation of uncertainty on velocity dispersion")

    for ii in range(n_bootstrap):

        # -- select random indices within redshift array

        R = np.random.randint(0, dim, size=dim)
        
        redshift_array_sim = redshift_array[R]

        # -- estimate velocity dispersion for bootstrap sample
        bootstrap_sigma[ii] = calc_cluster_velocity_dispersion(redshift_array_sim, escape_velocity, starting_redshift)[1]

    # -- computing symmetric width of the 68% confidence interval of the bootstrap distribution of velocity dispersion
    delta_sigma = utils.calc_result(bootstrap_sigma, 'symmetric')[1]

    # -- return output array
    return delta_sigma

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_clumberi_cluster_velocity_dispersion_error(input_redshift_array, starting_redshift, n_bootstrap):
    
    """ cluster_kinematics.calc_clumberi_cluster_velocity_dispersion_error(input_redshift_array, starting_redshift, n_bootstrap)

    This function was developed by D. Olave-Rojas (03/07/2021)

    Funcion that estimates the uncertainty on
    cluster velocity dispersion using a 
    boostrap technique
    
    We recomend the user use this function
    when cluster members are selected by 
    using CLUMBERI

	:param input_redshift_array: array with redshift 
        of spectroscopic members of a cluster
    :param escape_velocity: escape velocity of 
        the cluster
    :param starting_redshift: central redshift 
        of the galaxy cluster
    :param n_boostrap: number of bootstrap 
        simulations

	:type input_redshift_array: array
    :type escape_velocity: int, float
    :type starting_redshift: int, float
    :type n_bootstrap: int

    :returns: uncertainty on the cluster 
        velocity dispersion
    :rtype: array

    .. note::

	The returned uncertainty on the velocity 
        dispersion is in km/s
        
	"""

    #-- removing all bad values in redshift array
    good_values = np.where( input_redshift_array > 0.0 )[0]
    redshift_array = input_redshift_array[good_values]

    dim = redshift_array.size

    #-- defining arrays with output quantities and quantities useful for calculations
    bootstrap_sigma = np.zeros(n_bootstrap)
    peculiar_velocity  = np.zeros(n_bootstrap)

    print("starting boostrap estimation of uncertainty on velocity dispersion")

    for ii in range(n_bootstrap):

        # -- select random indices within redshift array

        R = np.random.randint(0, dim, size=dim)
        
        redshift_array_sim = redshift_array[R]

        # -- estimate velocity dispersion for bootstrap sample
        peculiar_velocity = calc_peculiar_velocity(redshift_array_sim, starting_redshift)
        bootstrap_sigma[ii] = biweight_scale(peculiar_velocity)

    # -- computing symmetric width of the 68% confidence interval of the bootstrap distribution of velocity dispersion
    delta_sigma = utils.calc_result(bootstrap_sigma, 'symmetric')[1]

    # -- return output array
    return delta_sigma

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def calc_caustic_escape_velocity(cluster_mass, cluster_radius, r, concentration):
    
    """ cluster_kinematics.calc_caustic_escape_velocity(m200, r200, r, concentration)

    This function was developed by D. Olave-Rojas (28/07/2023)

    Funcion developes to estimates the caustic 
    projected to the line of sight of the escape 
    velocity based on Navarro, Frenk & White 
    (1996, NFW) halo.

    :param cluster_mass: cluster mass defined as m_200.
        This parameter must be in M_sun
	:param cluster_radius: cluster radius defined as 
        r_200. This parameter must be in Mpc
    :param r: distance of the galaxies from the cluster 
        centre this parameter must be in Mpc
    :param concentration: concentration parameter

    :type cluster_mass      : float
    :type cluster_radius    : float
    :type r                 : array
    :type concentration     : float

    :returns: escape velocity caustic of the cluster as 
        a function of the m200, r200 and the distance 
        of the galaxies from the cluster centre
    :rtype: array

    .. note::

    The returned escape velocity is in km/s
    
    The user can fixed the concentration at 
    6 following Jaffé et al. (2015)
        
	"""

    # -- defining the gravitational constant
    grav_const = const.G
    g = grav_const.value # in Kg m-3 s-2

    # -- defining the solar mass
    solar_mass = const.M_sun
    ms = solar_mass.value

    # -- converting the mass of the clusters in solar units to mass in kilograms
    cluster_mass_kg =  cluster_mass * ms # cluster mass in kg units

    # -- converting the r200 in Mpc to r200 in meters
    cluster_radius_pc = cluster_radius*(10**6.)
    cluster_radius_meter = cluster_radius_pc*(u.pc.to(u.m))

    # -- converting the r in Mpc to r in  meters
    r_pc = r * (10**6.)
    r_meter = r_pc*(u.pc.to(u.m))
   
    # -- defining the output quantities
    dim = r.size
    s = np.zeros(dim)
    K = np.zeros(dim)
    v_esc = np.zeros(dim)
    escape_velocity = np.zeros(dim)

    # -- defining the g_c parameter
    g_c = ( np.log(1. + concentration)  - ((concentration)/(1. + concentration)) )**(-1.)
    
    # -- defining the s parameter
    for ii in range(dim):
        s[ii] = (np.pi /2.) * (r_meter[ii] / cluster_radius_meter)
    
    # -- defining the K parameter
    for ii in range(dim):
#        K[ii] = (g_c*(((np.log(1. + concentration*s[ii]))/s[ii]) - np.log(1 + concentration))) + 1. # -- Jaffé et al. 2015
        K[ii] = g_c *  ((np.log(1. + concentration*s[ii]))/s[ii]) # -- Rhee et al 2017

    # -- estimating the escape velocity
    for ii in range(dim):
        if r[ii] < cluster_radius:
            v_esc[ii] = np.sqrt( (2. * g * cluster_mass_kg * K[ii] )/ (3. * cluster_radius_meter) )
            escape_velocity[ii] = v_esc[ii]/1000. # escape velocity is in km s-1
        elif r[ii] >= cluster_radius:
            v_esc[ii] = np.sqrt( (2. * g * cluster_mass_kg * K[ii] )/ (3. * cluster_radius_meter) ) # -- Rhee et al. 2017
#            v_esc[ii] = np.sqrt( (2. * g * cluster_mass_kg)/ (3. * cluster_radius_meter**s[ii]) ) # -- Jaffé et al. 2015
            escape_velocity[ii] = v_esc[ii]/1000. # escape velocity is in km s-1
     
    # -- return output array
    return escape_velocity

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def peculiar_velocity_substructures(redshift_galaxy, label_galaxy, reshift_substructure, label_substructure, redshift_cluster):
	
	""" calsagos.cluster_kinematics.peculiar_velocity_substructures(redshift_galaxy, label_galaxy, reshift_substructure, label_substructure, redshift_cluster)
	
	Function that estimates peculiar velocities of galaxies
	with respect to its host sctructute. In the case of 
	galaxies inside substrucures, the peculiar velocity
	is estimated with respect to the central redshift of the
	substructure. However, in the case of galaxies out of
	substructures but are part of the cluster the peculiar 
	velocity of these galaxies is estimated with respect to 
	the cluster.
    
    This function estimate the peculiar velocity of galaxies
    using the equation published in Harrison 1974 and based 
    on peculiar_velocity function from cluster_kimenatics 
    module from calsagos. 

    This funcion was develop by D. E. Olave-Rojas
    (08/07/2024)
    
	:param redshift_galaxy: array with redshift of
        galaxies in the region of a cluster
    :param label_galaxy: array with the label of each
		galaxy indicating if are or not part of a
        substructure. Galaxies with label_galaxy == -1
        are part of a cluster however are not part of
        a substructure, and galaxies with label_galaxy >= 0
        are part of a substructure.
    :param redshift_substructure: array with central
		redshift of the substructure
    :param label_substructure: array with the label of each 
    	substructure without the label of principal halo
    :param redshift_cluster: central redshift of a galaxy 
    	cluster

	:type redshift_galaxy 			: array
    :type label_galaxy 				: array
    :type redshift_substructure 	: array
    :type label_substructure		: array
    :type cluster_redshift			: int, float

    :returns: peculiar velocity of galaxies with
		respect this host structure
	:rtype: array

    .. note::

	The returned velocity is in km/s 

	"""

    # -- defining the number of structures considering the principal halo
	dim_groups = len(label_substructure)

	# -- defining the number of galaxies in the calatog considering all of them
	dim_sample = len(label_galaxy)
    
    # -- START OF LOOP --
    # -- defining output quantities
	peculiar_velocity = np.zeros(dim_sample)

	for jj in range(dim_sample):

		if label_galaxy[jj] == -1:

			peculiar_velocity[jj] = c * ((redshift_galaxy[jj] - redshift_cluster)) / (1.0 + redshift_cluster)
		
		else:

			for ii in range(0,dim_groups): 

				if label_galaxy[jj] == label_substructure[ii]:

					peculiar_velocity[jj] = c * ((redshift_galaxy[jj] - reshift_substructure[ii])) / (1.0 + reshift_substructure[ii])
					
        
	# -- END OF LOOP
	# -- return cluster velocity dispersion
	return peculiar_velocity


#####################################################################################################################################################################################
#####################################################################################################################################################################################

def velocity_dispersion_substructures(id_galaxy, redshift_galaxy, velocity_galaxy, id_groups, redshift_groups, n_bootstrap):

    """ calsagos.velocity_dispersion_substructures(id_galaxy, redshift_galaxy, velocity_galaxy, id_groups, redshift_groups, n_bootstrap)
	
    Function that estimates velocity dispersion and it error
    in each substructure.

    Funcion that estimates the uncertainty on
    cluster velocity dispersion using a 
    boostrap technique

    This function estimate the peculiar velocity of galaxies
    using the equation published in Harrison 1974 and based 
    on peculiar_velocity function from cluster_kimenatics 
    module from calsagos. 

    This funcion was develop by D. E. Olave-Rojas
    (08/08/2024) and was based on the function
    calc_clumberi_cluster_velocity_dispersion_error
    from calsagos

    :param id_galaxy: array with the label of each
        galaxy indicating if are or not part of a
        substructure. Galaxies with label_galaxy == -1
        are part of a cluster however are not part of
        a substructure, and galaxies with label_galaxy >= 0
        are part of a substructure.    
    :param velocity_galaxy: array with central
        redshift of the substructure
    :param id_groups: array with the label of each 
        substructure without the label of principal halo
    :param redshift_groups: array with the redshift of each 
        substructure without the label of principal halo
    :param n_boostrap: number of bootstrap 
        simulations

    :type id_galaxy 			: array
    :type redshift_galaxy 		: array
    :type velocity_galaxy     	: array
    :type id_groups         	: array
    :type redshift_groups       : array
    :type n_bootstrap: int

    :type cluster_redshift		: int, float

    :returns: velocity dispersion of each substructure
        and their errors
    :rtype: array

    .. note::

    The returned dispersion velocity is in km/s 

    """
      
    # -- removing galaxies that are part of the principal halo
    good = np.where(id_galaxy >= 0)[0]

	# -- selecting only galaxies in a substructure
    good_velocity = velocity_galaxy[good]
    good_z = redshift_galaxy[good]
    good_label = id_galaxy[good]

	# -- defining unique label to each groups
    groups = np.unique(good_label) 
    
    # -- defining the number of groups
    dim_groups = len(groups)

    # -- defining output quantities
    sigma_group = np.zeros(dim_groups)
    sigma_uncertainty_group = np.zeros(dim_groups)

	# -- START OF LOOP --
    for ii in range(0,dim_groups): 

        # -- selecting a single substructure
        n_groups = np.where(good_label == ii)[0]
        single_group = np.where(id_groups == ii)[0] 

        # -- selecting parameter of the single structure
        single_group_velocity = good_velocity[n_groups]
        single_group_z = good_z[n_groups]

        # -- selecting central parameter of the single structure
        central_z_group = redshift_groups[single_group]
        single_group_id = id_groups[single_group]
        single_group_id = round(float(single_group_id))

        # -- Estimate the velocity dispersion of the substructure
        sigma = biweight_scale(single_group_velocity)

        print("estimating velocity dispersion on group :", single_group_id)

        # -- calculating uncertainty on velocity dispersion
        sigma_uncertainty = calc_clumberi_cluster_velocity_dispersion_error(single_group_z, central_z_group, n_bootstrap)
		
        if ii != 0:
            sigma_group = np.append(sigma_group, sigma)
            sigma_uncertainty_group = np.append(sigma_uncertainty_group, sigma_uncertainty)

        # -- process of the first iteration: defining variables
        else:                
            sigma_group = sigma
            sigma_uncertainty_group = sigma_uncertainty
        
	# -- building matrix with output quantities
    output_quantities = np.array([groups, sigma_group, sigma_uncertainty_group], dtype=object)

    # -- returning output quantity
    return output_quantities

#####################################################################################################################################################################################
#####################################################################################################################################################################################

def velocity_test(id_galaxy, velocity_galaxy):

    """ velocity_test(id_galaxy, velocity_galaxy)
	
    Function that perfoms the Kolmogorov-Smirnov and
    Shapiro-Wilk test to evaluate the normality of 
    the velocity distribution. 

    This funcion was develop by D. E. Olave-Rojas
    (08/08/2024)

    :param id_galaxy: array with the label of each
        galaxy indicating if are or not part of a
        substructure. Galaxies with label_galaxy == -1
        are part of a cluster however are not part of
        a substructure, and galaxies with label_galaxy >= 0
        are part of a substructure.    
    :param velocity_galaxy: array with central
        redshift of the substructure
    
    :type id_galaxy 			: array
    :type velocity_galaxy     	: array

    :returns: Kolmogorov-Smirnov and
        Shapiro-Wilk index
    :rtype: array

    """
    # -- removing galaxies that are part of the principal halo
    good = np.where(id_galaxy >= 0)[0]

    # -- selecting only galaxies in a substructure
    good_label = id_galaxy[good]
    good_velocity = velocity_galaxy[good]

    # -- defining unique label to each groups
    groups = np.unique(good_label) 

    # -- defining the number of groups
    dim_groups = len(groups)

    # -- defining output quantities
    gauss_ks_group = np.zeros(dim_groups)
    gauss_sw_group = np.zeros(dim_groups)

    # -- START OF LOOP --
    for ii in range(0,dim_groups): 

        # -- selecting a single substructure
        n_groups = np.where(good_label == ii)[0]

        # -- selecting parameter of the single structure
        single_group_velocity = good_velocity[n_groups]

        # -- Kolmogorov-Smirnov test
        gauss_ks = kstest(single_group_velocity, 'norm')[0]

        # -- Shapiro-Wilks test
        gauss_sw = stats.shapiro(single_group_velocity)[0]

        if ii != 0:
            gauss_ks_group = np.append(gauss_ks_group, gauss_ks)
            gauss_sw_group = np.append(gauss_sw_group, gauss_sw)

        # -- process of the first iteration: defining variables
        else:                
            gauss_ks_group = gauss_ks
            gauss_sw_group = gauss_sw

	# -- building matrix with output quantities
    central_values = np.array([groups, gauss_ks_group, gauss_sw_group], dtype=object)

    # -- returning output quantity
    return central_values

#####################################################################################################################################################################################
#####################################################################################################################################################################################
