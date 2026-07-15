'''
Hail Trajectory Growth Model code

Described in detail in Kumjian and Lombardo (2020, JAS)

Also used in Kumjian et al. (2021, JAS); Lin and Kumjian (2021, JAS), ...
Updated for Lin et al. (2024)

Changes for Fischer et al. (2026): July 10 2025

Rewritten into python with edited physics by Lydia Spychalla: July 14 2026
'''




'''
Import dependencies
'''
import numpy as np
import xarray as xr
from tqdm import tqdm
import glob
import scipy.io
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma


def define_constants():
    '''
    Defines constants and thresholds for thermodynamics, physics, and collection.
    
    Note: the thresholds defined in the "collection parameterization" and 
    "physics growth thresholds" section are rather arbitrary (albeit reasonable).
    Could be manually changed. Currently no flags to do this at setup.
    '''
    
    #----------------------------------------------------------------
    # Physical and Thermodynamic Constants
    #----------------------------------------------------------------

    global g, cpw, Rv, Rdry, p0, T0, lv, lf, ls, rhol, rhosolid, es0
    g        = 9.81      # gravitational acceleration               [m s^-2]
    cpw      = 4187.0    # specific heat capacity of liquid at constant pressure [J/kg/K] -- note generally is a function of T
    Rv       = 461.5     # gas constant for vapor                   [J/kg/K]
    Rdry     = 287.0     # gas constant for dry air                 [J/kg/K]
    p0       = 100000.0  # reference surface pressure                   [Pa]
    T0       = 273.15    # reference temperature                         [K]
    lv       = 2.50e6    # enthalpy of vaporization                   [J/kg]    **NOTE THESE ARE FUNCTIONS OF T, COULD
    lf       = 3.33e5    # enthalpy of fusion/melting                 [J/kg]    **ADD EXPRESSIONS TO MAKE MORE ACCURATE
    ls       = lv+lf     # enthalpy of sublimation                    [J/kg]    *****
    rhol     = 1000.0    # density of liquid water                  [kg/m^3]
    rhosolid =  917.0    # density of solid ice                     [kg/m^3]
    es0      = 611.0     # equilibrium vapor pressure at T=T0=273.15 K  [Pa]
    
    
    
    #----------------------------------------------------------------
    # Cloud Droplets, Rain Drops, and Collection Parameterization
    #----------------------------------------------------------------

    global CCN, NCC, Ecthresh, Dcthresh, Eci, vIMP
    CCN = 250       # number concentration of drops per CC from Morrison (250)
    NCC = CCN * 1e6 # convert number concentration to 1/m3
    
    Ecthresh = 0.1 # Threshold low value below which collection efficiency linearly drops off to zero
    Dcthresh = 5.0 # Threshold droplet diameter (microns) below which collection efficiency linearly drops off to zero
    Eci      = 1.0 # Collection efficiency of ice is unity
    
    vIMP     = 0.65 # impact velocity reduction following Rasmussen and Heymsfield 1985. Could be improved.
    
    #----------------------------------------------------------------
    # Physics growth thresholds
    #----------------------------------------------------------------

    global liquid_shell_growing_threshold_mass, liquid_shell_growing_threshold_radius
    global liquid_shell_physics_radius_threshold_1, liquid_shell_physics_radius_threshold_2
    global rhomin_rime, rhomin_sponge, rho_soak_freeze_thresh, rho_soak_freeze_fraction

    liquid_shell_growing_threshold_mass     =   1e-12  # Mass threshold for liquid shell growth physics (vs. dry hailstone)
    liquid_shell_growing_threshold_radius   =   1e-6   # Radius threshold for turning on thermal conduction through liquid (prevents blowup)
    
    # Liquid shell heat transfer regime radius thresholds
    liquid_shell_physics_radius_threshold_1 = 0.5e-3 # #1---thinner than this radius, the liquid shell sees all external heat transfer
    liquid_shell_physics_radius_threshold_2 =   1e-3 # #2---thicker than this radius, only thermal conduction through liquid
    
    rhomin_rime   =  100.0 # minimum density allowed for rime       [kg/m3]
    rhomin_sponge =  500.0 # minimum density allowed for wet growth [kg/m3]
    rho_soak_freeze_thresh = rhosolid # density at which to stop freezing soaked mass [kg/m3]
    rho_soak_freeze_fraction = 0.4 # fraction of heat transfer for freezing going towards freezing soaked mass in wet growth

    return


# 
# 

def drag_parameterization_velocity(vrel_hail, D_max_tumble, nu, Cdrag):
    '''
    Manual Reynold's Number calculation via fall speed with constant drag coefficient. (choose_drag = 0)
    '''
    # Reynolds Number
    N_Re = vrel_hail * (D_max_tumble / 1000) / nu

    # Set all drag coefficient values equal to constant Cdrag
    # No offset for terminal velocity vs. relative wind velocity because velocity and Reynolds number are not used in Cde definition
    CdeT = np.ones_like(D_max_tumble)*Cdrag
    Cde  = np.ones_like(D_max_tumble)*Cdrag
    
    return N_Re, Cde, CdeT

def drag_parameterization_constant(N_Be, velocity_ratio, lobes_volume_ratio, Cdrag):
    '''   
    Constant drag coefficient specified by setup Reynolds number calculated via Best-Davies Number (choose_drag = 1)
    '''

    # Set all drag coefficient values equal to constant Cdrag
    # No offset for terminal velocity vs. relative wind velocity because velocity and Reynolds number are not used in Cde definition
    CdeT = Cdrag / lobes_volume_ratio
    Cde  = Cdrag / lobes_volume_ratio
    
    N_ReT = (N_Be / CdeT)**0.5    #Reynolds number derived from Best number
    N_Re = N_ReT * velocity_ratio #Reynolds number adjustment for wind-relative velocity

    return N_Re, Cde, CdeT


def drag_parameterization_heymsfield(N_Be, velocity_ratio, lobes_volume_ratio):
    '''
    This parameterization uses the Heymsfield et al. (2020) "Corrigendum" parameterization for N_Be-N_ReT relationships to solve for N_ReT and CdeT using N-Be. (choose_drag = 2)
    '''

    # Calculate the drag coefficient via Best-Davies number following Heymsfield et al. (2020)
    # No offset for terminal velocity vs. relative wind velocity because velocity and Reynolds number are not used in Cde definition
    CdeT = 6.9252 * N_Be**(-0.1200) / lobes_volume_ratio
    Cde  = 6.9252 * N_Be**(-0.1200) / lobes_volume_ratio

    N_ReT = (N_Be / CdeT)**0.5     #Reynolds number derived from Best number  
    N_Re = N_ReT * velocity_ratio  #Reynolds number adjustment for wind-relative velocity

    return N_Re, Cde, CdeT


def Cde_3(N_Re, lobes_volume_ratio):
    '''
    Linear-log drag coefficient and Reynolds Number relationship from Clift and Gauvin (1971), used in choose_drag=3
    '''

    Cde = (24 / N_Re * (1 + 0.15 * N_Re**0.687) + 0.42 / (1 + 42500 / N_Re**1.16)) * (1/lobes_volume_ratio)
    return Cde


def Cde_4(N_Re, Ks, Kn):
    '''
    Linear-log drag coefficient and Reynolds Number relationship from Bagheri and Bonadonna (2016), used in choose_drag = 4
    '''
    Cde = 24 * Ks / N_Re * (1+ 0.125 * (N_Re * Kn / Ks)**(2/3)) + 0.46 * Kn / (1 + 5330 / (N_Re * Kn / Ks))
    return Cde



def drag_parameterization_clift_and_gauvin(N_Be, velocity_ratio, lobes_volume_ratio, Cdrag):
    '''
    Clift and Gauvin (1971) Drag, assumes sphere (choose_drag = 3)
    '''

    N_ReT = (N_Be / Cdrag)**0.5 #Reynolds number derived from Best number

    N_Re_prior = N_ReT * 2   # Force our iterative solve condition to be false for the first iteration 
    found = np.zeros_like(N_ReT).astype(bool) # No N_Re~Cde values have been found yet
    CdeT  = np. ones_like(N_Be )*Cdrag        # Initial drag guess = Cdrag
    
    i = 0    # Counter to prevent infinite loop
    
    # Iteratively solve for a N_Re and Cde that correspond to one another via the Best-Davies Number and abide by the linear-log drag coefficient relationship
    while np.any(~found):  # Iterate until all hailstones' Reynolds numbers have been found  
        CdeT     [~found] = Cde_3(N_ReT[~found], lobes_volume_ratio[~found]) # Drag coefficient (Clift and Gauvin 1971)
        N_ReT    [~found] = (N_Be[~found] / CdeT[~found])**0.5 #Reynolds number derived from Best number

        # If the change in N_Re over this iteration is less than 10% we've converged on the new N_Re and Cde
        found[~found] = abs(N_Re_prior[~found] / N_ReT[~found]) < 1.1
        N_Re_prior[~found] = N_ReT[~found] # Update prior N_Re holder
        
        #If the Reynolds Number and Drag Coefficients haven't been found after 1000 iterations, something is likely broken.
        if i > 1000:
            raise Exception(f"Could not find satisfactory CdeT and N_ReT for N_Be = {N_Be[~found]}")
        i += 1

    N_Re = N_ReT * velocity_ratio #Reynolds number adjustment for wind-relative velocity
    Cde = Cde_3(N_Re, lobes_volume_ratio) #Solve for wind-relative drag coefficient
    
    return N_Re, Cde, CdeT



def drag_parameterization_bagheri_and_bonadonna(N_Be, velocity_ratio, lobes_volume_ratio, Cdrag, rho_ice, rhoall, aspect_ratio, int_ratio):
    '''
    Bagheri and Bonadonna (2016) Drag, doesn't work well, not recommended!! (choose_drag = 4)
    '''
    
    Fs = np.real(aspect_ratio    * int_ratio**1.3 * (1 / lobes_volume_ratio))
    Fn = np.real(aspect_ratio**2 * int_ratio      * (1 / lobes_volume_ratio))
    
    density_frac  =   rho_ice / rhoall
    ac = 0.45 + 10 / (np.exp(2.5 * np.log(density_frac)) +  30)
    bc =    1 - 37 / (np.exp(3   * np.log(density_frac)) + 100)
    
    Kn = 10**(ac * (- np.log (Fn))**bc)
    Ks = (Fs**(1/3) + Fs**(-1/3)) / 2

    N_ReT = np.real((N_Be / Cdrag)**0.5) #Reynolds number derived from Best number    
    N_Re_prior = N_ReT * 2   # Force our iterative solve condition to be false for the first iteration 
    found = np.zeros_like(N_ReT).astype(bool) # No N_Re~Cde values have been found yet
    CdeT  = np. ones_like(N_Be )*Cdrag        # Initial drag guess = Cdrag
    
    i = 0
    while np.any(~found):
        CdeT[~found] = Cde_4(N_ReT[~found], Ks[~found], Kn[~found]) # Drag coefficient (Bagheri and Bonadonna 2016)
        N_ReT[~found] = np.real((N_Be[~found] / CdeT[~found])**0.5) # Reynolds number derived from Best number
        
        # If the change in N_Re over this iteration is less than 10% we've converged on the new N_Re and Cde
        found[~found] = abs(N_Re_prior[~found] / N_ReT[~found]) < 1.1
        N_Re_prior[~found] = N_ReT[~found] # Update prior N_Re holder
        
        #If the Reynolds Number and Drag Coefficients haven't been found after 1000 iterations, something is likely broken.
        if i > 1000:
            raise Exception(f"Could not find satisfactory CdeT and N_ReT for N_Be = {N_Be[~found]}")
        i += 1
        
    N_Re = N_ReT * velocity_ratio #Reynolds number adjustment for wind-relative velocity
    Cde = Cde_4(N_Re, lobes_volume_ratio) #Solve for wind-relative drag coefficient
    
    return N_Re, Cde, CdeT


def drag_parameterization_theis(N_Be, D_max, vrel_hail, vT_hail, nu):
    '''
    Theis et al. (2026) Drag. Default drag parameterization! (choose_drag = 5)
    '''
    
    # Hailstone sphericity parameterization from Theis et al. (2026)
    psi = 1 - 0.5/150*D_max
    
    # Reynolds Number from maximum dimension (will be an overestimate generally)
    N_Re_max_rel = vrel_hail * (D_max / 1000) / nu
    N_Re_max_T   =   vT_hail * (D_max / 1000) / nu

    # Theis et al. parameterization of Cde
    h_rel = np.exp(-N_Re_max_rel/6000)**(2.15 - 1.19*psi)
    h_T   = np.exp(-N_Re_max_T  /6000)**(2.15 - 1.19*psi)
    Cde  = (0.58 * (1 + 6.27/N_Re_max_rel**(1/2))**2)*h_rel + 0.58 * (1 - h_rel)
    CdeT = (0.58 * (1 + 6.27/N_Re_max_T  **(1/2))**2)*h_T   + 0.58 * (1 - h_T  )

    #Calcualte Reynolds number from Best number and the parameterized drag coefficient
    N_Re = (N_Be / CdeT)**0.5 
    
    return N_Re, Cde, CdeT
    


def drag_parameterization_wrapper(m_core, D_max, A_coll_tumble, aspect_ratio, int_ratio, vt_hail, vrel_hail, D_max_tumble, hailstone_total_density, rho_air, T_inf, choose_lobes, choose_drag, Cdrag):
    '''
    Functions to calculate and return Reynolds number, drag coefficient and lobes adjustment
    '''
    
    # Calculate the volume ratio if lobes are desired
    if choose_lobes == 1:
        lobes_volume_ratio = -0.0036*D_max + 1.0403 # New equation from IBHS and Soderholm data
        #lobes_volume_ratio = -0.002567 * D_max + 0.9853 #Old equation Lin et al. (2024), eqn. 30
    else:
        #No lobes---volume ratio equals 0
        lobes_volume_ratio = np.ones_like(D_max)

    # Calculate air properties for Best Number
    eta_a = (0.379565 + 0.0049*T_inf)*1e-5 # dynamic viscosity (kg/m/s)
    nu    = eta_a / rho_air    # kinematic viscosity
    # Calculate Best number, (e.g., Lin et al. 2024, eqn. 25)
    N_Be =  2 * m_core * g * D_max**2 * 1e-6 / (nu**2 * rho_air * A_coll_tumble)  

    # Calculate the ratio of velocities to scale terminal N_Re and wind-relateive N_Re
    velocity_ratio = vrel_hail / vt_hail

    if choose_drag == 0: # constant Cdrag, manual Reynolds
        N_Re, Cde, CdeT = drag_parameterization_velocity(vrel_hail, D_max_tumble, nu, Cdrag)
        
    elif choose_drag == 1: # constant Cdrag, Reynolds via Best-Davies Number
        N_Re, Cde, CdeT = drag_parameterization_constant(N_Be, velocity_ratio, lobes_volume_ratio, Cdrag)
        
    elif choose_drag == 2: # Heymsfield drag, Reynolds via Best-Davies Number
        N_Re, Cde, CdeT = drag_parameterization_heymsfield(N_Be, velocity_ratio, lobes_volume_ratio)
        
    elif choose_drag == 4: # Bagheri and Bonadonna drag, Reynolds via Best-Davies Number
        N_Re, Cde, CdeT = drag_parameterization_bagheri_and_bonadonna(N_Be, velocity_ratio, lobes_volume_ratio, Cdrag, hailstone_total_density, rho_air, aspect_ratio, int_ratio)
        
    elif choose_drag == 3: # (choose_drag == 3): Clift and Gauvin drag (spherical), Reynolds via Best-Davies Number
        N_Re, Cde, CdeT = drag_parameterization_clift_and_gauvin(N_Be, velocity_ratio, lobes_volume_ratio, Cdrag)
    else: # choose_drag = 5, Theis et al. (2026) drag parameterization
        N_Re, Cde, CdeT = drag_parameterization_theis(N_Be, D_max, vrel_hail, vt_hail, nu)
            
    return N_Re, Cde, CdeT, lobes_volume_ratio



def get_dimensionless_flow_numbers(T, rhoall, Dv):
    '''
    Function to return Prandtl and Schmidt numbers. Definitions can be found in RH87 in their List of Symbols
    '''
    
    eta_a = (0.379565 + 0.0049*T)*1e-5                    # dynamic viscosity [kg/m/s]
    Kair  = 9.1018e-11 * T**2 + 8.8197e-8 * T - 1.0654e-5 # thermal diffusivity of air [m2/s]
    nu    = eta_a / rhoall                                # kinematic viscosity [m2/s]
    
    N_Pr = nu / Kair   #Prandtl Number
    N_Sc = nu /   Dv   #Schmidt Number
    return N_Pr, N_Sc


    
def get_rh87_diffusion_and_conduction_coefficients(N_Re, aspect_ratio, N_Pr, N_Sc, choose_chi):
    '''
    Function to return generalized coefficients for thermal conduction in air and for vapor diffusion. These come from Rasmussen and  Heymsfield (1987a). Note that these coefficients are not specifically defined as such, but can be pulled from eqns. 3-5 of Table 1 for the respective Reynold's regimes. Beware: the names of intermediate  quantities here are also called "transfer coefficient", here and in RH87.
    '''
    
    # Reynolds regimes from RH87 Table 1
    reynolds_regime_3 = (N_Re < 6000)
    reynolds_regime_4 = (N_Re >= 6000)*(N_Re < 2e4)
    reynolds_regime_5 = (N_Re >= 2e4)
    
    # chi from Bailey and Macklin (1968), what RH87 calls the "heat transfer coefficient"
    chi = np.zeros_like(N_Re)*np.nan
    if choose_chi <= 0 or choose_chi > 1:
        # Increases with increasing oblatness - Maclin63.
        aspect_ratio_scale = (1-aspect_ratio) * 0.25
        chi[reynolds_regime_4] = (aspect_ratio_scale*np.ones_like(N_Re) + 0.76)[reynolds_regime_4]
        chi[reynolds_regime_5] = (aspect_ratio_scale + 0.57 + 9.0e-6 * N_Re)[reynolds_regime_5]
    else:
        # For constant chi equal to specified
        chi[reynolds_regime_4] = (choose_chi  *   np.ones_like(N_Re))[reynolds_regime_4]
        chi[reynolds_regime_5] = (choose_chi - 0.19 + 9.0e-6 * N_Re) [reynolds_regime_5]

    # Rasmussen and Heymsfield Table A1
    ventH = 0.78 + 0.308*np.sqrt(N_Re)*(N_Pr**(1.0/3.0)) #\overline{f_h}
    ventV = 0.78 + 0.308*np.sqrt(N_Re)*(N_Sc**(1.0/3.0)) #\overline{f_v}
    
    # Get Ventilation Coefficients as defined by RH87 in Table 1
    rh87_cond_coef = np.where(reynolds_regime_3, 2.0*ventH, chi*np.sqrt(N_Re)*(N_Pr**(1.0/3.0))) #ventilation coefficient for thermal energy
    rh87_diff_coef = np.where(reynolds_regime_3, 2.0*ventV, chi*np.sqrt(N_Re)*(N_Sc**(1.0/3.0))) #ventilation coefficient for vapor

    return rh87_diff_coef, rh87_cond_coef



def radii_for_liquid_covered_stone(D_max, D_int, D_min, m_shell):
    '''
    Get radii for the ice core and liquid shell (assumes spherical). Note that the sizes to this function are in [mm], but the output radii are in [m]. Assumes a sphere of equivalent surface area to the ellipsoidal hailstone.
    '''
    
    # Pull out the aspect and intermediate ratios
    aspect_ratio = D_min / D_max
    int_ratio    = D_int / D_max
    
    #Calculate total partical dimensions, adding liquid and solid
    liquid_volume   = m_shell / rhol * 1e9              # volume of liquid shell mass   (calculated via mass and density) [mm^3]
    ice_core_volume = D_max**3*int_ratio*aspect_ratio * np.pi / 6 # volume of hailstone ellipsoid (calculated by shape [density unknown]) [mm^3]
    full_particle_volume = liquid_volume + ice_core_volume        # volume of the full particle (ice core and liquid shell) [mm^3]

    # Solve for the maximum dimension of the ice+liquid assuming the same ellipsoidal shape for both
    liq_ice_D_max = (6/np.pi * full_particle_volume / (aspect_ratio * int_ratio))**(1/3) #total partical max dim [mm]

    # Use the equation for the surface area of an ellipsoid vs. surface area of a sphere as a scale for maximum dimension --> equivalent surface area diameter
    D_sph_eq         =         D_max * ((int_ratio**1.6 + aspect_ratio**1.6 + int_ratio**1.6*aspect_ratio**1.6)/3)**(1/3.2) # ice core only
    liq_ice_D_sph_eq = liq_ice_D_max * ((int_ratio**1.6 + aspect_ratio**1.6 + int_ratio**1.6*aspect_ratio**1.6)/3)**(1/3.2) # ice core + liquid shell 

    radius_ice_core     =         D_sph_eq / 1000 / 2 # convert diameters in mm to radii in m
    radius_liquid_shell = liq_ice_D_sph_eq / 1000 / 2 # convert diameters in mm to radii in m

    return radius_ice_core, radius_liquid_shell


def accretion(cloud_water_dm_dt, rain_water_dm_dt, cloud_ice_dm_dt, has_surface_liquid, T_liquid_shell, cpi, T_inf):
    '''
    Function to calculate the accreted mass transfer (ice and liquid) and corresponding heat transfer
    '''

    # MASS GROWTH FROM ACCRETION  dm/dt|acc#
    # Mass accretion rate from cloud and rain water (kg/s)
    dm_dt_accreted_liq = cloud_water_dm_dt + rain_water_dm_dt
    # Ice accretion rate (ice accretion only 
    dm_dt_accreted_ice = np.where(has_surface_liquid, cloud_ice_dm_dt, 0)

    # HEAT TRANSFER FROM HEATING ACCRETED MASS TO EQUILIBRIUM (dq/dt|acc)
    # Will be negative (net cooling for hailstone) if T_inf < hailstone_skin_temperature
    dq_dt_acc_dT = (dm_dt_accreted_liq * cpw + dm_dt_accreted_ice * cpi) * (T_inf - T_liquid_shell)

    # HEAT TRANSFER FROM FREEZING
    # Assume that all accreted ice melts into the liquid shell and then can refreeze later onto the ice core
    # Melting of accreted ice is always a heat sink (net cooling for hailstone)
    dq_dt_ice_acc_melt = -dm_dt_accreted_ice * lf

    # SUM UP ALL HEAT TRANSFER FROM ACCRETION
    dq_dt_accretion = dq_dt_acc_dT + dq_dt_ice_acc_melt

    return dm_dt_accreted_liq, dm_dt_accreted_ice, dq_dt_accretion


def continuous_collection(A_coll_tumble, fall_speed, sfcdens, rho_dry_air, Ecc, Ecr, Eci,
                             velrain_Q, LWCc, LWCr, IWC):
    '''
    Function to calculate the total collectable mass in the hailstone's sweep-out volume in (kg/s). Assumes continuous collection with prescribed efficiencies. Note: collected ice will only be accreted if the hailstone has a liquid shell
    '''
    
    # Collection kernels assuming continuous collection of LWC/IWC (m^3/s)
    # collection kernel for cloud droplets
    Kerc = Ecc * A_coll_tumble * fall_speed
    # collection kernel for raindrops, uses mass weighted fallspeed and density correction for rain. Ignores rain sizes for now.
    Kerr = Ecr * A_coll_tumble * np.abs(fall_speed-velrain_Q * np.sqrt(sfcdens/rho_dry_air))
    # collection kernel for ice particles
    Keri = Eci          * A_coll_tumble * fall_speed
    
    # mass collision rate from each hydrometeor type (kg/s)
    cloud_water_collection = Kerc * LWCc
    rain_water_collection  = Kerr * LWCr
    cloud_ice_collection   = Keri * IWC 

    return cloud_water_collection, rain_water_collection, cloud_ice_collection



def density_of_rime_accretion(fall_speed, T_core, T_inf, D_c, vIMP):
    '''
    Function to determine the density of accreted rime (Heymsfield and Pflaum 1985 parameterization of rime density)
    '''
    
    rime_density = np.zeros_like(T_core)

    # When T0 = T_core, the density is set to rhosolid: warm hailstone --> close to wet growth --> dense hailstone
    warm_hailstone = T_core >= T0
    rime_density[ warm_hailstone] = rhosolid
    
    # Otherwise, solve for the riming density following Kumjian and Lombardo (2020)
    A_rime = (0.5*D_c[~warm_hailstone] * (fall_speed[~warm_hailstone]*vIMP) / (T0 - T_core[~warm_hailstone]))
    
    rime_density[~warm_hailstone] = np.where((A_rime < 1.60)&(T_inf[~warm_hailstone] > 268.15), 1000.0*np.exp(-0.03115-1.7030*A_rime+0.9116*A_rime**2-0.1224*A_rime**3), 0.30 * A_rime**0.44 * 1000) #conversion to kg/m3

    # Catch any riming densities that are above or below the set thresholds
    rime_density[rime_density > rhosolid   ] = rhosolid     #Max allowed is solid ice
    rime_density[rime_density < rhomin_rime] = rhomin_rime  #Min allowed is min density threshold, default 500

    return rime_density


def diffusion_and_conduction(hail_radius, rh87_cond_coef, rh87_diff_coef, T_sfc, hailstone_physics_T, T_inf, Dv, rhov, kT, enthalpy, delt):
    '''
    Function to calculate mass and heat transfer by conduction and diffusion. Uses diffusion/conduction coefficients determined from the Reynolds number regime.
    '''
    
    # Calculate equilibrium vapor density at hailstone surface
    rhossfc = es0 * np.exp(ls / Rv * (1/T0-1/T_sfc)) / Rv / T_sfc

    # Heat transfer rate from conduction and diffusion (enthalpy portion only)
    dq_dt_conduction =  - 2 * np.pi * hail_radius * rh87_cond_coef *       kT      * (T_sfc - T_inf)
    dq_dt_diffusion_L  = -2 * np.pi * hail_radius * rh87_diff_coef * enthalpy * Dv * (rhossfc - rhov ) 

    # Mass transfer rate from diffusion
    dm_dt_diffusion = dq_dt_diffusion_L / enthalpy

    # Additional heat transfer to the liquid shell from the warm/cold boundary
    # Heat transfer by diffusion is more positive if the diffused mass is coming in at a warmer temperature than the desired physics temperature of the shell
    dq_dt_diffusion_dT = dm_dt_diffusion * cpw * (T_sfc - hailstone_physics_T)

    # Sum up total heat transfer by vapor diffusion
    dq_dt_diffusion = dq_dt_diffusion_L + dq_dt_diffusion_dT
    
    return dm_dt_diffusion, dq_dt_diffusion, dq_dt_conduction




def conduction_through_liquid(radius_ice, radius_liquid, T_average_liquid, Tsfc):
    '''
    Function to calculate heat transfer by conduction through liquid.
    '''
    
    # Only conduct heat through liquid if liquid shell exists (exceeds threshold)
    has_liquid = radius_liquid - radius_ice > liquid_shell_growing_threshold_radius

    # Thermal conductivity of water [cal cm^-1 s^-1, degC^-1], can be found ni RH87 (was called Kw2)
    kw  = 135.8e-5 * np.exp(3.473e-3*(T_average_liquid - 273.15) - 3.823e-5*(T_average_liquid - 273.15)**2 + 1.087e-6*(T_average_liquid - 273.15)**3) * 418.4

    # Set default thermal conduction through liquid to zero
    dq_dt_liq_conduction = np.zeros_like(Tsfc)
    # For hailstone with a liquid shell, calculate the thermal conduction through liquid
    dq_dt_liq_conduction[has_liquid] = -4* np.pi * kw[has_liquid] * radius_ice[has_liquid] * radius_liquid[has_liquid] * (T0-Tsfc[has_liquid]) / (radius_liquid[has_liquid]-radius_ice[has_liquid])
    
    return dq_dt_liq_conduction


def shed_excess_liquid(m_core, D_avg_eq_sfc, volume, liquid_mass_before_shed):
    '''
    Function to shed excess liquid from the liquid shell. Shedding follows non-spherical hailstone adjustment to RH87 done by Lin et al. (2024)
    '''
    
    # Critical mass of liquid retainable for a sphere following RH87
    mwcritsph = (2.68e-4 + 0.1389 * m_core)
    # Equivalent volume diameter
    D_avg_eq_vol = 2 * (volume * 3 / 4 / np.pi)**(1/3)
    # Adjustment to the spherical equation of RH87 for nonspherical hailstones by Lin et al. (2024)
    mwcrit = (D_avg_eq_sfc * 1e-3 / D_avg_eq_vol)**2 * mwcritsph

    #Shed anything in excess of mwcrit; this should be passed to next timestep
    liquid_mass_after_shed = np.minimum(mwcrit, liquid_mass_before_shed)  
    
    return liquid_mass_after_shed



def soaking(m_core, volume, m_shell, prior_soaked_mass, temperature_for_internal_physics, delt):
    '''
    Function to calculate soaking. Allows unfrozen water to soak into hailstone until all air pockets are filled and allows water to seep back onto hailstone surface if hailstone becomes over-soaked.
    '''
    
    # Compute how much mass is needed to fill in any air pockets with liquid water [kg]
    maximum_soakable = rhol * (volume - m_core/rhosolid - prior_soaked_mass/rhol)

    # Soak any available liquid mass up to the maximum_soakable limit (full soaked hailstone
    additional_soaked_liquid_mass = np.minimum(m_shell, maximum_soakable)

    # Calculate the heat transfer caused by equilibrating liquid to the ice core or liquid shell during soaking or unsoaking
    dq_dt_heat_soaked = (additional_soaked_liquid_mass/delt) * cpw * (temperature_for_internal_physics-T0)
    
    return additional_soaked_liquid_mass, dq_dt_heat_soaked


def density_of_spongy_growth(FrozenFrac):
    '''
    Function to calculate the density of spongy growth (wet growth). Uses RH87 spongy growth parameterization.
    '''
    
    growth_density = ((1.0 - 0.08*FrozenFrac)*FrozenFrac) * 1000      # RH87 spongy growth parameterization
    growth_density[growth_density > rhosolid       ] = rhosolid       # making sure we don't get super-dense ice.
    growth_density[growth_density < rhomin_sponge  ] = rhomin_sponge  # making sure we don't get super-fluff ice.

    return growth_density



def calculate_surface_area(D_max, D_int, D_min, choose_lobes, lobes_volume_ratio):
    '''
    Function to get hailstone surface area characteristics. Returns hailstone surface area and the equivalent surface area spherical diameter.
    '''
    
    # Maximum cross-sectional area of the ellipsoid
    hail_A = np.pi * D_int/2 * D_max/2 * 1e-6

    # Hailstone elipsoidal surface area
    hail_sfc = np.pi*(((D_max*D_int)**1.6+(D_max*D_min)**1.6+(D_int*D_min)**1.6)/3)**(1/1.6)

    # Lobiness correction to hailstone surface area
    if choose_lobes == 1:
        #New equation from IBHS and Soderholm data
        hail_sfc = (0.3264*lobes_volume_ratio + 0.7846) * hail_sfc
        #Old equation following Lin et al. (2024), eqn 27
        #hail_sfc = (0.05837 * lobes_volume_ratio + 0.5506) * hail_sfc

    # Equivalent surface area spherical diameter
    D_avg_eq_sfc = (hail_sfc / np.pi)**0.5
    
    return hail_A, D_avg_eq_sfc



def tumbling_parameterization(D_max, D_int, D_min, hail_A, choose_tumble):
    '''
    Function to calculate the tumbling adjustment to hailstone collection area. Returns the cross-sectional area and maximum dimension reduction for collection and fall speed due to hailstone tumbling.
    '''
    
    # If no tumbling, the tumbling maximum dimension and cross-sectional area are their maximum values
    if choose_tumble == 0:
        D_max_tumble = D_max
        A_coll_tumble = hail_A

    # If specified tumbling, reduce the cross-sectional area and maximum dimension by choose_tumble and sqrt(choose_tumble), respectively
    elif choose_tumble < 1 and choose_tumble > 0:
        D_max_tumble = D_max * choose_tumble**0.5
        A_coll_tumble = hail_A * choose_tumble

    # Explicit tumbling from the Lin et al. (2024) lookup table
    else: #choose_tumble == 1

        # NOTE THAT THIS CODE WORKS FOR THE SPECIFIC TUMBLING ADJUSTMENT LOOKUP TABLE FROM LIN ET AL. (2024)!!
        # IF THE LOOKUP TABLE CHANGES, THE CODE WILL NEED TO BE ADJUSTED

        # The rows in the lookup table correspond to 0.01 increments of aspect ratio, ranging from 0.228 to 0.998
        # Pull the nearest row to the current aspect ratio
        aspect_ratio = D_min/D_max
        idx_aspect_ratio = np.round((aspect_ratio - 0.228)*100).astype(int)
        # The columns of the lookup table correspond to 0.05 increments of intermediate ratio, ranging from 0.5 to 1
        # Pull the nearest column to the current intermediate ratio
        int_ratio = D_int/D_max
        idx_int_ratio    = np.round((int_ratio - 0.5  )* 20).astype(int)

        # Force anything outside the table range to its nearest edge
        idx_aspect_ratio[idx_aspect_ratio <  0] =  0
        idx_aspect_ratio[idx_aspect_ratio > 77] = 77
        idx_int_ratio   [idx_int_ratio    <  0] =  0
        idx_int_ratio   [idx_int_ratio    > 10] = 10
        
        # Pull a random int between 0 and 1000 to define this timestep's tumbing impact for each hailstone
        randidx   = np.random.randint(0, 1000, D_max.shape[0])

        # Sample the lookup table for the int_ratio, aspect_ratio, and random index
        hail_Ar = tumbling_A0   [idx_int_ratio, idx_aspect_ratio, randidx]
        hail_ar = tumbling_Dmax0[idx_int_ratio, idx_aspect_ratio, randidx]

        # Reduce hailstone maximum dimension and cross-sectional area corresponding to the sampled tumbling table
        D_max_tumble = hail_ar*D_max
        # D_int_tumble = hail_br*D_int 
        A_coll_tumble = hail_Ar*hail_A 

    return D_max_tumble, A_coll_tumble


def velocity_parameterization(D_max, m_core, A_coll_tumble, hailstone_density, sfcdens, rho_air, CdeT, randindV, choose_vt): 
    '''
    Function to calculate hailstone terminal velocity. Calculate the terminal fall speed for hailstones corresponding to the chosen terminal velocity parameterization.
    '''

    #Define Heymsfield aD^b equations for terminal fall speed for the chosen parameterizations
    Vaharray = np.array([6.73,7.25,7.79,8.33,8.55,8.78,9.00,9.23,9.45,9.67,9.89,10.10,10.32,10.54,10.88,11.23,11.58])
    Vbharray = np.array([0.68,0.66,0.64,0.62,0.62,0.62,0.62,0.62,0.62,0.62,0.62, 0.63, 0.63, 0.63, 0.61, 0.59, 0.58])
    
    
    if choose_vt == 1: 
        # Spherical fall speed based on gravity = drag
        vt_hail = (4/3 * g / CdeT * hailstone_density / rho_air * (D_max/1000))**(1/2)
        
    elif choose_vt > 1 or choose_vt < 0: 
        # Non-spherical fall speed based on gravity = drag
        vt_hail = (2 * m_core * g / (rho_air * A_coll_tumble * CdeT))**(1/2)
        
    else: 
        # Heymsfield et al. (2020) fallspeed
        
        ahV=Vaharray[randindV]*np.ones(n_traj)
        bhV=Vbharray[randindV]*np.ones(n_traj)
    
        vt_hail = (ahV * (D_max/10)**bhV)*np.sqrt(sfcdens/rho_air)

    return vt_hail


def inertial_velocity(D_max, vrel_hail, vt_hail, hailstone_total_density, storm_rel_w, storm_rel_v, storm_rel_u, v_wind_rel_z_prior, v_wind_rel_y_prior, v_wind_rel_x_prior, v_gr_hail_z_prior, v_gr_hail_y_prior, v_gr_hail_x_prior, rho_dry_air, Cde, CdeT, choose_inertia, delt):
    '''
    Function to calculate hailstone inertial velocities.
    '''
    
    if choose_inertia == 1:

        # Calculate the ground-relative hailstone acceleration due to relative wind velocity in each dimension following Stout et al. (2005) and Kumjian et al. (2025)
        acc_z = (Cde/CdeT*vrel_hail*v_wind_rel_z_prior/vt_hail**2 - 1) * ((hailstone_total_density-rho_dry_air)/hailstone_total_density) * g
        acc_y =  Cde/CdeT*vrel_hail*v_wind_rel_y_prior/vt_hail**2      * ((hailstone_total_density-rho_dry_air)/hailstone_total_density) * g
        acc_x =  Cde/CdeT*vrel_hail*v_wind_rel_x_prior/vt_hail**2      * ((hailstone_total_density-rho_dry_air)/hailstone_total_density) * g
        
        # Update hailstone position based on relative wind acceleration
        vz_storm_rel = v_gr_hail_z_prior + acc_z * delt
        vy_storm_rel = v_gr_hail_y_prior + acc_y * delt
        vx_storm_rel = v_gr_hail_x_prior + acc_x * delt

        # Calculate the inertial adjustment timescale
        tau = vt_hail / ( (hailstone_total_density-rho_dry_air) / hailstone_total_density * g )

        # If the timestep is small relative to the inertial adjustment timescale, use the calculated acceleration.
        # If the timestep is large relative to the inertial adjustment timescale, use instantaneous adjustment to the background winds
        vz_storm_rel = np.where(delt <= tau, vz_storm_rel, storm_rel_w - vt_hail)
        vy_storm_rel = np.where(delt <= tau, vy_storm_rel, storm_rel_v                      )
        vx_storm_rel = np.where(delt <= tau, vx_storm_rel, storm_rel_u                      )
        
    # If no inertial adjustment, hailstones follow the background winds with terminal velocity pointing downwards
    else:
        vz_storm_rel = storm_rel_w - vt_hail
        vy_storm_rel = storm_rel_v
        vx_storm_rel = storm_rel_u

    # Calculate the hailstone-relative background wind velocities
    hail_rel_w = storm_rel_w - vz_storm_rel
    hail_rel_v = storm_rel_v - vy_storm_rel
    hail_rel_u = storm_rel_u - vx_storm_rel

    # Calculate the hailstone-relative background wind speed
    vrel_hail = (hail_rel_u**2 + hail_rel_v**2 + hail_rel_w**2)**(1/2)

    return vrel_hail, vz_storm_rel, vy_storm_rel, vx_storm_rel, hail_rel_w, hail_rel_v, hail_rel_u
        
        

def advection_module(hail_x, hail_y, hail_z, vz_storm_rel, vy_storm_rel, vx_storm_rel, delt):
    '''
    Function to perform hailstone advection
    '''

    hail_x_next = hail_x + vx_storm_rel * delt * 0.001 #update x position in km
    hail_y_next = hail_y + vy_storm_rel * delt * 0.001 #udpate y position in km
    hail_z_next = hail_z + vz_storm_rel * delt * 0.001 #update z position in km
    
    return hail_x_next, hail_y_next, hail_z_next

    
def size_change_spherical(D_max, hail_mass_change_external, growth_density_external):
    '''
    Calculate hailstone size change for a spherical hailstone (choose_shape = 0)
    '''

    # Calculate the previous hailstone volume (ice core)
    volume        = np.pi / 6 * D_max**3 * 1e-9
    # Calculate the incremental hailstone volume change (ice core)
    volume_change = hail_mass_change_external / growth_density_external
    # Calculate the new hailstone volume (ice core)
    volume_next   = volume + volume_change

    # Update the hailstone size based on the new volume (sphere)
    D_max_next = (volume_next / (1e-9 * np.pi / 6))**(1/3)
    D_int_next = D_max_next
    D_min_next = D_max_next
    
    return D_max_next, D_int_next, D_min_next



def size_change_nonspherical_Deq(D_max, D_int, D_min, int_ratio, hail_mass_change_external, growth_density_external):
    '''
    Calculate hailstone size change following the Shedd et al. 2021 equivalent-volume spherical diameter with specified int ratio (choose_shape = 1)
    '''
    
    # Calculate the previous hailstone volume (ice core)
    volume        = np.pi / 6 *  D_max * D_int * D_min * 1e-9
    # Calculate the incremental hailstone volume change (ice core)
    volume_change = hail_mass_change_external / growth_density_external
    # Calculate the new hailstone volume (ice core)
    volume_next   = volume + volume_change

    # Calculate the new equivalent-volume spherical diameter
    D_eq_sph_vol = (volume_next / (1e-9 * np.pi / 6))**(1/3)
    # Calculate Shedd et al. (2021) parameterization for D_max from D_eq (capped so that D_max >= D_int)
    D_max_next = np.maximum(1.3978*D_eq_sph_vol - 0.1486, (D_eq_sph_vol**3/int_ratio**2)**(1/3))
    # Find minimum dimension from newly calculated maximum dimension and intermediate ratio
    D_int_next = D_max_next * int_ratio
    # Find the minimum dimension that gives the desired volume
    D_min_next = volume_next / (1e-9 * np.pi/6 * D_max_next * D_int_next)
    
    return D_max_next, D_int_next, D_min_next


def size_change_nonspherical_aspect(D_max, D_int, D_min, int_ratio, hail_mass_change_external, growth_density_external, randindA):    
    '''
    Calculate hailstone size change following Heymsfield et al. 2018 aspect ratio with specified int ratio (choose_shape)
    '''

    # Calculate Heymsfield et al. (2018) aspect ratio
    aspect_ratio = find_aspect_ratio(D_max, randindA)
    # Capped intermediate ratio so that D_int >= D_min
    int_ratio = np.maximum(int_ratio, aspect_ratio)
    
    # Calculate the previous hailstone volume (ice core)
    volume        = np.pi / 6 *  D_max * D_int * D_min * 1e-9
    # Calculate the incremental hailstone volume change (ice core)
    volume_change = hail_mass_change_external / growth_density_external
    # Calculate the new hailstone volume (ice core)
    volume_next   = volume + volume_change

    # Calculate the maximum dimension that gives the desired volume and int/aspect ratios
    D_max_next = (volume_next / (int_ratio * aspect_ratio * 1e-9 * np.pi / 6))**(1/3)
    D_int_next = D_max_next *    int_ratio
    D_min_next = D_max_next * aspect_ratio
    
    return D_max_next, D_int_next, D_min_next




def size_change_nonspherical_Deq_and_aspect(D_max, D_int, D_min, hail_mass_change_external, growth_density_external, randindA):
    '''
    Calculate hailstone size change following Shedd et al. 2021 equivalent-volume spherical diameter and Heymsfield et al. 2018 aspect ratio (choose_shape = 3)
    '''
    
    # Calculate Heymsfield et al. (2018) aspect ratio
    aspect_ratio = find_aspect_ratio(D_max, randindA)
    
    # Calculate the previous hailstone volume (ice core)
    volume        = np.pi / 6 *  D_max * D_int * D_min * 1e-9
    # Calculate the incremental hailstone volume change (ice core)
    volume_change = hail_mass_change_external / growth_density_external
    # Calculate the new hailstone volume (ice core)
    volume_next   = volume + volume_change

    # Calculate the new equivalent-volume spherical diameter
    D_eq_sph_vol = (volume_next / (1e-9 * np.pi / 6))**(1/3)
    # Calculate Shedd et al. (2021) parameterization for D_max from D_eq (capped so that D_max >= D_int)
    D_max_next = np.maximum(1.3978*D_eq_sph_vol - 0.1486, (D_eq_sph_vol**3/aspect_ratio)**(1/3))
    # Find minimum dimension from newly calculated maximum dimension and aspect ratio (capped so that D_min >= D_int)
    D_min_next = np.minimum(D_max_next * aspect_ratio, (D_eq_sph_vol**3 / D_max_next)**(1/2))
    # Find the intermediate dimension that gives the desired volume
    D_int_next = volume_next / (1e-9 * np.pi/6 * D_max_next * D_min_next)
    
    return D_max_next, D_int_next, D_min_next



def find_aspect_ratio(D_max, randindA):
    '''
    Get aspect ratio parameterization choice from Heymsfield et al. 2018 phi-D_max empirical relationships 
    '''
    
    # Define the possible phi-D_max coefficients
    # Small (D_max <= 60 mm) hail aspect ratio relationship slope
    Rp1array = np.array([-0.0032,-0.0033,-0.0035,-0.0036,-0.0031,-0.0032,-0.0035,-0.0036,-0.0035,-0.0035,-0.0036,-0.0038,-0.0037,-0.0037,-0.0042,-0.0037,-0.0037])
    # Small (D_max <= 60 mm) hail aspect ratio relationship intercept
    Rp2array = np.array([   0.42,   0.46,   0.50,   0.54,   0.56,   0.59,   0.62,   0.65,   0.66,   0.68,   0.71,   0.73,   0.75,   0.78,   0.82,   0.84,   0.88])
    # Large (D_max >  60 mm) hail aspect ratio values
    Rbharray = np.array([   0.23,   0.26,   0.29,   0.32,   0.37,   0.39,   0.41,   0.43,   0.45,   0.47,   0.49,   0.50,   0.53,   0.56,   0.57,   0.62,   0.66])

    # Plug in the chosen parameterizations
    aspect_ratio = np.where(D_max > 60, Rbharray[randindA], Rp1array[randindA] * D_max + Rp2array[randindA])
    
    return aspect_ratio



def shape_diagnostics_wrapper(D_max, D_int, D_min, int_ratio, freeze_shell_mass, ice_growth_diffusion, growth_density, rho_ice, randindA, choose_shape):
    '''
    Wrapper for hailstone size diagnostic parameterizations.
    
    This function updates the hailstone shape given growth at each timestep. Two separate growth steps are performed. One for diffusion, corresponding to the ice core's current density, and one for accretion/freezing/melting, corresponding to the growth/melt density.
    '''
    
    # Choose the non-spherical hailstone diagnostic function to use to calculate update hailstone dimensions:
    if choose_shape == 0:
        # Option 0: Spherical Hailstones
        D_max, D_int, D_min = size_change_spherical(D_max, ice_growth_diffusion, rho_ice)
        D_max, D_int, D_min = size_change_spherical(D_max,    freeze_shell_mass,        growth_density)
    
    elif choose_shape == 2:
        # Option 2: calculate the maximum dimension corresponding to Heymsfield et al. (2018)'s aspect ratio-maximum dimension relationship and the given intermediate ratio
        D_max, D_int, D_min = size_change_nonspherical_aspect(D_max, D_int, D_min, int_ratio, ice_growth_diffusion, rho_ice, randindA)
        D_max, D_int, D_min = size_change_nonspherical_aspect(D_max, D_int, D_min, int_ratio,    freeze_shell_mass,        growth_density, randindA)
        
    elif choose_shape == 3:
        # Option 3: calculate the maximum dimension corresponding to Shedd et al. (2021)'s equivalent spherical diameter-maximum dimension relationship and calculate the aspect ratio corresponding to Heymsfield et al. (2018)'s aspect ratio-maximum dimension relationships.
        D_max, D_int, D_min = size_change_nonspherical_Deq_and_aspect(D_max, D_int, D_min, ice_growth_diffusion, rho_ice, randindA)
        D_max, D_int, D_min = size_change_nonspherical_Deq_and_aspect(D_max, D_int, D_min,    freeze_shell_mass,        growth_density, randindA)
    
    else:
        # Option 1: calculate the maximum dimension corresponding to Shedd et al. (2021)'s equivalent spherical diameter-maximum dimension relationship, assuming the prescribed intermediate ratio
        D_max, D_int, D_min = size_change_nonspherical_Deq(D_max, D_int, D_min, int_ratio, ice_growth_diffusion, rho_ice)
        D_max, D_int, D_min = size_change_nonspherical_Deq(D_max, D_int, D_min, int_ratio,    freeze_shell_mass,        growth_density)

    volume = np.pi/6 * (D_max*D_int*D_min*1e-9)
    
    return D_max, D_int, D_min, volume




def surface_temperature_from_average(radius_liquid, radius_ice, T_average):
    '''
    Get hailstone surface temperature given constant-conduction temperature profile in the liquid shell.
    '''
    
    # Plug in for surface temperature
    Tsfc = 2*(T_average-T0) * (radius_liquid**3-radius_ice**3) / (2*radius_liquid**3 - radius_ice*radius_liquid*(radius_ice+radius_liquid)) + T0
    # If no liquid shell (radius_ice = radius_liquid), set surface temperature to T_average, which should be the ice core temperature when a liquid layer is not present
    Tsfc[np.isnan(Tsfc)] = T_average[np.isnan(Tsfc)]
    
    return Tsfc


def find_transport_fraction(r_liq, r_ice, r1, r2):
    '''
    Liquid shell physics parameterization. This function defines the ramp between full conduction physics (thick liquid shell) and no liquid shell physics for thin liquid shells. This ramp avoids blow up related to rapid adjustment timescales for thermal conduction through the liquid shell.
    '''

    shell_thickness = r_liq - r_ice

    if r1 == r2:
        # If there is no liquid shell, all external physics should transfer to the ice core
        transport_fraction = np.where(shell_thickness < r1, 1, 0)
    else:
        # If there is a liquid shell, ramp up to full liquid conduction between shell thickness of r1 and r2. 
        # Thinner shells than r1, should have full transfer of external physics to the ice core
        # Thicker shells than r2, should have full thermal conduction through the liquid shell
        transport_fraction = np.where((shell_thickness <= r2) & (shell_thickness >= r1), 0.5 + 0.5*np.sin(np.pi*((r1+r2)/2 + shell_thickness)/(r2-r1)), (shell_thickness < r1).astype(float))
        
    return transport_fraction




def setup_parameterizations(n_traj, choose_vt, choose_aspect):
    '''
    Initialize parameterizations for the hail embryos. When using Heymsfield parameterizations for aspect ratio and/or terminal fall speed, find the indecies of the arrays for Heysmfield coefficients that will be used for each hailstone.
    '''
    
    # Random choice of Heysmfield et al. (2020) velocity relationships for each particle
    if choose_vt == 0:
        # Sample a normal distribution (mean=0.5, std=0.1) of quantiles for Heysmfield terminal fall speed parameterization
        randpercV = 0.1*np.random.normal(size=n_traj) + 0.5
        # Sample the indicies that correspond to the randomly sampled quantiles (0.1,0.9,0.05)
        randindV  = (np.round(20*randpercV).astype(int) - 2) % 17
    elif choose_vt > 0 and choose_vt < 1:
        # choose the specified parameterization from choose_vt
        randindV = (np.ones_like(init_diam)*np.round(choose_vt/0.05-1)).astype(int)-1
        randindV[randindV <  0] =  0
        randindV[randindV > 16] = 16
        
    else:
        # If not using Heymsfield parameterizations, we don't need randindV
        randindV = np.full(n_traj, np.nan)

    
    if (choose_aspect <= 0) or (choose_aspect >= 1):
        # Sample a normal distribution (mean=0.5, std=0.1) of quantiles for Heysmfield aspect ratio parameterization
        randpercA = 0.1*np.random.normal(size=n_traj) + 0.5
        # Sample the indicies that correspond to the randomly sampled quantiles (0.1,0.9,0.05)
        randindA  = (np.round(20*randpercA).astype(int) - 2) % 17
    else:
        # If not randomly sampling, calculate the indicies corresponding to the specified quantile Heysmfield aspect ratio parameterization
        randindA = np.full(n_traj,np.round(choose_aspect / 0.05 - 2)).astype(int)
        # Make sure we don't go above 16 (q=0.9) or below 0 (q=0.1)
        randindA[randindA <  0] =  0
        randindA[randindA > 16] = 16
    
    return randindV, randindA



def setup_particles(n_traj, init_diam, init_rho, choose_int_ratio, choose_aspect, choose_vt, choose_shape, rho_air, Cdrag, sfcdens):
    '''
    Initialize hail embryos and their parameterizations
    '''
    
    #----------------------------------------------------------------
    # Initialize particle parameterizations
    #----------------------------------------------------------------
    
    # Define indicies for any Heymsfield parameterizations used herein
    randindV, randindA = setup_parameterizations(n_traj, choose_vt, choose_aspect)

    #----------------------------------------------------------------
    # Initialize particle shape
    #----------------------------------------------------------------
    
    # Define intermediate ratio
    if choose_int_ratio <= 0:
        # If random, choose from a uniform distribution between 0.8 and 0.9
        int_ratio = np.random.uniform(0.8,0.9,n_traj)
    elif choose_int_ratio >= 1:
        # Set maximum intermediate ratio to 1
        int_ratio = np.ones(n_traj)
    else:
        # Intermediate ratio was specified. Define an array full of the specified value
        int_ratio = np.full(n_traj, choose_int_ratio)

    
    # Define the hail embryo shape corresponding to the correct parameterization
    init_diam, init_dint, init_dmin, init_vol = shape_diagnostics_wrapper(init_diam, init_diam, init_diam, int_ratio, np.zeros(n_traj), np.zeros(n_traj), rhosolid, rhosolid, randindA, choose_shape)

    init_m    = init_vol  * init_rho                                 #initial mass of embryo (kg)
    init_area = init_diam * init_dint*1e-6                           #initial cross-sectional area of the hailstone (m^2)
    
        
    #----------------------------------------------------------------
    # Initialize particle fall speed
    #----------------------------------------------------------------
    init_vt = velocity_parameterization(init_diam, init_m, init_area, init_rho, sfcdens, rho_air, Cdrag, randindV, choose_vt)

    return init_diam, init_dint, init_dmin, init_vol, init_m, init_vt, randindA, int_ratio, randindV




def sample_locations_box(n_traj, xmin, xmax, ymin, ymax, zmin, zmax, seed=24):
    '''
    Sample n = n_traj random hail embryo locations from within a box defined by x: [xmin, xmax], y: [ymin, ymax], z: [zmin, zmax]

    Inputs:
    - n_traj: total number of trajectory locations to sample (float)
    - xmin: minimum x edge of the bounding box to sample (float)
    - xmax: maximum x edge of the bounding box to sample (float)
    - ymin: minimum y edge of the bounding box to sample (float)
    - ymax: maximum y edge of the bounding box to sample (float)
    - zmin: minimum z edge of the bounding box to sample (float)
    - zmax: maximum z edge of the bounding box to sample (float)
    - seed: (optional) random seed (int)

    Returns:
    - init_x: x locations of sampled embryos (1-D array of shape (n_traj,)) 
    - init_y: y locations of sampled embryos (1-D array of shape (n_traj,)) 
    - init_z: z locations of sampled embryos (1-D array of shape (n_traj,)) 
    '''

    np.random.seed(seed)
    
    init_x = np.random.uniform(xmin, xmax, n_traj)
    init_y = np.random.uniform(ymin, ymax, n_traj)
    init_z = np.random.uniform(zmin, zmax, n_traj)
    
    return init_x, init_y, init_z


def sample_locations_mask(mask, n_realizations, mask_zs, mask_ys, mask_xs, seed=24):
    '''
    Sample random hail embryo locations from a masked locations in a grid with n = n_realizations per mask == 1 location. Note that this function is meant for a regularly spaced grid.

    Inputs:
    - mask: boolean array of locations from which to sample hail embryos assuming (z,y,x) dimension ordering (3-D boolean array)
    - n_realizations: the number of samples to perform at each mask location (int)
    - mask_zs: mask z coordiates (1-D float array)
    - mask_ys: mask y coordiates (1-D float array)
    - mask_xs: mask x coordiates (1-D float array)
    - seed: (optional) random seed (int)

    Returns:
    - init_x: x locations of sampled embryos (1-D array of shape (n_realizations*np.count_nonzero(mask),)) 
    - init_y: y locations of sampled embryos (1-D array of shape (n_realizations*np.count_nonzero(mask),)) 
    - init_z: z locations of sampled embryos (1-D array of shape (n_realizations*np.count_nonzero(mask),)) 
    '''

    np.random.seed(seed)

    dz = np.diff(mask_zs)[0]
    dy = np.diff(mask_ys)[0]
    dx = np.diff(mask_xs)[0]

    perturbation_z = np.random.uniform(-dz/2, dz/2, np.count_nonzero(mask)*n_realizations)
    perturbation_y = np.random.uniform(-dy/2, dy/2, np.count_nonzero(mask)*n_realizations)
    perturbation_x = np.random.uniform(-dx/2, dx/2, np.count_nonzero(mask)*n_realizations)

    zzs, yys, xxs = np.meshgrid(mask_zs, mask_ys, mask_xs)
    init_z = np.repeat(zzs.ravel()[mask.ravel() == 1], n_realizations) + perturbation_z
    init_y = np.repeat(yys.ravel()[mask.ravel() == 1], n_realizations) + perturbation_y
    init_x = np.repeat(xxs.ravel()[mask.ravel() == 1], n_realizations) + perturbation_x
    
    return init_z, init_y, init_x


def sample_locations_weighted_mask(weights, n_traj, mask_zs, mask_ys, mask_xs, seed=24):
    '''
    Sample n_traj total random hail embryo locations from a masked grid randomly corresponding to the provided weights. Note that this function is meant for a regularly spaced grid.

    Inputs:
    - weights: array of desired sampling frequencies for each gridded location assuming (z,y,x) dimension ordering (3-D float array)
    - n_traj: the total number of samples to perform (int)
    - mask_zs: mask z coordiates (1-D float array)
    - mask_ys: mask y coordiates (1-D float array)
    - mask_xs: mask x coordiates (1-D float array)
    - seed: (optional) random seed (int)

    Returns:
    - init_x: x locations of sampled embryos (1-D array of shape (n_traj,)) 
    - init_y: y locations of sampled embryos (1-D array of shape (n_traj,)) 
    - init_z: z locations of sampled embryos (1-D array of shape (n_traj,)) 
    '''

    np.random.seed(seed)

    dz = np.diff(mask_zs)[0]
    dy = np.diff(mask_ys)[0]
    dx = np.diff(mask_xs)[0]

    perturbation_z = np.random.uniform(-dz/2, dz/2, n_traj)
    perturbation_y = np.random.uniform(-dy/2, dy/2, n_traj)
    perturbation_x = np.random.uniform(-dx/2, dx/2, n_traj)

    zzs, yys, xxs = np.meshgrid(mask_zs, mask_ys, mask_xs)
    sampled_idxs = np.random.choice(np.arange(zzs.shape[0]), size=n_traj, replace=True, p=weights/np.sum(weights).ravel())

    init_z = zzs.ravel()[sampled_idxs] + perturbation_z
    init_y = yys.ravel()[sampled_idxs] + perturbation_y
    init_x = xxs.ravel()[sampled_idxs] + perturbation_x
    
    return init_z, init_y, init_x
    

def hail_embryo_gamma(n, D_bar, D0_min=0, D0_max=0.01):
    '''
    Sample n hail embryo sizes from a gamma distribution defined by D_bar, bounded by D0_min and D0_max.

    Inputs:
    - n: number of samples to perform (int)
    - D_bar: characteristic size of the gamma distribution to sample (float or float array of shape (n,))
    - D0_min: minimum allowed size to sample (float)
    - D0_max: maximum allowed size to sample (float)

    Returns:
    - Ds: sampled hail sizes (1-D float array of size (n,))
    '''
    
    mu = 1/3
    alpha = 1
    nu = -1/3
    
    B  = ( gamma( (nu+1)/mu ) / gamma( (nu+2)/mu ) )**(-mu)
    Ds = np.random.gamma(shape=2, scale=1/(B/D_bar), size=n)

    wrong_size = (Ds > D0_max) | (Ds < D0_min)
    while np.any(wrong_size):
        if type(D_bar) in [float, int]:
            Ds[wrong_size] = np.random.gamma(shape=2, scale=1/(B/D_bar), size=np.count_nonzero(wrong_size))
        else:
            Ds[wrong_size] = np.random.gamma(shape=2, scale=1/(B/D_bar[wrong_size]), size=np.count_nonzero(wrong_size))
        wrong_size = (Ds > D0_max) | (Ds < D0_min)
        # print(np.count_nonzero(wrong_size))

    return Ds


def sample_sizes_gamma(n_traj, mean_size, D0_min=2, D0_max=10):
    '''
    Sample n_traj random sizes from the same gamma distribution, defined by mean_size, bounded by D0_min and D0_max

    Inputs:
    - n_traj: number of samples to perform (int)
    - mean_size: characteristic size of the gamma distribution to sample (float or float array of shape (n_traj,))
    - D0_min: minimum allowed size to sample (float)
    - D0_max: maximum allowed size to sample (float)

    Returns:
    - init_diam: sampled hail embryo sizes (1-D float array of size (n_traj,))
    '''

    init_diam = hail_embryo_gamma(n_traj, mean_size, D0_min=D0_min, D0_max=D0_max)

    return init_diam


def sample_sizes_spatial_gamma(n_realizations, mask, mean_size, D0_min=2, D0_max=20):
    '''
    Sample n_realizations from each valid location in a spatial mask. At each location, the a gamma distribution is sampled defined by each location's mean_size, bounded by D0_min and D0_max.
    
    Inputs:
    - n_realizations: number of samples to perform at each mask location (int)
    - mask: boolean array of locations to sample (N-D boolean array)
    - mean_size: characteristic size of the gamma distribution to sample for each mask location (float or N-D float array matching the shape of mask)
    - D0_min: minimum allowed size to sample (float)
    - D0_max: maximum allowed size to sample (float)

    Returns:
    - init_diam: sampled hail embryo sizes (1-D float array of size (n_realizations*np.count_nonzero(mask),))
    '''
    cropped_scaled_mean_size = mean_size.copy()
    cropped_scaled_mean_size[cropped_scaled_mean_size < D0_min] = D0_min
    cropped_scaled_mean_size[cropped_scaled_mean_size > D0_max] = D0_max

    
    sampling_mean_size = np.repeat(cropped_scaled_mean_size.ravel()[mask.ravel() == 1], n_realizations)
    
    init_diam = hail_embryo_gamma(n_realizations*np.count_nonzero(mask), sampling_mean_size, D0_min=D0_min, D0_max=D0_max)

    return init_diam

def sample_sizes_uniform(n_traj, D0_min=2, D0_max=10):
    '''
    Sample n_traj hail embryo sizes uniformly from the range (D0_min, D0_max)

    Inputs:
    - n_traj: total number of samples to perform (int)
    - D0_min: minimum allowed size in sample (float)
    - D0_min: maximum allowed size in sample (float)

    Returns:
    - init_diam: sampled hail embryo sizes (1-D float array of size (n_traj,))
    '''
    
    init_diam = np.random.uniform(D0_min, D0_max, n_traj)

    return init_diam


def sample_sizes_linear(n_traj, D0_min=2, D0_max=10):
    '''
    Sample n_traj hail embryo sizes from the range (D0_min, D0_max) with linearly decreasing frequency (fewer large embyros)

    Inputs:
    - n_traj: total number of samples to perform (int)
    - D0_min: minimum allowed size in sample (float)
    - D0_min: maximum allowed size in sample (float)

    Returns:
    - init_diam: sampled hail embryo sizes (1-D float array of size (n_traj,))
    '''

    # linear decreasing embryo size frequency (fewer large embryos)
    init_diam = (1 - np.sqrt(1 - np.random.random(size=n_traj))) * (D0_max - D0_min) + D0_min

    return init_diam



def sample_densities_uniform(n_traj, rho_min=300, rho_max=917):
    '''
    Sample n_traj hail embryo densities from the range (rho_min, rho_max).

    Inputs:
    - n_traj: total number of samples to perform (int)
    - rho_min: minimum allowed density in sample (float)
    - rho_min: maximum allowed density in sample (float)

    Returns:
    - init_rho: sampled hail embryo densities (1-D float array of size (n_traj,))
    '''
    
    init_rho = np.random.uniform(rho_min, rho_max, n_traj)

    return init_rho



def sample_densities_spatial(n_realizations, mask, rho_mean, rho_std=50, rho_min=300, rho_max=917):
    '''
    Sample n_realizations for each valid location in a spatial mask. For each location, sample the n_realizations sizes from a normal distribution defined by each location's rho_mean with standard deviation rho_std, bounded by rho_min and rho_max.

    Inputs:
    - n_realizations: number of samples to perform at each mask location (int)
    - mask: boolean array of locations to sample (N-D boolean array)
    - rho_mean: mean density of the gaussian distribution to sample for each mask location (float or N-D float array matching the shape of mask)
    - rho_std: standard deviation of the gaussian distribution to sample for each mask location (float)
    - rho_min: minimum allowed density to sample (float)
    - rho_max: maximum allowed density to sample (float)

    Returns:
    - init_rho: sampled hail embryo densities (1-D float array of size (n_realizations*np.count_nonzero(mask),))
    '''
    rho_mean_shaped =  np.repeat(rho_mean.ravel()[mask.ravel() == 1], n_realizations)
    
    init_rho = rho_mean_shaped + np.random.normal(loc=0, scale=rho_std, size=n_realizations*np.count_nonzero(mask))
    init_rho[init_rho < rho_min] = rho_min
    init_rho[init_rho > rho_max] = rho_max

    return init_rho



def load_tumbling_data(tumbling_file): 
    '''
    Load the lookup table defineing the tumbling dataset.
    '''

    # load file
    mat = scipy.io.loadmat(tumbling_file)

    # Parse variables and make them globally accessible
    global tumbling_A0, tumbling_Dmax0
    
    tumbling_A0    = mat['part_Ar0']    # Area
    tumbling_Dmax0 = mat['part_ar0']    # Maximum dimension

    return


def load_storm_data(sim_time, path_to_storm, vars, filetype='nc', sfcdens=None):
    '''
    Load in the storm dataset (assumes netcdf or zarr file formats for now)
    
    Note that sfcdens pulls the air density from the bottom south east corner of the domain to define the inflow air density if no sfcdens value is given.
    '''

    # Open the zarr or netcdf file, select the 
    if filetype == 'zarr':
        infile = glob.glob(f'{path_to_storm}/*{sim_time:06d}.zarr')[0]
        ds_storm_time = xr.open_zarr(infile)[vars]
    elif filetype == 'nc':
        infile = glob.glob(f'{path_to_storm}/*{sim_time:06d}.nc')[0]
        ds_storm_time = xr.open_dataset(infile, decode_timedelta=True)[vars]
    else:
        raise Exception('Unknown filetype, please specify from list: [\'nc\', \'zarr\']')

    # Remove the time dimension if present
    if 'time' in list(ds_storm_time.dims):
        ds_storm_time = ds_storm_time.isel(time=0)

    # Rearrange the data for the interpolator (n_var, nz, ny, nx)
    data_to_interpolate = np.moveaxis(ds_storm_time.to_array().to_numpy(), 0, 3)

    # Find the dimension names of the x, y, z dimensions (generalized for varients on x,y,z or i,j,k)
    dims = list(ds_storm_time.prs.dims)
    x_coord = [dim for dim in dims if ('x' in dim)|('i' in dim)|('lon' in dim)|('north' in dim)|('south'  in dim)][0]
    y_coord = [dim for dim in dims if ('y' in dim)|('j' in dim)|('lat' in dim)|('west'  in dim)|('east'   in dim)][0]
    z_coord = [dim for dim in dims if ('z' in dim)|('k' in dim)|('alt' in dim)|('top'   in dim)|('bottom' in dim)][0]

    # Find the inflow air density (used throughout). In case of weirdness in sampling the bottom south east corner, force air density to go no lower than 0.9 and no higher than 1.3, default 1.225.
    if sfcdens == None:
        sfcdens  = float(ds_storm_time.rho.isel({x_coord: -1, y_coord: 0, z_coord: 0}).values)
        if np.isnan(sfcdens):
            sfcdens = 1.225
        elif sfcdens < 0.9:
            sfcdens = 0.9
        elif sfcdens > 1.3:
            sfcdens = 1.3

    # Find the dimensions of the storm dataset
    storm_grid_size = (ds_storm_time[z_coord].shape[0], ds_storm_time[y_coord].shape[0], ds_storm_time[x_coord].shape[0])

    # Build the interpolator for this storm time
    interp = RegularGridInterpolator((ds_storm_time[z_coord].values, ds_storm_time[y_coord].values, ds_storm_time[x_coord].values), data_to_interpolate, method='linear', bounds_error=False, fill_value=None)

    # Identify the domain boundaries
    xmin = ds_storm_time[x_coord].values.min()
    xmax = ds_storm_time[x_coord].values.max()
    ymin = ds_storm_time[y_coord].values.min()
    ymax = ds_storm_time[y_coord].values.max()
    zmin = ds_storm_time[z_coord].values.min()
    zmax = ds_storm_time[z_coord].values.max()

    return storm_grid_size, interp, sfcdens, xmin, xmax, ymin, ymax, zmin, zmax
       

def load_data_dictionaries_for_output():
    '''
    Function to define all the metadata for the data we want to output
    '''
    
    global init_data_dict, final_data_dict, full_data_dict
    
    init_data_dict = {}
    init_data_dict['x'  ] = {'name': 'init_x'     , 'attrs': {'units':    'km', 'long_name': 'embryo x location in storm dataset coordinates',}}
    init_data_dict['y'  ] = {'name': 'init_y'     , 'attrs': {'units':    'km', 'long_name': 'embryo y location in storm dataset coordinates',}}
    init_data_dict['z'  ] = {'name': 'init_z'     , 'attrs': {'units':    'km', 'long_name': 'embryo z location in storm dataset coordinates',}}
    init_data_dict['D'  ] = {'name': 'init_diam'  , 'attrs': {'units':    'mm', 'long_name': 'given embryo diameter (may not be the same as initial D_max, depending on choose_shape)',}}
    init_data_dict['m_core'  ] = {'name': 'init_m'     , 'attrs': {'units':    'kg', 'long_name': 'embryo mass'                                   ,}}
    init_data_dict['rho_ice' ] = {'name': 'init_rho'   , 'attrs': {'units': 'kg/m3', 'long_name': 'given embryo ice core density'                 ,}}
    init_data_dict['vt_hail' ] = {'name': 'init_vt'    , 'attrs': {'units':   'm/s', 'long_name': 'embryo terminal fall speed'                    ,}}
    
    final_data_dict = {}
    final_data_dict['x'        ] = {'name': 'final_x'        , 'attrs':{'units': 'km'   , 'long_name': 'final x position in storm dataset coordinates', }}
    final_data_dict['y'        ] = {'name': 'final_y'        , 'attrs':{'units': 'km'   , 'long_name': 'final y position in storm dataset coordinates', }}
    final_data_dict['z'        ] = {'name': 'final_z'        , 'attrs':{'units': 'km'   , 'long_name': 'final z position in storm dataset coordinates', }}
    final_data_dict['D_max'    ] = {'name': 'final_D_max'    , 'attrs':{'units': 'mm'   , 'long_name': 'final maximum dimension', }}
    final_data_dict['D_int'    ] = {'name': 'final_D_int'    , 'attrs':{'units': 'mm'   , 'long_name': 'final intermediate dimension', }}
    final_data_dict['D_min'    ] = {'name': 'final_D_min'    , 'attrs':{'units': 'mm'   , 'long_name': 'final minimum dimension', }}
    final_data_dict['Ax_tumble'] = {'name': 'final_Ax_tumble', 'attrs':{'units': 'm2'   , 'long_name': 'final tumbling cross-sectional area', }}
    final_data_dict['volume'   ] = {'name': 'final_volume'   , 'attrs':{'units': 'm3'   , 'long_name': 'final hailstone volume', }}
    final_data_dict['m_core'   ] = {'name': 'final_m_core'   , 'attrs':{'units': 'kg'   , 'long_name': 'final ice core mass', }}
    final_data_dict['m_soaked' ] = {'name': 'final_m_soaked' , 'attrs':{'units': 'kg'   , 'long_name': 'final soaked liquid mass', }}
    final_data_dict['m_shell'  ] = {'name': 'final_m_shell'  , 'attrs':{'units': 'kg'   , 'long_name': 'final liquid shell mass', }}
    final_data_dict['rho_ice'  ] = {'name': 'final_rho_ice'  , 'attrs':{'units': 'kg/m3', 'long_name': 'final ice core density', }}
    final_data_dict['rho_tot'  ] = {'name': 'final_rho_tot'  , 'attrs':{'units': 'kg/m3', 'long_name': 'final total hailstone density (ice core + soaked mass)', }}
    final_data_dict['wet_frac' ] = {'name': 'final_wet_frac' , 'attrs':{'units': ''     , 'long_name': 'final hailstone wet fraction (liquid mass / (liquid mass + ice mass))', }}
    final_data_dict['vt_hail'  ] = {'name': 'final_terminal_fall_speed'     , 'attrs': {'units': 'm/s', 'long_name': 'final hailstone terminal fall speed', }}
    final_data_dict['vrel_hail'] = {'name': 'final_wind_relative_fall_speed', 'attrs': {'units': 'm/s', 'long_name': 'final hailstone wind-relative fall speed', }}
    final_data_dict['v_rel_x'   ] = {'name': 'v_rel_x'   , 'attrs': {'units': 'm/s'  , 'long_name': 'final hailstone-relative air motion (x component)',}}
    final_data_dict['v_rel_y'   ] = {'name': 'v_rel_y'   , 'attrs': {'units': 'm/s'  , 'long_name': 'final hailstone-relative air motion (y component)',}}
    final_data_dict['v_rel_z'   ] = {'name': 'v_rel_z'   , 'attrs': {'units': 'm/s'  , 'long_name': 'final hailstone-relative air motion (z component)',}}
    final_data_dict['v_hail_x'  ] = {'name': 'v_hail_x'  , 'attrs': {'units': 'm/s'  , 'long_name': 'final storm-relative hailstone velocity (x component)',}}
    final_data_dict['v_hail_y'  ] = {'name': 'v_hail_y'  , 'attrs': {'units': 'm/s'  , 'long_name': 'final storm-relative hailstone velocity (y component)',}}
    final_data_dict['v_hail_z'  ] = {'name': 'v_hail_z'  , 'attrs': {'units': 'm/s'  , 'long_name': 'final storm-relative hailstone velocity (z component)',}}
    final_data_dict['time_aloft'     ] = {'name': 'time_aloft'     ,  'attrs': {'units': 's', 'long_name': 'time spent aloft from seeding until complete melting or hitting the ground', }}
    final_data_dict['final_time_step'] = {'name': 'final_time_step', 'attrs': {'units': 's', 'long_name': 'final timestep of hailstone being aloft and unmelted', }}
    final_data_dict['melted'  ] = {'name': 'melted'  , 'attrs': {'units': '', 'long_name': 'did the hailstone melt completely before hitting the ground?', }}
    final_data_dict['grounded'] = {'name': 'grounded', 'attrs': {'units': '', 'long_name': 'did the hailstone hit the ground normally before melting?', }}
    final_data_dict['exited'  ] = {'name': 'exited'  , 'attrs': {'units': '', 'long_name': 'did the hailstone leave the domain without hitting the ground?', }}


    full_data_dict = {}
    full_data_dict['x'         ] = {'name': 'x'         , 'attrs': {'units': 'km', 'long_name': 'hailstone x location in storm dataset coordinates', }}
    full_data_dict['y'         ] = {'name': 'y'         , 'attrs': {'units': 'km', 'long_name': 'hailstone y location in storm dataset coordinates', }}
    full_data_dict['z'         ] = {'name': 'z'         , 'attrs': {'units': 'km', 'long_name': 'hailstone z location in storm dataset coordinates', }}
    full_data_dict['D_max'     ] = {'name': 'D_max'     , 'attrs': {'units': 'mm', 'long_name': 'maximum dimension', }}
    full_data_dict['D_int'     ] = {'name': 'D_int'     , 'attrs': {'units': 'mm', 'long_name': 'intermediate dimension', }}
    full_data_dict['D_min'     ] = {'name': 'D_min'     , 'attrs': {'units': 'mm', 'long_name': 'minimum dimension', }}
    full_data_dict['volume'    ] = {'name': 'volume'    , 'attrs': {'units': 'm3', 'long_name': 'hailstone volume', }}
    full_data_dict['T_core'    ] = {'name': 'T_core'    , 'attrs': {'units':  'K', 'long_name': 'hailstone ice core temperature', }}
    full_data_dict['T_avg_liq' ] = {'name': 'T_avg_liq' , 'attrs': {'units':  'K', 'long_name': 'average temperature of the liquid layer', }}
    full_data_dict['T_sfc'     ] = {'name': 'T_sfc'     , 'attrs': {'units':  'K', 'long_name': 'particle surface temperature', }}
    full_data_dict['T_inf'     ] = {'name': 'T_inf'     , 'attrs': {'units':  'K', 'long_name': 'background temperature at hailstone location', }}
    full_data_dict['m_core'    ] = {'name': 'm_core'    , 'attrs': {'units': 'kg', 'long_name': 'ice core mass',}}
    full_data_dict['m_soaked'  ] = {'name': 'm_soaked'  , 'attrs': {'units': 'kg', 'long_name': 'soaked liquid mass',}}
    full_data_dict['m_shell'   ] = {'name': 'm_shell'   , 'attrs': {'units': 'kg', 'long_name': 'liquid shell mass',}}
    full_data_dict['rho_ice'   ] = {'name': 'rho_ice'   , 'attrs': {'units': 'kg/m3', 'long_name': 'ice core density',}}
    full_data_dict['rho_tot'   ] = {'name': 'rho_tot'   , 'attrs': {'units': 'kg/m3', 'long_name': 'total hailstone density (ice core + soaked mass)',}}
    full_data_dict['rho_growth'] = {'name': 'rho_growth', 'attrs': {'units': 'kg/m3', 'long_name': 'density of growth/melt at this timestep',}}
    full_data_dict['vt_hail'   ] = {'name': 'vt_hail'   , 'attrs': {'units': 'm/s'  , 'long_name': 'hailstone terminal fall speed',}}
    full_data_dict['vrel_hail' ] = {'name': 'vrel_hail' , 'attrs': {'units': 'm/s'  , 'long_name': 'hailstone wind-relative fall speed',}}
    full_data_dict['v_rel_x'   ] = {'name': 'v_rel_x'   , 'attrs': {'units': 'm/s'  , 'long_name': 'Lagrangian relative air motion (x component, background winds relative to hailstone motion)',}}
    full_data_dict['v_rel_y'   ] = {'name': 'v_rel_y'   , 'attrs': {'units': 'm/s'  , 'long_name': 'Lagrangian relative air motion (y component, background winds relative to hailstone motion)',}}
    full_data_dict['v_rel_z'   ] = {'name': 'v_rel_z'   , 'attrs': {'units': 'm/s'  , 'long_name': 'Lagrangian relative air motion (z component, background winds relative to hailstone motion)',}}
    full_data_dict['v_hail_x'  ] = {'name': 'v_hail_x'  , 'attrs': {'units': 'm/s'  , 'long_name': 'Storm-relative hailstone velocity (x component)',}}
    full_data_dict['v_hail_y'  ] = {'name': 'v_hail_y'  , 'attrs': {'units': 'm/s'  , 'long_name': 'Storm-relative hailstone velocity (y component)',}}
    full_data_dict['v_hail_z'  ] = {'name': 'v_hail_z'  , 'attrs': {'units': 'm/s'  , 'long_name': 'Storm-relative hailstone velocity (z component)',}}
    full_data_dict['Ax_tumble'  ] = {'name': 'Ax_tumble'  , 'attrs': {'units': 'm2', 'long_name': 'tumbling cross-sectional area', }}
    full_data_dict['wet_frac'   ] = {'name': 'wet_frac'   , 'attrs': {'units':   '', 'long_name': 'liquid fraction of the hailstone (liquid mass / (ice mass + liquid mass))', }}
    full_data_dict['frozen_frac'] = {'name': 'frozen_frac', 'attrs': {'units':   '', 'long_name': 'fraction of liquid shell mass frozen at this timestep', }}
    full_data_dict['wet_growth_tracker'] = {'name': 'wet_growth_tracker','attrs':{'units': '', 'long_name': 'in wet growth regime?', }}
    full_data_dict['dry_growth_tracker'] = {'name': 'dry_growth_tracker','attrs':{'units': '', 'long_name': 'in dry growth regime?', }}
    full_data_dict['melting_tracker'   ] = {'name': 'melting_tracker'   ,'attrs':{'units': '', 'long_name': 'in melting growth regime?', }}
    full_data_dict['clear_air_tracker' ] = {'name': 'clear_air_tracker' ,'attrs':{'units': '', 'long_name': 'in diffusion only growth regime?', }}
    full_data_dict['growth_regime'     ] = {'name': 'growth_regime'     ,'attrs':{'units': '', 'long_name': 'growth regime: melting = 1; dry growth = 2; wet growth = 3; diffusion only = 4; n/a = 0', }}

    return

     
def main(init_x, init_y, init_z, init_diam, init_rho,
         path_to_storm, sim_time, filetype='nc', total_t=3601, delt=1, storm_delt=5, evolving=False, 
         vars=['rho', 'qc', 'qr', 'crw', 'qv', 'qi', 'qs', 'prs', 'uinterp', 'vinterp', 'winterp'], sfcdens=None,
         output_path=None, output_which='unmelted', 
         full_record_list=['x', 'y', 'z', 'D_max', 'growth_regime'], 
         final_record_list=['x', 'y', 'z', 'D_max', 'D_int', 'D_min', 'Ax_tumble', 'm_core', 'm_soaked', 'm_shell', 'vt_hail', 'vrel_hail', 'v_hail_x', 'v_hail_y', 'v_hail_z', 'time_aloft', 'final_time_step', 'melted', 'grounded', 'exited'], 
         initial_record_list=['x', 'y', 'z', 'D', 'rho_ice'], full_record_interval=1,
         choose_tumble=1, choose_inertia=1, choose_vt=2, choose_drag=5, Cdrag=0.58,
         choose_shape=1, choose_int_ratio=0, choose_aspect=0, choose_lobes=0, 
         choose_shell_T=1, choose_chi=0, choose_seed=24, tumbling_file='tumblerand.mat'
        ):   
    '''
    Main hailstone trajectory growth model. See README.md for documentation.

    Inputs:
    - init_x: (1D float array) hail embryo initial x positions [km]
    - init_y: (1D float array) hail embryo initial y positions [km]
    - init_z: (1D float array) hail embryo initial z positions [km]
    - init_diam: (1D float array) hail embryo initial equivalent volume spherical diameter [mm]
    - init_rho: (1D float array) hail embryo initial density [kg/m3]
    - path_to_storm: (string) path to the directory containing the storm dataset files
    - sim_time: (string or int) file number of the storm dataset file to use for hail trajectory model initalization
    - filetype: (string) filetype of the storm dataset (must be 'nc' or 'zarr')
    - total_t: (int or float) maximum trajectory duration [s]
    - delt: (int or float) model integration timestep [s]
    - storm_delt: (int or float) time between storm simulation output times [s]
    - evolving: (boolean) compute trajectories with evolving (evolving=True) or static (evolving=False) storm data
    - vars: (list or array of shape (10,)) variable names corresponding to the following variables in the storm dataset **in order**: [dry air density, cloud water mixing ratio, rain water mixing ratio, rain water number concentration, water vapor mixing ratio, cloud ice mixing ratio, snow mixing ratio, air pressure, u-component winds, v-component winds, w-component winds]
    - sfcdens: (float, optional) storm inflow surface dry air density
    - output_path: (string) path to save trajectory model output
    - output_which: (string) trajectory termination condition for model output, must be one of ['unmelted', 'grounded', 'all']
    - full_record_list: (list) hail trajectory time series variables to output
    - final_record_list: (list) hail swath variables to output
    - initial_record_list: (list) hail embryo initial conditions to output
    - full_record_interval: (int) frequency of time series output [timesteps]
    - choose_tumble: (float) tumbling parameterization choice
    - choose_inertia: (int) inertial adjustment choice
    - choose_vt: (float) terminal fall speed parameterization choice
    - choose_drag: (int) drag parameterization choice
    - Cdrag: (float) default constant drag coefficient
    - choose_shape: (int) shape parameterization choice
    - choose_int_ratio: (float) hailstone intermediate ratio choice
    - choose_aspect: (float) aspect ratio parameterization choice
    - choose_lobes: (int) lobes parameterization choice
    - choose_shell_T: (int) liquid shell temperature for heat transfer choice
    - choose_chi: (int) chi parameterization choice
    - choose_seed: (int) random seed
    - tumbling_file: (string) path to the tumbling lookup table file (e.g., 'tumblerand.mat')

    Returns: 
    - output_ds: xarray dataset defining all the trajectory model output
    '''
    
    nt = np.floor(total_t/delt).astype(int) # Total number of time steps
    n_traj = init_x.shape[0]                # Total nubmer of trajectories

    # Set random seed
    np.random.seed(choose_seed)

    # Prepare the model configuration
    define_constants()
    load_tumbling_data(tumbling_file)
    storm_grid_size, interp, sfcdens, xmin, xmax, ymin, ymax, zmin, zmax = load_storm_data(sim_time, path_to_storm, vars, filetype, sfcdens)
    load_data_dictionaries_for_output()

    
    #----------------------------------------------------------------
    # Allocate arrays for hailstone trackers
    #----------------------------------------------------------------
    hail_x        = np.zeros(n_traj) # Hailstone x location [km]
    hail_y        = np.zeros(n_traj) # Hailstone y location [km]
    hail_z        = np.zeros(n_traj) # Hailstone z location [km]
    D_max         = np.zeros(n_traj) # Hailstone      maximum dimension [mm], (was part_si)
    D_int         = np.zeros(n_traj) # Hailstone intermediate dimension [mm], (was part_b)
    D_min         = np.zeros(n_traj) # Hailstone      minimum dimension [mm], (was part_c)
    volume        = np.zeros(n_traj) # Hailstone ellipsoidal volume [m^3]
    m_core        = np.zeros(n_traj) # Hailstone ice core mass [kg], (was part_mass)
    m_shell       = np.zeros(n_traj) # Hailstone liquid shell mass [kg]
    m_soaked      = np.zeros(n_traj) # Hailstone soaked liquid mass [kg]
    rho_ice       = np.zeros(n_traj) # Density of the ice core [kg/m3]
    vt_hail       = np.zeros(n_traj) # Hailstone terminal fall speed [m/s], (was part_vel)
    vrel_hail     = np.zeros(n_traj) # Hailstone wind-relative speed magnitude [m/s]
    vx_storm_rel  = np.zeros(n_traj) # Hailstone storm-relative velocity x componant [m/s]
    vy_storm_rel  = np.zeros(n_traj) # Hailstone storm-relative velocity y componant [m/s]
    vz_storm_rel  = np.zeros(n_traj) # Hailstone storm-relative velocity z componant [m/s]
    hail_rel_u    = np.zeros(n_traj) # Hailstone-relative background wind velocity x componant [m/s]
    hail_rel_v    = np.zeros(n_traj) # Hailstone-relative background wind velocity y componant [m/s]
    hail_rel_w    = np.zeros(n_traj) # Hailstone-relative background wind velocity z componant [m/s]
    A_coll_tumble = np.zeros(n_traj) # Hailstone collection area, accounting for tumble [m^2], (was part_At)
    D_max_tumble  = np.zeros(n_traj) # Hailstone collection maximum dimension, accounting for tumble [mm], (was part_at)
    D_avg_eq_sfc  = np.zeros(n_traj) # Hailstone equivalent-surface-area sphrical diameter
    D_avg_eq_vol  = np.zeros(n_traj) # Hailstone equivalent-surface-area sphrical diameter
    lobes_ratio   = np.zeros(n_traj) # Hailstone lobes volume ratio, if lobey
    T_core        = np.zeros(n_traj) # Hailstone surface temperature [K], (was part_Ts)
    T_sfc         = np.zeros(n_traj) # Hailstone liquid shell average temperature [K]
    T_liq_avg     = np.zeros(n_traj) # The temperature of the outside of the surface liquid layer [K]

    
    #----------------------------------------------------------------
    # Sample the interpolators to get storm data
    #----------------------------------------------------------------
    # Using copy functions liberally because getting overwriting somewhere mysterious
    hail_x[:]     = init_x.copy() # Initial embryo location from user
    hail_y[:]     = init_y.copy() # Initial embryo location from user
    hail_z[:]     = init_z.copy() # Initial embryo location from user

    # Evaluate the imterpolator at the current positions
    out = interp((hail_z,hail_y,hail_x))
        
    rho_dry_air = out[:, 0]  #                   dry air density at the hailstone locations
    qc          = out[:, 1]  #          cloud water mixing ratio at the hailstone locations
    qr          = out[:, 2]  #           rain water mixing ratio at the hailstone locations
    nr          = out[:, 3]  #    rain drop number concentration at the hailstone locations
    qv          = out[:, 4]  #          water vapor mixing ratio at the hailstone locations
    qi          = out[:, 5]  #            cloud ice mixing ratio at the hailstone locations
    qs          = out[:, 6]  #                 snow mixing ratio at the hailstone locations
    p_air       = out[:, 7]  #                      air pressure at the hailstone locations
    storm_rel_u = out[:, 8]  # ground relative x-component winds at the hailstone locations
    storm_rel_v = out[:, 9]  # ground relative y-component winds at the hailstone locations
    storm_rel_w = out[:,10]  # ground relative z-component winds at the hailstone locations
    
    rho_air = rho_dry_air * (1 + qv)    # total air density (kg/m3)
    R = (1 - qv)  * Rdry + qv * Rv      # gas constant for moist air
    T_inf = (1 / (rho_air * R)) * p_air # temperature (K)

    
    #----------------------------------------------------------------
    # Initialize Particle Properties
    #----------------------------------------------------------------
    
    # Initial ice core temperature is ambient at initial location
    T_core   [:] = np.where(T_inf <= T0, T_inf, T0)
    T_sfc    [:] = T_core.copy()
    T_liq_avg[:] = T_core.copy()

    
    # Initialize embryo size, demsity, mass
    init_dmax, init_dint, init_dmin, init_vol, init_m, init_vt, randindA, int_ratio, randindV = setup_particles(n_traj, init_diam, init_rho, choose_int_ratio, choose_aspect, choose_vt, choose_shape, rho_air, Cdrag, sfcdens)
    D_max  [:] = init_dmax.copy()
    D_int  [:] = init_dint.copy()
    D_min  [:] = init_dmin.copy()
    volume [:] = init_vol .copy()
    m_core [:] = init_m   .copy()
    vt_hail[:] = init_vt  .copy()
    rho_ice[:] = init_rho .copy()


    # Initial relative fall speed equals terminal fall speed in z, otherwise hail embryo is following background winds
    vrel_hail [:] = init_vt.copy()
    hail_rel_w[:] = init_vt.copy()
    hail_rel_v[:] = np.zeros(n_traj)
    hail_rel_u[:] = np.zeros(n_traj)

    # Calculate the ground-relative hailstone velocities from the wind-relative hailstone velocities
    vz_storm_rel[:] = storm_rel_w - hail_rel_w
    vy_storm_rel[:] = storm_rel_v - hail_rel_v
    vx_storm_rel[:] = storm_rel_u - hail_rel_u
    
    # No tumbling adjustment or soaking  on first timestep
    D_max_tumble [:]    = D_max.copy()
    D_avg_eq_vol [:] = 2 * (volume * 3 / 4 / np.pi)**(1/3)
    A_coll_tumble[:], D_avg_eq_sfc[:] = calculate_surface_area(D_max, D_int, D_min, choose_lobes, np.ones(n_traj))
    dq_dt_heat_soaked   = np.zeros(n_traj)

    # All particles are initially massive and aloft
    massive = np.ones (n_traj).astype(bool) #tracker for hailstones that have non-zero ice mass
    done    = np.zeros(n_traj).astype(bool) #tracker for if hailstone has melted/hit the ground

    # Calculate initial Reynolds Number and Drag coefficient
    N_Re, Cde, CdeT, lobes_ratio[:] = drag_parameterization_wrapper(m_core, D_max, A_coll_tumble, D_min/D_max, int_ratio, vt_hail, vrel_hail, D_max_tumble, rho_ice, rho_air, T_inf, choose_lobes, choose_drag, Cdrag)

    # Tracker arrays for how and when hailstones meet their end
    total_times      = np.full(n_traj, -1)             # Hailstones' total time aloft for hailstone [s]
    final_timestep   = np.full(n_traj, -1).astype(int) # Hailstones' final timestep aloft [delt]
    gonewiththewinds = np.zeros(n_traj).astype(bool)   # Flag for if particle completely melts
    grounded         = np.zeros(n_traj).astype(bool)   # Flag for if particle hits the ground
    exited           = np.zeros(n_traj).astype(bool)   # Flag for if particle exits the domain before hitting the ground (also can flag particles that got inertially through by a bad inertial adjustment timescale)


    # Define data dictionaries for output
    # This is done through big dictionaries because we only want to create tracker arrays if we will output them
    # Big save on RAM!
    
    # Record the desired array structure for the output

    
    prior_data_dict = {
        'x'                 : {'fill_value': np.nan, 'dtype': float, 'data': init_x       [:]}, 
        'y'                 : {'fill_value': np.nan, 'dtype': float, 'data': init_y       [:]}, 
        'z'                 : {'fill_value': np.nan, 'dtype': float, 'data': init_z       [:]}, 
        'D'                 : {'fill_value': np.nan, 'dtype': float, 'data': init_diam    [:]}, 
        'D_max'             : {'fill_value': np.nan, 'dtype': float, 'data': init_dmax    [:]}, 
        'D_int'             : {'fill_value': np.nan, 'dtype': float, 'data': init_dint    [:]}, 
        'D_min'             : {'fill_value': np.nan, 'dtype': float, 'data': init_dmin    [:]}, 
        'volume'            : {'fill_value': np.nan, 'dtype': float, 'data': init_vol     [:]}, 
        'T_core'            : {'fill_value': np.nan, 'dtype': float, 'data': T_core   .copy()}, 
        'T_avg_liq'         : {'fill_value': np.nan, 'dtype': float, 'data': T_liq_avg.copy()}, 
        'T_sfc'             : {'fill_value': np.nan, 'dtype': float, 'data': T_sfc    .copy()}, 
        'T_inf'             : {'fill_value': np.nan, 'dtype': float, 'data': T_inf    .copy()}, 
        'm_core'            : {'fill_value': np.nan, 'dtype': float, 'data': init_m       [:]}, 
        'm_soaked'          : {'fill_value': np.nan, 'dtype': float, 'data': np.zeros(n_traj)}, 
        'm_shell'           : {'fill_value': np.nan, 'dtype': float, 'data': np.zeros(n_traj)}, 
        'rho_ice'           : {'fill_value': np.nan, 'dtype': float, 'data': init_rho     [:]}, 
        'rho_tot'           : {'fill_value': np.nan, 'dtype': float, 'data': init_rho     [:]}, 
        'rho_growth'        : {'fill_value': np.nan, 'dtype': float, 'data': init_rho     [:]}, 
        'vt_hail'           : {'fill_value': np.nan, 'dtype': float, 'data': init_vt      [:]}, 
        'vrel_hail'         : {'fill_value': np.nan, 'dtype': float, 'data': init_vt      [:]}, 
        'v_rel_z'           : {'fill_value': np.nan, 'dtype': float, 'data': hail_rel_w   [:]}, 
        'v_rel_y'           : {'fill_value': np.nan, 'dtype': float, 'data': hail_rel_v   [:]}, 
        'v_rel_x'           : {'fill_value': np.nan, 'dtype': float, 'data': hail_rel_u   [:]}, 
        'v_hail_x'          : {'fill_value': np.nan, 'dtype': float, 'data': vx_storm_rel [:]}, 
        'v_hail_y'          : {'fill_value': np.nan, 'dtype': float, 'data': vy_storm_rel [:]}, 
        'v_hail_z'          : {'fill_value': np.nan, 'dtype': float, 'data': vz_storm_rel [:]}, 
        'Ax_tumble'         : {'fill_value': np.nan, 'dtype': float, 'data': A_coll_tumble[:]}, 
        'wet_frac'          : {'fill_value': np.nan, 'dtype': float, 'data': np.zeros(n_traj)}, 
        'frozen_frac'       : {'fill_value': np.nan, 'dtype': float, 'data': np. ones(n_traj)}, 
        'wet_growth_tracker': {'fill_value':      0, 'dtype':  bool, 'data': np.zeros(n_traj)}, 
        'dry_growth_tracker': {'fill_value':      0, 'dtype':  bool, 'data': np.zeros(n_traj)}, 
        'melting_tracker'   : {'fill_value':      0, 'dtype':  bool, 'data': np.zeros(n_traj)}, 
        'clear_air_tracker' : {'fill_value':      0, 'dtype':  bool, 'data': np.zeros(n_traj)}, 
        'growth_regime'     : {'fill_value':      0, 'dtype':   int, 'data': np.zeros(n_traj)}, 
        'time_aloft'        : {'fill_value':     -1, 'dtype':  int}, 
        'final_time_step'   : {'fill_value':     -1, 'dtype':  int}, 
        'melted'            : {'fill_value':      0, 'dtype': bool}, 
        'grounded'          : {'fill_value':      0, 'dtype': bool}, 
        'exited'            : {'fill_value':      0, 'dtype': bool},
    }
    # Create dictionary of all of the variables to be output
    full_record_dict    = {}
    final_record_dict   = {}

    for key in prior_data_dict:
        
        # Build dictionary for full time tracker output
        if key in full_record_list:
            full_record_dict[key] = np.full((n_traj, int(nt/full_record_interval)+1), prior_data_dict[key]['fill_value']).astype(prior_data_dict[key]['dtype'])
            full_record_dict[key][:,0] = prior_data_dict[key]['data']

        # Build dictionary for final condition output
        if key in final_record_list:
            final_record_dict[key] = np.full(n_traj, prior_data_dict[key]['fill_value']).astype(prior_data_dict[key]['dtype'])   

    #----------------------------------------------------------------
    # Time loop structure wrapper beginning
    #----------------------------------------------------------------
    
    loaded_storm_time = sim_time # Record current storm timestep for temporal evolution
    
    #Continue to loop as long as hailstones have not hit the ground, with nt maximum iterations
    for tt in tqdm(np.arange(0,nt)):

        #----------------------------------------------------------------
        # Check if our particles have terminated
        #----------------------------------------------------------------
        
        done = np.logical_or(np.logical_or(np.logical_or(exited, grounded), gonewiththewinds), done)
        if np.all(done):
            break # If all hailstones have melted, hit the ground, or exited the domain, stop iterating
        n_aloft = np.count_nonzero(~done) # Number of unfinished hailstones

        
        #----------------------------------------------------------------
        # Load in the storm data
        #----------------------------------------------------------------

        # What storm output time should we be using for trajectory calculation?
        current_storm_time = sim_time + int(tt*delt/storm_delt) 

        # If we are doing evolving trajectories and the storm time we should be using is not what we currently have loaded, load the new storm time
        if evolving and (current_storm_time != loaded_storm_time):
            _, interp, _, xmin, xmax, ymin, ymax, zmin, zmax = load_storm_data(current_storm_time, path_to_storm, vars, filetype) # Load in the new storm timestep
            loaded_storm_time = current_storm_time # Update the storm time tracker
            
    
        #----------------------------------------------------------------
        # Interpolate our storm arrays to the current hailstone location
        #----------------------------------------------------------------
        
        # Interpolate all the storm arrays for the current hailstone positions
        out = interp((hail_z[~done], hail_y[~done], hail_x[~done]))
        
        rho_dry_air = out[:, 0]  #                   dry air density at the hailstone locations
        qc          = out[:, 1]  #          cloud water mixing ratio at the hailstone locations
        qr          = out[:, 2]  #           rain water mixing ratio at the hailstone locations
        nr          = out[:, 3]  #    rain drop number concentration at the hailstone locations
        qv          = out[:, 4]  #          water vapor mixing ratio at the hailstone locations
        qi          = out[:, 5]  #            cloud ice mixing ratio at the hailstone locations
        qs          = out[:, 6]  #                 snow mixing ratio at the hailstone locations
        p_air       = out[:, 7]  #                      air pressure at the hailstone locations
        storm_rel_u = out[:, 8]  # ground relative x-component winds at the hailstone locations
        storm_rel_v = out[:, 9]  # ground relative y-component winds at the hailstone locations
        storm_rel_w = out[:,10]  # ground relative z-component winds at the hailstone locations
    
        # Calcualte thermodynamic properties of the air at the hailstone locations
        rho_air = rho_dry_air * (1 + qv)  # total air density (kg/m3)    
        R = (1 - qv)  * Rdry + qv * Rv  # gas constant for moist air
        T_inf = (1 / (rho_air * R)) * p_air    # temperature (K)   
        
        cpi   = (-2.0572 + 0.14644*T_inf + 0.06163*T_inf*np.exp(-(T_inf/125.1)**2))*1000/18 # specific heat capacity of ice at constant pressure (J/kg/K) #Yuzu modded
        rhov  = qv * rho_dry_air                          # Vapor density (kg/m3)
        kT    = (2.381 + 0.0071 * (T_inf-T0)) * 1e-2      # thermal conductivity (J/m/s/K)
        Dv    = (2.11e-5) * (T_inf/T0)**1.94 * (p0/p_air) # diffusivity of vapor in air (m2/s)
        
        LWCc =  qc       * rho_dry_air # kg/m3 of cloud mass content
        LWCr =  qr       * rho_dry_air # kg/m3 of rain mass content
        IWC  = (qi + qs) * rho_dry_air # kg/m3 of IWC (cloud and snow ice)
   
    
        #----------------------------------------------------------------
        # Calculate collection parameters at the hailstone locations
        #----------------------------------------------------------------
        
        # mean droplet size in micrometers
        D_c    = 1e6 * (6.0 * LWCc / NCC / np.pi / rhol)**(1.0/3.0) # Mean cloud droplet diameter in the volume
        D_c_43 = D_c * 1.19 # correction to change r_m-->r_43 FOR NSSL MICROPHYSICS!! Volume-weighted mean cloud droplet diameter in the volume
        
        # Collection Efficiency for cloud droplets, when mean droplet size is smaller than Dcthresh, collection efficiencies < 1, otherwise collection efficiencies = 1
        Ecc = np.where(D_c < Dcthresh, (Ecthresh / Dcthresh) * D_c, 1)
        # Collection Efficiency for rain drops, everywhere equal to 0.8
        Ecr = np.ones(n_aloft)*0.8   
                
        # Slope parameter for raindrops (1/mm) - leading coef is conversion to 1/mm
        lamr = np.zeros_like(nr) # Slope parameter: initialize at zero for non-rainy grid cells
        has_rain = qr > 0
        lamr[has_rain] = (0.001) * (np.pi * rhol * nr[has_rain] / qr[has_rain])**(1.0/3.0) # Rainy cells, initialize slope parameter
        # Intercept parameter for raindrop size distribution. (1/m3 mm)
        n0r  = nr * lamr * rho_dry_air  
        
        # Construct DSD bins
        deldr     = 0.1 # increment of raindrop sizes [mm]
        drmm      = np.arange(deldr,8.0+deldr,deldr)  # raindrop diameters [mm]

        # Build arrays to quickly sum over the DSD
        n0r_1d  = np.tile( n0r[has_rain], (drmm.shape[0], 1)).T
        lamr_1d = np.tile(lamr[has_rain], (drmm.shape[0], 1)).T
        # Rain DSD
        rainND_1d = n0r_1d * np.exp(-lamr_1d * drmm) 

        # Drop fall velocity (Brandes et al. 2002)
        velrain   = -0.102 + 4.932*drmm - 0.9551*drmm**2 + 0.07934*drmm**3 - 0.002362*drmm**4 

        # Calculate the mean rain drop velocity (zero if no rain)
        velrain_Q = np.zeros(n_aloft)
        velrain_Q[has_rain] = np.sum(rainND_1d * velrain * drmm**3 * deldr, axis=-1) / np.sum(rainND_1d * drmm**3 * deldr)
               

        #----------------------------------------------------------------
        # Determine the liquid shell characteristics for the hailstones
        #----------------------------------------------------------------
        
        # Determine if the hailstone has a liquid shell. Flag used to determine ice collection, diffusion to ice or liquid, and enthalpy used for heat balance
        has_surface_liquid =  m_shell[~done] > liquid_shell_growing_threshold_mass
        enthalpy = np.where(has_surface_liquid, lv, ls)
    
        
        # Calculate the spherical equivalent radii (converted to m) of the full particle (ice and liquid, if relevant). Diffusion should happen at this distance
        radius_ice_core, radius_liquid_shell = radii_for_liquid_covered_stone(D_max[~done], D_int[~done], D_min[~done], m_shell[~done])
        sfc_radius = np.where(has_surface_liquid, radius_liquid_shell, radius_ice_core) # Pull the radius that should be used for conduction/diffusion
    
        # Identify the temperatures that we should assume for liquid mass coming from/going to the liquid shell at its innermost and outermost layers
        if choose_shell_T in [0,2]:
            # Boundary conditions mode. Internal physics takes the hailstone core temperature and external physics takes the surface temperature
            # When choose_shell_T = 0, T_sfc == T_core
            temperature_for_external_physics = T_sfc [~done]
            temperature_for_internal_physics = T_core[~done]
        else:
            # Average liquid temperature mode. Internal and external physics use the average temperature of the liquid layer
            temperature_for_external_physics = T_liq_avg[~done]
            temperature_for_internal_physics = T_liq_avg[~done]      
    
        
        #----------------------------------------------------------------
        # External Physics
        #----------------------------------------------------------------
        
        
        # DIFFUSION AND CONDUCTION (dm/dt|diff), (dq/dt|diff), (dq/dt|cond)
        # Hailstone aspect ratio
        aspect_ratio = D_min[~done]/D_max[~done] 
        # Prandtl and Schmidt numbers
        N_Pr, N_Sc = get_dimensionless_flow_numbers(T_inf, rho_air, Dv) 
        # Diffusion and conduction coefficients
        rh87_diff_coef, rh87_cond_coef = get_rh87_diffusion_and_conduction_coefficients(N_Re[~done[massive]], aspect_ratio, N_Pr, N_Sc, choose_chi) 
        # Calculate heat and mass transfer by conduction and diffusion
        dm_dt_diffusion, dq_dt_diffusion, dq_dt_conduction = diffusion_and_conduction(sfc_radius, rh87_cond_coef, rh87_diff_coef, T_sfc[~done], temperature_for_external_physics, T_inf, Dv, rhov, kT, enthalpy, delt) 
    
        # CONTINUOUS COLLECTION
        # Calculate the hydrometeor mass in the sweep-out volume (not necessarily collected)---kept as a kg/s rate
        cloud_water_dm_dt, rain_water_dm_dt, cloud_ice_dm_dt = continuous_collection(A_coll_tumble[~done], vrel_hail[~done], sfcdens, rho_dry_air, Ecc, Ecr, Eci, velrain_Q, LWCc, LWCr, IWC)
        # Calculate mass and heat transfer from accretion
        dm_dt_accreted_liq, dm_dt_accreted_ice, dq_dt_accretion = accretion(cloud_water_dm_dt, rain_water_dm_dt, cloud_ice_dm_dt, has_surface_liquid, temperature_for_external_physics, cpi, T_inf)
        
        
        #----------------------------------------------------------------
        # Liquid Shell Physics
        #----------------------------------------------------------------
        
        # Calculate the total conduction through the liquid shell (dq/dt|liq cond)
        if choose_shell_T == 0:
            # If we are in T_shell = T0 mode, all external heating is transported to the ice core
            dq_dt_liq_transport = dq_dt_conduction + dq_dt_diffusion + dq_dt_accretion
        else:
            # Calculate the ramping function between full conduction (thick shell) and full transfer (thin shell) modes
            transport_fraction = find_transport_fraction(radius_liquid_shell, radius_ice_core, liquid_shell_physics_radius_threshold_1, liquid_shell_physics_radius_threshold_2)

            # Calculate heat transfer by conduction through liquid (assuming full conduction)
            dq_dt_liq_conduction = conduction_through_liquid(radius_ice_core, radius_liquid_shell, T_liq_avg[~done], T_sfc[~done])

            # Calculate total heat transferred through the external shell to the ice core
            dq_dt_liq_transport = (1-transport_fraction)*dq_dt_liq_conduction + (transport_fraction)*(dq_dt_conduction + dq_dt_diffusion + dq_dt_accretion)
    
        
        
        #----------------------------------------------------------------
        # Compute the hailstone heat balance equations
        #----------------------------------------------------------------

        #Define the dependent heat transfer arrays
        dq_dt_frz_melt   = np.zeros(n_aloft)
        dq_dt_frz_soaked = np.zeros(n_aloft)
        dq_dt_heat_hail  = np.zeros(n_aloft)
        dq_dt_heat_shell = np.zeros(n_aloft)
        
        # Calculate the heating rate required to heat the hailstone to T0 during this timestep
        # This is the maximum amount of heat that the hailstone can absorb before it begins to melt
        dq_dt_heat_hail_total =  (m_core[~done] / delt) * cpi * (T0 - T_core[~done])

        # Sum up the total intermittantly liquid mass (this includes prior liquid shell mass, new accretion, and mass diffused to liquid even if it will be frozen this timestep).
        all_shell_mass = m_shell[~done] + dm_dt_accreted_liq*delt + dm_dt_accreted_ice*delt + np.where(has_surface_liquid, dm_dt_diffusion, 0)*delt
        
        # Calculate the total heat rate that would result from the complete heating of various liquid sources:
        dq_dt_frz_liquid_total = (all_shell_mass  / delt) * (lf + cpw * (temperature_for_internal_physics - T0)) # The preexisting liquid shell
        dq_dt_frz_soaked_total = (m_soaked[~done] / delt) *  lf  # Preexisting soaked mass
        
        # Calculate how much excess heat source/sink there would be if all_shell_mass liquid was frozen and the hailstone was heated to T0 during this timestep given the precalculated physical processes
        # Flag for wet/dry hailstone. If there is excess heat, the hailstone is not cold enough to freeze all liquid at this time (results in a wet hailstone)
        full_freeze = (dq_dt_conduction + dq_dt_diffusion + dq_dt_accretion + dq_dt_heat_soaked[~done[massive]] + dq_dt_frz_liquid_total <= dq_dt_heat_hail_total)

        #----------------------------------------------------------------
        # BALANCING HEAT TRANSFER FOR DRY HAILSTONE
        #----------------------------------------------------------------

        # Nno liquid shell heating in dry growth
        dq_dt_heat_shell[full_freeze] = 0
        # If the hailstone is dry, we are able to freeze all surface and growth liquid. Set heat transfer for melting/freezing equal to that of complete freezing of surface and growth liquid
        dq_dt_frz_melt  [full_freeze] =  dq_dt_frz_liquid_total[full_freeze]
        # All excess heat goes towards first freezing soaked liquid and second cooling the hailstone
        # The total heat transfer from these two processes must balance the heat transfer from the physical processes and freezing/melting
        dq_dt_frz_soaked_and_heat_hailstone = dq_dt_conduction[full_freeze] + dq_dt_diffusion[full_freeze] + dq_dt_accretion[full_freeze] + dq_dt_heat_soaked[~done[massive]][full_freeze] + dq_dt_frz_melt[full_freeze]
        # If there is a heat sink, first use it to freeze soaked liquid
        dq_dt_frz_soaked[full_freeze] = np.minimum(np.maximum(-dq_dt_frz_soaked_and_heat_hailstone, 0), dq_dt_frz_soaked_total[full_freeze])
        # Any excess heat source or sink after the freezing of soaked mass goes towards heating/cooling the hailstone
        dq_dt_heat_hail [full_freeze] = dq_dt_frz_soaked[full_freeze] + dq_dt_frz_soaked_and_heat_hailstone

        #----------------------------------------------------------------
        # BALANCING HEAT TRANSFER FOR WET HAILSTONE
        #----------------------------------------------------------------
    
        # If the hailstone cannot freeze the full amount of surface liquid, the hailstone will be heated to T0
        dq_dt_heat_hail [~full_freeze] = dq_dt_heat_hail_total[~full_freeze]
        # Calculate the total amount of heat transfer that will go towards freezing/melting
        dq_dt_all_freezing = dq_dt_heat_hail[~full_freeze] - (dq_dt_heat_soaked[~done[massive]][~full_freeze] + dq_dt_liq_transport[~full_freeze])
        # Compute the maximum freezing of soaked mass that could occur regardless of heat transfer at this time (just by volume)
        # It is whichever is smaller of the soaked mass required to densify the hailstone to rho_soak_freeze_thresh or the amount of soaked mass
        max_frz_soaked_mass_wet_growth = np.maximum(np.minimum((rho_soak_freeze_thresh - rho_ice[~done][~full_freeze])*volume[~done][~full_freeze], m_soaked[~done][~full_freeze]), 0)
        dq_dt_frz_soaked_maximum = max_frz_soaked_mass_wet_growth / delt * lf
        # Dedicate rho_soak_freeze_fraction portion of the heat transfer going towards freezing at this time to the freezing of soaked mass, until max_frz_soaked_mass_wet_growth has been reached
        dq_dt_frz_soaked[~full_freeze] = np.minimum(np.maximum(dq_dt_all_freezing, 0)*rho_soak_freeze_fraction, dq_dt_frz_soaked_maximum)
        # The amount of freezing going towards freezing/melting of the liquid shell This value may be positive or negative, indicating net freezing and melting, respectively.
        dq_dt_frz_melt  [~full_freeze] = dq_dt_all_freezing - dq_dt_frz_soaked[~full_freeze]
        # Any excess heat after accounting for the physical processes and raising the hailstone temperature will go towards freezing/melting between the liquid shell and the ice core
        dq_dt_heat_shell[~full_freeze] = dq_dt_conduction[~full_freeze] + dq_dt_diffusion[~full_freeze] + dq_dt_accretion[~full_freeze] - dq_dt_liq_transport[~full_freeze]
    
        
        #----------------------------------------------------------------
        # Compute growth regimes and fractional freezing/melting
        #----------------------------------------------------------------
    
        # Designate growth regimes:
        # If the hailstone is not able to freeze all liquid and there is net melting: "melting"
        is_melting    = ~full_freeze & (dq_dt_frz_melt <  0)
        # If the hailstone is not able to freeze but there is net freezing: "wet growth"
        in_wet_growth = ~full_freeze & (dq_dt_frz_melt >= 0)
        # If the hailstone can fully freeze liquid, "dry growth" and is accreting 
        in_dry_growth =  full_freeze & ~(dm_dt_accreted_liq + dm_dt_accreted_ice == 0)
        # If the hailstone can fully freeze liquid, but is not accreting
        in_clear_air  =  full_freeze &  (dm_dt_accreted_liq + dm_dt_accreted_ice == 0)
        
        
        
        #----------------------------------------------------------------
        # Compute the hailstone mass transfer from melting/freezing
        #----------------------------------------------------------------
    
        # timestep mass transfer between the liquid shell and the ice core
        freeze_shell_mass = (dq_dt_frz_melt / (lf + cpw * (temperature_for_internal_physics - T0))) * delt 
        # timestep mass transfer between the soaked mass and the ice core
        freeze_soaked_mass = (dq_dt_frz_soaked / lf) * delt      

        # timestep mass transfer from diffusion to/from various locations
        liq_growth_diffusion = np.where( has_surface_liquid, dm_dt_diffusion * delt, 0)
        ice_growth_diffusion = np.where(~has_surface_liquid, dm_dt_diffusion * delt, 0)
        tot_growth_accretion = (dm_dt_accreted_liq + dm_dt_accreted_ice)*delt
        
    
        #----------------------------------------------------------------
        # Sum up the total mass changes and iterate for the liquid shell/soaked mass/ice core
        #----------------------------------------------------------------
        
    
        # Add the mass changes to each respective mass tracker
        m_shell [~done] = m_shell [~done] - freeze_shell_mass + liq_growth_diffusion + tot_growth_accretion 
        m_core  [~done] = m_core  [~done] + freeze_shell_mass + ice_growth_diffusion + freeze_soaked_mass
        m_soaked[~done] = m_soaked[~done] - freeze_soaked_mass
        # Clear out any finished tracker spots
        m_shell [ done] = np.nan
        m_soaked[ done] = np.nan
        m_core  [ done] = np.nan

        # Isolate the hailstones that still have non-zero ice mass after the mass changes
        massive   = (~done) & (m_core > 0)
        
        
        #----------------------------------------------------------------
        # Update the hailstone sizes
        #----------------------------------------------------------------

        # Designate the density of growth to be used for each growth regime
        use_spongy_density = in_wet_growth
        use_rime_density   = in_dry_growth
        use_same_density   = is_melting | in_clear_air

        
        # Calculate the Frozen Fraction
        FrozenFrac = np.zeros(n_aloft)
        # FrozenFrac is the proportion of liquid shell (including new accretion/diffusion) that freezes during this timestep
        FrozenFrac[in_wet_growth] = (freeze_shell_mass[in_wet_growth]) / (freeze_shell_mass[in_wet_growth] + m_shell[~done][in_wet_growth])
        FrozenFrac[in_dry_growth] = 1
        FrozenFrac[in_clear_air ] = 1
        FrozenFrac[   is_melting] = 0
        
        # Clean up fractional trackers to stay between 0 and 1
        FrozenFrac[FrozenFrac > 1] = 1
        FrozenFrac[FrozenFrac < 0] = 0
        
    
        # Growth Density: Decide rime vs. spongy growth density based on wet vs. dry growth
        growth_density = np.zeros(np.count_nonzero(massive))
        # Calculate the density of the ice particle mass change during this timestep
        # If the hailstone is melting, keep the ice density constant (assumes uniform density/uniform soaking)
        growth_density[  use_same_density[massive[~done]]] = rho_ice[~done][use_same_density&massive[~done]]
        # Calculate the growth densities for riming and spongy particles
        growth_density[  use_rime_density[massive[~done]]] = density_of_rime_accretion(vrel_hail[~done][use_rime_density&massive[~done]], T_core[~done][use_rime_density&massive[~done]], T_inf[(use_rime_density&massive[~done])], D_c_43[(use_rime_density&massive[~done])], vIMP)
        growth_density[use_spongy_density[massive[~done]]] = density_of_spongy_growth(FrozenFrac[ use_spongy_density&massive[~done]])

        # Update the hailstones' sizes
        D_max[massive], D_int[massive], D_min[massive], volume[massive] = shape_diagnostics_wrapper(D_max[massive], D_int[massive], D_min[massive], int_ratio[massive], freeze_shell_mass[massive[~done]], ice_growth_diffusion[massive[~done]], growth_density, rho_ice[massive], randindA[massive], choose_shape)
        # Clear size arrays for non-massive hailstones
        D_max [~massive] = np.nan
        D_int [~massive] = np.nan
        D_min [~massive] = np.nan
        volume[~massive] = np.nan
    
    
    
        #----------------------------------------------------------------
        # Soaking the outer growth layer/unsoaking any melted layers
        #----------------------------------------------------------------
        
        # For dry hailstones that are freezing soaked liquid and oozing that soaked liquid back to their exterior, let that liquid be T0
        unsoaking_dry_hailstones = (m_shell[massive] <= 0) & (m_soaked[massive] > 0)
        temperature_for_internal_physics[massive[~done]] = np.where(unsoaking_dry_hailstones, T0, temperature_for_internal_physics[massive[~done]])
        
        # Soak any remaining liquid mass into the new hailstone growth layer
        new_soaked_mass, dq_dt_heat_soaked = soaking(m_core[massive], volume[massive], m_shell[massive], m_soaked[massive], temperature_for_internal_physics[massive[~done]], delt)
        
        # Remove the newly soaked mass from the liquid shell and add it to the soaked mass
        m_shell [massive] -= new_soaked_mass
        m_soaked[massive] += new_soaked_mass
    
        # Calculate the new hailstone density, both for the ice core only and for the soaked particle
        rho_ice  [ massive] = m_core[massive] / volume[massive]
        rho_ice  [~massive] = np.nan
        hailstone_total_density = (m_core[massive] + m_soaked[massive]) / volume[massive]
        
    
    
            
        #----------------------------------------------------------------
        # Update the hailstone temperature
        #----------------------------------------------------------------
    
        if choose_shell_T == 0:
            # Use the calculated heat transfer for heating/cooling the hailstone to update the hailstone temperature
            # When liquid shell temperature is set to T0, this all goes to the ice core
            T_core   [massive] = T_core   [massive] + dq_dt_heat_hail[massive[~done]] * delt / (m_core[massive] * cpi[massive[~done]])
            T_liq_avg[massive] = T_core   [massive]
        else:
            # Identify the new liquid shell characteristics of hailstones post-soaking
            has_liquid_mass = massive & (m_shell >  liquid_shell_growing_threshold_mass)
            no_liquid_mass  = massive & (m_shell <= liquid_shell_growing_threshold_mass)
            radius_ice_core, radius_liquid_shell = radii_for_liquid_covered_stone(D_max[has_liquid_mass], D_int[has_liquid_mass], D_min[has_liquid_mass], m_shell[has_liquid_mass])
            
            # Use the calculated heat transfer for heating/cooling the hailstone to update the hailstone temperature
            T_core   [massive] = T_core[massive] + dq_dt_heat_hail [massive[~done]] * delt / (m_core[massive] * cpi[massive[~done]])

            # Calculate the heated liquid shell temperature when a liquid shell exists, otherwise, all liquid temperature trackers are equal to T_core    
            T_liq_avg[has_liquid_mass] = T_liq_avg[has_liquid_mass] + dq_dt_heat_shell[has_liquid_mass[~done]] * delt / (cpw * m_shell[has_liquid_mass])
            T_liq_avg[ no_liquid_mass] =    T_core[ no_liquid_mass]

            # Make sure that dry growth hailstones (whose liquid shell froze this timestep, but may have gained a new liquid shell from soaking) have liquid shell temperature of T0
            T_liq_avg[massive] = np.where(unsoaking_dry_hailstones, T0, T_liq_avg[massive])
    
        # Clear out temperature arrays for non-massive particles
        T_core   [~massive] = np.nan
        T_liq_avg[~massive] = np.nan
        
            
        
        #----------------------------------------------------------------
        # Hailstone tumbling, surface area, and fall speed updates 
        #----------------------------------------------------------------
    
        # Calculate the collection area and equivalent spherical diameter for the next timestep
        collection_area_max, D_avg_eq_sfc = calculate_surface_area(D_max[massive], D_int[massive], D_min[massive], choose_lobes, lobes_ratio[massive])
    
        # Calculate the tumbling area and dimensions for the next timestep
        D_max_tumble [ massive], A_coll_tumble[massive] = tumbling_parameterization(D_max[massive], D_int[massive], D_min[massive], collection_area_max, choose_tumble)
        D_max_tumble [~massive] = np.nan
        A_coll_tumble[~massive] = np.nan
    
        # We want to use the total hailstone mass to calculate its fall speed, including soaked liquid
        total_particle_mass = m_core[massive] + m_soaked[massive]
    
        # Calculate the hailstones fall speeds
        aspect_ratio = D_min[massive]/D_max[massive] # Aspect ratio
        N_Re, Cde, CdeT, lobes_ratio[massive] = drag_parameterization_wrapper(m_core[massive], D_max[massive], A_coll_tumble[massive], aspect_ratio, int_ratio[massive], vt_hail[massive], vrel_hail[massive], D_max_tumble[massive], hailstone_total_density, rho_air[massive[~done]], T_inf[massive[~done]], choose_lobes, choose_drag, Cdrag) # Drag coefficient, reynolds number and lobe characteristics
        lobes_ratio[~massive] = np.nan

        # Update hailstone terminal fall speed
        vt_hail[ massive] = velocity_parameterization(D_max[massive], total_particle_mass, A_coll_tumble[massive], hailstone_total_density, sfcdens, rho_air[massive[~done]], CdeT, randindV[massive], choose_vt)

        # Calculate the inertial velocities of the hailstones
        vrel_hail[massive], vz_storm_rel[massive], vy_storm_rel[massive], vx_storm_rel[massive], hail_rel_w[massive], hail_rel_v[massive], hail_rel_u[massive] = inertial_velocity(D_max[massive], vrel_hail[massive], vt_hail[massive], hailstone_total_density, storm_rel_w[massive[~done]], storm_rel_v[massive[~done]], storm_rel_u[massive[~done]], hail_rel_w[massive], hail_rel_v[massive], hail_rel_u[massive], vz_storm_rel[massive], vy_storm_rel[massive], vx_storm_rel[massive], rho_dry_air[massive[~done]], Cde, CdeT, choose_inertia, delt)
        
        # Clear out velocity arrays for non-massive particles
        vt_hail     [~massive] = np.nan
        vrel_hail   [~massive] = np.nan
        vz_storm_rel[~massive] = np.nan
        vy_storm_rel[~massive] = np.nan
        vx_storm_rel[~massive] = np.nan
        hail_rel_w  [~massive] = np.nan
        hail_rel_v  [~massive] = np.nan
        hail_rel_u  [~massive] = np.nan
    
        
        #----------------------------------------------------------------
        # Shedding module
        #----------------------------------------------------------------
    
        # Shed any liquid shell mass that exceeds the critical amount of retainable liquid
        m_shell [massive] = shed_excess_liquid(m_core[massive], D_avg_eq_sfc, volume[massive], m_shell[massive])

        # Calculate new liquid shell characteristics post-shedding
        has_liquid_mass = (m_shell > liquid_shell_growing_threshold_mass) & massive & ~done
        radius_ice_core, radius_liquid_shell = radii_for_liquid_covered_stone(D_max[has_liquid_mass], D_int[has_liquid_mass], D_min[has_liquid_mass], m_shell[has_liquid_mass])

        # Calculate new hailstone surface temperature when there is a liquid layer
        T_sfc[has_liquid_mass] = surface_temperature_from_average(radius_liquid_shell, radius_ice_core, T_liq_avg[has_liquid_mass])
        # If no liquid layer, the surface temperature is the ice core temperature
        T_sfc[massive&(~has_liquid_mass)] = T_core[massive&(~has_liquid_mass)]
        T_sfc[~massive] = np.nan
        
        #Update Water fraction: This tracks the proportion of the total particle mass that is liquid (soaked or in the shell)
        WetFrac = (m_shell[massive] + m_soaked[massive]) / (m_core[massive] + m_shell[massive] + m_soaked[massive])
    
    
        #----------------------------------------------------------------
        # Hailstone Advection
        #----------------------------------------------------------------
        
        # Advect the hailstones to determine their new (x, y, z) locations
        hail_x[massive], hail_y[massive], hail_z[massive] = advection_module(hail_x[massive], hail_y[massive], hail_z[massive], vz_storm_rel[massive], vy_storm_rel[massive], vx_storm_rel[massive], delt)
    
        hail_x[~massive] = np.nan
        hail_y[~massive] = np.nan
        hail_z[~massive] = np.nan
        
        #----------------------------------------------------------------
        # Update completed hailstone trackers
        #----------------------------------------------------------------

        # Did the hailstone hit the ground?
        ground_exit = ~ (hail_z >= zmin)
        # Did the hailstone exit the domain not at the ground? This can be a sign of sketchiness in inertial velocities!!
        other_exit  = ~((hail_x >= xmin) & (hail_x <= xmax) & (hail_y >= ymin) & (hail_y <= ymax) & (hail_z <= zmax))
        # Did the hailstone melt or exit the domain this timestep?
        finished_this_timestep = ~done & np.logical_or(np.logical_or(ground_exit, other_exit), ~massive)

        # Identify and track the reason for hail trajectory termination
        gonewiththewinds[~done & ~massive   ] = 1 # gonewiththewinds identifies the fully melted hailstones
        grounded        [~done & ground_exit & ~gonewiththewinds] = 1 # Tracker for hailstones that hit the ground
        exited          [~done &  other_exit & ~gonewiththewinds] = 1 # Tracker for hailstones that exit the domain not at the ground
        
        #Set the final time for any hailstones that hit the ground
        total_times   [finished_this_timestep] = tt*delt
        final_timestep[finished_this_timestep] = tt

        # Data dictionary for mid-run trackers
        mid_run_data_dict = {
            'x'                 : hail_x[:], 
            'y'                 : hail_y[:], 
            'z'                 : hail_z[:], 
            'D_max'             : D_max[:], 
            'D_int'             : D_int[:], 
            'D_min'             : D_min[:], 
            'volume'            : volume[:],
            'm_core'            : m_core[:], 
            'm_soaked'          : m_soaked[:], 
            'm_shell'           : m_shell[:], 
            'rho_ice'           : rho_ice[:], 
            'rho_tot'           : hailstone_total_density[:], 
            'rho_growth'        : growth_density[:], 
            'wet_frac'          : WetFrac[:], 
            'frozen_frac'       : FrozenFrac[:], 
            'vt_hail'           : vt_hail[:], 
            'vrel_hail'         : vrel_hail[:], 
            'v_hail_x'          : vx_storm_rel[:], 
            'v_hail_y'          : vy_storm_rel[:], 
            'v_hail_z'          : vz_storm_rel[:], 
            'v_rel_x'           : hail_rel_u[:], 
            'v_rel_y'           : hail_rel_v[:], 
            'v_rel_z'           : hail_rel_w[:], 
            'T_core'            : T_core[:],
            'T_avg_liq'         : T_liq_avg[:],
            'T_sfc'             : T_sfc[:],
            'T_inf'             : T_inf[:],
            'Ax_tumble'         : A_coll_tumble[:],
            'wet_growth_tracker': in_wet_growth[:],
            'dry_growth_tracker': in_dry_growth[:],
            'melting_tracker'   : is_melting[:],
            'clear_air_tracker' : in_clear_air[:],
            'growth_regime'     : (is_melting*1 + in_dry_growth*2 + in_wet_growth*3 + in_clear_air*4)[:],
            'grounded'          : grounded,
            'exited'            : exited,
            'melted'            : gonewiththewinds,
            'time_aloft'        : total_times,
            'final_time_step'   : final_timestep,
        }

        

        for key in mid_run_data_dict:
            # Update our mid-run final hailstone characteristics trackers for hailstones that fell this timestep
            if key in final_record_list:
                
                data = mid_run_data_dict[key]
                
                # Shape checking in case we are saving arrays that do not have dimension (n_traj)
                if data.shape[0] == n_traj:
                    final_record_dict[key][finished_this_timestep        ] = data[finished_this_timestep         ]
                elif data.shape[0] == np.count_nonzero(massive):
                    final_record_dict[key][finished_this_timestep&massive] = data[finished_this_timestep[massive]]
                elif data.shape[0] == np.count_nonzero(~done):
                    final_record_dict[key][finished_this_timestep&~done  ] = data[finished_this_timestep[~done  ]]
                else:
                    final_record_dict[key][finished_this_timestep        ] = data[finished_this_timestep         ]

        if ((tt+1) % full_record_interval) == 0:
            for key in mid_run_data_dict:
                # Update our mid-run full hailstone characteristics trackers for all hailstones
                if key in full_record_list:
                    
                    data = mid_run_data_dict[key]
                    
                    # Shape checking in case we are saving arrays that do not have dimension (n_traj)
                    if data.shape[0] == n_traj:
                        full_record_dict[key][      :,int(tt/full_record_interval)+1] = data
                    elif data.shape[0] == np.count_nonzero(massive):
                        full_record_dict[key][massive,int(tt/full_record_interval)+1] = data
                    elif data.shape[0] == np.count_nonzero(~done):
                        full_record_dict[key][  ~done,int(tt/full_record_interval)+1] = data
                    else:
                        full_record_dict[key][      :,int(tt/full_record_interval)+1] = data


    # -------------------------------------------------------
    # Time loop finished! Start exit processes
    # -------------------------------------------------------
    
    print('Finished Loop at', tt)
    print('# Melted  ', np.count_nonzero(gonewiththewinds), np.round( np.count_nonzero(gonewiththewinds) / n_traj * 100, 2), '%')
    print('# Grounded', np.count_nonzero(grounded        ), np.round( np.count_nonzero(grounded        ) / n_traj * 100, 2), '%')
    print('# Exited  ', np.count_nonzero(exited          ), np.round( np.count_nonzero(exited          ) / n_traj * 100, 2), '%')
    print('# Aloft   ', np.count_nonzero(~done           ), np.round( np.count_nonzero(~done           ) / n_traj * 100, 2), '%')

    output_ds = xr.Dataset()
    if output_which == 'unmelted':
        save_out = ~gonewiththewinds # Save data for unmelted hailstones only
    elif output_which == 'grounded':
        save_out = grounded          # Save data for grounded hailstones only
    elif output_which == 'all':
        save_out = np.ones(n_traj)   # Save data for all trajcetories
    else:
        save_out = np.ones(n_traj)   # Default is to output all trajectories
        
    traj = np.arange(np.count_nonzero(save_out))

    # Build a dataset with all of the desired output
    # Loop through the variables to save out at each time and record their properties from the lookup dictionaries
    for key in init_data_dict:
        if key in initial_record_list:
            output_ds[init_data_dict[key]['name']] = xr.DataArray(prior_data_dict[key]['data'][save_out], **init_data_dict[key], dims=['traj'], coords={'traj': traj}, )

    for key in final_data_dict:
        if key in final_record_list:
            output_ds[final_data_dict[key]['name']] = xr.DataArray(final_record_dict[key][save_out], **final_data_dict[key], dims=['traj'], coords={'traj': traj}, )

    for key in full_data_dict:
        if key in full_record_list:
            output_ds[full_data_dict[key]['name']] = xr.DataArray(full_record_dict[key][save_out,:int(tt/full_record_interval)+1], **full_data_dict[key], dims=['traj', 'timestep'], coords={'traj': traj, 'timestep': np.arange(0,tt+1, full_record_interval)}, )

    # Add initial configuration to the attributes of the dataset for later reference
    output_ds = output_ds.assign_attrs(
        storm_file       = path_to_storm,
        sim_time         = sim_time,
        evolving         = int(evolving),
        total_t          = total_t,
        delt             = delt,
        choose_tumble    = choose_tumble,
        choose_vt        = choose_vt,
        choose_chi       = choose_chi,
        choose_drag      = choose_drag,
        choose_int_ratio = choose_int_ratio,
        choose_aspect    = choose_aspect,
        choose_lobes     = choose_lobes,
        choose_seed      = choose_seed,
        Cdrag            = Cdrag,
        choose_shell_T = choose_shell_T,
        choose_shape   = choose_shape,
    )

    # Save the output dataset        
    if output_path not in ['', 'none', None]:
        output_ds.to_netcdf(output_path)

    # Return the output dataset
    return output_ds
    

if __name__ == "__main__":
    main()
