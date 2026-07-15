# Hail Trajectory and Growth Model
## Overview

This model simulates the trajectories, growth, and melting of non-spherical hailstones through a three-dimensional storm simulation. The original hail trajectory model was built and documented in Kumjian and Lombardo (2020). Lin et al. (2024) updated the model to simulate non-spherical hailstones. Spychalla et al. (in prep) further edited the hailstone growth and melting physics to better match the physical behavior of growing hailstones.

Current Model Version 1.0. Last Updated: July 15, 2026.

## Model Requirements
The model is defined by trajectory_model.py.

Model dependencies: numpy, xarray, scipy, glob, and tqdm.

If random tumbling is desired, the file "tumblerand.mat" is necessary, which contains the lookup table used to define the impact of hailstone tumbling on its orientation. The path of tumblerand.mat should be provided as an input to the model in *tumbling_file* if not in the same directory as the trajectory model itself.

## Running the Model

To run the model, a user will provide the initial conditions of the hail embryos to be simulated, a storm simulation in which hail growth will be computed, and specification about desired output. The user may also choose to modify the model configuration and desired parameterizations. More information is provided below regarding each of these components.

An example script to run the hail model is provided in run_hail_trajectories.py and is provided at the bottom of this document.

## Hail Embryo Initial Conditions

Hail trajectories are computed for hail embryos specified by the user. The trajectory model **main** function requires five hail embryo input arrays:
- *init_x*: (1D float array) hail embryo initial x locations in storm coordinates \[km\]
- *init_y*: (1D float array) hail embryo initial y locations in storm coordinates \[km\]
- *init_z*: (1D float array) hail embryo initial z locations in storm coordinates \[km\]
- *init_diam*: (1D float array) hail embryo initial diameter \[mm\]
  - Note that *init_diam* is assumed to be the equivalent volume spherical diameter of the hailstone at the first timestep. Depending on *choose_shape*, *init_diam* may or may not be exactly equal to any of $D_\mathrm{max}$, $D_\mathrm{int}$, or $D_\mathrm{min}$. In many cases, *init_diam* will be less than $D_\mathrm{max}$ at the first timestep.
- *init_rho*: (1D float array) hail embryo initial ice core density \[kg m$^{-3}$\]

Hail embryos may be manually chosen and provided by the user. The trajectory model module also contains a collection of functions to fascilitate embryo specification. See individual function documentation for more details on each particular sampling function.

Functions to fascilitate embryo location sampling:
- To sample embryo locations from a random uniform distribution within a provided bounding box: **sample_locations_box**
- To sample embryo locations uniformly from only masked locations within an array: **sample_locations_mask**
- To sample embryo locations from a weighted mask array: **sample_locations_weighted_mask**

Functions to fascilitate embryo size sampling:
- To sample sizes from a uniform distribution: **sample_sizes_uniform**
- To sample sizes from a distribution with linearly decreasing frequency with increasing size: **sample_sizes_linear**
- To sample sizes from a single gamma distribution: **sample_sizes_gamma**
- To sample sizes from a variable gamma distribution defined by a spatial mask and characteristic sizes: **sample_sizes_spatial_gamma**

Functions to fascilitate embryo density sampling:
- To sample densities from a uniform distribution: **sample_densities_uniform**
- To sample densities from a spatially varying gaussian distribution: **sample_densities_spatial**

## Storm Simulation
The user is required to provide a storm simulation (e.g., from CM1, WRF, ...) in which to compute hail trajectories.

### Storm Dataset File Paths
The parent directory of the storm datasets should be provided by the user as a model input in *path_to_storm*. 

The provided storm data must be stored in netcdf or zarr format. The filetype should be specified by the input *filetype*='nc' for netcdf files and *filetype*='zarr' for zarr files. 

The input parameter *sim_time* defines the name index of the initial storm file in which trajectories will be seeded. 

The file paths of the storm dataset should be discoverable at {*path_to_storm*}/\*{*sim_time*}.{*filetype*}.

If evolving trajectories are desired, the user should specify *evolving*=True. In this case, subsequent storm times should be discoverable at
{*path_to_storm*}/\*{*sim_time*+1}.{*filetype*}
{*path_to_storm*}/\*{*sim_time*+2}.{*filetype*}
{*path_to_storm*}/\*{*sim_time*+3}.{*filetype*}.
The output time length between output times of the storm should be specified by the input *storm_delt* with units of seconds. It is currently not possible for trajectory model simulations to skip over storm output times during trajectory computation. If a value of *storm_delt* is provided that does not correspond to subsequent output times,  storm evolution will not be correctly simulated.

### File Format and Contents
Each storm dataset must contain the following variables:

| Quantity                          | Units   | Likely Variable Names | *vars* index   |
| --------------------------------- | ------- | --------------------- | -------------- |
| Dry air density                   | kg m⁻³  | rho                   | 0              |
| Pressure                          | Pa      | prs, p                | 1              |
| Water vapor mixing ratio          | kg kg⁻¹ | qv                    | 2              |
| Cloud water mixing ratio          | kg kg⁻¹ | qc                    | 3              |
| Rain water mixing ratio           | kg kg⁻¹ | qr                    | 4              |
| Rain water number concentration   | kg kg⁻¹ | nr, crw               | 5              |
| Cloud ice mixing ratio            | kg kg⁻¹ | qi                    | 6              |
| Snow mixing ratio                 | kg kg⁻¹ | qs                    | 7              |
| x-component wind                  | m s⁻¹   | u, uinterp            | 8              |
| y-component wind                  | m s⁻¹   | v, vinterp            | 9              |
| z-component wind                  | m s⁻¹   | w, winterp            | 10             |

| Coordinate                        | Units   | Likely Variable Names |
| --------------------------------- | ------- | --------------------- |
| x coordinate                      | km      | x, i,                 |
| y coordinate                      | km      | y, j,                 |
| z coordinate                      | km      | z, k,                 |


Because specific variable names vary for these quantities depending on the storm dataset, the input parameter *vars* allows a user to specify the names of needed storm variables unique to their dataset. *vars* must be a 10-item list of the variable names for the 10 needed storm variables **in the order of their *vars* index listed above**. Incorrect ordering will cause fatal errors to the trajectory model. 

All storm variables are assumed to be on the same grid. Providing variables on differently staggered grids will result in failure to properly load the storm data. Dimensions of provided data may appear in any order; however, the respective dimensions **must be in units of km** and must contain one of the following character strings to be identified: {'i', 'x', 'lon', 'north', 'south'}, {'j', 'y', 'lat', 'west', 'east'}, and {'k', 'z', 'alt', 'top', 'bottom'} (e.g., valid dimensions include {\['zh', 'yh', 'xh'\], \['north_south', 'west_east', 'top_bottom'\], \['latitude', 'longitude', 'altitude'\], \['ni', 'nj', 'nk'\]}. 

The trajectory model will linearly interpolate the given storm data. This interpolation step is one of the slowest steps in trajectory model. Thus, it is advantageous to supply data that is cropped around the storm of interest to reduce model runtime. 

### Missing Variables
Depending on the microphysics scheme used in the storm simulation, it is possible that the specified hydrometeor species do not all exist separately. If only one cloud ice category exists in the storm simulation, a user may simply provide an array of all zeros for missing ice mixing ratio array. 

If pressure or dry air density is missing, the user should calculate the field and provide it.

If rain drop number concentration is missing, the user may combine the cloud water and rain water mixing ratio fields into the cloud water mixing ratio field only. Then arrays of zeros may be passed in for rain water mixing ratio and number concentration. Similary, if rain drops do not exist as a separate liquid hydrometeor species, arrays of zeros may be passed in for rain water mixing ratio and number concentration. This solution is not recommended, but may be done if necessary. In this case, all storm water will be treated as cloud water, with collection efficiencies following that expected for cloud droplets and with assumed fall speed relative to the air stream = 0 m/s for all cloud water.

All other variable fields are required and the model will not function correctly if they are not provided.

### Surface inflow air density

The input parameter *sfcdens* defines the air density at the ground within the inflow air. By default, the dry air density at the ($z_i=0$,$y_i=0$,$x_i=-1$) location in the initial storm dataset file is used as the dry air density of the surface inflow. *sfcdens* may optionally be provided by the user. If the density value at this corner is not appropriate for defining the inflow surface dry air density, an appropriate value may be provided by the user in the input *sfcdens*, which should have units of \[kg m$^{-3}$\] and should be within the range (0.9, 1.3) kg m$^{-3}$.

## Trajectory Model Output

The trajectory model allows for a variety of output options that the user may specify, including the trajectories to output, the output variables, and the output frequency.

By default, any trajectory model output is returned to the user by the **main** function as an xarray dataset. However, an output filepath may be provided by the input *output_path*. If *output_path* is provided, the output that is returned by the **main** function will also be saved to *output_path*'s location as a netcdf dataset. If no *output_path* is provided, no data is saved.

### Which Trajectories to Output

The parameter *output_which* specifies which trajectories should be saved and/or returned by the trajectory model. By default, hail trajectories are only returned or saved if their trajectories terminate without melting completely. This selection is done to cut down on the total amount of storage used by trajectories. A typical trajectory model run may result in 90% or more of trajectories fully melting before hitting the ground. Because typical trajectory model applications do not care about the trajectories of melted hailstones, those trajectories are neglected for output. However, some applications may want to output the trajectories of melted hailstones, or they may want to further constrain the number of returned trajectories. The valid options for *output_which* in the trajectory model are 
- *output_which* = 'unmelted': Hailstones that terminate for any reason besides melting are returned or saved. This include hailstones that hit the ground, hailstone that exit the domain laterally, and hailstones that are still aloft at the end of the simulation.
- *output_which* = 'grounded': Only hailstones that hit the ground are returned or saved. Hailstones whose trajectories have not terminated by the end of the simulation, those that exit the domain laterally, and those that melt completely before hitting the ground are not returned.
- *output_which* = 'all': All simulated hail trajectories are returned regardless of their fate. Note that this option is not recommended if the user has limited RAM/storage and does not care about the trajectories of melted hailstones.

### Output Variables

The user may specify which variables they want to output in the record of hail trajectories. The variable options are separated into hail embryo initial conditions, hailstone final characteristics, and the full time series record of hail trajectories. Note that adding output variables can greatly increase the RAM requirements of the trajectory model, particularly when frequent output is requested for the full time series variables. If RAM becomes a limiting factor, the number of output variables should be reduced and/or the frequency of full time series output should be reduced.

#### Output Initial Conditions 

The initial condition information that will be returned or saved is specified by the *init_record_list* parameter.

The variables available for initial hail embryo condition output are:
- 'x': hail embryo initial x-location \[km\]
- 'y': hail embryo initial y-location \[km\]
- 'z': hail embryo initial z-location \[km\]
- 'D': hail embryo initial diameter \[mm\]
- 'rho_ice': hail embryo initial density \[kg m$^{-3}$\]
- 'm_core': hail embryo initial mass \[kg\]
- 'vt_hail': hail embryo initial fall speed \[m s$^{-1}$\]

The recommended *initial_record_list* = \['x', 'y', 'z', 'D', 'rho_ice'\].

#### Output Final Hail Swath Characteristics

Final hailstone fall out characteristics to be returned and saved are specified by *final_record_list*.

The variables available for final hail swath output are:
- 'x': final x position \[km\]
- 'y': final y position \[km\]
- 'z': final z position \[km\]
- 'D_max': final maximum dimension \[mm\]
- 'D_int': final intermediate dimension \[mm\]
- 'D_min': final minimum dimension \[mm\]
- 'Ax_tumble': final-timestep tumbling cross-sectional area \[m$^2$\]
- 'volume': final ellipsoidal volume \[m$^3$\]
- 'm_core': final ice core mass \[kg\]
- 'm_soaked': final soaked liquid mass \[kg\]
- 'm_shell': final liquid shell mass \[kg\]
- 'rho_ice': final ice core density \[kg m$^3$\]
- 'rho_tot': final soaked density (soaked mass and ice core) \[kg m$^3$\]
- 'wet_frac': final fraction of liquid mass ((liquid shell + soaked liquid) / (liquid shell + soaked liquid + ice core) mass) \[unitless\]
- 'vt_hail': final terminal fall speed \[m s$^{-1}$\]
- 'vrel_hail': final background wind speed relative to the hailstone's Lagrangian motion \[m s$^{-1}$\]
- 'v_rel_x': final background x-component wind speed relative to the hailstone's Lagrangian motion \[m s$^{-1}$\]
- 'v_rel_y': final background y-component wind speed relative to the hailstone's Lagrangian motion \[m s$^{-1}$\]
- 'v_rel_z': final background z-component wind speed relative to the hailstone's Lagrangian motion \[m s$^{-1}$\]
- 'v_hail_x': final hailstone ground-relative x-component velocity \[m s$^{-1}$\]
- 'v_hail_y': final hailstone ground-relative y-component velocity \[m s$^{-1}$\]
- 'v_hail_z': final hailstone ground-relative z-component velocity \[m s$^{-1}$\]
- 'time_aloft': total time from seeding to trajectory termination \[s\]
- 'final_time_step': 
- 'melted': flag for hailstones that terminated their trajectory by melting \[unitless\]
- 'grounded': flag for hailstones that terminated their trajectory by hitting the ground \[unitless\]
- 'exited': flag for hailstones that terminated their trajectory by exiting the domain laterally or through the top boundary \[unitless\]

The recommended *final_record_list* = \['x', 'y', 'z', 'D_max', 'D_int', 'D_min', 'Ax_tumble', 'm_core', 'm_soaked', 'm_shell', 'vt_hail', 'vrel_hail', 'v_hail_x', 'v_hail_y', 'v_hail_z', 'time_aloft', 'final_time_step', 'melted', 'grounded', 'exited'\]


#### Output Full Time Series Record

Outputting the full time series record of hail trajectory quantities should be done sparingly! Increasing the number of output variables will greatly increase the needed RAM for computing trajectories and will greatly increase the needed storage to save trajectory output. Note that many model variables can be calculated from a smaller subset (e.g., volume  may be calculated from rho_ice and m_core). The recommended full_record_list should be used unless other full time series variables are specifically needed.

The variables available for full time series output are:
- 'x': x position \[km\]
- 'y': y position \[km\]
- 'z': z position \[km\]
- 'D_max': maximum dimension \[mm\]
- 'D_int': intermediate dimension \[mm\]
- 'D_min': minimum dimension \[mm\]
- 'volume': ellipsoidal volume \[m$^3$\]
- 'T_core': ice core temperature \[K\]
- 'T_avg_liq': average liquid shell temperature \[K\]
- 'T_sfc': outermost surface temperautre \[K\]
- 'T_inf': background air temperature at hailstone location \[K\]
- 'm_core': mass of ice core \[kg\]
- 'm_soaked': mass of soaked liquid \[kg\]
- 'm_shell': mass in liquid shell \[kg\]
- 'rho_ice': density of the ice core \[kg m$^{-3}$\]
- 'rho_tot': soaked hailstone density (ice core + soaked mass) \[kg m$^{-3}$\]
- 'rho_growth': density of new growth at current timestep \[kg m$^{-3}$\]
- 'vt_hail': terminal fall speed \[m s$^{-1}$\]
- 'vrel_hail': background wind speed relative to the hailstone's Lagrangian motion \[m s$^{-1}$\]
- 'v_rel_x': background x-component wind speed relative to the hailstone's Lagrangian motion \[m s$^{-1}$\]
- 'v_rel_y': background y-component wind speed relative to the hailstone's Lagrangian motion \[m s$^{-1}$\]
- 'v_rel_z': background z-component wind speed relative to the hailstone's Lagrangian motion \[m s$^{-1}$\]
- 'v_hail_x': hailstone ground-relative x-component velocity \[m s$^{-1}$\]
- 'v_hail_y': hailstone ground-relative y-component velocity \[m s$^{-1}$\]
- 'v_hail_z': hailstone ground-relative z-component velocity \[m s$^{-1}$\]
- 'Ax_tumble': tumbling cross-sectional area (instantaneous) \[m$^2$\]
- 'wet_frac': fraction of total hailstone mass in liquid phase ((liquid shell + soaked liquid) / (liquid shell + soaked liquid + ice core) mass) \[unitless\]
- 'frozen_frac': fraction of new accretion that is frozen at this timestep (instantaneous) \[unitless\]
- 'wet_growth_tracker': flag for hailstones in wet growth (instantaneous) \[unitless\]
- 'dry_growth_tracker': flag for hailstones in dry growth (instantaneous) \[unitless\]
- 'melting_tracker': flag for melting hailstones (instantaneous) \[unitless\]
- 'clear_air_tracker': flag for dry hailstones in clear air (instantaneous) \[unitless\]
- 'growth_regime': hailstone's instantaneous growth regime (0: N/A, 1: melting, 2: dry growth, 3: wet growth, 4: clear air) \[unitless\]

The recommended *full_record_list* = \['x', 'y', 'z', 'D_max', 'growth_regime'\]

The time series record outputs every model integration timestep by default. However, the user may specify a longer interval if sparser time series output is desired (this may be desirable to reduce total storage and RAM requirements). The parameter *full_record_interval* defines the number of model integration timesteps between output timesteps (e.g., for a model $\Delta t = 1$ s, *full_record_interval* = 5 would result in full time series record output every 5 s).


## Model Configuration

### Integration Timestep

The model integration timestep $\Delta t$ = 1 s by default. A timestep of 1 s is appropriate for storm model grid spacings of ~100 m. However, the timestep may be manually set by the *delt* parameter in units of \[s\]. It is not recommended to set *delt* > 1.

### Integration Time Limit

The maximum total trajectory model compute time may be set through *total_t*, which defines the maximum length of time a hailstone's trajectory will be computed in \[s\]. By default, the maximum integration time is set to 3601 s = 1 hr. It is very common for a small subset of hail trajectories to remain aloft after this integration time limit. Increasing the integration time limit can allow more of the trajectories to terminate by melting or exiting the domain. However, increasing the integration time limit will also increase the length of the trajectory model time series output and the needed storage/RAM.

If all hail trajectories terminate before the integration time limit is reached, the main funnction will stop and will return the trajectories through the final timestep that any hailstone was aloft. Note that this can result in different lengths of trajectory output time series: If one trajectory simulations terminates after 2000 s and another terminates after 2400 s, their outputs will have different shapes (assuming *full_record_interval* = 1, (n_traj, 2000) vs. (n_traj, 2400)).

### Random Seed

Some parameterizations use random sampling to define hailstone characteristics (i.e., *choose_int_ratio* = 0, *choose_aspect* = 0, *choose_vt* = 0, and *choose_tumble* = 1). A random seed may be provided for reproducability in *choose_seed*.


### Choosing Parameterizations

When running the hail trajectory model, the user is able to choose parameterizations defining a hailstone's shape, fall behavior, and thermodynamics. These choices may specified when the hail trajectory model **main** function is called. The parameterization defaults will be used unless otherwise specified by the user. Note that if any parameterization inputs are given that do not correspond to the options listed below, the model will force the default choice for that input.

#### Hailstone Shape Parameterization
All model configurations assume ellipsoidal hailstones. Here, *choose_shape* defines the relationship between the simulated hailstones' maximum, intermediate, and minimum dimensions.
- *choose_shape* = 0: Spherical Hailstones (i.e., D_max = D_int = D_min).
- *choose_shape* = 1: Hailstone shapes following the maximum dimension-equivalent volume spherical diameter reltaionship of Shedd et al. (2021) with a fixed intermediate ratio.
- *choose_shape* = 2: Hailstone shapes following the maximum dimension-aspect ratio relationship of Heymsfield et al. (2018) with a fixed intermediate ratio.
- *choose_shape* = 3: Hailstone shapes following the maximum dimension-equivalent volume spherical diameter reltaionship of Shedd et al. (2021) and the maximum dimension-aspect ratio relationship of Heymsfield et al. (2018).
**Default: choose_shape = 1**

#### Aspect Ratio Percentile for Heysmfield Aspect Ratios
Hailstone aspect ratio is equal to the ratio of its minimum dimension and its maximum dimension (assuming ellipsoids). The parameter *choose_aspect* defines the aspect ratio parameterization, and is only used when *choose_shape* is in {2,3} (i.e., when using the maximum dimension-aspect ratio relationship from Heymsfield et al. 2018).
- *choose_aspect* = 0: Each hailstone's aspect ratio relationship will be chosen at random from the 0.1-0.9 quantile relationships (incrementing every 0.05).
- *choose_aspect* in range (0,1): All hailstones will use the same maximum dimension-aspect ratio relationship, specified by the nearest 0.05 quantile, with minimum quantile = 0.1 and maximum quantile = 0.9 (e.g., for *choose_aspect* = 0.76, all hailstones will use the 75th percentile maximum dimension-aspect ratio relationship; for *choose_aspect* = 0.02, all hailstones will use the 10th percentile maximum dimension-aspect ratio relationship; for *choose_aspect* = 0.5, all hailstones will use the 50th percentile maximum dimension-aspect ratio relationship).
**Default: choose_aspect = 0**

#### Intermediate Ratio 
The intermediate ratio is the ratio of a hailstone's intermediate dimension to its maximum dimension (assuming ellipsoids). The parameter *choose_int_ratio* specifies the intermediate ratio for the hailstones, and is only used when *choose_shape* is in {1,2} (i.e., when using a fixed intermediate ratio).
- *choose_int_ratio* = 0: A random intermediate ratio is chosen for each individual hailstone uniformly from the range \[0.8, 0.9\].
- *choose_int_ratio* in range (0,1\]: Intermediate ratios equal *choose_int_ratio* for all hailstones. Note that the intermediate ratio is bounded by the aspect ratio and unity within the hail model. If a *choose_int_ratio* is given that does not maintain these relationships, the intermediate ratio will change to compensate. In this case, the model will continue to work fine and should produce no unrealistic behavior. Nevertheless, a *choose_int_ratio* value between 0.8 and 0.9 is recommended if this option is used.
**Default: choose_int_ratio = 0**

#### Terminal Fall Speeds
The relationship between hailstone size and hailstone fall speed is specified by *choose_vt* and is parameterized either by maximum dimension only (*choose_vt* in range [0,1)) or by mass and shape (*choose_vt* in {1,2}).
- *choose_vt* = 0: Individual hailstone terminal fall speeds are specified from randomly selected fall speed-maximum dimension relationships from Heymsfield et al. (2020), quantiles ranging from 0.1 to 0.9 with an increment of 0.05. As hailstones' sizes increase, their terminal fall speeds follow the respectively chosen terminal fall speed relationships.
- *choose_vt* in range (0,1): Manually specified fall speed-maximum dimension relationship from Heymsfield et al. (2020). The given *choose_vt* value is rounded to the nearest 0.05 to choose the quantile relationship for all hailstones (maximum quantile = 0.9, minimum quantile = 0.1).
- *choose_vt* = 1: Spherical fall speed relationship derived from a balance between drag and gravitational forces. Maximum dimension is assumed to be the spherical diameter for this equation. Using this option for non-spherical hailstones will generally overestimate realistic hailstone fall speeds.
- *choose_vt* = 2: Non-spherical fall speed relationship derived from a balance between drag and gravitational forces.
**Default choose_vt = 2**

#### Drag coeff
Hailstone drag coefficient and Reynolds Numbers are calculated together according to *choose_drag*. For *choose_drag* in {0,1}, the input parameter *Cdrag* is used as the constant value for drag coefficient. The recommended value for *Cdrag* is 0.58, which is the function's default.
- *choose_drag* = 0: Constant drag coefficient for all hailstones of all sizes equal to the specified *Cdrag*. Reynolds number is manually calculated assuming length scale = $D_\mathrm{max}$ and velocity scale = $v_T$. Note that these drag coefficients greatly underestimate drag coefficients for small hailstones.
- *choose_drag* = 1: Constant drag coefficient for all hailstones of all sizes equal to the specified *Cdrag*. Reynolds number is calculated via the Best Number. Note that these drag coefficients greatly underestimate drag coefficients for small hailstones.
- *choose_drag* = 2: Drag coefficient calculated from the Heymsfield et al. (2020) empirical maximum dimension-Best Number power-law relationship. Reynolds number calculated via the Best Number. Note that these drag coefficients can generally overestimate drag coefficients for small hailstones and underestimate drag coefficients for large hailstones.
- *choose_drag* = 3: Drag coefficient solved iteratively with the Reynolds Number following Clift and Gauvin (1971). Note that these drag coefficients are generally smaller than realistic hailstone drag coefficients for all hailstone sizes.
- *choose_drag* = 4: Drag coefficient solved iteratively with the Reynolds Number following Bagheri and Bardonna (2016). Note that these drag coefficients are generally far greater than realistic hailstone drag coefficients for all hail sizes.
- *choose_drag* = 5: Drag coefficient solved with the parameterization of Theis et al. (2026) for a variety of natural hailstones depending on their sphericity and maximum dimension. Reynolds Number is solved via the Best Number. This is the recommended drag and Reynolds Number parameterization.
**Default: choose_drag = 5**

#### Hailstone lobes 
Hailstones in the model are assumed to be smooth ellipsoids unless *choose_lobes* = 1 is specified. Note that hailstone lobiness has been included following Lin et al. (2024); however, updates to the model from Spychalla et al. (in prep) have not comprehensively changed and/or added lobe adjustments to any changed hail growth physics.  Users should beware that lobeiness has not been carefully checked for correctness in the newest model version. For this reason we recommend running the trajectory model without lobes.
- *choose_lobes* = 0: No parameterized hailstone lobes.
- *choose_lobes* = 1: A lobes adjustment parameter is used to modify the total heat transfer by modifying hailstone surface area and to modify the drag coefficient for choose_drag parameterizations in \[1,2,3\] following the expected impact of lobes to hailstone heat transfer and fall speed (Lin et al. 2024).
**Default: choose_lobes = 0**

#### Hailstone tumble mode
Hailstone tumbling may be turned on or off by *choose_tumble*. From Lin et al. (2024), running the hail model with non-spherical hailstone shapes without tumbling is likely to produce erroneously large hail sizes. Tumbling should be turned on unless spherical hailstones are manually specified.
- *choose_tumble* = 1: Explicit hailstone tumbling from Lin et al. (2024). A randomly chosen adjustment to hailstone fall dimensions are applied to each hailstone at each time.
- *choose_tumble* in range (0,1): Fixed adjustment from hailstone tumbling. The given *choose_tumble* is used as a fixed muliplier on hailstone cross-sectional area. Note that *choose_tumble*$\rightarrow$0 are quite unrealistic and should not be used.
- *choose_tumble* = 0: No hailstone tumbling. Hailstones are assumed to fall with their largest cross-sectional area normal to their fall direction. Note that this choice will generally overestimate hailstone collection of liquid water and is not recommended.
**Note that *choose_tumble* = \{0,1\} have been flipped since their original implementation!**
**Default: choose_tumble = 1**
  
If *choose_tumble* = 1 (default tumbling mode for explicit hailstone tumbling), the tumbling lookup table must be provided. This tumbling file tumblerand.mat is provided here with the model source code. This tumbling file should be discoverable by the hail model with the path to the tumbling file defined by the input parameter *tumbling_file* to the **main** function. Note that the function **tumbling_parameterization** is hard coded to specifically sample the provided tumblerand.mat file. **Swapping out this tumbling file for another requires a complete rewritting of the tumbling_parameterization function.**

#### Hailstone inertial adjustment
Hailstone inertial adjustment may be turned on or off by *choose_inertia*. Note that if no spatial interpolation is used, computing hail trajectories with inertia is not recommended. As long as linear interpolation is used (default), inertial adjustment is recommended.
- *choose_inertia* = 1: Hailstone motions are calculated with physical inertial adjustment to changing background winds following Kumjian et al. (2025).
- *choose_inertia* = 0: Hailstone motions assume instantaneous adjustment to their background winds (no inertia).
**Default: choose_inertia = 1**

#### Heat transfer coefficient
The parameterization of $\chi$, which is used by Macklin (1963) and Rasmussen and Heymsfield (1987) to modify heat transfer processes corresponding to a hailstone's aspect ratio may be chosen by *choose_chi*. 
- *choose_chi* = 0: Axis-ratio dependent heat transfer coefficient adjustment $\chi$ from Macklin (1963).
- *choose_chi* in range (0,1): Manual specification of heat transfer coefficient adjustment $\chi$ for Reynolds Numbers less than 20000 (becomes a linear function of *choose_chi* for Reynolds Numbers greater or equal to 20000).
**Default: choose_chi = 0**

#### Liquid shell temperature
The liquid shell temperature assumes an equilibrium profile for thermal conduction unless otherwise specified. When heat transfer processes occur, sometimes a temperature for the liquid shell must be assumed. This parameter defines what temperature is used for the various heat transfer processes.
- *choose_shell_T* = 0: All liquid shell mass is forced to have temperature $T_0$ = 273.15 K in all growth/melting regimes. This option generally produces too much wet-growth freezing and too much melting and is not recommended.
- *choose_shell_T* = 1: All heat transfer is calculated using the average temperature of the liquid shell.
- *choose_shell_T* = 2: Heat transfer processes are calculated using the nearest boundary temperature for the liquid shell. Melting, freezing, and soaking are calculated with the internal boundary temperature $T = T_0$. Accretion, conduction, and diffusion are calculated with the surface boundary temperature $T = T_\mathrm{sfc}$.
**Default: choose_shell_T = 1**

#### Depreciated Parameterizations
- *choose_D*: The parameter *choose_D* used to be available to specify the Heymsfield et al. (2018) relationship used to specify hail embryo sizes. During the 2026 rewrite, hail embryo size was modified to be a model input.


## Example Trajectory Model Script

The following script will compute hail trajectories in a storm dataset stored at 'path_to_storm_simulation/storm_data_*.nc' and will save output to 'path_to_save/trajectories.nc'. An example to compute trajectories in the first static storm time is shown first followed by the code to compute trajectories in the evolving storm.

    ```
    import trajectory_model as model
    
    
    # For files stored at 'path_to_storm_simulation/storm_data_{i}.nc', starting with i=1
    path_to_storm = 'path_to_storm_simulation/'
    sim_time = 1
    filetype='nc'
    
    # Save trajectory model output to 'path_to_save/trajectories.nc'
    output_path = 'path_to_save/trajectories.nc'
    
    # Sample 10000 trajectories
    n_traj = 10000
    
    # Sample hail embryo locations
    init_z, init_y, init_x = model.sample_locations_box(n_traj=n_traj, xmin=-10, xmax=10, ymin=-10, ymax=10, zmin=4, zmax=10,)
    
    # Sample hail embryo sizes
    init_diam  = model.sample_sizes_gamma(n_traj=n_traj, mean_size=7.5, D0_min=2, D0_max=10)

    # Sample hail embryo densities
    init_rho   = model.sample_densities_uniform(n_traj=n_traj, rho_min=300, rho_max=917)
    
    # To run the hail model in the first static snapshot
    output_ds = model.main(init_x        = init_x       , 
                           init_y        = init_y       , 
                           init_z        = init_z       , 
                           init_diam     = init_diam    , 
                           init_rho      = init_rho     , 
                           path_to_storm = path_to_storm, 
                           sim_time      = sim_time     , 
                           filetype      = filetype     , 
                           output_path   = output_path  , )

                           
    # To run the hail model in an evolving storm with storm output every 5 min
    output_ds = model.main(init_x        = init_x       , 
                           init_y        = init_y       , 
                           init_z        = init_z       , 
                           init_diam     = init_diam    , 
                           init_rho      = init_rho     , 
                           path_to_storm = path_to_storm, 
                           sim_time      = sim_time     , 
                           filetype      = filetype     , 
                           output_path   = output_path  , 
                           evolving      = True         , 
                           storm_delt    = 300          , ) # 300 s = 5 min
    
    ```
