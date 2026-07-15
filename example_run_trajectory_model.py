"""

The following script will compute hail trajectories in a storm dataset stored at 
'path_to_storm_simulation/storm_data_*.nc' and will save output to 'path_to_save/trajectories.nc'. 
An example to compute trajectories in the first static storm time is shown first followed by the 
code to compute trajectories in the evolving storm.


By Lydia Spychalla
July 15, 2026

"""
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