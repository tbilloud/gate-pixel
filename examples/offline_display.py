# Shows how to display the reconstruction in 3D with napari
# It is shown as offline processing since macOS cannot do it with Gate simulation

from tools.CCevents import pixelClusters2CCevents, local2global, gHits2CCevents
from tools.reconstruction import reconstruct
from tools.utils import charge_speed_mm_ns
from tools.utils_opengate import copy_sim_from_script
from tools.pixelClusters import pixelHits2pixelClusters
from tools.allpix import gHits2allpix2pixelHits
from tools.utils_plot import plot_reco

# INPUT
sim = copy_sim_from_script('examples/main.py')
sensor = sim.volume_manager.get_volume("sensor")
source = sim.source_manager.get_source("source")
thick = sensor.size[2]
file_ref = sim.output_dir / sim.actor_manager.get_actor("Singles").output_filename
npix, pitch = 256, 0.055
spd = charge_speed_mm_ns(mobility_cm2_Vs=1000, bias_V=-500, thick_mm=thick)

# REFERENCE
ev_ref = gHits2CCevents('output/gateHits.root', source_MeV=source.energy.mono)

# ALLPIX
hits_allpix = gHits2allpix2pixelHits(sim,
                                     entry_stop=100,
                                     npix=npix,
                                     config='precise',
                                     bias_V=-500,
                                     mobility_electron_cm2_Vs=1000,
                                     mobility_hole_cm2_Vs=500,
                                     threshold_smearing=30,
                                     electronics_noise=110,
                                     charge_per_step=10,
                                     )
clstr_allpix = pixelHits2pixelClusters(hits_allpix, npix=npix, window_ns=100)
ev_allp = pixelClusters2CCevents(clstr_allpix, thick=thick, speed=spd, twindow=100)

# RECONSTRUCTION
ev_allp = local2global(ev_allp, sensor.translation, sensor.rotation, npix, pitch, thick)
reco_params = {'vpitch': 0.1, 'vsize': [256, 256, 256], 'cone_width': 0.01,
               'energies_MeV': [source.energy.mono], 'tol_MeV': 0.01}
vol_ref = reconstruct(ev_ref, **reco_params)
vol_allp = reconstruct(ev_allp, **reco_params)

# DISPLAY
plot_reco(vol_ref, reco_params['vpitch'])
plot_reco(vol_allp, reco_params['vpitch'])
# Or both together:
# plot_reco([vol_ref,vol_allp], reco_params['vpitch'])


