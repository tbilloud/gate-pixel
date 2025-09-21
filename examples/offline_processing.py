# Shows how to process Gate simulation data offline
# Based on output from the main.py -> it needs to be run first
# Plots are the same as in main.py if the Allpix/clustering parameters are the same

from tools.CCevents import pixelClusters2CCevents, local2global
from tools.pixelHits import singles2pixelHits
from tools.utils import charge_speed_mm_ns
from tools.utils_opengate import copy_sim_from_script
from tools.pixelClusters import pixelHits2pixelClusters
from tools.allpix import gHits2allpix2pixelHits
from tools.utils_plot import plot_energies

# INPUT
sim = copy_sim_from_script('examples/main.py')
sensor = sim.volume_manager.get_volume("sensor")
source = sim.source_manager.get_source("source")
thick = sensor.size[2]
file_sgl = sim.output_dir / sim.actor_manager.get_actor("Singles").output_filename
npix, pitch = 256, 0.055
spd = charge_speed_mm_ns(mobility_cm2_Vs=1000, bias_V=-500, thick_mm=thick)

# SINGLES
hits_sgl = singles2pixelHits(file_sgl, speed=spd, thick=thick, actor='Singles')
clstr_sgl = pixelHits2pixelClusters(hits_sgl, npix=npix, window_ns=100)
evt_sgl = pixelClusters2CCevents(clstr_sgl, thick=thick, speed=spd, twindow=100)

# ALLPIX
hits_allpix = gHits2allpix2pixelHits(sim,
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
evt_allp = pixelClusters2CCevents(clstr_allpix, thick=thick, speed=spd, twindow=100)

# ENERGY SPECTRA
plot_energies(max_keV=160,
              hits_list=[hits_sgl, hits_allpix],
              clusters_list=[clstr_sgl, clstr_allpix],
              CCevents_list=[evt_sgl, evt_allp],
              names=['singles', 'allpix'],
              alphas=[0.5, 0.5, 0.5])

