# Shows how to process Gate simulation data offline
# Based on output from the example.py -> it needs to be run first
# Plots are the same as in example.py if the Allpix/clustering parameters are the same

from tools.pixelCoincidences import pixelClusters2pixelCoincidences, local2global
from tools.pixelHits import singles2pixelHits
from tools.utils import charge_speed_mm_ns
from tools.utils_opengate import copy_sim_from_script
from tools.pixelClusters import pixelHits2pixelClusters
from tools.allpix import gHits2allpix2pixelHits
from tools.utils_plot import plot_energies

# INPUT
sim = copy_sim_from_script('example.py')
sensor = sim.volume_manager.get_volume("sensor")
source = sim.source_manager.get_source("source")
npix, pitch = 256, 0.055

coord_transform = dict(translation=sensor.translation, rotation=sensor.rotation,
                       npix=npix, pitch=pitch, thickness=sensor.size[2])

# SINGLES
spd = charge_speed_mm_ns(mobility_cm2_Vs=1000, bias_V=-500, thick_mm=sensor.size[2])
hits_single = singles2pixelHits('output/gateSingles_b.root',
                                charge_speed_mm_ns=spd,
                                thickness_mm=sensor.size[2],
                                actor_name='Singles_b')
clstr_single = pixelHits2pixelClusters(hits_single, npix=npix, window_ns=100)
coin_single = pixelClusters2pixelCoincidences(clstr_single,
                                              source_MeV=source.energy.mono,
                                              thickness_mm=sensor.size[2],
                                              charge_speed_mm_ns=spd,
                                              )
coin_single = local2global(coin_single, **coord_transform)

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
coin_allpix = pixelClusters2pixelCoincidences(clstr_allpix,
                                              source_MeV=source.energy.mono,
                                              thickness_mm=sensor.size[2],
                                              charge_speed_mm_ns=spd,
                                              )
coin_allpix = local2global(coin_allpix, **coord_transform)

# PLOT
plot_energies(max_keV=160,
              hits_list=[hits_single, hits_allpix],
              clusters_list=[clstr_single, clstr_allpix],
              coincidences_list=[coin_single, coin_allpix],
              names=['singles', 'allpix'],
              alphas=[0.5, 0.5, 0.5])

