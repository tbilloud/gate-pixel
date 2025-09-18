# Shows how to:
# - simulate Timepix3 response with Allpix2
# - cluster Timepix3 hits
# - identify Compton camera events
# - reconstruct the source position
# - compare with Gate's blurred singles

from opengate.utility import g4_units
from examples.gate_simu import gate_simu
from tools.pixelHits import singles2pixelHits
from tools.reconstruction import valid_psource
from tools.CCevents import gHits2CCevents_prototype, local2global, \
    pixelClusters2CCevents
from tools.utils import charge_speed_mm_ns
from tools.allpix import gHits2allpix2pixelHits
from tools.pixelClusters import pixelHits2pixelClusters
from tools.utils_plot import plot_energies

um, mm, keV, Bq, ms = g4_units.um, g4_units.mm, g4_units.keV, g4_units.Bq, g4_units.ms

if __name__ == "__main__":
    ## ============================
    ## ==  RUN GATE              ==
    ## ============================
    sim = gate_simu()
    sim.random_engine, sim.random_seed = "MersenneTwister", 1
    singles = sim.add_actor("DigitizerReadoutActor", "Singles")
    singles.authorize_repeated_volumes = True
    singles.discretize_volume = 'sensor'
    singles.input_digi_collection = "Hits"
    singles.policy = "EnergyWeightedCentroidPosition"
    singles.output_filename = 'gateSingles.root'
    singles_b = sim.add_actor("DigitizerBlurringActor", "Singles_b")
    singles_b.authorize_repeated_volumes = True
    singles_b.input_digi_collection = singles.name
    singles_b.output_filename = 'gateSingles_b.root'
    singles_b.blur_attribute = "TotalEnergyDeposit"
    singles_b.blur_method = "Gaussian"
    singles_b.blur_fwhm = 5 * keV
    sim.run()

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    sensor = sim.volume_manager.get_volume("sensor")
    hits = sim.actor_manager.get_actor("Hits")
    singles_b = sim.actor_manager.get_actor("Singles_b")
    source = sim.source_manager.get_source("source")
    npix, pitch, thick = 256, 55 * um, 1 * mm
    reco_params = {'vpitch': 0.1, 'vsize': [256, 256, 256], 'cone_width': 0.01,
                   'energies_MeV': [source.energy.mono], 'tol_MeV': 0.01}
    bias = -500  # in V
    mobility_e = 1000  # main charge carrier mobility in cm^2/Vs
    spd = charge_speed_mm_ns(mobility_cm2_Vs=mobility_e, bias_V=bias, thick_mm=thick)
    coord_transform = dict(translation=sensor.translation, rotation=sensor.rotation,
                           npix=npix, pitch=pitch, thickness=thick)

    # ######## REFERENCE ##############
    coinc_ref = gHits2CCevents_prototype(sim.output_dir / hits.output_filename,
                                                  source.energy.mono)

    # #########  SINGLES ##############
    hits_single = singles2pixelHits(sim.output_dir / singles_b.output_filename,
                                    charge_speed_mm_ns=spd,
                                    thickness_mm=thick,
                                    actor_name='Singles_b')
    clstr_single = pixelHits2pixelClusters(hits_single, npix=npix, window_ns=100)
    coin_single = pixelClusters2CCevents(clstr_single,
                                                  thickness_mm=thick,
                                                  charge_speed_mm_ns=spd,
                                                  )
    coin_single = local2global(coin_single, **coord_transform)

    # ########### ALLPIX ##############
    hits_allpix = gHits2allpix2pixelHits(sim,
                                         npix=npix,
                                         config='precise',
                                         log_level='FATAL',
                                         skip_hitless_events=False,
                                         bias_V=bias,
                                         mobility_electron_cm2_Vs=mobility_e,
                                         mobility_hole_cm2_Vs=500,
                                         threshold_smearing=30,
                                         electronics_noise=110,
                                         charge_per_step=10,  # more speeds Allpix up
                                         )
    clstr_allpix = pixelHits2pixelClusters(hits_allpix, npix=npix, window_ns=100)
    coin_allpix = pixelClusters2CCevents(clstr_allpix,
                                                  thickness_mm=thick,
                                                  charge_speed_mm_ns=spd,
                                                  )
    coin_allpix = local2global(coin_allpix, **coord_transform)

    # ##### SINGLES VS ALLPIX #####
    plot_energies(max_keV=160,
                  hits_list=[hits_single, hits_allpix],
                  clusters_list=[clstr_single, clstr_allpix],
                  CCevents_list=[coin_single, coin_allpix],
                  names=['singles', 'allpix'],
                  alphas=[0.5, 0.5, 0.5])

    # ###### VALIDATION  ##########
    vs, vp = (256, 256, 256), 0.1
    sp = source.position.translation
    valid_psource(coinc_ref, src_pos=sp, **reco_params)
    valid_psource(coin_single, src_pos=sp, **reco_params)
    valid_psource(coin_allpix, src_pos=sp, **reco_params)
