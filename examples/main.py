# Shows how to:
# - simulate Timepix3 response with Allpix2
# - group Timepix3 hits into clusters
# - identify Compton camera events
# - reconstruct the source position
# - compare with Gate's blurred singles
#
# There also is a comparison with Gate's 'singles', to justify the use of Allpix2.
# 'Singles' sum energy deposits per pixel, but do not account for charge sharing.
# Hence, energy spectra of the raw data (pixel hits) are different.
# Moreover, 'singles' do not simulate time-of-arrival, which is crucial for single-layer
# semiconductor Compton cameras. The function singles2pixelHits() gives a TOA based on
# interactions' depth, but might not be realistic in some cases (see doc).

from opengate.utility import g4_units
from examples.gate_simu import gate_simu
from tools.pixelHits import singles2pixelHits
from tools.reconstruction import valid_psource
from tools.CCevents import gHits2CCevents, local2global, pixelClusters2CCevents
from tools.utils import charge_speed_mm_ns
from tools.allpix import gHits2allpix2pixelHits
from tools.pixelClusters import pixelHits2pixelClusters
from tools.utils_plot import plot_energies

if __name__ == "__main__":
    ## ============================
    ## ==  RUN GATE              ==
    ## ============================
    sim = gate_simu()
    singles = sim.add_actor("DigitizerBlurringActor", "Singles")
    singles.authorize_repeated_volumes = True
    singles.output_filename = 'gateSingles.root'
    singles.blur_attribute = "TotalEnergyDeposit"
    singles.blur_method = "Gaussian"
    singles.blur_fwhm = 5 * g4_units.keV
    sim.run()

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    sensor = sim.volume_manager.get_volume("sensor")
    source = sim.source_manager.get_source("source")
    file_hits = sim.output_dir / sim.actor_manager.get_actor("Hits").output_filename
    file_sgl = sim.output_dir / sim.actor_manager.get_actor("Singles").output_filename
    npix, pitch, thick = 256, 55 * g4_units.um, 1 * g4_units.mm
    bias = -500  # in V
    mobility_e = 1000  # main charge carrier mobility in cm^2/Vs
    spd = charge_speed_mm_ns(mobility_cm2_Vs=mobility_e, bias_V=bias, thick_mm=thick)

    # ######## REFERENCE ##############
    evt_ref = gHits2CCevents(file_hits, source.energy.mono)

    # #########  SINGLES ##############
    hits_sgl = singles2pixelHits(file_sgl, speed=spd, thick=thick, actor='Singles')
    clstr_sgl = pixelHits2pixelClusters(hits_sgl, npix=npix, window_ns=100)
    evt_sgl = pixelClusters2CCevents(clstr_sgl, thick=thick, speed=spd, twindow=100)

    # ########### ALLPIX ##############
    hits_allp = gHits2allpix2pixelHits(sim,
                                       npix=npix,
                                       config='precise',
                                       log_level='FATAL',
                                       skip_hitless_events=False,
                                       bias_V=bias,
                                       mobility_electron_cm2_Vs=mobility_e,
                                       mobility_hole_cm2_Vs=500,
                                       threshold_smearing=30,
                                       electronics_noise=110,
                                       charge_per_step=100,
                                       )
    clstr_allp = pixelHits2pixelClusters(hits_allp, npix=npix, window_ns=100)
    evt_allp = pixelClusters2CCevents(clstr_allp, thick=thick, speed=spd, twindow=100)

    # ####### ENERGY SPECTRA ##########
    plot_energies(max_keV=160,
                  hits_list=[hits_sgl, hits_allp],
                  clusters_list=[clstr_sgl, clstr_allp],
                  CCevents_list=[evt_sgl, evt_allp],
                  names=['singles', 'allpix'],
                  alphas=[0.5, 0.5, 0.5])

    # ##### CONES INTERSECTIONS  ######
    coord_transform = dict(translation=sensor.translation, rotation=sensor.rotation,
                           npix=npix, pitch=pitch, thickness=thick)
    reco_params = {'vpitch': 0.1, 'vsize': [256, 256, 256], 'cone_width': 0.01,
                   'energies_MeV': [source.energy.mono], 'tol_MeV': 0.01}
    sp = source.position.translation
    valid_psource(evt_ref, src_pos=sp, **reco_params)
    valid_psource(local2global(evt_sgl, **coord_transform), src_pos=sp, **reco_params)
    valid_psource(local2global(evt_allp, **coord_transform), src_pos=sp, **reco_params)
