# Example showing how to simulate a radioactive source and its cones in case of:
# - an ideal detector (with function gHits2cones_byEvtID)
# - a Timepix detector (with Allpix and hits/cluster processing)

# WARNING:
# source.direction.acceptance_angle and theta_phi() modify activity with ion sources!

import json

from opengate import g4_units
from opengate.utility import g4_units
from tools.allpix import *
from tools.utils import metric_num, charge_speed_mm_ns
from tools.utils_opengate import get_isotope_data
from tools.reconstruction import valid_psource
from tools.utils_plot import plot_energies
from tools.pixelClusters import *
from tools.CCevents import pixelClusters2CCevents, local2global, gHits2CCevents
from examples.gate_simu import gate_simu

if __name__ == "__main__":
    ## ============================
    ## == RUN GATE               ==
    ## ============================

    sim = gate_simu()

    # Add statistics actor to get number of events
    stats = sim.add_actor('SimulationStatisticsActor', 'Stats')
    stats.output_filename = 'gateStats.txt'

    # Enable radioactive decay
    sim.physics_manager.enable_decay = True

    # Erase the monoenergetic source defined in gate_simu() and switch to ion
    source = sim.source_manager.get_source("source")
    source.particle, source.half_life = "ion 71 177", 6.65 * g4_units.day  # Lu177
    # source.particle, source.half_life = 'ion 49 111', 2.81 * g4_units.day # In111
    # source.particle, source.half_life = 'ion 42 99', 2.75 * g4_units.day # Mo99/Tc99m
    source.energy.mono = 0  # erase energy that was set in gate_simu()

    # Increase run time to get few cones
    sim.run_timing_intervals[0][1] *= 10

    sim.run()

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    hits = sim.actor_manager.get_actor("Hits")
    sensor = sim.volume_manager.get_volume("sensor")
    npix, pitch, thick = 256, 55 * g4_units.um, 1 * g4_units.mm
    ghits_file = sim.output_dir / hits.output_filename
    ghits_df = uproot.open(ghits_file)[hits.name].arrays(library='pd')
    nevents = json.load(open(sim.output_dir / 'gateStats.txt'))['events']['value']
    global_log.info(
        f"{metric_num(nevents)} events, {metric_num(len(ghits_df))} hits\n{'-' * 80}")

    # ############################### REFERENCE  #######################################

    # Find information about the isotope by
    #   1. Listing excited states of decay daughters:
    global_log.info(get_isotope_data(source, filter_excited_daughters=True))
    #   2. Listing particles emitted by the source:
    global_log.info(f"{ghits_df['ParentParticleName'].unique()}\n{'-' * 80}")
    #   3. Listing gamma energies emitted by daughter states:
    mask = (ghits_df['ParentParticleName'].to_numpy() == 'Hf177[321.316]') & \
           (ghits_df['ParticleName'].to_numpy() == 'gamma')
    vc = ghits_df[mask]['KineticEnergy'].value_counts()
    global_log.info(f"{vc[vc > 1]}\n{'-' * 80}")

    # To get reference cones, select an excited state and its decay energy, e.g.:
    # For Lu177
    p_keV = 208.366
    source_str = 'Hf177[321.316]_' + str(p_keV)
    coin_ref = gHits2CCevents(ghits_file, source_MeV=source_str)
    # For In111
    # p_keV = 245.390
    # source_str ='Cd111[245.390]_' + str(p_keV)
    # coin_ref = gHits2CCevents(ghits_file, source_MeV=source_str)

    # ################################ ALLPIX  #########################################

    # DETECTOR PARAMETERS
    bias = -500  # in V
    mobility_e = 1000  # main charge carrier mobility in cm^2/Vs

    # PIXEL HITS
    pixelHits = gHits2allpix2pixelHits(sim,
                                       npix=npix,
                                       config='precise',
                                       log_level='FATAL',
                                       bias_V=bias,
                                       mobility_electron_cm2_Vs=mobility_e,
                                       charge_per_step=1000  # speeds Allpix simulation
                                       )

    # PIXEL CLUSTERS
    pixelClusters = pixelHits2pixelClusters(pixelHits, npix=npix, window_ns=100)

    # CCevents
    spd = charge_speed_mm_ns(mobility_cm2_Vs=mobility_e, bias_V=bias, thick_mm=thick)
    coin = pixelClusters2CCevents(pixelClusters, thick=thick, speed=spd, twindow=100)
    coin = local2global(coin, sensor.translation, sensor.rotation, npix, pitch, thick)

    # ################################ COMPARISON  #####################################

    # ENERGY HISTOGRAMS
    plot_energies(max_keV=300, hits_list=[pixelHits],
                  clusters_list=[pixelClusters],
                  CCevents_list=[coin])

    # CONE VALIDATION
    reco_params = {'vpitch': 0.1, 'vsize': [256, 256, 256], 'cone_width': 0.01,
                   'energies_MeV': [p_keV / 1000], 'tol_MeV': 0.01,
                   'method': 'torch'  # remove if torch is not installed
                   }
    valid_psource(coin_ref, src_pos=source.position.translation, **reco_params)
    valid_psource(coin, src_pos=source.position.translation, **reco_params)
