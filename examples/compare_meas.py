# Show how to compare measurement with simulation

from pathlib import Path
from opengate import g4_units, Simulation
from pandas import read_csv
from tools.CCevents import local2global, pixelClusters2CCevents, gHits2CCevents
from tools.allpix import gHits2allpix2pixelHits
from tools.utils import charge_speed_mm_ns
from tools.pixelHits import ENERGY_keV, remove_edge_pixels
from tools.pixelClusters_custom import pixelHits2pixelClusters
from tools.reconstruction import valid_psource, reconstruct
from tools.utils_opengate import setup_pixels, setup_hits, set_fluorescence
from tools.utils_plot import compare_pixelClusters, plot_energies, compare_recos, \
    plot_reco

path = '/media/billoud/029A94FF9A94F101/2nd_DRIVE/TIMEPIX3/Advapix_TPX3_2mm_CdTe/'
path_meas = path + 'In_90deg_pixelHits.csv'
um, mm, keV, Bq, s = g4_units.um, g4_units.mm, g4_units.keV, g4_units.Bq, g4_units.s

if __name__ == "__main__":
    ## ============================
    ## ==  GATE SIMULATION       ==
    ## ============================
    sim, sim.output_dir = Simulation(), Path("output")
    sim.random_engine, sim.random_seed = "MersenneTwister", 1
    sim.world.size = [300 * mm, 300 * mm, 300 * mm]
    sim.world.material = "G4_AIR"
    sensor = sim.add_volume("Box", "sensor")
    sensor.material = 'G4_CADMIUM_TELLURIDE'
    npix, pitch, t = 256, 55 * um, 2 * mm
    sensor.size = [npix * pitch, npix * pitch, t]
    setup_pixels(sim, npix, sensor, pitch, t)
    hits = setup_hits(sim, sensor_name=sensor.name)
    source = sim.add_source("GenericSource", "source")
    source.particle = "gamma"
    source.position.translation = [0 * mm, 0 * mm, 0 * mm]
    source.activity = 4.5e6 * Bq

    # TUNE
    # sim.physics_manager.physics_list_name = 'G4EmLivermorePhysics'  # Doppler effect
    # set_fluorescence(sim)
    sensor.translation = [-128 * mm, 0 * mm, 0 * mm]  # [0, -11.33, 128.5]
    source.energy.mono = 245 * keV
    sim.run_timing_intervals = [[0, 1 * s]]
    hits.output_filename = f"gateHits_{sim.physics_manager.physics_list_name}_{int(sim.run_timing_intervals[0][1] / s)}s.root"
    sim.run()

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    bias = -500  # in V
    mobility_e = 1000  # main charge carrier mobility in cm^2/Vs
    spd = charge_speed_mm_ns(mobility_cm2_Vs=1000, bias_V=bias, thick_mm=t)

    # ########################## HITS  ##################################
    hit_allp = gHits2allpix2pixelHits(sim,
                                      npix=npix,
                                      config='precise',
                                      skip_hitless_events=True,
                                      bias_V=bias,
                                      mobility_electron_cm2_Vs=mobility_e,
                                      mobility_hole_cm2_Vs=500,
                                      threshold_smearing=0,
                                      electronics_noise=0,
                                      charge_per_step=1000,
                                      )
    # hits_meas = pixet2pixelHit(path_meas, path + 'CALIBRATION')
    # hits_meas.to_csv(path_meas+'_pixelHits.csv', index=False)
    hit_meas = read_csv(path_meas, nrows=10 * len(hit_allp))

    # TOA CUTS
    hit_allp = hit_allp[hit_allp['ToA (ns)'] <= 2.1e10]  # outliers in Gate global time

    # ENERGY CUTS
    emin, emax = 10, source.energy.mono * 1000
    hit_meas = hit_meas[(hit_meas[ENERGY_keV] > emin) & (hit_meas[ENERGY_keV] < emax)]
    hit_allp = hit_allp[(hit_allp[ENERGY_keV] > emin) & (hit_allp[ENERGY_keV] < emax)]

    # BORDER PIXEL CUTS
    cut_edge = 5
    hit_meas = remove_edge_pixels(hit_meas, npix, edge_thickness=cut_edge)
    hit_allp = remove_edge_pixels(hit_allp, npix, edge_thickness=cut_edge)

    # ######################### CLUSTERS  ###############################
    clust_meas = pixelHits2pixelClusters(hit_meas, window_ns=100, npix=256)
    clust_allp = pixelHits2pixelClusters(hit_allp, window_ns=100, npix=256)

    # ENERGY CUTS
    eClstrMax = 260
    clust_meas = clust_meas[clust_meas[ENERGY_keV] < eClstrMax]
    clust_allp = clust_allp[clust_allp[ENERGY_keV] < eClstrMax]

    # compare_pixelClusters(clust_meas, clust_allp, name_a='Measurement', name_b='Allpix', energy_bins=eClstrMax)

    # ######################## CCevents  ################################
    ev_meas = pixelClusters2CCevents(clust_meas, thick=t, speed=spd, twindow=100)
    ev_allp = pixelClusters2CCevents(clust_allp, thick=t, speed=spd, twindow=100)
    ev_refc = gHits2CCevents(sim.output_dir / hits.output_filename, source.energy.mono)
    # 'Cd111[245.390]_245.390' if isotope source

    plot_energies([hit_meas, hit_allp], [clust_meas, clust_allp], [ev_meas, ev_allp],
                  max_keV=260, names=['Measurement', 'Allpix'])

    # ############################## RECO  ###############################
    ev_meas = local2global(ev_meas, sensor.translation, sensor.rotation, npix, pitch, t)
    ev_allp = local2global(ev_allp, sensor.translation, sensor.rotation, npix, pitch, t)
    reco_params = {'method': 'torch',
                   'vpitch': 1,
                   'vsize': [128 * 4, 200, 200],
                   'cone_width': 0.01,
                   'energies_MeV': [source.energy.mono], 'tol_MeV': 0.05}
    vm, va, vr = (reconstruct(e, **reco_params) for e in (ev_meas, ev_allp, ev_refc))

    # # ############################ DISPLAY  ###############################
    # plot_reco(va, colormap='inferno')
    compare_recos([vm, va, vr], ['Measurement', 'Allpix', 'Reference'], 0)
