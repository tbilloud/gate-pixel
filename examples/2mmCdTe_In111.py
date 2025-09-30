from pathlib import Path
from opengate import Simulation
from opengate.utility import g4_units
from tools.CCevents import gHits2CCevents, local2global, pixelClusters2CCevents
from tools.utils import charge_speed_mm_ns
from tools.allpix import gHits2allpix2pixelHits
from tools.pixelClusters import pixelHits2pixelClusters
from tools.utils_opengate import setup_pixels, setup_hits
from tools.utils_plot import plot_energies
from tools.utils_plot import compare_recos
from tools.reconstruction import reconstruct

um, mm, keV, Bq, ms = g4_units.um, g4_units.mm, g4_units.keV, g4_units.Bq, g4_units.ms

if __name__ == "__main__":
    ## ============================
    ## ==  GATE SIMULATION       ==
    ## ============================
    sim, sim.output_dir = Simulation(), Path("output")
    sim.random_engine, sim.random_seed = "MersenneTwister", 1
    sim.world.size = [15 * mm, 15 * mm, 300 * mm]
    sim.world.material = "G4_AIR"
    sensor = sim.add_volume("Box", "sensor")
    sensor.material = 'G4_CADMIUM_TELLURIDE'  # 'G4_Si', 'G4_CADMIUM_TELLURIDE'
    npix, pitch, thick = 256, 55 * um, 2 * mm
    sensor.size = [npix * pitch, npix * pitch, thick]
    setup_pixels(sim, npix, sensor, pitch, thick)
    hits = setup_hits(sim, sensor_name=sensor.name)
    source = sim.add_source("GenericSource", "source")
    source.particle = "gamma"
    source.position.translation = [0 * mm, 0 * mm, 0 * mm]
    source.activity = 1e6 * Bq

    # TUNE
    # sim.physics_manager.physics_list_name = 'G4EmLivermorePhysics' # for Doppler effect
    # set_fluorescence(sim) # for fluorescence (important for CdTe/GaAs sensors)
    sensor.translation = [0 * mm, 0 * mm, 10 * mm]
    source.energy.mono = 245 * keV
    sim.run_timing_intervals = [[0, 0.1 * g4_units.s]]
    sim.run()

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    bias = -500  # in V
    mobility_e = 1000  # main charge carrier mobility in cm^2/Vs
    spd = charge_speed_mm_ns(mobility_cm2_Vs=mobility_e, bias_V=bias, thick_mm=thick)

    # ######## REFERENCE ##############
    evt_ref = gHits2CCevents(sim.output_dir / hits.output_filename, source.energy.mono)

    # ########### ALLPIX ##############
    hits_allp = gHits2allpix2pixelHits(sim,
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
    clstr_allp = pixelHits2pixelClusters(hits_allp, npix=npix, window_ns=100)
    evt_allp = pixelClusters2CCevents(clstr_allp, thick=thick, speed=spd, twindow=100)
    evt_allp = local2global(evt_allp, sensor.translation, sensor.rotation, npix, pitch, thick)

    # ####### ENERGY SPECTRA ##########
    plot_energies(max_keV=160, hits_list=[hits_allp], clusters_list=[clstr_allp],
                  CCevents_list=[evt_allp])

    # ##### CONES INTERSECTIONS  ######
    reco_params = {'method': 'torch',
                   'vpitch': 1,
                   'vsize': [60, 60, 4 * int(abs(sensor.translation[2]))],
                   'cone_width': 0.002,
                   'energies_MeV': [source.energy.mono], 'tol_MeV': 0.05}
    v_allp = reconstruct(evt_allp, **reco_params)
    v_ref = reconstruct(evt_ref, **reco_params)
    compare_recos([v_ref, v_allp], names=['Reference', 'Allpix'])
