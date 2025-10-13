# Shows how to generate Compton cones from Gate hits
# - reconstruct the source
# - visualize the result in 3D

from opengate.utility import g4_units
from examples.gate_simu import gate_simu
from tools.reconstruction import reconstruct
from tools.CCevents import gHits2CCevents
from tools.utils_opengate import set_fluorescence
from tools.utils_plot import compare_recos

if __name__ == "__main__":
    ## ============================
    ## == RUN GATE               ==
    ## ============================
    sim = gate_simu('G4_CADMIUM_TELLURIDE')  # 'G4_Si', 'G4_CADMIUM_TELLURIDE'
    sim.world.material = "G4_AIR"
    sim.physics_manager.physics_list_name = 'G4EmLivermorePhysics' # for Doppler
    # set_fluorescence(sim)
    sim.run_timing_intervals = [[0, 1e4 * g4_units.ms]]
    sim.run()

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    hits = sim.actor_manager.get_actor("Hits")
    sensor = sim.volume_manager.get_volume("sensor")
    source = sim.source_manager.get_source("source")

    # ###### CCevents #########
    events = gHits2CCevents(sim.output_dir / hits.output_filename, source.energy.mono)

    # ###### RECONSTRUCTION  ##########
    reco_params = {'vpitch': 0.2, 'vsize': [32, 32, 16], 'cone_width': 0.01,
                   'energies_MeV': [source.energy.mono], 'tol_MeV': 0.01,
                   'sensor_position': sensor.translation,
                   'sensor_rotation': sensor.rotation, 'sensor_size': sensor.size,
                   'cone_thickness': 'parallel'  # 'parallel' or 'angular'
                   }

    v_torch = reconstruct(events, method='torch', **reco_params)
    v_coresi = reconstruct(events, method='coresi', **reco_params)

    # ###### DISPLAY  ##########
    compare_recos([v_torch, v_coresi], names=['Torch', 'CoreSi'])
