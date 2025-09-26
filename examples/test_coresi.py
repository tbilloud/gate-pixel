# Shows how to generate Compton cones from Gate hits
# - reconstruct the source
# - visualize the result in 3D

import numpy as np
from matplotlib import pyplot as plt
from opengate.utility import g4_units
from examples.gate_simu import gate_simu
from tools.reconstruction import reconstruct
from tools.CCevents import gHits2CCevents
from tools.utils_plot import compare_recos

if __name__ == "__main__":
    ## ============================
    ## == RUN GATE               ==
    ## ============================
    sim = gate_simu()
    sim.run_timing_intervals=[[0, 2000*g4_units.ms]]
    sim.run()

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    sensor = sim.volume_manager.get_volume("sensor")
    hits = sim.actor_manager.get_actor("Hits")
    source = sim.source_manager.get_source("source")

    # ###### CCevents #########
    events = gHits2CCevents(sim.output_dir / hits.output_filename, source.energy.mono)

    # ###### RECONSTRUCTION  ##########
    reco_params = {'vpitch': 0.2, 'vsize': [64, 64, 32], 'cone_width': 0.01,
                   'energies_MeV': [source.energy.mono], 'tol_MeV': 0.01,
                   'sensor_position': sensor.translation, 'sensor_rotation': sensor.rotation, 'sensor_size': sensor.size,
                   'cone_thickness':'parallel' # 'parallel' or 'angular'
                   }

    v_torch = reconstruct(events, method='torch', **reco_params)
    v_coresi = reconstruct(events, method='coresi', **reco_params)

    # ###### DISPLAY  ##########
    compare_recos([v_torch, v_coresi], names=['Torch', 'CoreSi'])