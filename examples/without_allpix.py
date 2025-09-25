# Shows how to generate Compton cones from Gate hits
# - reconstruct the source
# - visualize the result in 3D

from examples.gate_simu import gate_simu
from tools.reconstruction import valid_psource, reconstruct
from tools.CCevents import gHits2CCevents

if __name__ == "__main__":
    ## ============================
    ## == RUN GATE               ==
    ## ============================
    sim = gate_simu()
    sim.run()

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    sensor = sim.volume_manager.get_volume("sensor")
    hits = sim.actor_manager.get_actor("Hits")
    source = sim.source_manager.get_source("source")

    # ###### CCevents #########
    events = gHits2CCevents(sim.output_dir / hits.output_filename, source.energy.mono)

    # ###### VALIDATION  ##########
    reco_params = {'vpitch': 0.1, 'vsize': [256, 256, 256], 'cone_width': 0.01,
                   'energies_MeV': [source.energy.mono], 'tol_MeV': 0.01}
    sp = source.position.translation
    valid_psource(events, method='numpy', src_pos=sp, **reco_params)
