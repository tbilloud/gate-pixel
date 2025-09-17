# Shows how to generate Compton cones from Gate hits
# - reconstruct the source
# - visualize the result in 3D

from opengate.utility import g4_units
from examples.gate_simu import gate_simu
from tools.reconstruction import valid_psource, reconstruct
from tools.pixelCoincidences import gHits2pixelCoincidences

um, mm, keV, Bq, ms = g4_units.um, g4_units.mm, g4_units.keV, g4_units.Bq, g4_units.ms

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
    reco_params = {'vpitch': 0.1, 'vsize': [256, 256, 256], 'cone_width': 0.01}

    # ######## CONES ##############
    coin = gHits2pixelCoincidences(sim.output_dir / hits.output_filename, source.energy.mono)

    # ###### RECONSTRUCTION #######
    volume = reconstruct(coin, **reco_params)

    # ###### VALIDATION  ##########
    sp = source.position.translation
    valid_psource(coin, method='numpy', src_pos=sp, **reco_params)