# Show the different reconstruction methods
# All methods are cross-platform except cupy which is not available on macOS
# Speed depends on the platform

from pathlib import Path
from opengate.utility import g4_units

from examples.gate_simu import gate_simu
from tools.pixelCoincidences import gHits2pixelCoincidences
from tools.reconstruction import valid_psource

um, mm, keV, MeV, deg, Bq, ms, sec = g4_units.um, g4_units.mm, g4_units.keV, g4_units.MeV, g4_units.deg, g4_units.Bq, g4_units.ms, g4_units.s

if __name__ == "__main__":
    ## ============================
    ## == RUN GATE               ==
    ## ============================
    sim = gate_simu()
    sim.run()

    ##=====================================================
    ##   OFFLINE PROCESSING
    ##=====================================================
    source = sim.source_manager.get_source("source")
    hits = sim.actor_manager.get_actor("Hits")
    sensor = sim.volume_manager.get_volume("sensor")
    hits_path = Path(sim.output_dir) / hits.output_filename
    sp = source.position.translation
    pixelCoincidences = gHits2pixelCoincidences(hits_path, source.energy.mono)
    reco_params = {'vpitch': 0.2, 'vsize': [256, 256, 256], 'cone_width': 0.1}
    
    # #################### NUMPY ##########################
    valid_psource(pixelCoincidences, **reco_params, src_pos=sp, method='numpy')

    # ##################### CUPY ##########################
    valid_psource(pixelCoincidences, **reco_params, src_pos=sp, method='cupy')

    # ################### PYTORCH #########################
    valid_psource(pixelCoincidences, **reco_params, src_pos=sp, method='torch')

    # #################### CoReSi #########################
    # WARNING: with coresi, SENSOR CANNOT BE IN VOLUME !
    vsize_coresi = [reco_params['vsize'][0], reco_params['vsize'][1], 90]
    reco_params_coresi = {**reco_params, 'vsize': vsize_coresi}
    valid_psource(pixelCoincidences, **reco_params_coresi, src_pos=sp, method='coresi',
                  sensor_size=sensor.size,
                  sensor_position=sensor.translation,
                  sensor_rotation=sensor.rotation,
                  energies_MeV=[source.energy.mono],
                  tol_MeV=0.01,
                  log_scale=True  # helps see cone profiles in the background
                  )

    # ################### CUSTOM ##########################
    # To add a custom reconstruction function:
    # 1) give it the same input parameters, plus, if needed, extras as kwargs (as coresi)
    # 2) put it in a separate python file in the tools subdirectory
    # 3) update the reconstruct() function in tools/reconstruction.py as shown with the 3 lines starting with `elif method == "custom"`
    # valid_psource(pixelCoincidences, **reco_params, src_pos=sp, method='custom')
