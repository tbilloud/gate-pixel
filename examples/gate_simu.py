# Gate simulation serving as base for other examples
# Point source and a semiconductor sensor with Timepix3 dimensions

from pathlib import Path
import opengate_core
from opengate.managers import Simulation
from opengate.utility import g4_units
from tools.utils_opengate import setup_pixels

um, mm, keV, Bq, ms = g4_units.um, g4_units.mm, g4_units.keV, g4_units.Bq, g4_units.ms

def gate_simu(sensor_material="G4_CADMIUM_TELLURIDE"):
    sim, sim.output_dir = Simulation(), Path("output")
    sim.random_seed = 1
    sim.visu = False # if needed, install pyvista
    sim.verbose_level = 'DEBUG'  # DEBUG for data preview, INFO for algo timing only

    # ===========================
    # ==   GEOMETRY            ==
    # ===========================
    npix, pitch, thick = 256, 55 * um, 1 * mm
    sim.world.size = [15 * mm, 15 * mm, 30 * mm]
    sim.world.material = "G4_AIR"
    sensor = sim.add_volume("Box", "sensor")
    sensor.material = sensor_material  # 'G4_Si', 'G4_CADMIUM_TELLURIDE'
    sensor.size = [npix * pitch, npix * pitch, thick]
    sensor.translation = [0 * mm, 0 * mm, 10 * mm]
    setup_pixels(sim, npix, sensor, pitch, thick)

    ## ===========================
    ## ==  PHYSICS              ==
    ## ===========================
    # sim.physics_manager.physics_list_name = 'G4EmLivermorePhysics' # for Doppler effect
    # set_fluorescence(sim) # for fluorescence (important for CdTe/GaAs sensors)

    ## =============================
    ## == ACTORS                  ==
    ## =============================
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'Hits')
    hits.attached_to = sensor.name
    hits.authorize_repeated_volumes = True
    hits.attributes = opengate_core.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames()
    hits.output_filename = 'gateHits.root'

    ## ============================
    ## == SOURCE                 ==
    ## ============================
    source = sim.add_source("GenericSource", "source")
    source.particle = "gamma"
    source.energy.mono = 140 * keV
    source.position.translation = [0 * mm, 0 * mm, 0 * mm]
    source.activity = 100e3 * Bq

    sim.run_timing_intervals = [[0, 20 * ms]]

    return sim