# Simulate the path of electrons in semi-conductor sensor typical for Timepix3 chips

import humanize
from opengate.utility import g4_units
from tools.allpix import *
import opengate_core
from opengate.managers import Simulation
from tools.utils_plot import electron_tracks_3D_viewer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

mm, keV, Bq = g4_units.mm, g4_units.keV, g4_units.Bq

if __name__ == "__main__":
    # ============================
    # == SIMULATION             ==
    # ============================
    sim = Simulation()
    sim.random_seed = 1
    sim.visu = 0
    sim.volume_manager.add_material_database('GateMaterials.db')
    sim.output_dir = Path("/media/billoud/029A94FF9A94F101/2nd_DRIVE/CC/electron_range/")

    # ===========================
    # ==   GEOMETRY            ==
    # ===========================
    sim.world.material = "G4_CADMIUM_TELLURIDE"  # Vacuum G4_AIR
    sim.world.size = [10 * mm, 10 * mm, 10 * mm]

    ## ===========================
    ## ==  PHYSICS              ==
    ## ===========================
    sim.physics_manager.global_production_cuts.electron = 1 * g4_units.um

    ## =============================
    ## == ACTORS                  ==
    ## =============================
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'Hits')
    hits.attached_to = sim.world
    hits.attributes = opengate_core.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames()

    ## ============================
    ## == SOURCE                 ==
    ## ============================
    src = sim.add_source("GenericSource", f"src")
    src.n, src.particle, src.energy.mono = 10, 'e-', 1000 * keV
    src.direction.type, src.direction.momentum = "momentum", [0, 0, 1]
    src.position.translation = [0 * mm, 0 * mm, 0 * mm]
    hits.output_filename = f"Hits_{humanize.metric(src.n)}.root"

    # ============================
    # == RUN                    ==
    # ============================
    sim.run()

    # ============================
    # == ANALYSIS               ==
    # ============================
    gateHits = uproot.open(sim.output_dir / hits.output_filename)[hits.name].arrays(library='pd')
    electron_tracks_3D_viewer(gateHits, sensor_half_size_mm=(7.04, 7.04, 0.5),
                              min_vertex_keV=100,
                              pixel_pitch_mm=0.055,
                              show_pixel_wireframe=True)
