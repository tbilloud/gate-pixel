# Shows how to reconstruct a source from Compton camera measurements at multiple positions.

import opengate_core
from opengate.managers import Simulation
from tools.CCevents import gHits2CCevents
from tools.reconstruction import reconstruct
from tools.utils_plot import plot_reco
from opengate.utility import g4_units
from tools.allpix import *
from tools.utils import metric_num
import concurrent.futures

mm, keV, Bq = g4_units.mm, g4_units.keV, g4_units.Bq
pars = opengate_core.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames()


def run_sim(output, material='G4_CADMIUM_TELLURIDE', thick_mm=1, source_keV=200, n=1e4, translation=[0, 0, 0], vis=False):
    sim = Simulation()
    sim.output_dir = Path(output)
    sim.random_seed = 1
    sim.visu = vis
    sim.volume_manager.add_material_database('GateMaterials.db')

    # ===========================
    # ==   GEOMETRY            ==
    # ===========================
    sim.world.material = "Vacuum"  # Vacuum G4_AIR
    sim.world.size = [60 * mm, 60 * mm, 40 * mm]
    npix, pitch, thick = 1, 55 * 256 * g4_units.um, thick_mm * mm
    sensor = sim.add_volume("Box", "sensor")
    sensor.material = material  # 'G4_CADMIUM_TELLURIDE'  # 'G4_Si'
    sensor.size = [14.08 * mm, 14.08 * mm, thick]
    sensor.translation = translation

    ## =============================
    ## == ACTORS                  ==
    ## =============================
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'Hits')
    # hits.keep_zero_edep = True
    hits.attached_to = sensor
    hits.attributes = pars

    ## ============================
    ## == SOURCE                 ==
    ## ============================
    source = sim.add_source("GenericSource", "source")
    source.energy.mono = source_keV * keV
    source.position.type = "box"
    source.position.size = [10 * mm, 10 * mm, 4 * mm]
    source.n = 10 if sim.visu else n

    ## ============================
    ## == FILE NAMES + RUN       ==
    ## ============================
    sim.output_dir = sim.output_dir / f'{source_keV}keV' / f'{sim.physics_manager.physics_list_name}'
    info = [f'{sensor.material[3:]}_{round(thick)}mm', f'{source_keV}keV', f'{translation}', metric_num(n)]
    fname = '_'.join(info) + '.root'
    hits.output_filename = hits.name + '_' + fname
    sim.run(start_new_process=True)
    if sim.visu: return

    # # ============================
    # # == COMPTON RECONSTRUCTION ==
    # # ============================
    events = gHits2CCevents(sim.output_dir / hits.output_filename, source_MeV=source.energy.mono)
    events.to_csv(sim.output_dir / f'CCevents_{fname}.csv', index=False)
    reco_pars = {'vpitch': 0.1, 'vsize': [256, 256, 256], 'cone_width': 0.01, 'tol_MeV': 0.01}
    vol = reconstruct(events, energies_MeV=[source.energy.mono], method='torch', **reco_pars)
    np.save(sim.output_dir / f'volume_{fname}.npy', vol)

    # ============================
    # == CLEAN LOW STAT FILES   ==
    # ============================
    if source.n < 1e6:
        f = sim.output_dir / hits.output_filename
        if f.exists(): f.unlink()

    return vol


if __name__ == "__main__":
    params = {
        'material': 'G4_CADMIUM_TELLURIDE',
        'thick_mm': 10,
        'source_keV': 200,
        'n': 1e5,
        'output': "/media/billoud/029A94FF9A94F101/2nd_DRIVE/temp/multi_runs",
        'vis': False
    }

    z = 10
    translations = [[0, 0, z], [10, 0, z], [-10, 0, z], [0, 10, z], [0, -10, z], [10, 10, z]]

    jobs = [{**params, 'translation': [t[0] * mm, t[1] * mm, t[2] * mm]} for t in translations]
    workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_sim, **job) for job in jobs]
        vols = [np.asarray(f.result()) for f in futures]

    vol = np.add.reduce(vols)
    out_file = Path(params['output']) / f"vol_multi_{metric_num(params['n'])}.npy"
    np.save(out_file, vol)
    plot_reco(vol, vpitch=0.1)