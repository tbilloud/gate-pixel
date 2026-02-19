# Shows how to reconstruct a source from Compton camera measurements at multiple positions.

import opengate_core
from opengate.managers import Simulation
from tools.CCevents import gHits2CCevents
from tools.reconstruction import reconstruct
from tools.utils_plot import plot_reco
from opengate.utility import g4_units
from tools.allpix import *
from tools.utils import metric_num, translation_and_facing_rotation
import concurrent.futures

mm, keV, Bq = g4_units.mm, g4_units.keV, g4_units.Bq
pars = opengate_core.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames()


def run(d, theta, phi, material, thick_mm, source_keV, n, output, vis=False):
    translation, rotation = translation_and_facing_rotation(d, theta, phi)

    sim = Simulation()
    sim.output_dir = Path(output)
    sim.random_seed = 1
    sim.visu = vis
    sim.volume_manager.add_material_database('GateMaterials.db')

    # ===========================
    # ==   SOURCE & GEOMETRY   ==
    # ===========================
    sim.world.material = "Vacuum"  # Vacuum G4_AIR
    sim.world.size = [60 * mm, 60 * mm, 60 * mm]

    source = sim.add_source("GenericSource", "source")
    source.energy.mono = source_keV * keV
    source.position.type = "box"
    source.position.size = [10 * mm, 10 * mm, 10 * mm]

    npix, pitch, thick = 1, 55 * 256 * g4_units.um, thick_mm * mm
    sensor = sim.add_volume("Box", "sensor")
    sensor.material = material  # 'G4_CADMIUM_TELLURIDE'  # 'G4_Si'
    sensor.size = [14.08 * mm, 14.08 * mm, thick]
    sensor.translation = translation
    sensor.rotation = rotation

    ## =============================
    ## == ACTORS                  ==
    ## =============================
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'Hits')
    # hits.keep_zero_edep = True
    hits.attached_to = sensor
    hits.attributes = pars

    ## ============================
    ## == EVENTS, OUTPUT, RUN    ==
    ## ============================
    source.n = 10 if sim.visu else n
    sim.output_dir += f'/sim.physics_manager.physics_list_name'
    info = [f'{sensor.material[3:]}_{round(thick)}mm', f'{source_keV}keV', str(d), str(theta), str(phi), metric_num(n)]
    fname = '_'.join(info) + '.root'
    hits.output_filename = hits.name + '_' + fname
    sim.run(start_new_process=True)
    if sim.visu: return

    # # ============================
    # # == COMPTON EVENTS & RECO  ==
    # # ============================
    events = gHits2CCevents(sim.output_dir / hits.output_filename, source_MeV=source.energy.mono)
    events.to_csv(sim.output_dir / f'CCevents_{fname}.csv', index=False)
    reco_pars = {'vpitch': 0.1, 'vsize': [256, 256, 256], 'cone_width': 0.01, 'tol_MeV': 0.01}
    vol = reconstruct(events, energies_MeV=[source.energy.mono], method='torch', **reco_pars)
    np.save(sim.output_dir / f'volume_{fname}.npy')

    # ============================
    # == CLEAN LOW STAT FILES   ==
    # ============================
    if source.n < 1e6:
        f = sim.output_dir / hits.output_filename
        if f.exists(): f.unlink()

    return vol


if __name__ == "__main__":
    o = "/media/billoud/029A94FF9A94F101/2nd_DRIVE/temp/multi_runs/200keV"
    d_theta_phi = [[15, 0, 0], [15, 45, 0], [15, 90, 0], [15, 135, 0], [15, 180, 0]]
    pars = {'material': 'G4_CADMIUM_TELLURIDE', 'thick_mm': 10, 'source_keV': 200, 'n': 1e7, 'output': o, 'vis': 0}


    def _run_wrapper(args):
        d, theta, phi, out = args
        return run(d, theta, phi, **pars)

    workers = min(len(d_theta_phi), max(1, (os.cpu_count() or 2) - 1))
    args_list = [(d, th, ph, o) for d, th, ph in d_theta_phi]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_wrapper, a): a for a in args_list}
        arrs = []
        for fut in concurrent.futures.as_completed(futures):
            a = futures[fut]
            try:
                res = fut.result()
                if res is None:
                    print(f"No result from run {a}")
                    continue
                arr = np.asarray(res)
                if arr.size == 0:
                    print(f"Empty array from run {a}")
                    continue
                arrs.append(arr)
                print(f"Completed run {a}")
            except Exception as e:
                print(f"Run {a} failed: {e}")
        if arrs:
            try:
                vol = np.add.reduce(arrs)
            except ValueError:
                vol = np.sum(np.stack(arrs, axis=0), axis=0)
        else:
            vol = np.array([])

    np.save(Path(o) / f"vol_cube10mm_{metric_num(pars['n'])}.npy", vol)
    plot_reco(vol, vpitch=0.1)
