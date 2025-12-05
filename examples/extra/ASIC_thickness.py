from matplotlib import pyplot as plt
from opengate.actors.digitizers import ProcessDefinedStepInVolumeAttribute
from opengate.managers import Simulation
from scipy.spatial.transform import Rotation
from tools.CCevents import gHits2CCevents_0edep, gHits2CCevents
from tools.utils import metric_num
from tools.utils_opengate import setup_pixels
from opengate.utility import g4_units
from tools.allpix import *
from tools.pixelClusters_custom import *
import pandas as pd

um, mm, keV, Bq, s = g4_units.um, g4_units.mm, g4_units.keV, g4_units.Bq, g4_units.s

def run(asic_mm=1, energy_kev = 140, rot_deg = 0):
    sim, sim.output_dir = Simulation(), Path("output")
    sim.volume_manager.add_material_database('GateMaterials.db')
    sim.random_seed = 1
    sim.visu = True

    npix, pitch, thick = 256, 55 * um, 1 * mm
    sim.world.material = "Vacuum"  # "Vacuum" # "G4_AIR"
    sim.world.color = [0, 0, 1, 0]  # blue, semi-transparent
    sim.world.size = [15 * mm, 15 * mm, 65 * mm]
    d = (asic_mm * mm + thick) / 2
    mother = sim.add_volume("Box", "mother")
    mother.material = "G4_AIR"
    mother.size = [npix * pitch, npix * pitch, thick * 2 + asic_mm]
    mother.translation = [0 * mm, 0 * mm, 25 * mm]
    mother.rotation = Rotation.from_euler("x", rot_deg, degrees=True).as_matrix()
    mother.color = [0.5, 0.5, 0.5, 0.05]
    s1 = sim.add_volume("Box", "s1")
    s2 = sim.add_volume("Box", "s2")
    s1.mother = s2.mother = mother.name
    s1.material = s2.material = 'G4_CADMIUM_TELLURIDE'
    s1.size = s2.size = [npix * pitch, npix * pitch, thick]
    s1.translation = [0 * mm, 0 * mm, -d]
    s2.translation = [0 * mm, 0 * mm, d]
    asic = sim.add_volume("Box", "asic")
    asic.mother = mother.name
    asic.material = "G4_Si"
    asic.size = [npix * pitch, npix * pitch, asic_mm * mm]
    asic.translation = [0 * mm, 0 * mm, 0 * mm]  # centered in mother
    asic.color = [1, 0, 0, 0.5]
    if not sim.visu:
        setup_pixels(sim, npix, s1, pitch, thick, 'pixel1')
        setup_pixels(sim, npix, s2, pitch, thick, 'pixel2')

    sim.physics_manager.physics_list_name = 'G4EmLivermorePhysics'  # G4EmStandardPhysics_option4  G4EmLivermorePhysics

    cols = ['EventID', 'TrackID', 'TrackCreatorProcess', "ProcessDefinedStep",
            "ParticleName", "ParentID", "ParentParticleName", "TotalEnergyDeposit",
            'KineticEnergy', "PreKineticEnergy", "PostKineticEnergy", "GlobalTime",
            'PreGlobalTime', "PrePosition", "PostPosition",
            "PDGCode", "UnscatteredPrimaryFlag"]
    name = ProcessDefinedStepInVolumeAttribute(sim, "Rayl", asic.name).name
    cols.append(name)
    hits_phsp = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
    hits_phsp.attached_to = ["s1", "s2"]
    hits_phsp.authorize_repeated_volumes = True
    hits_phsp.attributes = cols
    hits_phsp.output_filename = hn = "gateHits.root"
    f = sim.add_filter("ParticleFilter", "f")
    f.particle = "gamma"
    hits_phsp.filters.append(f)

    source = sim.add_source("GenericSource", "source")
    source.particle, source.energy.mono = "gamma", energy_kev * keV
    source.direction.acceptance_angle.volumes = ["mother"]
    source.direction.acceptance_angle.intersection_flag = True
    source.direction.acceptance_angle.max_rejection = 1e8 # avoid crash
    source.position.translation = [0 * mm, 0 * mm, -25 * mm]
    source.n = 1e0

    sim.run(start_new_process=True)

    hits_phsp = uproot.open(sim.output_dir / hn)["Hits"].arrays(library="pd")

    CCevents = gHits2CCevents_0edep(sim.output_dir / hn)
    CCevents['TotalEnergy (keV)'] = CCevents['Energy (keV)_1'] + CCevents['Energy (keV)_2']
    print(len(CCevents), 'CCevents')
    mask = np.isclose(CCevents['TotalEnergy (keV)'], source.energy.mono / keV, atol=0.1)
    CCevents = CCevents.loc[mask].reset_index(drop=True)
    print(len(CCevents), 'CCevents with full energy deposited')

    s_asic = hits_phsp.groupby("EventID")["ProcessDefinedStep__Rayl__asic"].any()
    events_asic = s_asic[s_asic].index
    CCeventsIDs_rayleigh_asic = CCevents['EventID'].isin(events_asic)
    c_asic = CCeventsIDs_rayleigh_asic.sum()
    print(f"CCevents - rayleigh in asic: {c_asic} ({100 * c_asic / len(CCevents):.1f}%)")

    return len(CCevents), c_asic, sim

if __name__ == "__main__":

    angles = [90]  # degrees
    thicknesses = [1]  # mm (must be sorted or will use min for normalization)
    energies = [140, 511]  # keV
    rows = []

    # SIMULATE, STORE AND PLOT
    if 1:
        for rot_deg in angles:
            for energy in energies:
                for t in thicknesses:
                    print(f"\n--- {rot_deg} deg, asic {t} mm, {energy} keV ---")
                    n_evt, n_rayl, sim = run(asic_mm=t, energy_kev=energy, rot_deg=rot_deg)
                    rows.append({"rotation_deg": rot_deg, "asic_mm": t, "energy_kev": int(energy), "n_events": int(n_evt), "n_rayleigh": int(n_rayl)})

        df = pd.DataFrame(rows).sort_values(["rotation_deg", "energy_kev", "asic_mm"])
        df = df.reset_index(drop=True)
        n = sim.source_manager.get_source("source").n
        df.to_csv(f"output_{metric_num(n)}.csv", index=False)
    # PLOT FROM STORED
    else:
        df = pd.read_csv("output_100K.csv")
        print(df)

    # create a 2 x N_angles grid: top row = normalized CCevents, bottom row = Rayleigh fraction
    n_ang = len(angles)
    fig, axes = plt.subplots(nrows=2, ncols=n_ang, figsize=(6 * n_ang, 8), sharex=True)

    # ensure axes is 2D array even when n_angles == 1
    if n_ang == 1: axes = np.array([[axes[0]], [axes[1]]]).reshape(2, 1)

    for j, rot_deg in enumerate(angles):
        df_a = df[df.rotation_deg == rot_deg]
        cce = df_a.pivot(index="asic_mm", columns="energy_kev", values="n_events")
        ray = df_a.pivot(index="asic_mm", columns="energy_kev", values="n_rayleigh")

        ray_frac = 100 * ray.divide(cce)

        min_thick = cce.index.min()
        denom_cce = cce.loc[min_thick]
        cce_norm = cce.divide(denom_cce, axis=1) * 100

        cce_err = np.sqrt(cce).divide(denom_cce, axis=1) * 100
        ray_err = np.sqrt(ray).divide(cce) * 100

        ax_top = axes[0, j]
        ax_bot = axes[1, j]

        for e in energies:
            cce_norm[e].plot(ax=ax_top, yerr=cce_err[e], marker="o", label=f"{e} keV")
            ray_frac[e].plot(ax=ax_bot, yerr=ray_err[e], marker="o", label=f"{e} keV")

        ax_top.set_title(f"rotation = {rot_deg}°")
        ax_top.set_ylabel("n events (%)")
        ax_top.legend(title="Energy")

        ax_bot.set_xlabel("ASIC thickness (mm)")
        ax_bot.set_ylabel("Rayleigh fraction (%)")
        ax_bot.legend(title="Energy")

    plt.tight_layout()
    plt.savefig("curves_normalized_angles.png", dpi=150)
    plt.show()
