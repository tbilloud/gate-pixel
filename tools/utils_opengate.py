# Utility function when using opengate

from pathlib import Path
import SimpleITK as sitk
import opengate_core
import pandas as pd
import uproot
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
import logging
import json
import shutil
import time

from opengate.utility import g4_units
from opengate.logger import global_log
from opengate.geometry.volumes import RepeatParametrisedVolume

from tools.utils import sum_time_intervals, metric_num

um, mm, keV, MeV, deg, Bq, ns, sec = g4_units.um, g4_units.mm, g4_units.keV, g4_units.MeV, g4_units.deg, g4_units.Bq, g4_units.ns, g4_units.s


def setup_pixels(sim, npix, sensor, pitch, thickness):
    if not sim.visu:  # because 256 x 256 pixels are too heavy for visualization
        pixel = sim.add_volume("Box", "pixel")
        pixel.mother, pixel.size = sensor.name, [pitch, pitch, thickness]
        pixel.material = sensor.material
        par = RepeatParametrisedVolume(repeated_volume=pixel)
        par.linear_repeat, par.translation = [npix, npix, 1], [pitch, pitch, 0]
        sim.volume_manager.add_volume(par)
    else:
        global_log.warning("VISUALIZATION MODE. DETECTOR NOT SIMULATED.")

def setup_hits(sim, sensor_name, output_filename='gateHits.root'):
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'Hits')
    hits.attached_to = sensor_name
    hits.authorize_repeated_volumes = True
    hits.attributes = opengate_core.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames()
    hits.output_filename = output_filename
    return hits


def theta_phi(sensor, source, sim = None, extra_spread=1):
    """
    Computes the emission angles of a source so that it focuses on the sensor.
    This speeds up the simulation.
    Only works in some cases:
    - if the source and sensor are not attached to a mother volume AND they have the
      same x,y coordinates -> leave the 'sim' parameter to None.
    - if the source or sensor is attached to a mother volume:
        Set the 'sim' parameter to the simulate object (so that global coordinates can be calculated).
        => WARNING This case has not been thoroughly validated. TODO

    Returns:
        A pair of theta,phi ranges that can be given directly to the simulation:
        source.direction.theta, source.direction.phi = theta_phi(sensor,source)
    """

    if not sim:
        # If the source and sensor are not attached to a mother volume
        # Only works if source and sensor have same x,y coordinates
        sensor_position = np.array(sensor.translation)
        source_position = np.array(source.position.translation)
        sensor_size = np.max(sensor.size[0:1])
        distance = np.linalg.norm(sensor_position - source_position) - sensor.size[
            2] / 2
        phi_deg = 180 - np.degrees(np.arctan(sensor_size / (2 * distance)))
        return [phi_deg * deg, 180 * deg], [0, 360 * deg]
    else:
        # If the source or sensor is attached to a mother volume
        sensor_position = np.array(get_global_translation(sensor))
        source_position = np.array(get_global_translation(sim.volume_manager.get_volume(source.attached_to)) if sim else source.position.translation)
        sensor_size = np.max(sensor.size[0:2])  # Use max of x/y size for coverage
        direction = source_position - sensor_position  # Focus from source to sensor
        distance = np.linalg.norm(direction)
        if distance == 0:
            raise ValueError("Sensor and source positions are identical.")

        theta = np.degrees(np.arccos(direction[2] / distance))
        phi = np.degrees(np.arctan2(direction[1], direction[0]))

        half_size = sensor_size / 2
        spread = np.degrees(np.arctan(half_size / distance)) * extra_spread

        theta_range = [max(0, theta - spread) * deg, min(180, theta + spread) * deg]
        phi_range = [((phi - spread) % 360) * deg, ((phi + spread) % 360) * deg]
        return theta_range, phi_range

def set_fluorescence(sim):
    sim.physics_manager.global_production_cuts.gamma = 1 * um
    sim.physics_manager.global_production_cuts.electron = 100 * um
    sim.physics_manager.em_parameters.update(
        {'fluo': True, 'pixe': True, 'deexcitation_ignore_cut': False,
         'auger': True, 'auger_cascade': True})
    # TODO: deexcitation_ignore_cut impacts number of hits, and depends on cuts
    
def get_isotope_data(source, filter_excited_daughters= False):
    """
    Fetches isotope data from specified source.

    The function retrieves decay data from the environment variable 'G4RADIOACTIVEDATA'.
    It can additionally filter information about daughter nuclei.

    Parameters:
        source: either the opengate source object or a string in format 'ion Z A'
        filter_excited_daughters: bool (optional)
            If True, filters and processes decay information of daughter nuclei.

    Returns:
        str or pandas.DataFrame
            Returns the processed isotope data as a concatenated string or, if `filter_excited_daughters` is True,
            as a pandas DataFrame with processed decay data details.

    Note:
        - The function expects the environment variable 'G4RADIOACTIVEDATA' to be set to a valid path where 
          isotope data files are stored.
    """
    import os
    from pathlib import Path
    from opengate.sources.generic import GenericSource

    if isinstance(source, GenericSource):
        Z, A = source.ion['Z'], source.ion['A']
    else:
        parts = str(source).split()
        if parts[0] == "ion" and len(parts) == 3:
            Z, A = int(parts[1]), int(parts[2])
        else:
            raise ValueError("Source must be a GenericSource or a string 'ion Z A'")

    messages = []

    g4_data = os.environ.get('G4RADIOACTIVEDATA')
    if not g4_data:
        messages.append("Warning: G4RADIOACTIVEDATA not set, try to initialize simulation first")
        return "\n".join(messages)

    radioactive_data_path = Path(g4_data)
    isotope_file = f"z{Z}.a{A}"
    possible_files = [
        radioactive_data_path / f"{isotope_file}",
        radioactive_data_path / f"{isotope_file}.z",
        radioactive_data_path / f"{isotope_file}.txt"
    ]

    found_file = None
    for file_path in possible_files:
        if file_path.exists():
            found_file = file_path
            break

    if found_file is None:
        messages.append(f"No decay data found for Z={Z}, A={A}")
        return "\n".join(messages)

    messages.append(f"Found decay data in: {found_file}")

    try:
        with open(found_file, 'r') as f:
            content = f.read()
            messages.append(content)
    except Exception as e:
        messages.append(f"Error reading file: {e}")

    if not filter_excited_daughters:
        return "\n".join(messages)
    else:
        data = "\n".join(messages)
        lines = data.splitlines()
        result = []
        i = 0

        while i < len(lines):
            line = lines[i]
            if line.startswith('P'):
                parts = line.split()
                if len(parts) > 1 and parts[1] == '0':
                    # Keep this block
                    result.append(line.rstrip('\n'))
                    i += 1
                    # Add indented lines below
                    while i < len(lines) and (lines[i].startswith(' ') or lines[i].startswith('\t')):
                        result.append(lines[i].rstrip('\n'))
                        i += 1
                else:
                    # Skip this block
                    i += 1
                    while i < len(lines) and (lines[i].startswith(' ') or lines[i].startswith('\t')):
                        i += 1
            else:
                i += 1


        filtered = []
        for l in result:
            parts = l.split()
            if len(parts) > 1:
                try:
                    daughter_ex = float(parts[1])
                    if daughter_ex != 0:
                        filtered.append(l)
                except ValueError:
                    continue
        result = filtered

        # Make a DataFrame from the result
        df_rows = []
        for l in result:
            parts = l.split()
            if len(parts) >= 5:
                row = {
                    "Mode": parts[0],
                    "Daughter Ex": float(parts[1]),
                    "flag": parts[2],
                    "Intensity": float(parts[3]),
                    "Q": float(parts[4]),
                    "Comment": " ".join(parts[5:]) if len(parts) > 5 else ""
                }
                df_rows.append(row)
        df = pd.DataFrame(df_rows)
        return df

        return "\n".join(messages + result)


def format_activity_int(activity_bq):
    units = [("Bq", 1), ("kBq", 1_000), ("MBq", 1_000_000), ("GBq", 1_000_000_000)]
    value = activity_bq
    for i in range(len(units)-1, -1, -1):
        unit, factor = units[i]
        if value % factor == 0:
            return f"{int(value // factor)}{unit}"
    return f"{int(value)}Bq"

def format_time_int(duration_ns):
    units = [
        ("years", 3_155_760_000_000_000_0), ("days", 86_400_000_000_000),
        ("h", 3_600_000_000_000), ("min", 60_000_000_000),
        ("s", 1_000_000_000), ("ms", 1_000_000),
        ("us", 1_000), ("ns", 1)
    ]
    for unit, factor in units:
        if duration_ns >= factor and duration_ns % factor == 0:
            return f"{int(duration_ns // factor)}{unit}"
    return f"{int(duration_ns)}ns"

def format_energy_int(energy_keV):
    """
    Formats an energy value given in keV to a string with the most appropriate unit.
    If the value is an integer and less than 1000 keV, it is shown as keV with one decimal.

    Args:
        energy_keV (float or int): Energy value in keV.

    Returns:
        str: Formatted energy string (e.g., '1MeV', '500.0keV').
    """
    if isinstance(energy_keV, int) and energy_keV < 1000:
        return f"{energy_keV:.1f}keV"
    units = [("GeV", 1_000_000), ("MeV", 1_000), ("keV", 1)]
    value = energy_keV
    for unit, factor in units:
        if value >= factor and value % factor == 0:
            return f"{int(value // factor)}{unit}"
    return f"{float(value):.1f}keV"

def subdir_output(base_dir, sim, geo_name):
    """
    Generates a unique directory/sub-directory path based on some simulation parameters:
     - directory: source and geometry names
     - sub-directory: source activity and measurement time

   The generated path ensures that it is uniquely determinable, avoiding overwriting existing directories. It depends on

    Arguments:
        base_dir (str): Base directory path where the output directory will be created.
        sim: Simulation object containing source data and run timing intervals.
        geo_name (str): Name of the geometry

    Returns:
        Path: The generated unique directory path for output storage.
    """
    sources = sim.source_manager.sources.values()
    if len(sources) != 1:
        global_log.error("This function only works if there's one source only.")
        sys.exit()
    else:
        source = next(iter(sources))
        name = str(source.particle)
        src = name.replace(' ', '_') if name.startswith('ion') else f'{name}_{format_energy_int(source.energy.mono / keV)}'
        activity_str = format_activity_int(activity_bq=int(round(source.activity / Bq)))
        time_str = format_time_int(duration_ns=sim.run_timing_intervals[0][1] / g4_units.ns)
        dose = f'n{source.n}' if source.n else f'{activity_str}_{time_str}'
        base_subdir = f"{src}_{geo_name}"
        dose_dir = dose
        suffix = 1
        while os.path.exists(os.path.join(base_dir, base_subdir, dose_dir)):
            dose_dir = f"{dose}_{suffix}"
            suffix += 1
        return Path(os.path.join(base_dir,base_subdir, dose_dir))


def copy_sim_from_script(input_script):
    with open(input_script, 'r') as f:
        lines = f.readlines()

    sim_lines = []
    for line in lines:
        if 'sim.run()' in line:
            break

        sim_lines.append(line)

    ns = {'__name__': '__main__'}
    exec('from opengate.managers import Simulation', ns)
    exec('from opengate.utility import g4_units', ns)
    exec('import opengate_core', ns)
    exec(''.join(sim_lines), ns)

    return ns['sim']

def get_global_translation(volume):
    """
    Calculate the global translation of a given volume by accumulating the
    translations of the volume and its hierarchical parent volumes.

    Parameters:
    volume: Volume
        The volume object for which the global translation is being calculated.
        This object is expected to have a 'translation' attribute representing
        its local translation as well as an optional 'mother' attribute linking
        it to its parent volume.

    Returns:
    numpy.ndarray
        The global translation of the volume as a NumPy array derived by
        summing up all translations in its hierarchy.
    """
    pos = np.array(volume.translation)
    mother = getattr(volume, 'mother', None)
    while mother:
        mother_volume = volume.simulation.volume_manager.get_volume(mother)
        pos += np.array(mother_volume.translation)
        mother = getattr(mother_volume, 'mother', None)
    return pos

def set_all_volumes_to_vacuum_except_sensor(sim,sensor_name):
    """
    Set the material of all volumes in the simulation to a specified material.

    Parameters:
    """
    for vol in sim.volume_manager.volumes:
        volume = sim.volume_manager.get_volume(vol)
        if volume.name not in (sensor_name, 'pixel', 'pixel_param'):
            volume.material = "Vacuum"


def set_minipix(sim, thickness, translation = [0 * mm, 0 * mm, 0 * mm]):
    al = 1 * mm  # Aluminum layer thickness
    gap = 5 * mm  # Air gap thickness
    npix = 256
    pitch = 55 * um  # Pixel pitch
    minipix = sim.add_volume("Box", "minipix")
    minipix.translation = translation
    minipix.size = [npix * pitch * 3, npix * pitch, thickness + al + gap]
    # tip = sim.add_volume("Box", "assembly_tip")
    # tip.mother = minipix
    # tip.size = [npix * pitch, npix * pitch * 2, thickness + al + gap]
    # tip.material = "Air"
    # body = sim.add_volume("Box", "body")
    # body.mother = minipix
    # body.size = tip.size
    # body.translation = [0 * mm, 0 * mm, - (thickness + al + gap) / 2 + (thickness + al + gap) / 2]
    # body.material = "Air"
    # al_layer = sim.add_volume("Box", "al_layer")
    # al_layer.mother = tip
    # al_layer.material = "Aluminium"
    # al_layer.size = [npix * pitch, npix * pitch, al]
    # al_layer.translation = [0 * mm, 0 * mm, - (thickness + al + gap) / 2 + al / 2]
    sensor = sim.add_volume("Box", "sensor")
    sensor.mother = minipix
    sensor.material = "cadmium_telluride"  # or 'Silicon'
    sensor.size = [npix * pitch, npix * pitch, thickness]
    sensor.translation = [-10 * mm, 0 * mm, 0 * mm]
    sensor.color = [1.0, 0.0, 0.0, 1.0]
    setup_pixels(sim, npix, sensor, pitch, thickness)


def plot_DigitizerProjectionActor(sim):
    Bq, sec = g4_units.Bq, g4_units.s
    proj = sim.actor_manager.get_actor("Projection")
    file_path = sim.output_dir + '/' + proj.output_filename  # Replace with the actual path to your .mhd file
    image = sitk.ReadImage(file_path)
    im = sitk.GetArrayFromImage(image)[0, :, :].astype(int)
    plt.imshow(im, cmap='gray',vmax=2)
    source = sim.source_manager.get_source("source")
    events = f'{source.n} events' if source.n else f'{int(source.activity / Bq)}Bq {sum_time_intervals(sim.run_timing_intervals) / sec} sec'
    plt.title(f'{source.particle} {source.energy.mono} MeV \n {events}')
    cbar = plt.colorbar(label='number of pixel hits summed over all events')
    cbar.set_ticks(np.arange(np.min(im), np.max(im) + 1))
    plt.show()

def get_global_log():
    try:
        from opengate.logger import global_log
    except ImportError:
        global_log = logging.getLogger("dummy")
        global_log.addHandler(logging.NullHandler())
    global_log.setLevel(logging.DEBUG)
    return global_log


def run_multi(sim, hit_actor, stat_actor):
    """
    Run the simulation and log key metrics.
    Useful when the simulation is run in a function, not in the __name__ == '__main__'.
    """
    start_time = time.time()
    global_log.info('Simulation START')
    sim.run(start_new_process=True)
    hp = sim.output_dir / hit_actor.output_filename
    nhits = metric_num(len(uproot.open(hp)['Hits'].arrays(library='pd')))
    nev = metric_num(json.load(open(stat_actor.output_filename))['events']['value'])
    tm = round(time.time() - start_time)
    sstr = 'Simulation STOP. '
    global_log.info(f"{sstr}Time: {tm} seconds. {nev} events, {nhits} hits\n{'-' * 80}")
    shutil.copy2(os.path.abspath(sys.argv[0]), sim.output_dir)

