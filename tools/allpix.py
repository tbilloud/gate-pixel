import subprocess
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import warnings
from tools.pixelHits import *
import opengate

FNAME_MODEL_CONF = 'allpix_detector_model.conf'
FNAME_MAIN_CONF = 'allpix_main.conf'
FNAME_GEO_CONF = 'allpix_geometry.conf'
FNAME_DATA_TXT = 'allpixHits'
FNAME_MODULES = 'allpix_modules.root'

def run_allpix(sim,
               binary_path='allpix/allpix-squared/install-noG4/bin/',
               output_dir='allpix/',
               log_level='FATAL',
               config='default',
               entry_stop=None,
               skip_hitless_events=False,
               bias_V=-500,
               mobility_electron_cm2_Vs=1000,
               mobility_hole_cm2_Vs=500,
               threshold=1000,
               threshold_smearing=30,
               electronics_noise=110,
               charge_per_step=10  # Allpix default for all propagation modules
               ):
    """
    Runs the Allpix2 simulation framework with the specified parameters.

    Args:
        sim: Simulation object containing Gate simulation data.
        binary_path (str): Path to the Allpix2 binaries.
        output_dir (str): Directory to store the output files.
        log_level (str): Logging level for Allpix2.
        config (str): Configuration type ('default', 'fast', or 'precise').
        entry_stop (int, optional): Number of entries to process from the input file.
        skip_hitless_events (bool): Whether to skip events without hits.
            -> Speeds up simulation considerably if detector is far from the source.
            -> But event IDs in allpix/pixel hits, etc. do not match those in gate hits.
                -> Actually, it should if you used
                   source.direction.acceptance_angle.skip_policy = 'SkipEvents'
                   If not:
                -> event IDs of allpix/pixel hits can still be match to event IDs in
                   gate hits, since the former correspond to the index of the later when
                   grouping grouping gate hits by eventID.
        bias_V (int): Bias voltage applied to the sensor.
        mobility_electron_cm2_Vs (float): Electron mobility in cm²/V/s.
        mobility_hole_cm2_Vs (float): Hole mobility in cm²/V/s.
        threshold (int): Threshold for digitizer in electrons.
        threshold_smearing (int): Smearing of the threshold in electrons.
        electronics_noise (int): Electronics noise in electrons.
        charge_per_step (int): Charge per step for propagation modules. Larger values speed up simulation.
    Raises:
        SystemExit: If required files or configurations are missing.
    """
    # ==========================
    # == INPUTS & INIT        ==
    # ==========================
    binary_path = Path(binary_path)
    output_dir = Path(output_dir)

    # Prevent two competing visualizations
    if sim.visu is True:
        sys.exit("Allpix cannot be run with Gate visualization enabled")

    # Fetch inputs
    hits_actor = sim.actor_manager.get_actor("Hits")
    gHits = Path(sim.output_dir) / hits_actor.output_filename
    gHits = os.path.join(os.getcwd(), gHits)
    gHits_df = uproot.open(gHits)['Hits'].arrays(library='pd', entry_stop=entry_stop)
    global_log.debug(f"Input {gHits}, {len(gHits_df)} gHits")
    sensor = sim.volume_manager.get_volume("sensor")
    source = sim.source_manager.get_source("source")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = R.from_matrix(sensor.rotation).as_euler('xyz', degrees=True)

    # Check that Gate geometry was adapted to Allpix
    try:
        pixel = sim.volume_manager.get_volume("pixel_param")
    except Exception:
        global_log.error(
            f"Pixels must be defined with RepeatParametrisedVolume() in Gate. Call the volume 'pixel'.")
        sys.exit()

    # Prepare the weighting potential file
    wp_fname, wp_suffix = '', '_weightingpotential.apf'
    if config == 'precise':
        wp_fname = f"pitch{int(pixel.translation[0] * 1000)}um_thick{int(sensor.size[2] * 1000)}um"
        wp_fname = os.path.join(os.getcwd(), 'allpix', wp_fname) # Allpix adds suffix
    elif config == 'fast':
        if sensor.material != 'G4_Si':
            global_log.error(
                f"The 'fast' configuration only works with Silicon sensors.")
            sys.exit()

    # ==========================
    # == PRODUCE CONFIG FILES ==
    # ==========================
    sensor_translation = \
        opengate.geometry.utility.get_transform_world_to_local(sensor)[0][0]

    geometry_conf_content = f"""[0_0]
type = "{FNAME_MODEL_CONF.replace('.conf', '')}"
position = {" ".join([f"{sensor_translation[i]}mm" for i in range(3)])}
orientation = {" ".join([f"{angles[i]}deg" for i in range(3)])}
    """

    detector_model_conf_content = f"""type = "hybrid"
geometry = "pixel"
number_of_pixels = {pixel.linear_repeat[0]} {pixel.linear_repeat[1]}
pixel_size = {pixel.translation[0]}mm {pixel.translation[1]}mm
sensor_thickness = {sensor.size[2]}mm
sensor_material = "{get_allpix_material_name(sensor.material)}"
bump_sphere_radius = 9.0um
bump_cylinder_radius = 7.0um
bump_height = 20.0um
    """

    # TODO: is Jacoboni model valid for CdTe / GaAs ?
    # TODO: use CSADigitizer in 'precise' config
    # TODO: speed up precise config (e.g. charge groups etc)
    # TODO: deal with different units for pixelHits (energy/charge/bits for TOT, time/bits for TOA)
    configurations = {
        "fast": f"""
    [ElectricFieldReader]
    model = "linear"
    bias_voltage = {bias_V}V # - to collect electrons, + to collect holes
    [ProjectionPropagation] # mobility model is Jacoboni 
    temperature = 293K
    integration_time = 1us # default 25ns might stop charge propagation
    charge_per_step = {charge_per_step}
    [PulseTransfer]
    timestep = 1.6ns # 0.01ns by default, but Timepix3 clock is 1.6ns
    [DefaultDigitizer]
    threshold = {threshold}e # 0e turns off ToA
    threshold_smearing = {threshold_smearing}e
    electronics_noise = {electronics_noise}e
    """,
        "default": f"""
    [ElectricFieldReader]
    model = "constant"
    bias_voltage = {bias_V}V # - to collect electrons, + to collect holes
    [GenericPropagation]
    integration_time = 1us  # default 25ns might stop propagation
    mobility_model = "constant"
    mobility_electron = {mobility_electron_cm2_Vs}cm*cm/V/s
    mobility_hole = {mobility_hole_cm2_Vs}cm*cm/V/s
    charge_per_step = {charge_per_step}
    [PulseTransfer]
    timestep = 1.6ns # 0.01ns by default, but Timepix3 clock is 1.6ns
    [DefaultDigitizer]
    threshold = {threshold}e # 0e turns off ToA
    threshold_smearing = {threshold_smearing}e
    electronics_noise = {electronics_noise}e
    """,
        "precise": f"""
    [ElectricFieldReader]
    model="constant"
    bias_voltage={bias_V}V
    [WeightingPotentialReader]
    model = "mesh"
    file_name = "{wp_fname + wp_suffix}"
    field_mapping = "PIXEL_FULL"
    [TransientPropagation]
    mobility_model = "constant"
    mobility_electron = {mobility_electron_cm2_Vs}cm*cm/V/s
    mobility_hole = {mobility_hole_cm2_Vs}cm*cm/V/s # holes always propagated
    integration_time = 1us # default 25ns might stop propagation
    timestep = 1.6ns
    distance = 0 # 0 means no transient signal on neighboring pixels
    charge_per_step = {charge_per_step}
    [PulseTransfer]
    [DefaultDigitizer]
    threshold = {threshold}e # 0e turns off ToA
    threshold_smearing = {threshold_smearing}e
    electronics_noise = {electronics_noise}e
    """
    }

    main_conf_content = f"""[Allpix]
log_level = {log_level}
log_format = "DEFAULT"
detectors_file = "{FNAME_GEO_CONF}"
root_file = "{FNAME_MODULES}"
number_of_events = {source.n if source.n else (gHits_df['EventID'].nunique() if skip_hitless_events else gHits_df['EventID'].max() + 1)}
model_paths = ["."]
output_directory = "."
random_seed = 1
[DepositionReader]
model = "root"
file_name = "{gHits}"
tree_name = "Hits"
detector_name_chars = 3
require_sequential_events = {'false' if skip_hitless_events else 'true'} # avoids storing events w/o hits, but looses event IDs
branch_names = ["EventID", "TotalEnergyDeposit", "GlobalTime", "Position_X", "Position_Y", "Position_Z", "HitUniqueVolumeID", "PDGCode", "TrackID", "ParentID"]
{configurations[config]}
[TextWriter]
include = "PixelHit"
file_name = "{FNAME_DATA_TXT}"
    """

    with open(output_dir / FNAME_GEO_CONF, 'w') as geometry_conf_file:
        geometry_conf_file.write(geometry_conf_content)

    with open(output_dir / FNAME_MODEL_CONF,
              'w') as detector_model_conf_file:
        detector_model_conf_file.write(detector_model_conf_content)

    with open(output_dir / FNAME_MAIN_CONF, 'w') as main_conf_file:
        main_conf_file.write(main_conf_content)

    # ===========================
    # === RUN BINARIES        ===
    # ===========================

    if config == 'precise':
        if os.path.isfile(wp_fname + wp_suffix):
            global_log.debug(f"Offline [Allpix2]: Using {wp_fname + wp_suffix}")
        else:
            global_log.warning(f"Weighting potential file not found. Generating it...")
            subprocess.run([binary_path / 'generate_potential', '--model',
                            output_dir / FNAME_MODEL_CONF, '--output',
                            wp_fname, '-v', log_level],
                           check=True)

    subprocess.run([binary_path / 'allpix', '-c', output_dir / FNAME_MAIN_CONF],
                   check=True)

    if source.n:
        txt = "Using source.n leads to data processing issues:\nAll events start with the same global time -> pixel hits will have the same ToA.\nThis pile-up will prevent proper pixel hit clustering and lead to wrong cones."
        global_log.warning(txt)


# TODO: I've seen negative ToT values in data.txt
@log_offline_process('pixelHits', input_type = 'sim')
def gHits2allpix2pixelHits(sim, npix,
                           binary_path='allpix/allpix-squared/install-noG4/bin/',
                           config='default',
                           log_level='FATAL',
                           allpix_dir='allpix/',
                           entry_stop=None,
                           skip_hitless_events=False,
                           bias_V=-500,
                           mobility_electron_cm2_Vs=1000,
                           mobility_hole_cm2_Vs=500,
                           threshold=1000,
                           threshold_smearing=30,
                           electronics_noise=110,
                           charge_per_step=10
                           ):
    run_allpix(sim, binary_path=binary_path,
               output_dir=allpix_dir,
               log_level=log_level,
               config=config,
               entry_stop=entry_stop,
               skip_hitless_events=skip_hitless_events,
               bias_V=bias_V,
               mobility_electron_cm2_Vs=mobility_electron_cm2_Vs,
               mobility_hole_cm2_Vs=mobility_hole_cm2_Vs,
               threshold=threshold,
               threshold_smearing=threshold_smearing,
               electronics_noise=electronics_noise,
               charge_per_step=charge_per_step
               )
    pixelHits = allpixTxt2pixelHit(Path(allpix_dir) / FNAME_DATA_TXT, n_pixels=npix)
    return pixelHits
    # Lines starting with PixelHit in data.txt have:
    # PixelHit X_ID, Y_ID, TOT, TOA, global_time, X_global, Y_global, Z_global


def get_allpix_material_name(gate_material_name):
    mapping = {
        "G4_Si": "Silicon",
        "G4_CADMIUM_TELLURIDE": "cadmium_telluride"
    }
    return mapping.get(gate_material_name, None)