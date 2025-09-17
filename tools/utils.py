# Some utility functions
# WARNING: For print functions, make sure that dataframe columns are present in simulation settings (c.f. actor attribtues)
import sys
import time
import importlib.metadata
import numpy as np

try:
    from opengate.logger import global_log
except ImportError:
    import logging

    global_log = logging.getLogger("dummy")
    global_log.addHandler(logging.NullHandler())


# Prints hits like G4 steps are logged via sim.g4_verbose_level_tracking
# If pandas.set_option('display.float_format'...) is used in script calling the function, remove it
def print_hits_inG4format(hits_df):
    print(
        hits_df[['EventID', 'PostPosition_X', 'PostPosition_Y', 'PostPosition_Z',
                 'KineticEnergy', 'TotalEnergyDeposit',
                 'StepLength', 'TrackLength', 'HitUniqueVolumeID',
                 'ProcessDefinedStep', 'ParticleName', 'TrackID',
                 'ParentID', 'ParentParticleName',
                 'TrackCreatorProcess', 'TrackCreatorModelName'
                 ]])


# Prints only few relevant columns from hits tree
def print_hits_short(hits_df):
    print(hits_df[['EventID', 'TrackID', 'ParticleName', 'ParentID',
                   'ParentParticleName', 'KineticEnergy',
                   'TotalEnergyDeposit', 'ProcessDefinedStep',
                   'TrackCreatorProcess', 'GlobalTime',
                   'HitUniqueVolumeID']].to_string(index=False))


def print_hits_long(hits_df):
    print(hits_df[[
        'EventID', 'TrackID', 'ParticleName', 'ParentID', 'ParentParticleName',
        'KineticEnergy',
        'TotalEnergyDeposit', 'ProcessDefinedStep', 'TrackCreatorProcess',
        'PrePosition_X', 'PrePosition_Y', 'PrePosition_Z', 'PostPosition_X',
        'PostPosition_Y', 'PostPosition_Z',
        'PreDirection_X', 'PreDirection_Y', 'PreDirection_Z',
        'PostDirection_X', 'PostDirection_Y', 'PostDirection_Z'
    ]].to_string(index=False))


# Prints directional info from hits tree
# PreDirection and PostDirection seem to be the same very frequently but not always
def print_hits_direction(hits_df):
    print(hits_df[[
        'EventID', 'TrackID', 'ParticleName', 'ProcessDefinedStep',
        'PrePosition_X', 'PrePosition_Y', 'PrePosition_Z', 'PostPosition_X',
        'PostPosition_Y', 'PostPosition_Z',
        'PreDirection_X', 'PreDirection_Y', 'PreDirection_Z',
        'PostDirection_X', 'PostDirection_Y', 'PostDirection_Z'
    ]].to_string(index=False))


# Prints gamma interactions
def print_hits_gammas(hits_df):
    hits_df = hits_df[hits_df['ParticleName'] == 'gamma']
    print(hits_df[[
        'EventID', 'TrackID', 'ParentID', 'KineticEnergy',
        'TotalEnergyDeposit', 'ProcessDefinedStep', 'TrackCreatorProcess',
        'PostPosition_Z'
    ]].to_string(index=False))


# Prints time info
def print_hits_time(hits_df):
    print(hits_df[[
        'EventID', 'TrackID', 'ParticleName', 'GlobalTime', 'PreGlobalTime',
        'LocalTime', 'TimeFromBeginOfEvent',
        'TrackProperTime',
    ]].to_string(index=False))


# Prints processes info
def print_hits_processes(hits_df):
    print(hits_df[[
        'EventID', 'TrackID', 'ParticleName', 'ProcessDefinedStep',
        'TrackCreatorProcess', 'TrackCreatorModelName'
    ]].to_string(index=False))


def print_hits_inG4format_sortedByGlobalTime(hits_df):
    hits_df = hits_df.groupby('EventID').apply(
        lambda x: x.sort_values('GlobalTime'))
    print_hits_inG4format(hits_df)


def print_hits_long_sortedByGlobalTime(hits_df):
    hits_df = hits_df.groupby('EventID').apply(
        lambda x: x.sort_values('GlobalTime'))
    print_hits_long(hits_df)


def get_pixID(x, y, n_pixels=256):
    return x * n_pixels + y


def get_pixID_2D(pixel_id, n_pixels=256):
    x = pixel_id // n_pixels
    y = pixel_id % n_pixels
    return x, y


def sum_time_intervals(time_intervals):
    return sum([time_interval[1] - time_interval[0] for time_interval in
                time_intervals])


def get_worldSize(sensor, source, margin=0.1):
    stype = source.position.type
    if stype not in ["point", "sphere", "box"]:
        raise ValueError(
            "Function get_worldSize() is only implemented for point/sphere/box sources")
    ssize = source.position.size if stype in ['point', 'box'] else [
                                                                       source.position.radius] * 3
    return [np.max([abs(st) + sz / 2, abs(sp) + ss / 2]) * (2 + margin) for
            st, sz, sp, ss in
            zip(sensor.translation, sensor.size, source.position.translation,
                ssize)]


def coordinateOrigin2arrayCenter(cp_array, vpitch, vsize):
    cp_array[:, 0] = cp_array[:, 0] + vpitch * vsize[0] / 2
    cp_array[:, 1] = cp_array[:, 1] + vpitch * vsize[1] / 2
    cp_array[:, 2] = cp_array[:, 2] + vpitch * vsize[2] / 2
    return cp_array


def get_stop_string(stime):
    return f"STOP. Time: {time.time() - stime:.1f} seconds.\n" + '-' * 80


def global_log_debug_df(df):
    """
    Print preview (head) of the dataframe
    """
    if not df.empty:
        global_log.debug(f"Output preview:\n{df.head().to_string(index=False)}")

def global2localFractionalCoordinates(g, sensor, npix):
    pitch = sensor.size[0] / npix  # mm

    a = sensor.translation
    b = [-(npix / 2 - 0.5)] * 2 + [0]

    a, b, g = np.array(a), np.array(b), np.array(g)

    bc = -a + g
    bc = np.dot(sensor.rotation.T, bc) / [pitch, pitch, sensor.size[2]]

    c = bc - b

    return c.tolist()

def localFractional2globalCoordinates(c, translation, rotation, npix, pitch, thickness):
    """
    Converts local fractional coordinates to global coordinates.
    See doc/coordinate_transformation.md for a drawing.

    In the local fractional pixel coordinate system, the origin is the center of the lower
    left pixel in the grid, i.e. the pixel with indices (0,0), whereas the z-axis pointing
    towards the readout connected to the sensor.
    This simplifies calculations in the local coordinate system as all positions can either
    be stated in absolute numbers or in fractions of the pixel pitch.
    See the Allpix2 documentation for more details.

    Parameters:
    c : list[float]
        Coordinates in local fractional pixel space. Should be a list of three values.
    translation : list[float]
        Translation vector representing the position of the sensor in global
        coordinates [x, y, z].
    rotation : list[list[float]]
        3x3 rotation matrix representing the orientation of the sensor.
    npix : int
        Number of pixels along the width (and height) of the sensor.
    pitch : float
        Sensor pitch
    thickness : float
        Sensor thickness

    Returns:
    list[float]
        Global spatial coordinates corresponding to the input local fractional pixel
        coordinates.
    """
    a = translation
    b = [-(npix / 2 - 0.5)] * 2 + [0]

    a, b, c = np.array(a), np.array(b), np.array(c)

    bc = b + c
    bc = np.dot(rotation, bc) * [pitch, pitch, thickness]

    g = a + bc

    return g.tolist()

def localFractional2globalVector(v, rotation, pitch, thickness):
    v = np.array(v)
    v_scaled = v * [pitch, pitch, thickness]
    v_rotated = np.dot(rotation, v_scaled)
    v_normalized = v_rotated / np.linalg.norm(v_rotated)
    return v_normalized.tolist()


def charge_speed_mm_ns(mobility_cm2_Vs, bias_V, thick_mm):
    """
    Assuming constant electric field (ohmic type sensors) and mobility, calculate speed of charges
    In mm per ns
    """

    efield = abs(bias_V) / (thick_mm / 10)  # [V/cm]
    elec_speed = mobility_cm2_Vs * efield  # [cm*cm/V/s] * [V/cm] => [cm/s]
    return elec_speed * 1e-8  # [mm/ns]


def check_gate_version():
    if importlib.metadata.version("opengate") != "10.0.1":
        global_log.error("opengate version not supported: pip install opengate==10.0.1")
        sys.exit()

def create_sensor_object(size,translation,rotation=np.eye(3)):
    """
    Create a dummy sensor object with specified size, translation, and rotation.
    This is useful for testing purposes.
    """
    class Sensor:
        def __init__(self, size, translation, rotation=np.eye(3)):
            self.size = size
            self.translation = translation
            self.rotation = rotation

    return Sensor(size=size, translation=translation, rotation=rotation)

def metric_num(n, decimals=0):
    abs_n = abs(n)
    if abs_n >= 1_000_000_000_000:
        return f"{n/1_000_000_000_000:.{decimals}f}T"
    elif abs_n >= 1_000_000_000:
        return f"{n/1_000_000_000:.{decimals}f}B"
    elif abs_n >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    elif abs_n >= 1_000:
        return f"{n/1_000:.{decimals}f}K"
    else:
        return str(n)


def compton_cos_theta(E1_MeV, source_MeV):
    """
    Calculate the cosine of the Compton scattering angle.

    Parameters
    ----------
    E1_MeV : float or np.ndarray
        Energy of the scattered photon [MeV]
    source_MeV : float or np.ndarray
        Energy of the incident photon [MeV]

    Returns
    -------
    cos_theta : float or np.ndarray
        Cosine of the scattering angle
    """
    numerator = 0.511 * E1_MeV
    denominator = source_MeV * (source_MeV - E1_MeV)
    cos_theta = 1 - numerator / denominator
    return cos_theta
