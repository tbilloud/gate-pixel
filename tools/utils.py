# General utility functions

import os
import time
from pathlib import Path
import numpy as np
import pandas
from tools.logging_custom import global_log


def get_pixID(x, y, n_pixels=256):
    return x * n_pixels + y


def get_pixID_2D(pixel_id, n_pixels=256):
    x = pixel_id // n_pixels
    y = pixel_id % n_pixels
    return x, y


def global2localFractionalCoordinates(g, sensor, npix):
    pitch = sensor.size[0] / npix  # mm

    a = sensor.translation
    b = [-(npix / 2 - 0.5)] * 2 + [0]

    a, b, g = np.array(a), np.array(b), np.array(g)

    bc = -a + g
    bc = np.dot(sensor.rotation.T, bc) # rotate first
    bc = bc / [pitch, pitch, sensor.size[2]] # then scale (inverse of localFractional2globalCoordinates)

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
    bc = bc * [pitch, pitch, thickness] # scale first
    bc = np.dot(rotation, bc) # then rotate (inverse of global2localFractionalCoordinates)

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




def log_offline_process(object_name, input_type):
    def decorator(func):
        def wrapper(*args, **kwargs):
            stime = time.time()
            global_log.info(f"Offline [{object_name}]: START")

            if input_type == 'file':
                if not os.path.isfile(args[0]):
                    global_log.error(f"Input file '{args[0]}' not found.")
            elif input_type == 'file_or_dataframe':
                if not isinstance(args[0], pandas.DataFrame) and not os.path.isfile(args[0]):
                    global_log.error(f"Input '{args[0]}' is neither a DataFrame nor a valid file.")
            elif input_type == 'dataframe':
                if args[0].empty:
                    global_log.error(f"Input dataframe is empty.")
            elif input_type == 'sim':
                f = Path(args[0].output_dir) / args[0].actor_manager.get_actor(
                    "Hits").output_filename
                f = os.path.join(os.getcwd(), f)
                if not os.path.isfile(f):
                    global_log.error(f"Input simulation not found.")
            else:
                raise ValueError("input_type must be 'file', 'dataframe' or 'sim'")

            result = func(*args, **kwargs)

            if isinstance(result, pandas.DataFrame):
                global_log.debug(f"{len(result)} output rows.")
                global_log.debug(f"Preview:\n{result.head().to_string(index=False)}")

            stop_string = f"STOP. Time: {time.time() - stime:.1f} seconds.\n" + '-' * 80
            global_log.info(f"Offline [{object_name}]: {stop_string}")

            return result

        return wrapper

    return decorator


