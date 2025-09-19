# Functions to process cones
import os
import time
import numpy as np
import pandas
import pandas as pd
import uproot

from tools.CCevents import global_log
from tools.pixelHits import EVENTID
from tools.utils import get_stop_string, global_log_debug_df, \
    localFractional2globalCoordinates, localFractional2globalVector, metric_num

# Cone format definition
APEX_X, APEX_Y, APEX_Z = 'Apex_X', 'Apex_Y', 'Apex_Z'
DIRECTION_X, DIRECTION_Y, DIRECTION_Z = 'Direction_X', 'Direction_Y', 'Direction_Z'
COS = 'cosT'
ERROR = 'error'


def CCevents2CCcones(CCevents, log=True):
    stime = time.time()

    if log:
        global_log.info("Offline [cones]: START")
        if not len(CCevents):
            global_log.error("Empty input.")
            global_log.info(f"Offline [cones]: {get_stop_string(stime)}")
            return pandas.DataFrame()
        global_log.debug(f"Input: {len(CCevents)} entries")
    elif not len(CCevents):
        return pandas.DataFrame()

    pos_compton = CCevents[
        ['PositionX_1', 'PositionY_1', 'PositionZ_1']].to_numpy()
    pos_photoel = CCevents[
        ['PositionX_2', 'PositionY_2', 'PositionZ_2']].to_numpy()
    E1_keV = CCevents['Energy (keV)_1'].to_numpy()
    E2_keV = CCevents['Energy (keV)_2'].to_numpy()
    eventid = CCevents['evt_1'].to_numpy()

    # Compute direction vectors
    direction = pos_compton - pos_photoel
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    # Avoid division by zero
    norm[norm == 0] = 1
    direction = direction / norm

    # Compute cosT
    E1_MeV = E1_keV / 1000
    E2_MeV = E2_keV / 1000
    source_MeV = E1_MeV + E2_MeV
    denom = source_MeV * (source_MeV - E1_MeV)
    # Avoid division by zero
    denom[denom == 0] = np.nan
    cosT = 1 - (0.511 * E1_MeV) / denom

    # Filter out invalid cones (cosT not finite or out of range)
    valid = np.isfinite(cosT) & (np.abs(cosT) <= 1)
    eventid = eventid[valid]
    pos_compton = pos_compton[valid]
    direction = direction[valid]
    cosT = cosT[valid]

    # Build DataFrame
    cones = pandas.DataFrame({
        EVENTID: eventid,
        APEX_X: pos_compton[:, 0],
        APEX_Y: pos_compton[:, 1],
        APEX_Z: pos_compton[:, 2],
        DIRECTION_X: direction[:, 0],
        DIRECTION_Y: direction[:, 1],
        DIRECTION_Z: direction[:, 2],
        COS: cosT,
    })

    if log:
        global_log.info(f"Offline [cones]: {len(cones)} cones")
        global_log_debug_df(cones)
        global_log.info(f"Offline [cones]: {get_stop_string(stime)}")
    return cones


def local2global_cones(cones, translation, rotation, npix, pitch, thickness):
    """
    Converts local fractional coordinates in a cones DataFrame to global coordinates.

    WARNING: when using Gate, sensor translation is not global if it has a mother volume
    => in this case use tools.utils_opengate.get_global_translation() !

    The function processes each row of the DataFrame to compute the global coordinates
    from local fractional coordinates and updates the corresponding columns in the
    DataFrame.

    Args:
        cones (DataFrame): Input DataFrame containing cones
        translation (list[float] or tuple[float]): Translation vector for converting
            coordinates.
        rotation (list[list[float]]): Rotation matrix to apply during the conversion
            process.
        npix (int): Number of pixels for the grid.
        pitch (float): The pitch distance used in the grid layout.
        thickness (float): Thickness of the sensor.

    Returns:
        DataFrame: Updated DataFrame with apex columns replaced by global coordinates.
    """

    global_log.info(f"Offline [transform coord]: START")
    stime = time.time()

    df_copy = cones.copy()
    if df_copy.empty:
        global_log.error('Input DataFrame is empty. No coordinates to convert.')
        return pd.DataFrame()

    # Transform apex coordinates
    apex_cols = [APEX_X, APEX_Y, APEX_Z]

    def transform_apex(row):
        c = [row[col] for col in apex_cols]
        return localFractional2globalCoordinates(c, translation, rotation, npix,
                                                 pitch, thickness)

    coords = df_copy.apply(transform_apex, axis=1, result_type='expand')
    df_copy[apex_cols] = coords

    # Transform direction vectors
    direction_cols = [DIRECTION_X, DIRECTION_Y, DIRECTION_Z]

    def transform_direction(row):
        v = [row[col] for col in direction_cols]
        return localFractional2globalVector(v, rotation, pitch, thickness)

    vectors = df_copy.apply(transform_direction, axis=1, result_type='expand')
    df_copy[direction_cols] = vectors

    global_log_debug_df(df_copy)
    global_log.info(f"Offline [transform coord]: {get_stop_string(stime)}")
    return df_copy
