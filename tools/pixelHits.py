# Functions to process pixelHits dataframes

import os
from pathlib import Path
import pandas
import pandas as pd
import uproot
import sys
import xml.etree.ElementTree as ET
import base64
import numpy as np
from tools.utils import get_pixID, get_pixID_2D, log_offline_process

try:
    from opengate.logger import global_log
except ImportError:
    import logging

    global_log = logging.getLogger("dummy")
    global_log.addHandler(logging.NullHandler())

# Pixel hit format definition
PIXEL_ID = 'PixelID (int16)'
TOA = 'ToA (ns)'
ENERGY_keV = 'Energy (keV)'
TOT = 'ToT'
pixelHits_columns = [PIXEL_ID, TOA, ENERGY_keV]  # ADD / REMOVE columns as needed
EVENTID = 'EventID'  # optional, used for simulated hits only


@log_offline_process('pixelHits', input_type='file')
def singles2pixelHits(file_path, speed, thick, actor='Singles', nrows=None):
    """
    Converts a ROOT file containing Gate singles into a DataFrame of pixelHits.

    Args:
        file_path (str): Path to the Gate's ROOT file containing singles.
        speed (float): Charge propagation speed in the sensor (unit must be consistent with the 'thick' parameter).
        thick (float): Sensor thickness (unit must be consistent with the 'speed' parameter).
        actor (str, optional): Name of the Gate actor used to get singles. The actor can be e.g. DigitizerReadoutActor or DigitizerBlurringActor, but its name is user defined. Defaults to 'Singles'.
        nrows (int, optional): Maximum number of rows to read from the file. If None, reads all rows.

    Returns:
        pandas.DataFrame: DataFrame containing pixelHits.
    """

    singles = uproot.open(file_path)[actor].arrays(library='pd', entry_stop=nrows)
    singles['HitUniqueVolumeID'] = singles['HitUniqueVolumeID'].astype(
        str).str.replace(r'0_', '', regex=True)
    singles.rename(columns={'HitUniqueVolumeID': PIXEL_ID}, inplace=True)
    singles[PIXEL_ID] = singles[PIXEL_ID].str.replace('pixel_param-', '', regex=False)
    singles[PIXEL_ID] = singles[PIXEL_ID].astype(int)
    singles.rename(columns={'TotalEnergyDeposit': ENERGY_keV}, inplace=True)
    singles[ENERGY_keV] = singles[ENERGY_keV] * 1e3  # Convert MeV to keV
    singles['GlobalTime'] += (-singles['PostPositionLocal_Z'] + thick / 2) / speed
    singles.rename(columns={'GlobalTime': TOA}, inplace=True)
    singles[TOT] = singles[ENERGY_keV] * 1e3  # TODO temporary
    singles = singles[[EVENTID] + pixelHits_columns]

    return singles[[EVENTID] + pixelHits_columns]


# TODO adapt to different simulation chains
def allpixTxt2pixelHit(text_file, n_pixels=256):
    rows = []
    with open(text_file.with_suffix('.txt'), "r") as file:
        event_id = None
        for line in file:
            line = line.strip()

            if line.startswith("==="):
                event_id = int(line.split()[1]) - 1  # allpix adds 1 to event ID
                continue

            if line.startswith("---"):
                continue

            if line.startswith("PixelHit"):
                parts = line.split()
                x, y = int(parts[1].strip(',')), int(parts[2].strip(','))
                pixel_id = get_pixID(x, y, n_pixels=n_pixels)
                tot = float(parts[3].strip(','))
                # parts[4] is ToA from event start
                global_time = float(parts[5].strip(','))  # ToA from simu start
                # parts[6] is position_x
                # parts[7] is position_y
                # parts[8] is position_z

                rows.append({
                    EVENTID: event_id,
                    PIXEL_ID: pixel_id,
                    TOT: tot,
                    ENERGY_keV: tot * 4.43 / 1000,
                    # TODO: adapt to qdc_resolution (on/off) in DefaultDigitizer
                    TOA: global_time
                })

    df = pd.DataFrame(rows, columns=[EVENTID] + pixelHits_columns)

    return df


# TODO: check for ToA overflow
@log_offline_process('pixelHits', input_type='file')
def pixet2pixelHit(t3pa_file, calib, nrows=None):
    """
    Convert pixel hits and calibration from ADVACAM/PIXET to a pixelHit DataFrame.

    calib must be a directory containing the files caliba.txt, calibb.txt, calibc.txt, calibt.txt
      => In Pixet: Detector Setting -> More Detector Settings -> Chips -> Save

    The measurement must be done with:
    * Measurement -> Type -> Pixels
    * Detector Setting -> Mode -> ToA + ToT
    => This stores a .t3pa and a .t3pa.info file. Only the .t3pa file is needed here.
    """
    if not Path(calib).exists():
        raise FileNotFoundError(f"Calibration directory not found")

    df = pd.read_csv(t3pa_file, sep='\t', index_col='Index', nrows=nrows)

    # ===========================
    # ==  TIME CALIBRATION     ==
    # ===========================

    df['ToA (ns)'] = 25 * df['ToA'] - (25 / 16) * df['FToA']

    # ===========================
    # == ENERGY CALIBRATION    ==
    # ===========================

    calib_names = ['caliba', 'calibb', 'calibc', 'calibt']
    calib_dict = {}

    global_log.debug(f"Offline [pixelHits]: Searching {calib} for calib files")

    for name in calib_names:
        file_path = Path(calib) / f"{name}.txt"
        if not file_path.is_file():
            raise FileNotFoundError(f"`{file_path}` not found")
        arr = np.loadtxt(file_path)
        if arr.shape != (256, 256):
            raise ValueError(f"{file_path} does not have shape (256, 256)")
        calib_dict[name] = arr.flatten()

    def tot_to_energy(tot, a, b, c, t):
        A = a
        B = b - tot - a * t
        C = -t * (b - tot) - c
        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0 or A == 0:
            return np.nan
        E = (-B + np.sqrt(discriminant)) / (2 * A)
        return E

    for name in calib_names:
        global_log.debug(f"Mean of {name}: {np.mean(calib_dict[name])}")

    def compute_energy(row):
        idx = int(row['Matrix Index'])
        a = calib_dict['caliba'][idx]
        b = calib_dict['calibb'][idx]
        c = calib_dict['calibc'][idx]
        t = calib_dict['calibt'][idx]
        return tot_to_energy(row['ToT'], a, b, c, t)

    df['Energy (keV)'] = df.apply(compute_energy, axis=1)

    # ===========================
    # ==  FORMAT DATAFRAME     ==
    # ===========================
    df = df.drop(columns=['ToA', 'ToT', 'FToA', 'Overflow'])
    df = df.rename(columns={'Matrix Index': 'PixelID (int16)'})

    return df


def remove_edge_pixels(df, n_pixels=256, edge_thickness=1):
    """
    Remove edge pixels from a DataFrame.
    edge_thickness: number of pixels to exclude from each edge (default=1).
    """

    x, y = zip(*df[PIXEL_ID].apply(get_pixID_2D, args=(n_pixels,)))
    x = np.array(x)
    y = np.array(y)
    mask = (
            (x >= edge_thickness) & (x < n_pixels - edge_thickness) &
            (y >= edge_thickness) & (y < n_pixels - edge_thickness)
    )
    return df[mask].reset_index(drop=True)
