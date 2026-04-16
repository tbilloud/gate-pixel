# Functions to process pixelHits dataframes

from pathlib import Path
import pandas
import pandas as pd
import uproot
import numpy as np
from tools.utils import get_pixID, get_pixID_2D, log_offline_process

# Pixel hit format definition
PIXEL_ID = 'PixelID (int16)'
TOA = 'ToA (ns)'
ENERGY_keV = 'Energy (keV)'
TOT = 'ToT'
pixelHits_columns = [PIXEL_ID, TOA, ENERGY_keV]  # ADD / REMOVE columns as needed
EVENTID = 'EventID'  # optional, used for simulated hits only


@log_offline_process('pixelHits', input_type='file_or_dataframe')
def singles2pixelHits(file_path_or_df, speed, thick, actor='Singles', nrows=None):
    """
    Converts Gate singles into a DataFrame of pixelHits.

    Args:
        file_path_or_df (str | Path | pd.DataFrame): Path to the Gate ROOT file, or a DataFrame already loaded.
        speed (float): Charge propagation speed in the sensor (unit must be consistent with the 'thick' parameter).
        thick (float): Sensor thickness (unit must be consistent with the 'speed' parameter).
        actor (str, optional): Name of the Gate actor used to get singles. Only used when a file path is given. Defaults to 'Singles'.
        nrows (int, optional): Maximum number of rows to read from the file. If None, reads all rows. Ignored when a DataFrame is given.

    Returns:
        pandas.DataFrame: DataFrame containing pixelHits.
    """

    if isinstance(file_path_or_df, pd.DataFrame):
        singles = file_path_or_df.copy()
    else:
        singles = uproot.open(file_path_or_df)[actor].arrays(library='pd', entry_stop=nrows)

    # Deal with pixel IDs
    # When a track ends at a border pixel, 'PostStepUniqueVolumeID' might be whatever volume is behind the pixel (e.g.
    # the world). In those cases I replace it with the pre-step. TODO is this the best solution?
    singles['PostStepUniqueVolumeID'] = singles['PostStepUniqueVolumeID'].astype(str)
    mask = ~singles['PostStepUniqueVolumeID'].str.contains('0_0_')
    singles.loc[mask, 'PostStepUniqueVolumeID'] = singles.loc[mask, 'PreStepUniqueVolumeID']
    singles['PostStepUniqueVolumeID'] = singles['PostStepUniqueVolumeID'].astype(str).str.extract(r'0_0_(\d+)').astype(int) # 'PostStepUniqueVolumeID' is in the format '0_0_X' with opengate 10.0.1 and 'pixel_param-0_0_X' with 10.0.3 (where X is the pixel ID). This works with both formats.
    singles.rename(columns={'PostStepUniqueVolumeID': PIXEL_ID}, inplace=True)
    singles[PIXEL_ID] = singles[PIXEL_ID].astype(int)

    # Deal with energy and time
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
