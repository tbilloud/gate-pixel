# Functions to process pixelHits dataframes

import os
import pandas
import pandas as pd
import uproot
import sys
import time
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import xml.etree.ElementTree as ET
import base64
import numpy as np

from tools.utils import get_pixID, global_log_debug_df, get_stop_string, get_pixID_2D

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
pixelHits_columns = [PIXEL_ID, TOA, ENERGY_keV] # ADD / REMOVE columns as needed
EVENTID = 'EventID' # optional, used for simulated hits only


def singles2pixelHits(file_path, charge_speed_mm_ns, thickness_mm, actor_name='Singles', nrows=None):
    if not os.path.isfile(file_path):
        sys.exit(f"Offline [pixelHits]: {file_path} does not exist, probably no hit produced...")
    else:
        global_log.info(f"Offline [pixelHits]: START")
    stime = time.time()
    singles = uproot.open(file_path)[actor_name].arrays(library='pd', entry_stop=nrows)
    global_log.debug(f"Input {file_path}, {len(singles)} entries")
    singles['HitUniqueVolumeID'] = singles['HitUniqueVolumeID'].astype(
        str).str.replace(r'0_', '', regex=True)
    singles.rename(columns={'HitUniqueVolumeID': PIXEL_ID}, inplace=True)
    singles[PIXEL_ID] = singles[PIXEL_ID].str.replace('pixel_param-', '', regex=False)
    singles[PIXEL_ID] = singles[PIXEL_ID].astype(int)
    singles.rename(columns={'TotalEnergyDeposit': ENERGY_keV}, inplace=True)
    singles[ENERGY_keV] = singles[ENERGY_keV] * 1e3  # Convert MeV to keV
    singles['GlobalTime'] += (-singles['PostPositionLocal_Z'] + thickness_mm/2) / charge_speed_mm_ns
    singles.rename(columns={'GlobalTime': TOA}, inplace=True)
    singles[TOT] = singles[ENERGY_keV] * 1e3  # TODO temporary
    singles = singles[[EVENTID] + pixelHits_columns]
    global_log_debug_df(singles)
    global_log.info(f"Offline [pixelHits]: {get_stop_string(stime)}")
    return singles[[EVENTID] + pixelHits_columns]


def pixelHits_fig_ax(pixelHits_df, n_pixels, fig, ax,
                     log_scale=[False, False, False]):
    df, np = pixelHits_df, n_pixels
    x, y = zip(*df[PIXEL_ID].apply(get_pixID_2D, args=(np,)))

    nc, ne, nt = [mcolors.LogNorm() if log else None for log in log_scale]

    hc = ax[0].hist2d(x, y, bins=[np] * 2, range=[[0, n_pixels]] * 2, norm=nc)
    cb = fig.colorbar(hc[3], ax=ax[0], label='Count')
    cb.locator = MaxNLocator(integer=True)
    cb.update_ticks()
    ax[0].set_title('Counts')

    he = ax[1].hist2d(x, y, bins=[np] * 2, weights=df[ENERGY_keV],
                      range=[[0, np]] * 2, norm=ne,
                      vmin=0.5 * df[ENERGY_keV].min() if not ne else None)
    fig.colorbar(he[3], ax=ax[1], label='Energy (keV)')
    ax[1].set_title('Energy')

    ht = ax[2].hist2d(x, y, bins=[np] * 2, weights=df[TOA],
                      range=[[0, np]] * 2, norm=nt,
                      vmin=0.9 * df[TOA].min() if not nt else None)
    fig.colorbar(ht[3], ax=ax[2], label='ToA (ns)')
    ax[2].set_title('Time')

    for a in ax:
        a.set_aspect('equal')
        a.set_xlabel('Pixel x')
        a.set_ylabel('Pixel y')
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.yaxis.set_major_locator(MaxNLocator(integer=True))

    return fig, ax


def plot_pixelHits_perEventID(pixelHits_df, n_pixels,
                              log_scale=[False, False, False]):
    unique_event_ids = pixelHits_df[EVENTID].unique()
    for event_id in unique_event_ids:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        df = pixelHits_df[pixelHits_df[EVENTID] == event_id]
        pixelHits_fig_ax(df, n_pixels, fig, ax, log_scale)
        plt.suptitle(f'Event ID: {event_id}')
        plt.tight_layout()
        plt.show()


def plot_pixelHits_comparison(pixelHits_df1, pixelHits_df2, n_pixels,
                              log_scale=[False, False, False]):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    pixelHits_fig_ax(pixelHits_df1, n_pixels, fig, ax[0], log_scale)
    pixelHits_fig_ax(pixelHits_df2, n_pixels, fig, ax[1], log_scale)
    plt.tight_layout()
    plt.show()


def plot_pixelHits_comparison_perEventID(pixelHits_df1, pixelHits_df2,
                                         n_pixels,
                                         log_scale=[False, False, False]):
    unique_event_ids = pixelHits_df1[EVENTID].unique()
    assert (unique_event_ids == pixelHits_df2[EVENTID].unique()).all()
    for event_id in unique_event_ids:
        df1 = pixelHits_df1[pixelHits_df1[EVENTID] == event_id]
        df2 = pixelHits_df2[pixelHits_df2[EVENTID] == event_id]
        fig, ax = plt.subplots(2, 3, figsize=(11, 6))
        pixelHits_fig_ax(df1, n_pixels, fig, ax[0], log_scale)
        pixelHits_fig_ax(df2, n_pixels, fig, ax[1], log_scale)
        plt.suptitle(f'Event ID: {event_id}')
        plt.tight_layout()
        plt.show()


def pixelHits2burdaman(pixelHits_df, out_path):
    # TODO set types correctly (else visu with TrackLab will not work)
    # => https://software.utef.cvut.cz/tracklab/manual/a01627.html
    # insert a column with 0s at the 3rd position
    pixelHits_df.insert(2, 'fTOA', 0)
    print(pixelHits_df)
    pixelHits_df.to_csv(out_path, header=False, index=False, sep='\t')

    # Dummy header
    # TODO replace values with NaNs
    custom_header = """# Start of measurement: 10/1/2017 17:34:41.8467094
# Start of measurement - unix time: 1506872081.846
# Chip ID: H3-W00036
# Readout IP address: 192.168.1.105
# Back-end location: Satigny, CH
# Detector mode: ToA & ToT
# Readout mode: Data-Driven Mode
# Bias voltage: 229.72V
# THL = 1570 (0.875V)
# Sensor temperature: 58.9°C
# Readout temperature: 42.9°C
# ------- Internal DAC values ---------------
# Ibias_Preamp_ON:\t128\t(1.208V)
# Ibias_Preamp_OFF:\t8\t(1.350V)
# VPreamp_NCAS:\t\t128\t(0.702V)
# Ikrum:\t\t15\t(1.128V)
# Vfbk:\t\t164\t(0.891V)
# Vthreshold_fine:\t505\t(0.877V)
# Vthreshold_coarse:\t7\t(0.875V)
# Ibias_DiscS1_ON:\t100\t(1.109V)
# Ibias_DiscS1_OFF:\t8\t(1.321V)
# Ibias_DiscS2_ON:\t128\t(0.396V)
# Ibias_DiscS2_OFF:\t8\t(0.256V)
# Ibias_PixelDAC:\t128\t(0.984V)
# Ibias_TPbufferIn:\t128\t(1.169V)
# Ibias_TPbufferOut:\t128\t(1.077V)
# VTP_coarse:\t\t128\t(0.693V)
# VTP_fine:\t\t256\t(0.724V)
# Ibias_CP_PLL:\t\t128\t(0.557V)
# PLL_Vcntrl:\t\t128\t(0.874V)
# BandGap output:\t--- \t(0.684V)
# BandGap_Temp:\t\t--- \t(0.733V)
# Ibias_dac:\t\t--- \t(1.241V)
# Ibias_dac_cas:\t\t--- \t(1.004V)
# DACs: \t128\t8\t128\t15\t164\t505\t7\t100\t8\t128\t8\t128\t128\t128\t128\t256\t128\t128
# DACs Scans: \t1.208V\t1.350V\t0.702V\t1.128V\t0.891V\t0.877V\t0.875V\t1.109V\t1.321V\t0.396V\t0.256V\t0.984V\t1.169V\t1.077V\t0.693V\t0.724V\t0.557V\t0.874V\t0.684V\t0.733V\t1.241V\t1.004V
# -----------------------------------------------------------------------------------------------------------------------------
"""

    # Read the CSV file and add the custom header
    with open(out_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    lines.insert(0, custom_header)

    # Write the modified content back to the file
    with open(out_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


# TODO adapt to different simulation chains
def allpixTxt2pixelHit(text_file, n_pixels=256):

    stime = time.time()
    global_log.info(f"Offline [pixelHits]: START")
    if os.path.isfile(text_file.with_suffix('.txt')):
        global_log.debug(f"Input {text_file}")
    else:
        global_log.error(f"{text_file} does not exist.")
        global_log.info(f"Offline [pixelHits]: {get_stop_string(stime)}")
        return pandas.DataFrame()

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
    if len(df) == 0:
        global_log.error(f"Offline [pixelHits]: Empty pixel hits dataframe, probably no hit produced.")
    global_log_debug_df(df)
    global_log.info(f"Offline [pixelHits]: {get_stop_string(stime)}")
    return df


# TODO: check for ToA overflow
def pixet2pixelHit(t3pa_file, calib, chipID=None, nrows=None):
    """
    Convert pixel hits and calibration from ADVACAM/PIXET to a pixelHit DataFrame.

    calib can be:
    * A directory containing the files caliba.txt, calibb.txt, calibc.txt, calibt.txt
      => In Pixet: Detector Setting -> More Detector Settings -> Chips -> Save
    * An XML file containing the calibration data for the chipID

    The measurement must be done with:
    * Measurement -> Type -> Pixels
    * Detector Setting -> Mode -> ToA + ToT
    => This stores a .t3pa and a .t3pa.info file. Only the .t3pa file is needed here.

    The XML file and chip ID are provided when purchasing a detector.
    """
    df = pd.read_csv(t3pa_file, sep='\t', index_col='Index', nrows=nrows)

    global_log.info(f"Offline [pixelHits]: START")
    global_log.debug(f"Inputs:\n{t3pa_file}\n{calib}, {len(df)} entries")
    stime = time.time()

    # ===========================
    # ==  TIME CALIBRATION     ==
    # ===========================

    df['ToA (ns)'] = 25 * df['ToA'] - (25 / 16) * df['FToA']

    # ===========================
    # == ENERGY CALIBRATION    ==
    # ===========================

    calib_names = ['caliba', 'calibb', 'calibc', 'calibt']
    calib_dict = {}

    if calib.endswith('xml'):
        global_log.info(f"Offline [pixelHits]: Using XML file for calibration")
        global_log.error("Reading calibration from XML seems wrong with current decoding")
        tree = ET.parse(calib)
        root = tree.getroot()
        chip = root.find(chipID)
        if not chip:
            global_log.error(f"Chip {chipID} not found in XML file {calib_dict}")
            sys.exit(1)
        for name in calib_names:
            calib_str = chip.find(name).text
            if calib_str is not None:
                decoded = base64.b64decode(calib_str)
                arr = np.frombuffer(decoded, dtype=np.float32)
                values = arr[1::2]
                calib_dict[name] = values
    elif os.path.isdir(calib):
        global_log.info(f"Offline [pixelHits]: Searching {calib} for calib files")
        for name in calib_names:
            file_path = os.path.join(calib, f"{name}.txt")
            arr = np.loadtxt(file_path)
            if arr.shape != (256, 256):
                raise ValueError(f"{file_path} does not have shape (256, 256)")
            calib_dict[name] = arr.flatten()  # row-major order

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

    if len(df) == 0:
        global_log.error(f"Offline [pixelHits]: Empty pixel hits dataframe, probably no hit produced.")
    global_log_debug_df(df)
    global_log.info(f"Offline [pixelHits]: {get_stop_string(stime)}")
    return df

def remove_edge_pixels(df, n_pixels=256, edge_thickness=1):
    """
    Remove edge pixels from a DataFrame.
    Works with either PIX_X_ID/PIX_Y_ID or PIX_ID columns.
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