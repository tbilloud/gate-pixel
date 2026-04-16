import os
import re
import time
from pathlib import Path

import humanize
import numpy as np
import pandas as pd
from loguru import logger

from tools.logging_custom import global_log

from tools.pixelClusters import PIX_X_ID, PIX_Y_ID, SIZE, DELTA_TOA
from tools.pixelHits import ENERGY_keV, TOA, PIXEL_ID
from tools.utils import log_offline_process, get_pixID


def _parse_clog(file_path, max_lines=None, max_bytes=None):
    """
    Core clog parser. Yields (hits, frame_time) for each cluster line, where
    hits is a list of (x, y, energy, toa) tuples.

    Works on raw bytes to avoid the overhead of decoding every line.
    """
    frame_time = None
    frame_re = re.compile(rb'^Frame\s+\d+\s+\(\s*([^,]+)\s*,')
    bracket_re = re.compile(rb'\[([^\]]+)\]')

    if max_bytes is not None:
        max_bytes = int(max_bytes)

    bytes_read = 0

    with open(file_path, 'rb') as f:
        for lineno, raw in enumerate(f, start=1):
            bytes_read += len(raw)
            if max_lines is not None and lineno > max_lines:
                break
            if max_bytes is not None and bytes_read > max_bytes:
                break

            stripped = raw.strip()
            if not stripped:
                continue

            m = frame_re.match(stripped)
            if m:
                try:
                    frame_time = float(m.group(1))
                except ValueError:
                    frame_time = 0.0
                continue

            groups = bracket_re.findall(stripped)
            if not groups:
                continue

            hits = []
            for g in groups:
                parts = g.split(b',')
                if len(parts) < 4:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    e = float(parts[2])
                    t = float(parts[3])
                except ValueError:
                    continue
                hits.append((x, y, e, t))

            if hits:
                yield hits, frame_time


def clog2clogEnergyFiltered(input_path, output_path, min_energy_keV=0.0, max_energy_keV=float('inf')):
    """
    Read a clog file and write a new one keeping only clusters whose total
    pixel-hit energy (sum of all hit energies) is in [min_energy_keV, max_energy_keV].

    Args:
        input_path:      Path to the source .clog file.
        output_path:     Path for the filtered output .clog file.
        min_energy_keV:  Minimum cluster energy in keV (default 0).
        max_energy_keV:  Maximum cluster energy in keV (default inf).

    Returns:
        (kept, total) cluster counts.
    """
    t0 = time.time()
    frame_re_b = re.compile(rb'^Frame\s+\d+\s+\(')
    bracket_re_b = re.compile(rb'\[([^\]]+)\]')

    kept = total = 0
    pending_frame = None
    wrote_any = False

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for lineno, raw in enumerate(fin, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            if frame_re_b.match(stripped):
                pending_frame = raw
                continue
            groups = bracket_re_b.findall(stripped)
            if not groups:
                continue

            total += 1
            total_e = 0.0
            for g in groups:
                parts = g.split(b',')
                if len(parts) != 4:
                    print(f"filter_clog_by_energy: malformed hit on line {lineno}")
                    return kept, total
                total_e += float(parts[2])

            if min_energy_keV <= total_e <= max_energy_keV:
                if pending_frame is not None:
                    if wrote_any:
                        fout.write(b'\n')
                    fout.write(pending_frame)
                    pending_frame = None
                    wrote_any = True
                fout.write(raw)
                kept += 1

    print(f"filter_clog_by_energy: kept {kept}/{total} clusters "
          f"({min_energy_keV}\u2013{max_energy_keV} keV) in {time.time() - t0:.2f}s")
    return kept, total

@log_offline_process('clog2clogBorderFiltered', input_type='file')
def clog2clogBorderFiltered(input_path, output_path, border_values=(0, 255)):
    """
    Read a clog file and write a new one, dropping any cluster that has at least
    one pixel hit whose x or y coordinate is on a border.

    Streams line-by-line in raw bytes so memory usage stays constant regardless
    of file size.

    Args:
        input_path:    Path to the source .clog file.
        output_path:   Path for the filtered output .clog file.
        border_values: Set/tuple of x/y pixel values considered border (default (0, 255)).

    Returns:
        (kept, total) cluster counts.
    """
    border_set = set(border_values)
    frame_re = re.compile(rb'^Frame\s+\d+\s+\(')
    bracket_re = re.compile(rb'\[([^\]]+)\]')

    kept = total = 0
    pending_frame = None
    wrote_any = False

    in_size = os.path.getsize(input_path)

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for raw in fin:
            stripped = raw.strip()
            if not stripped:
                continue

            if frame_re.match(stripped):
                pending_frame = raw
                continue

            groups = bracket_re.findall(stripped)
            if not groups:
                continue

            total += 1

            # Check if any hit touches a border pixel
            on_border = False
            for g in groups:
                parts = g.split(b',')
                if len(parts) < 4:
                    continue
                try:
                    x = int(float(parts[0]))
                    y = int(float(parts[1]))
                except ValueError:
                    continue
                if x in border_set or y in border_set:
                    on_border = True
                    break

            if not on_border:
                if pending_frame is not None:
                    if wrote_any:
                        fout.write(b'\n')
                    fout.write(pending_frame)
                    pending_frame = None
                    wrote_any = True
                fout.write(raw)
                kept += 1

    out_size = os.path.getsize(output_path)
    global_log.info(
        f"kept {humanize.metric(kept)}/{humanize.metric(total)} clusters, "
        f"{humanize.naturalsize(in_size)} → {humanize.naturalsize(out_size)}")
    return kept, total

@log_offline_process('pixelClusters', input_type='file')
def clog2pixelClusters(file_path, max_lines=None, max_bytes=None, omit_border=False, border_values=(0, 255),
                       chunk_size=100_000):
    """
    Convert a clog file from the Pixet software (Advacam) to a DataFrame of pixel clusters.
    See: https://wiki.advacam.cz/index.php/PIXet

    Processes the file in chunks to keep memory usage bounded for large files.
    """
    columns = [PIX_X_ID, PIX_Y_ID, ENERGY_keV, TOA, SIZE, DELTA_TOA]
    border_set = set(border_values)
    chunks = []
    events = []

    file_size = os.path.getsize(file_path)
    global_log.info(f"clog2pixelClusters: reading {file_path} ({humanize.naturalsize(file_size)})")

    for hits, frame_time in _parse_clog(file_path, max_lines, max_bytes):
        total_e = sum(h[2] for h in hits)
        if total_e == 0:
            x_w = sum(h[0] for h in hits) / len(hits)
            y_w = sum(h[1] for h in hits) / len(hits)
        else:
            x_w = sum(h[0] * h[2] for h in hits) / total_e
            y_w = sum(h[1] * h[2] for h in hits) / total_e

        if omit_border and (int(round(x_w)) in border_set or int(round(y_w)) in border_set):
            continue

        toa = (frame_time if frame_time is not None else 0.0) + min(h[3] for h in hits)
        size = len(hits)
        delta_toa = max(h[3] for h in hits) - min(h[3] for h in hits) if size > 1 else float('nan')

        events.append((x_w, y_w, total_e, toa, size, delta_toa))

        if len(events) >= chunk_size:
            chunks.append(pd.DataFrame(events, columns=columns))
            events = []

    if events:
        chunks.append(pd.DataFrame(events, columns=columns))

    if not chunks:
        return pd.DataFrame(columns=columns)

    df = pd.concat(chunks, ignore_index=True)
    df_size = df.memory_usage(deep=True).sum()
    global_log.info(f"clog2pixelClusters: {humanize.metric(len(df))} clusters ({humanize.naturalsize(df_size)})")
    return df


@log_offline_process('pixelHitsTagged', input_type='file')
def clog2pixelHitsTagged(file_path, npix, max_lines=None, max_bytes=None):
    """
    Convert a clog file to a DataFrame of individual pixel hits, each tagged with a cluster_id.
    Columns: PixelID (int16), Energy (keV), ToA (ns), cluster_id
    """
    rows = []
    cluster_id = 0

    for hits, frame_time in _parse_clog(file_path, max_lines, max_bytes):
        t_offset = frame_time if frame_time is not None else 0.0
        for x, y, e, t in hits:
            rows.append({
                PIXEL_ID: get_pixID(int(x), int(y), npix), ENERGY_keV: e,
                TOA: t_offset + t, 'cluster_id': cluster_id,
            })
        cluster_id += 1

    return pd.DataFrame(rows, columns=[PIXEL_ID, ENERGY_keV, TOA, 'cluster_id'])


@log_offline_process('pixelHits', input_type='file')
def t3pa2pixelHits(t3pa_file, calib, nrows=None):
    """
    Convert pixel hits and calibration from ADVACAM/PIXET to a pixelHit DataFrame.

    calib must be a directory containing the files caliba.txt, calibb.txt, calibc.txt, calibt.txt
      => In Pixet: Detector Setting -> More Detector Settings -> Chips -> Save

    The measurement must be done with:
    * Measurement -> Type -> Pixels
    * Detector Setting -> Mode -> ToA + ToT
    => This stores a .t3pa and a .t3pa.info file. Only the .t3pa file is needed here.

    # TODO: check for ToA overflow
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
