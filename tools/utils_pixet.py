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


@log_offline_process('clog2clogEnergyFiltered', input_type='file')
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

def find_noisy_pixels(file_path, npix=256, outlier_sigma_counts=None,
                      outlier_sigma_energy=None, outlier_sigma_toa=None,
                      max_lines=None, max_bytes=None):
    """
    Scan a clog file and return a set of (x, y) pixel coordinates that are noisy:
      - pixels with negative coordinate/energy/time values
      - pixels whose hit count, total energy, or median ToA exceeds mean + sigma * std

    At least one of outlier_sigma_counts, outlier_sigma_energy, outlier_sigma_toa
    should be set, otherwise only negative-value pixels are detected.

    Args:
        outlier_sigma_counts: Sigma threshold for hit count outliers.
        outlier_sigma_energy: Sigma threshold for total energy outliers.
        outlier_sigma_toa:    Sigma threshold for median ToA outliers.

    Returns:
        set of (x, y) tuples
    """
    if outlier_sigma_counts is None and outlier_sigma_energy is None and outlier_sigma_toa is None:
        global_log.warning("find_noisy_pixels: no outlier_sigma_* set — only negative-value detection active. "
                           "Set at least one of outlier_sigma_counts, outlier_sigma_energy, outlier_sigma_toa.")

    counts = np.zeros((npix, npix), dtype=np.int64)
    energy = np.zeros((npix, npix), dtype=np.float64)
    toa_lists = [[[] for _ in range(npix)] for _ in range(npix)]
    neg_pixels = set()

    for hits, frame_time in _parse_clog(file_path, max_lines, max_bytes):
        t_offset = frame_time if frame_time is not None else 0.0
        for x, y, e, t in hits:
            xi, yi = int(x), int(y)
            if x < 0 or y < 0 or e < 0 or t < 0:
                neg_pixels.add((xi, yi))
            if 0 <= xi < npix and 0 <= yi < npix:
                counts[yi, xi] += 1
                energy[yi, xi] += e
                toa_lists[yi][xi].append(t_offset + t)

    if neg_pixels:
        global_log.warning(f"Negative values at {len(neg_pixels)} pixel(s): "
                           f"{sorted(neg_pixels)[:20]}{'...' if len(neg_pixels) > 20 else ''}")

    median_toa = np.full((npix, npix), np.nan)
    for yi in range(npix):
        for xi in range(npix):
            if toa_lists[yi][xi]:
                median_toa[yi, xi] = np.median(toa_lists[yi][xi])

    noisy = set(neg_pixels)
    for name, d, sigma in [('counts', counts, outlier_sigma_counts),
                            ('energy', energy, outlier_sigma_energy),
                            ('median_toa', median_toa, outlier_sigma_toa)]:
        if sigma is None:
            continue
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size == 0:
            continue
        mean, std = valid.mean(), valid.std()
        threshold = mean + sigma * std
        outlier_yx = np.argwhere(np.isfinite(d) & (d > threshold))
        if outlier_yx.size > 0:
            outlier_pixels = {(int(xi), int(yi)) for yi, xi in outlier_yx}
            noisy |= outlier_pixels
            global_log.warning(f"Outliers in {name} (>{sigma}σ, threshold={threshold:.1f}): "
                               f"{len(outlier_pixels)} pixel(s): "
                               f"{sorted(outlier_pixels)[:20]}{'...' if len(outlier_pixels) > 20 else ''}")

    global_log.info(f"find_noisy_pixels: {len(noisy)} noisy pixel(s) found")
    return noisy


def load_noisy_pixels_from_file(path):
    """
    Load noisy pixel coordinates from a text file.
    Expected format: one pixel per line, 'x y' or 'x,y' (whitespace or comma separated).
    Lines starting with '#' are ignored.
    """
    noisy = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r'[,\s]+', line)
            if len(parts) >= 2:
                noisy.add((int(parts[0]), int(parts[1])))
    global_log.info(f"Loaded {len(noisy)} noisy pixel(s) from {path}")
    return noisy


@log_offline_process('clog2clogNoiseFree', input_type='file')
def clog2clogNoiseFree(input_path, output_path=None, npix=256, noisy_pixels=None,
                       noisy_pixels_file=None, outlier_sigma_counts=None,
                       outlier_sigma_energy=None, outlier_sigma_toa=None):
    """
    Read a clog file and write a new one with clusters touching noisy pixels removed.

    Noisy pixels are determined by (combined with union):
      1. noisy_pixels: a set/list of (x, y) tuples provided directly
      2. noisy_pixels_file: path to a text file with 'x y' per line
      3. Auto-detection via find_noisy_pixels() using per-map sigma thresholds

    If none of the above yields any noisy pixels, the file is copied unchanged.

    Args:
        input_path:           Path to the source .clog file.
        output_path:          Path for the output file (default: input with '_noisefree' suffix).
        npix:                 Number of pixels per side (default 256).
        noisy_pixels:         Set/list of (x,y) tuples to remove.
        noisy_pixels_file:    Path to a text file listing noisy pixel coordinates.
        outlier_sigma_counts: Sigma threshold for hit count outliers (default None).
        outlier_sigma_energy: Sigma threshold for total energy outliers (default None).
        outlier_sigma_toa:    Sigma threshold for median ToA outliers (default None).
                              Set at least one to enable auto-detection.

    Returns:
        (kept, total) cluster counts.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_noisefree{ext}"

    # Build the noisy pixel set
    bad = set()
    if noisy_pixels is not None:
        bad |= set(noisy_pixels)
        global_log.info(f"User-provided noisy pixels: {len(noisy_pixels)}")
    if noisy_pixels_file is not None:
        bad |= load_noisy_pixels_from_file(noisy_pixels_file)
    if outlier_sigma_counts is not None or outlier_sigma_energy is not None or outlier_sigma_toa is not None:
        bad |= find_noisy_pixels(input_path, npix=npix,
                                 outlier_sigma_counts=outlier_sigma_counts or float('inf'),
                                 outlier_sigma_energy=outlier_sigma_energy or float('inf'),
                                 outlier_sigma_toa=outlier_sigma_toa or float('inf'))

    if not bad:
        global_log.info("No noisy pixels found, nothing to filter.")
        import shutil
        shutil.copy2(input_path, output_path)
        return 0, 0

    global_log.info(f"Filtering {len(bad)} noisy pixel(s): {sorted(bad)[:20]}{'...' if len(bad) > 20 else ''}")

    # Stream-filter
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
            touches_noisy = False
            for g in groups:
                parts = g.split(b',')
                if len(parts) < 4:
                    continue
                try:
                    x = int(float(parts[0]))
                    y = int(float(parts[1]))
                except ValueError:
                    continue
                if (x, y) in bad:
                    touches_noisy = True
                    break

            if not touches_noisy:
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


@log_offline_process('plot_clog_pixel_maps', input_type='file')
def plot_clog_pixel_maps(file_path, npix=256, max_lines=None, max_bytes=None, cmap='jet',
                         log_scale=False, show_1d=False):
    """
    Read a clog file and display three 2D maps side by side:
      1. Hit count per pixel
      2. Total energy per pixel (keV)
      3. Median ToA per pixel (ns)

    Streams the file so memory stays bounded even for very large files.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    counts = np.zeros((npix, npix), dtype=np.int64)
    energy = np.zeros((npix, npix), dtype=np.float64)
    toa_lists = [[[] for _ in range(npix)] for _ in range(npix)]

    for hits, frame_time in _parse_clog(file_path, max_lines, max_bytes):
        t_offset = frame_time if frame_time is not None else 0.0
        for x, y, e, t in hits:
            xi, yi = int(x), int(y)
            if 0 <= xi < npix and 0 <= yi < npix:
                counts[yi, xi] += 1
                energy[yi, xi] += e
                toa_lists[yi][xi].append(t_offset + t)

    median_toa = np.full((npix, npix), np.nan)
    for yi in range(npix):
        for xi in range(npix):
            if toa_lists[yi][xi]:
                median_toa[yi, xi] = np.median(toa_lists[yi][xi])


    nrows = 2 if show_1d else 1
    ratios = {'height_ratios': [3, 1]} if show_1d else {}
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 10 if show_1d else 6),
                             gridspec_kw=ratios, squeeze=False)

    titles = ['Hit count per pixel', 'Total energy per pixel (keV)', 'Median ToA per pixel (ns)']
    data = [counts, energy, median_toa]

    for col, (d, title) in enumerate(zip(data, titles)):
        ax2d = axes[0, col]

        valid = d[np.isfinite(d) & (d > 0)]
        if log_scale and valid.size > 0:
            norm = LogNorm(vmin=max(valid.min(), 1e-10), vmax=valid.max())
        else:
            norm = None
        im = ax2d.imshow(d, origin='lower', aspect='equal', cmap=cmap, norm=norm)
        ax2d.set_title(title)
        ax2d.set_xlabel('X (pixel)')
        ax2d.set_ylabel('Y (pixel)')
        fig.colorbar(im, ax=ax2d, shrink=0.8)

        if show_1d:
            ax1d = axes[1, col]
            # Vectorized: build 1D pixel IDs = xi * npix + yi
            xi_grid, yi_grid = np.meshgrid(np.arange(npix), np.arange(npix))
            pids = (xi_grid * npix + yi_grid).ravel()
            flat = d.ravel()
            # Use fill_between for fast rendering (bar is too slow for 65K points)
            order = np.argsort(pids)
            pids, flat = pids[order], flat[order]
            ax1d.fill_between(pids, flat, step='mid', color='steelblue', alpha=0.8)
            if log_scale:
                ax1d.set_yscale('log')
            ax1d.set_xlabel('Pixel ID')
            ax1d.set_ylabel(title.split(' per pixel')[0])
            ax1d.set_xlim(0, npix * npix)

    fig.suptitle(str(file_path), fontsize=9)
    plt.tight_layout()
    plt.show()
