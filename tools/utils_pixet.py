import os
import re
import time
from collections import namedtuple
from pathlib import Path

import humanize
import numpy as np
import pandas as pd
from loguru import logger

from tools.logging_custom import global_log

from tools.pixelClusters import PIX_X_ID, PIX_Y_ID, SIZE, DELTA_TOA
from tools.pixelHits import ENERGY_keV, TOA, PIXEL_ID
from tools.utils import log_offline_process, get_pixID


FrameInfo = namedtuple('FrameInfo', ['frame_num', 'time_ns', 'time_s'])

def _parse_clog(file_path, max_lines=None, max_bytes=None):
    """
    Core clog parser. Yields (hits, frame_info) for each cluster line, where
    hits is a list of (x, y, energy, toa) tuples and frame_info is a FrameInfo
    namedtuple with fields:
      - frame_num (int):  frame index from the header line
      - time_ns  (float): first timestamp in the header (nanoseconds)
      - time_s   (float): second timestamp in the header (seconds)

    Example frame header:
        Frame 29903 (688317573795.312500, 0.000000 s)

    Warning:
         - sometimes TOA values are negative
         - the frame time is reset at regular intervals, but not to zero
    """
    frame_info = None
    frame_re = re.compile(rb'^Frame\s+(\d+)\s+\(\s*([^,]+)\s*,\s*([^\s)]+)')
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
                    frame_info = FrameInfo(
                        frame_num=int(m.group(1)),
                        time_ns=float(m.group(2)),
                        time_s=float(m.group(3)),
                    )
                except ValueError:
                    frame_info = FrameInfo(frame_num=0, time_ns=0.0, time_s=0.0)
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
                yield hits, frame_info

def debug_clog(file_path):
    """
    Scan a clog file and report:
      1. Frame number gaps (not incremented by exactly 1)
      2. Time resets (frame time_ns decreasing vs previous frame)
    """
    prev = None
    n_clusters = 0
    frame_count = 0  # number of unique frames seen
    frames_since_reset = 0
    gaps = []
    resets = []

    # Track per-frame info (one entry per frame, not per cluster)
    seen_frames = {}  # frame_num -> frame_info, to avoid counting duplicates

    for hits, frame_info in _parse_clog(file_path):
        n_clusters += 1
        fn = frame_info.frame_num

        if fn not in seen_frames:
            seen_frames[fn] = frame_info
            frame_count += 1
            frames_since_reset += 1

            if prev is not None:
                # Check frame_num increment
                expected = prev.frame_num + 1
                if fn != expected:
                    gaps.append({'at_frame': fn, 'expected': expected, 'got': fn, 'skip': fn - expected})

                # Check time monotonicity
                if frame_info.time_ns < prev.time_ns:
                    resets.append({
                        'at_frame': fn,
                        'frames_since_last_reset': frames_since_reset,
                        'prev_time_ns': prev.time_ns,
                        'new_time_ns': frame_info.time_ns,
                        'decrease_ns': prev.time_ns - frame_info.time_ns,
                        'decrease_s': (prev.time_ns - frame_info.time_ns) / 1e9,
                    })
                    frames_since_reset = 0

            prev = frame_info

    print(f"Total clusters: {n_clusters}, unique frames: {frame_count} (frames contain coincident clusters)")

    if gaps:
        print(f"\nFrame number gaps ({len(gaps)} total):")
        for g in gaps:
            print(f"  Frame {g['at_frame']}: expected {g['expected']}, skipped {g['skip']} frame(s)")
    else:
        print("Frame numbers: OK (always +1)")

    if resets:
        print(f"\nTime resets ({len(resets)} total):")
        for r in resets:
            print(f"  Frame {r['at_frame']} (after {r['frames_since_last_reset']} frames since last reset): "
                  f"{r['prev_time_ns']:.3f} ns → {r['new_time_ns']:.3f} ns "
                  f"(decrease: {r['decrease_ns']:.3f} ns / {r['decrease_s']:.6f} s)")
    else:
        print("Frame times: OK (always increasing)")

    # Plot histogram of time deltas between consecutive frames
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    sorted_frames = sorted(seen_frames.values(), key=lambda fi: fi.frame_num)
    frame_indices = np.array([fi.frame_num for fi in sorted_frames])
    times_ns = np.array([fi.time_ns for fi in sorted_frames])
    deltas_ns = np.diff(times_ns)
    delta_frame_indices = frame_indices[1:]  # x-axis for the delta-vs-index plot

    # Exclude negative deltas (resets) from histograms
    pos_mask = deltas_ns > 0
    pos_deltas = deltas_ns[pos_mask]
    n_resets = int((~pos_mask).sum())
    reset_label = f'  [{n_resets} resets excluded]' if n_resets else ''

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0: frame time vs frame index (full width)
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.plot(frame_indices, times_ns / 1e9, '.', markersize=1, alpha=0.5, color='steelblue')
    ax_t.set_xlabel('Frame index')
    ax_t.set_ylabel('Frame time (s)')
    ax_t.set_title('Frame time vs frame index')
    ax_t.grid(True, alpha=0.3)

    hist_configs = [
        (gs[1, 0], 'linear', 'log'),
        (gs[1, 1], 'linear', 'linear'),
    ]
    for spec, xscale, yscale in hist_configs:
        ax = fig.add_subplot(spec)
        ax.hist(pos_deltas / 1e6, bins=200, color='steelblue', alpha=0.8)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel('Frame time delta (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'Inter-frame Δt histogram {reset_label}')
        ax.grid(True, alpha=0.3)

    # Row 2: Δt vs frame index (positive deltas only)
    ax_dt = fig.add_subplot(gs[2, :])
    ax_dt.plot(delta_frame_indices[pos_mask], pos_deltas / 1e6,
               '.', markersize=1, alpha=0.5, color='steelblue')
    ax_dt.set_xlabel('Frame index')
    ax_dt.set_ylabel('Frame time delta (ms)')
    ax_dt.set_title('Inter-frame Δt vs frame index')
    ax_dt.grid(True, alpha=0.3)

    # Row 3: resets vs frame index
    ax_rst = fig.add_subplot(gs[3, :])
    if n_resets:
        neg_mask = ~pos_mask
        ax_rst.plot(delta_frame_indices[neg_mask], deltas_ns[neg_mask] / 1e9,
                    'rx', markersize=6)
    ax_rst.set_xlabel('Frame index')
    ax_rst.set_ylabel('Time decrease (s)')
    ax_rst.set_title(f'Time resets vs frame index ({n_resets} total)')
    ax_rst.grid(True, alpha=0.3)

    fig.suptitle(str(file_path), fontsize=9)
    plt.show()

@log_offline_process('clog2clogTimeFixed', input_type='file')
def clog2clogTimeFixed(input_path, output_path=None):
    """
    Correct frame-time resets in a clog file and write a new file with monotonically
    increasing frame timestamps.

    When a reset is detected (time_ns[i] < time_ns[i-1]), an offset equal to
    corrected[i-1] - time_ns[i] is added to all subsequent frames, making the
    corrected series continuous (zero gap at the reset point).

    Streams line-by-line so memory usage stays constant for large files.

    Args:
        input_path:   Path to the source .clog file.
        output_path:  Path for the corrected output (default: input with '_timefixed' suffix).

    Returns:
        Number of resets corrected.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_timefixed{ext}"

    # ── Pass 1: build per-frame correction offsets ──────────────────────────
    frame_re_parse = re.compile(rb'^Frame\s+(\d+)\s+\(\s*([^,]+)\s*,\s*([^\s)]+)')
    seen: dict[int, float] = {}   # frame_num -> original time_ns
    ordered: list[tuple[int, float]] = []  # (frame_num, time_ns) in encounter order

    with open(input_path, 'rb') as f:
        for raw in f:
            m = frame_re_parse.match(raw.strip())
            if m:
                fn = int(m.group(1))
                t  = float(m.group(2))
                if fn not in seen:
                    seen[fn] = t
                    ordered.append((fn, t))

    # compute cumulative offsets
    offsets: dict[int, float] = {}  # frame_num -> offset to add
    cumulative_offset = 0.0
    prev_corrected = None
    for fn, t_orig in ordered:
        if prev_corrected is not None and (t_orig + cumulative_offset) < prev_corrected:
            cumulative_offset += prev_corrected - (t_orig + cumulative_offset)
        offsets[fn] = cumulative_offset
        prev_corrected = t_orig + cumulative_offset

    n_resets = sum(1 for o in offsets.values() if o > 0)
    global_log.info(f"clog2clogTimeFixed: {n_resets} reset(s) to correct")

    # ── Pass 2: stream-copy, replacing timestamps in Frame header lines ──────
    frame_re_sub = re.compile(rb'^(Frame\s+\d+\s+\()([^,]+)(,.+)$')
    in_size = os.path.getsize(input_path)

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for raw in fin:
            m = frame_re_parse.match(raw.strip())
            if m:
                fn = int(m.group(1))
                t_orig = float(m.group(2))
                corrected_t = t_orig + offsets.get(fn, 0.0)
                # replace only the first timestamp field
                raw = frame_re_sub.sub(
                    lambda mo, ct=corrected_t: mo.group(1) + f'{ct:.6f}'.encode() + mo.group(3),
                    raw.rstrip(b'\n')
                ) + b'\n'
            fout.write(raw)

    out_size = os.path.getsize(output_path)
    global_log.info(
        f"clog2clogTimeFixed: {humanize.naturalsize(in_size)} → {humanize.naturalsize(out_size)}, "
        f"{n_resets} reset(s) corrected")
    return n_resets


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

    for hits, frame_info in _parse_clog(file_path, max_lines, max_bytes):
        total_e = sum(h[2] for h in hits)
        if total_e == 0:
            x_w = sum(h[0] for h in hits) / len(hits)
            y_w = sum(h[1] for h in hits) / len(hits)
        else:
            x_w = sum(h[0] * h[2] for h in hits) / total_e
            y_w = sum(h[1] * h[2] for h in hits) / total_e

        if omit_border and (int(round(x_w)) in border_set or int(round(y_w)) in border_set):
            continue

        toa = (frame_info.time_ns if frame_info is not None else 0.0) + min(h[3] for h in hits)
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

    for hits, frame_info in _parse_clog(file_path, max_lines, max_bytes):
        t_offset = frame_info.time_ns if frame_info is not None else 0.0
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

    for hits, frame_info in _parse_clog(file_path, max_lines, max_bytes):
        t_offset = frame_info.time_ns if frame_info is not None else 0.0
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

    for hits, frame_info in _parse_clog(file_path, max_lines, max_bytes):
        t_offset = frame_info.time_ns if frame_info is not None else 0.0
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
