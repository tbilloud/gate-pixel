# Functions to process pixelClusters dataframes
# Easy to read, but not performant. See pixelClusters_custom.py for faster clustering.
# Time of pixelHits2pixelClusters() is not linear with number of hits (e.g. 100k 200 sec, 1M 13000 sec on my machine)

import os
import humanize
import pandas as pd
from tools.utils import get_pixID_2D, log_offline_process
from tools.logging_custom import global_log
from tools.pixelHits import PIXEL_ID, TOA, ENERGY_keV, EVENTID
import re

# Pixel cluster format definition
PIX_X_ID = 'X'  # pixel X index (starts from 0, bottom left)
PIX_Y_ID = 'Y'  # pixel Y index (starts from 0, bottom left)
SIZE = 'size'
DELTA_TOA = 'Delta_TOA'  # ns

def pixelHits2onePixelCluster(cluster, n_pixels):
    """
    X and Y are in the sensor's local coordinates system, as in Allpix2
    => origin = center of the lower-left pixel
    """
    cluster_total_energy = cluster[ENERGY_keV].sum()
    cluster_first_TOA = cluster[TOA].min()

    pixX, pixY = zip(*cluster[PIXEL_ID].apply(get_pixID_2D, args=(n_pixels,)))

    x = sum(pixX * cluster[ENERGY_keV]) / cluster_total_energy
    y = sum(pixY * cluster[ENERGY_keV]) / cluster_total_energy

    size = len(cluster)
    delta_toa = cluster[TOA].max() - cluster_first_TOA if size > 1 else float('nan')

    data = {
        PIX_X_ID: [x],
        PIX_Y_ID: [y],
        ENERGY_keV: [cluster_total_energy],
        TOA: [cluster_first_TOA],
        SIZE: [size],
        DELTA_TOA: [delta_toa],
    }
    if EVENTID in cluster.columns:
        data[EVENTID] = [int(cluster[EVENTID].min())]

    return pd.DataFrame(data)


def is_adjacent(hit, cluster, n_pix):
    x1, y1 = get_pixID_2D(hit[PIXEL_ID], n_pix)
    return any(
        abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1
        for x2, y2 in
        (get_pixID_2D(hit[PIXEL_ID], n_pix) for _, hit in cluster.iterrows())
    )


@log_offline_process('pixelClusters', input_type = 'dataframe')
def pixelHits2pixelClusters(pixelHits, npix, window_ns):
    """
    Simple clustering prototype for demo, but:
    - It's slow
    - If hit A and hit C are not adjacent, but hit B (arriving later) bridges them, A and C will end up in separate clusters.
    - the time window is relative to the TOA of the first hit in the cluster -> better use a rolling window
    => Better use pixelClusters_custom.py
    """

    # Initialization
    clusters = []
    sorted_hits = pixelHits.sort_values(by=TOA).reset_index(drop=True)

    def new_cluster(clust_list, cluster, hit, n_pixels):
        clust_list.append(pixelHits2onePixelCluster(cluster, n_pixels))
        new_cluster_df = pd.DataFrame([hit])
        new_time_window_start = hit[TOA]
        return new_cluster_df, new_time_window_start

    # 1st cluster starts with 1st hit
    clust = pd.DataFrame([sorted_hits.iloc[0]])  # clust is a cluster being built
    wst = sorted_hits.iloc[0][TOA]  # window start

    # Loop over hits
    for index, hit in sorted_hits.iloc[1:].iterrows():
        if hit[TOA] - wst <= window_ns and is_adjacent(hit, clust, npix):
            clust = pd.concat([clust, hit.to_frame().T], ignore_index=True)
        else:
            clust, wst = new_cluster(clusters, clust, hit, npix)

    # Last cluster
    clusters.append(pixelHits2onePixelCluster(clust, npix))

    df = pd.concat(clusters, ignore_index=True)

    return df


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


def clog2pixelClusters(file_path, max_lines=None, max_bytes=None, omit_border=False, border_values=(1, 256),
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
