# Functions to process pixelClusters dataframes
# Easy to read, but not performant. See pixelClusters_custom.py for faster clustering.
# Time of pixelHits2pixelClusters() is not linear with number of hits (e.g. 100k 200 sec, 1M 13000 sec on my machine)

import time
import pandas as pd
from tools.utils import get_pixID_2D, log_offline_process
from tools.pixelHits import PIXEL_ID, TOA, ENERGY_keV, EVENTID
import re

# Pixel cluster format definition
PIX_X_ID = 'X'  # pixel X index (starts from 0, bottom left)
PIX_Y_ID = 'Y'  # pixel Y index (starts from 0, bottom left)
SIZE = 'size'
DELTA_TOA = 'Delta_TOA'  # ns

def process_func(cluster, n_pixels):
    """
    X and Y are in the sensor's local coordinates system, as in Allpix2
    => origin = center of the lower-left pixel
    """
    cluster_total_energy = cluster[ENERGY_keV].sum()
    cluster_first_TOA = cluster[TOA].min()

    pixX, pixY = zip(*cluster[PIXEL_ID].apply(get_pixID_2D, args=(n_pixels,)))

    x = sum(pixX * cluster[ENERGY_keV]) / cluster_total_energy
    y = sum(pixY * cluster[ENERGY_keV]) / cluster_total_energy

    data = {
        PIX_X_ID: [x],
        PIX_Y_ID: [y],
        ENERGY_keV: [cluster_total_energy],
        TOA: [cluster_first_TOA]
    }
    if EVENTID in cluster.columns:
        data[EVENTID] = [int(cluster[EVENTID].min())]

    return pd.DataFrame(data)


def new_cluster(clust_list, cluster, hit, n_pixels):
    clust_list.append(process_func(cluster, n_pixels))
    new_cluster_df = pd.DataFrame([hit])
    new_time_window_start = hit[TOA]
    return new_cluster_df, new_time_window_start


def is_adjacent(hit, cluster, n_pix):
    x1, y1 = get_pixID_2D(hit[PIXEL_ID], n_pix)
    return any(
        abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1
        for x2, y2 in
        (get_pixID_2D(hit[PIXEL_ID], n_pix) for _, hit in cluster.iterrows())
    )


@log_offline_process('pixelClusters', input_type = 'dataframe')
def pixelHits2pixelClusters(pixelHits, npix, window_ns):

    # Initialization
    clusters = []
    pixelHits = pixelHits.sort_values(by=TOA)
    hits_df = pixelHits.copy()
    hits_df.index = range(len(hits_df))  # Ensure integer index

    # 1st cluster starts with 1st hit
    clust = pd.DataFrame([pixelHits.iloc[0]])  # clust is a cluster being built
    wst = pixelHits.iloc[0][TOA]  # window start

    # Loop over hits
    for index, hit in pixelHits.iloc[1:].iterrows():
        if hit[TOA] - wst <= window_ns and is_adjacent(hit, clust, npix):
            clust = pd.concat([clust, hit.to_frame().T], ignore_index=True)
        else:
            clust, wst = new_cluster(clusters, clust, hit, npix)

    # Last cluster
    new_cluster(clusters, clust, hit, npix)

    df = pd.concat(clusters, ignore_index=True)

    return df


def clog2pixelClusters(file_path):
    """
    Convert a clog file from the Pixet software (Advacam) to a DataFrame of pixel clusters.
    See: https://wiki.advacam.cz/index.php/PIXet
    TODO: validate
    """
    events = []
    frame_time = None
    frame_re = re.compile(r'^Frame\s+\d+\s+\(\s*([^,]+)\s*,')
    bracket_re = re.compile(r'\[([^\]]+)\]')

    with open(file_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = frame_re.match(line)
            if m:
                # parse frame start time (ns)
                try:
                    frame_time = float(m.group(1))
                except ValueError:
                    frame_time = 0.0
                continue

            # line is an event: find all bracket groups
            groups = bracket_re.findall(line)
            if not groups:
                continue

            hits = []
            for g in groups:
                parts = [p.strip() for p in g.split(',')]
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

            if not hits:
                continue

            # compute totals and weighted coords
            total_e = sum(h[2] for h in hits)
            if total_e == 0:
                # avoid division by zero: fall back to mean pixel coords
                x_w = sum(h[0] for h in hits) / len(hits)
                y_w = sum(h[1] for h in hits) / len(hits)
            else:
                x_w = sum(h[0] * h[2] for h in hits) / total_e
                y_w = sum(h[1] * h[2] for h in hits) / total_e

            toa = (frame_time if frame_time is not None else 0.0) + min(h[3] for h in hits)

            events.append({
                'X': x_w,
                'Y': y_w,
                'Energy (keV)': total_e,
                'ToA (ns)': toa
            })

    df = pd.DataFrame(events, columns=['X', 'Y', 'Energy (keV)', 'ToA (ns)'])
    return df
