# Functions to process pixelClusters dataframes
# Easy to read, but not performant. See pixelClusters_custom.py for faster clustering.
# Time of pixelHits2pixelClusters() is not linear with number of hits (e.g. 100k 200 sec, 1M 13000 sec on my machine)

import pandas as pd
from tools.utils import get_pixID_2D, log_offline_process
from tools.pixelHits import PIXEL_ID, TOA, ENERGY_keV, EVENTID

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


