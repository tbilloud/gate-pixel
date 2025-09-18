import matplotlib.pyplot as plt
import numpy as np
from opengate.logger import global_log
from tools.pixelHits import ENERGY_keV


def plot_energies(
        *,
        max_keV,
        min_keV=0,
        hits_list=None,
        clusters_list=None,
        CCevents_list=None,
        names=None,
        colors=None,
        alphas=None,
        max_y=None,
        ylog=False,  # NEW: enable log scale on Y axis
        output_filename=None,
):
    """
    Plots histograms of multiple pixel hits and clusters vertically.
    Each dataset is overlaid with a different color and label.

    Args:
        max_keV (int): Maximum energy (keV) for histogram range and bins.
        min_keV (int): Minimum energy (keV) for histogram range and bins.
        hits_list (list): List of DataFrames for pixel hits.
        CCevents_list (list): List of DataFrames for Compton camera events.
        names (list): List of labels for each dataset.
        colors (list): List of colors for each dataset.
        alphas (list): List of alpha values for each dataset.
        max_y (float or int, optional): Maximum y-axis value for both plots.
        ylog (bool): If True, use logarithmic scale on the y-axis.
    """

    if hits_list is None or clusters_list is None:
        raise ValueError(
            "hits_list and clusters_list must be provided as lists of DataFrames.")
    n = len(hits_list)
    if names is None:
        names = [f"Dataset {i + 1}" for i in range(n)]
    if colors is None:
        colors = [None] * n
    if alphas is None:
        alphas = [0.7] * n

    nr = 3 if CCevents_list else 2
    fig, axes = plt.subplots(nrows=nr, ncols=1, figsize=(8, 10), sharex=True)

    def plot_histogram(ax, data_list, title, xlab=False):
        for data, name, color, alpha in zip(data_list, names, colors, alphas):
            if data.empty:
                print(f"Warning: empty dataframe for plot '{title}'.")
            else:
                ax.hist(
                    data[ENERGY_keV],
                    bins=max_keV - min_keV,
                    range=(min_keV, max_keV),
                    alpha=alpha,
                    color=color,
                    label=name,
                    log=ylog,  # NEW: request log scaling in histogram
                )
        if xlab:
            ax.set_xlabel(ENERGY_keV)
        ax.set_ylabel('Counts')
        ax.set_title(title)
        if len(data_list) > 1:
            ax.legend()
        ax.grid(True, which='both', alpha=0.4)
        if ylog:
            ax.set_yscale(
                'log')  # ensure axis is log even if all counts were zero in a call
        if max_y is not None:
            ax.set_ylim(top=max_y)

    plot_histogram(axes[0], hits_list, title='Pixel Hits')
    plot_histogram(axes[1], clusters_list, title='Pixel Clusters',
                   xlab=False if CCevents_list else True)
    if CCevents_list:
        for df in CCevents_list:
            df[ENERGY_keV] = df['Energy (keV)_1'] + df['Energy (keV)_2']
        plot_histogram(axes[2], CCevents_list, title='CCevents sum', xlab=True)

    if output_filename:
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    else:
        plt.tight_layout()
        plt.show()

    return fig, axes


# Python
def _mark_detector_in_volume(vol, size, position, vpitch):
    center = np.array(vol.shape) // 2
    start = ((np.array(position) - np.array(size) / 2) / vpitch + center).astype(int)
    end = ((np.array(position) + np.array(size) / 2) / vpitch + center).astype(int)
    start = np.maximum(start, 0)
    end = np.minimum(end, vol.shape)
    start = [int(x) for x in start]
    end = [int(x) for x in end]
    end = [e + 1 if e == s else e for s, e in zip(start, end)]
    vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = np.inf


def _setup_napari_viewer(viewer, axes_order, orientation2d):
    viewer.axes.visible = True
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'mm'
    viewer.scale_bar.length = 10
    viewer.scale_bar.font_size = 20
    viewer.scale_bar.colored = True
    viewer.scale_bar.color = 'red'
    viewer.scale_bar.position = 'bottom_center'
    viewer.dims.order = axes_order
    viewer.camera.orientation2d = orientation2d


def plot_reco(vol, vpitch, detector=False, colormap='gray_r', axes_order=(0, 1, 2),
              orientation2d=('up', 'right')):
    try:
        import napari
    except ImportError:
        global_log.warning("Napari is not installed, cannot use plot_reco.")
        return

    if detector:
        _mark_detector_in_volume(vol, detector['size'], detector['position'], vpitch)

    viewer = napari.view_image(
        vol,
        translate=tuple(-(v * vpitch) / 2 for v in vol.shape),
        axis_labels=['x', 'y', 'z'],
        scale=[vpitch, vpitch, vpitch],
        colormap=colormap
    )
    _setup_napari_viewer(viewer, axes_order, orientation2d)
    napari.run()


def plot_recos(
        vols,
        vpitch,
        detectors=None,
        axes_order=(0, 1, 2),
        orientation2d=("up", "right"),
        spacing_mm=5.0,
        colormap="gray_r",
        names=None,
):
    try:
        import napari
    except ImportError:
        global_log.warning("Napari is not installed, cannot use plot_reco.")
        return

    n = len(vols)
    if detectors is None:
        detectors = [False] * n
    if len(detectors) != n:
        raise ValueError("`detectors` must be a list with the same length as `vols`.")
    if names is None:
        names = [f"vol_{i + 1}" for i in range(n)]

    pitch = (float(vpitch),) * 3 if np.isscalar(vpitch) else tuple(map(float, vpitch))
    vsize = vols[0].shape
    base_translate = tuple(-(s * p) / 2.0 for s, p in zip(vsize, pitch))

    viewer = napari.Viewer()

    for i, (vol, det, name) in enumerate(zip(vols, detectors, names)):
        vol_i = np.array(vol, copy=True)
        if det:
            _mark_detector_in_volume(vol_i, det["size"], det["position"], pitch)
        translate_i = list(base_translate)
        translate_i[0] += i * (vsize[0] * pitch[0] + float(spacing_mm))
        viewer.add_image(
            vol_i,
            name=name,
            translate=tuple(translate_i),
            scale=pitch,
            colormap=colormap,
        )

    _setup_napari_viewer(viewer, axes_order, orientation2d)
    napari.run()


def compare_clusters(
        clusters1, clusters2, name_a='A', name_b='B', xlog=False, ylog=False, npix=256,
        energy_bins=None, toa_bins=None, dtoa_bins=None, size_bins=None,
        energy_range=None, toa_range=None, dtoa_range=None, size_range=None
):
    from tools.pixelHits import ENERGY_keV, TOA
    from tools.pixelClusters import PIX_X_ID, PIX_Y_ID, SIZE, DELTA_TOA
    props = [
        (ENERGY_keV, energy_bins, energy_range),
        (TOA, toa_bins, toa_range),
        (DELTA_TOA, dtoa_bins, dtoa_range),
        (SIZE, size_bins, size_range),
        (PIX_X_ID, npix, (0, npix)),
        (PIX_Y_ID, npix, (0, npix)),
    ]
    fig, axes = plt.subplots(nrows=len(props), ncols=1, figsize=(6, 10))
    for i, (col, bins, rng) in enumerate(props):
        ax = axes[i]
        ax.hist(
            [clusters1[col], clusters2[col]],
            bins=bins,
            range=rng,
            alpha=0.5,
            label=[name_a, name_b],
            log=ylog,
            histtype='stepfilled'  # or 'step'
        )
        ax.set_xlabel(f'{col}')
        ax.set_ylabel('Counts')
        ax.legend()
        ax.grid(True, which='both', alpha=0.4)
        if xlog and col not in [PIX_X_ID, PIX_Y_ID]:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
    plt.tight_layout()
    plt.show()
    return fig, axes
