import math
import sys
import numpy as np
from tools.logging_custom import global_log
from tools.pixelHits import ENERGY_keV
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

if sys.platform == "darwin": matplotlib.use("macosx")


def plot_energies(hits_list, clusters_list, CCevents_list, max_keV, min_keV=0,
                  names=None, colors=None, alphas=None, max_y=None, ylog=False,
                  output_filename=None,
                  ):
    """
    Plots histograms of multiple pixel hits and clusters vertically.
    Each dataset is overlaid with a different color and label.

    Args:
        max_keV (int): Maximum energy (keV) for histogram range and bins.
        min_keV (int): Minimum energy (keV) for histogram range and bins.
        hits_list (list): List of DataFrames for pixel hits.
        clusters_list (list): List of DataFrames for pixel clusters.
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
                global_log.warning(f"Warning: empty dataframe for plot '{title}'.")
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
        CCevents_list = [df.copy() for df in CCevents_list]
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


def plot_reco(
        vol,
        vpitch=None,
        sensor_size=None,
        sensor_translation=None,
        axes_order=(0, 1, 2),
        orientation2d=("up", "right"),
        colormap="gray_r",
        name=None,
):
    """
    Displays a single 3D volume with napari or matplotlib (if napari is not available).
    Can mark a sensor region if specified.

    Args:
        vol (np.ndarray): Volume to display.
        vpitch (float, optional): Voxel size (mm), used to show scale and sensor.
        sensor_size (list, optional): sensor info to mark.
        sensor_translation (list, optional): sensor info to mark.
        axes_order (tuple): Axes order for napari.
        orientation2d (tuple): 2D orientation for napari.
        colormap (str): Colormap for display.
        name (str, optional): Name of the volume.
    """

    try:
        import napari
    except ImportError:
        global_log.warning("Napari not installed, using matplotlib.")

        # 2D with slider
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        img = ax.imshow(vol[:, :, 0], cmap='gray')
        ax.set_title('z=0')
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'z', 0, vol.shape[2] - 1, valinit=0, valstep=1)

        def update(val):
            z = int(slider.val)
            img.set_data(vol[:, :, z])
            ax.set_title(f'z={z}')
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
        return

    def _mark_sensor_in_volume(vol, size, position, vpitch):
        center = np.array(vol.shape) // 2
        start = ((np.array(position) - np.array(size) / 2) / vpitch + center).astype(
            int)
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

    if vpitch:
        pitch = (float(vpitch),) * 3
        vsize = vol.shape
        base_translate = tuple(-(s * p) / 2.0 for s, p in zip(vsize, pitch))

    viewer = napari.Viewer()
    vol_i = np.array(vol, copy=True)
    if sensor_size is not None and sensor_translation is not None:
        if vpitch:
            _mark_sensor_in_volume(vol_i, sensor_size, sensor_translation, pitch)
        else:
            global_log.error("vpitch must be specified to mark sensor.")
    viewer.add_image(
        vol_i,
        name=name or "volume",
        translate=base_translate if vpitch else None,
        scale=pitch if vpitch else None,
        colormap=colormap,
    )
    _setup_napari_viewer(viewer, axes_order, orientation2d)
    napari.run()


def compare_pixelClusters(
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


def compare_recos(volumes, names=None, slice_axis=2):
    """
    Compare multiple volumes side-by-side.
    For each volume, 5 slices along the specified axis are plotted.
    Volumes are normalized to their maximum intensity value.

    Args:
        volumes (list of np.ndarray): List of 3D volumes to compare.
        names (list of str, optional): List of names for each volume.
        slice_axis (int): Axis along which to take slices (default: 2).
    """
    if len({vol.shape for vol in volumes}) != 1:
        raise ValueError("In volumes must have the same shape.")

    volumes = [vol / np.max(vol) if np.max(vol) else vol for vol in volumes]
    names = names if names else [f'Volume {i + 1}' for i in range(len(volumes))]
    axis_size = volumes[0].shape[slice_axis]
    slices = np.linspace(0, axis_size - 1, 5, dtype=int).tolist()
    vmin = min(np.nanmin(vol) for vol in volumes)
    vmax = max(np.nanmax(vol) for vol in volumes)
    fig, axes = plt.subplots(len(slices), len(volumes), figsize=(8, 3 * len(slices)))
    if len(volumes) == 1:
        axes = np.array(axes).reshape(-1, 1)
    for i, idx in enumerate(slices):
        for j, (vol, name) in enumerate(zip(volumes, names)):
            ax = axes[i, j]
            if slice_axis == 0:
                img = vol[idx, :, :]
            elif slice_axis == 1:
                img = vol[:, idx, :]
            else:
                img = vol[:, :, idx]
            ax.imshow(img, cmap='inferno', vmin=vmin, vmax=vmax)
            ax.set_title(f'{name} - {["x", "y", "z"][slice_axis]}={idx}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def add_line_to_spectrum(ax, text, energy, color, fontsize=12, rotation=45):
    """
    Adds a vertical line and a label to a spectrum plot. Useful to mark known emission lines.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib Axes object to draw on.
        text (str): The label text to display near the line.
        energy (float): The x-coordinate (energy value) where the line is drawn.
        color (str): The color of the line and text.
        fontsize (int, optional): The font size of the label text. Default is 12.
        rotation (int, optional): The rotation angle of the label text. Default is 45.
    """
    ax.axvline(energy, color=color, linestyle='--', linewidth=1.2, alpha=0.6)
    ax.text(energy, -0.02, text,
            transform=ax.get_xaxis_transform(),
            color=color, fontsize=fontsize, rotation=rotation,
            ha='center', va='top', clip_on=False)


def plot_decay_products(df_hits, min_keV=1, max_keV=np.inf, bins=100, hist_range_keV=None, figsize=(10, 6)):
    """
    Plot spectra of decay products from an isotope source.

    Use case: run a simulation with a large world made of a high-Z material surrounding the source, e.g.:
        sim = Simulation()
        sim.physics_manager.enable_decay = True
        sim.world.material = "G4_Ac"
        sim.world.size = [1 * g4_units.m, 1 * g4_units.m, 1 * g4_units.m]
        hits = sim.add_actor('DigitizerHitsCollectionActor', 'Hits')
        hits.attached_to = sim.world
        hits.attributes = ['EventID', 'TrackID', 'ParticleName', 'TrackCreatorProcess', 'KineticEnergy']
        hits.output_filename = 'test.root'
        source = sim.add_source("GenericSource", "source")
        #source.particle, source.half_life = "ion 92 238", 4.463e9 * g4_units.year  # U238
        source.particle, source.half_life = "ion 71 177", 6.65 * g4_units.day # Lu177
        source.n = 1e4
        sim.run()
        df_hits = uproot.open('./test.root')[hits.name].arrays(library='pd')
        plot_radioactive_decay_spectra(df_hits, min_keV=0, hist_range_keV=[0, 1000], bins=1000)

    Notes:

        Some sources (e.g. U238) emit lots of:
         - Different excited states for the daughter nucleus(i), leading to a cluterred legend
         - low energy electrons (internal conversion, Auger)
         => Set min_keV > 1 to avoid

        In case of beta decays, one will see neutrinos if `hits.keep_zero_edep = True`


    """

    # build mask with NumPy arrays to avoid awkward-pandas issues
    mask = (df_hits['TrackCreatorProcess'].to_numpy() == 'RadioactiveDecay') & \
           (df_hits['KineticEnergy'].to_numpy() > (min_keV/1000)) & \
           (df_hits['KineticEnergy'].to_numpy() < (max_keV/1000))

    cols = ['EventID', 'TrackID', 'ParticleName', 'KineticEnergy']
    df_r = df_hits.loc[mask, cols].copy()
    if df_r.empty:
        print("No RadioactiveDecay tracks with KineticEnergy > 0 found.")
        return

    # ensure numeric float array for KineticEnergy
    df_r['KineticEnergy'] = df_r['KineticEnergy'].to_numpy().astype(float)

    # for each track take the step with the highest kinetic energy
    idx = df_r.groupby(['EventID', 'TrackID'])['KineticEnergy'].idxmax()
    df_first = df_r.loc[idx].reset_index(drop=True)
    df_first['KineticEnergy'] = df_first['KineticEnergy'].mul(1000)

    # prepare bins / range
    ke_min, ke_max = df_first['KineticEnergy'].min(), df_first['KineticEnergy'].max()
    if hist_range_keV is None:
        hist_range_keV = (max(0.0, ke_min * 0.9), ke_max * 1.1)
    bin_edges = np.linspace(hist_range_keV[0], hist_range_keV[1], bins + 1)

    # plot overall histogram (filled, faint)
    plt.figure(figsize=figsize)
    plt.hist(df_first['KineticEnergy'].to_numpy(), bins=bin_edges,
             histtype='stepfilled', alpha=0.18, label='all particles')

    # overlay per-ParticleName histograms (e- in red and gamma in green, as in default opengate visualization)
    for pname, grp in df_first.groupby('ParticleName'):
        data = grp['KineticEnergy'].to_numpy()
        if data.size == 0:
            continue
        color = None
        if pname == 'e-':
            color = 'red'
        elif pname == 'gamma':
            color = 'green'
        plt.hist(data, bins=bin_edges, histtype='step', linewidth=1.5, label=str(pname), color=color)

    plt.xlim(hist_range_keV[0], hist_range_keV[1])
    plt.xlabel('KineticEnergy (keV)')
    plt.ylabel('Counts')
    plt.title('Kinetic energy of tracks from RadioactiveDecay (first/highest step per track)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_hist_by_time(df, interval_ns, bins=100, x_range=None, max_plots=20, ncols=4, cmap=plt.cm.viridis):
    """
    For pixelClusters, plot energy histograms for each ToA time interval of width `interval_ns` (ns).
    Useful in case of long measurements with time-varying conditions (e.g. source decay, sensor instability, heating).

    - df: DataFrame with columns 'Energy (keV)' and 'ToA (ns)'.
    - interval_ns: width of each time interval in ns (float).
    - bins: histogram bins (int or sequence).
    - x_range: tuple (xmin, xmax) for histogram x-axis; None to auto.
    - max_plots: maximum number of subplots to draw (skips later intervals).
    - ncols: number of columns in subplot grid.
    Returns the matplotlib Figure and list of Axes.
    """
    toa = df['ToA (ns)'].to_numpy()
    energies = df['Energy (keV)'].to_numpy()

    tmin = float(np.nanmin(toa))
    tmax = float(np.nanmax(toa))
    # build interval edges (right open)
    edges = np.arange(tmin, tmax + interval_ns, interval_ns)
    if len(edges) < 2:
        raise ValueError("interval_ns too large or DataFrame ToA has insufficient range")

    # assign interval index
    idx = np.digitize(toa, edges, right=False) - 1  # 0-based interval index
    n_intervals = len(edges) - 1

    # prepare plotting
    interval_plots = []
    for i in range(n_intervals):
        mask = (idx == i)
        if not np.any(mask):
            continue
        interval_plots.append((i, edges[i], edges[i+1], energies[mask]))

    if not interval_plots:
        raise ValueError("No events found in any interval")

    # limit number of plots
    interval_plots = interval_plots[:max_plots]
    nplots = len(interval_plots)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
    axes_flat = axes.flatten()

    cmap_vals = cmap(np.linspace(0, 1, nplots))
    for ax, (k, t0, t1, ev) in zip(axes_flat, interval_plots):
        ax.hist(ev, bins=bins, range=x_range, color=cmap_vals[k % len(cmap_vals)], edgecolor='k', alpha=0.8)
        ax.set_title(f"t ∈ [{t0:.1f}, {t1:.1f}) ns\nN={len(ev)}")
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Counts')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # hide unused axes
    for ax in axes_flat[nplots:]:
        ax.axis('off')

    plt.tight_layout()
    return fig, axes_flat[:nplots]
