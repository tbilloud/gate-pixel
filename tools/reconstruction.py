# Reconstruction functions based on different methods

import time
from pathlib import Path
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

from tools.CCevents import select_CCevents_energies
from tools.utils import get_stop_string
from tools.pixelHits import EVENTID
from tools.CCcones import APEX_X, APEX_Y, APEX_Z, DIRECTION_X, DIRECTION_Y, \
    DIRECTION_Z, COS, CCevents2CCcones

try:
    from opengate.logger import global_log
except ImportError:
    import logging

    global_log = logging.getLogger("dummy")
    global_log.addHandler(logging.NullHandler())


def reconstruct(CCevents, vpitch, vsize, energies_MeV=False, tol_MeV=0.01,
                cone_width=0.01, log=True, method="numpy",
                **kwargs):
    """
    Unified reconstruction interface.

     Args:
        CCevents (pandas.DataFrame): DataFrame containing the cone information used for validation.
        vpitch (float): The voxel size or pitch of the volume.
        vsize (tuple[int, int, int]): The size of the volume in voxels (X, Y, Z).
        cone_width (float): cone width (the larger the value the thicker the cones).
        log (bool): whether to log performance info
        method: "numpy", "cupy", "torch", "coresi" (see README for details)
        energies_MeV: (list[float]): select energy peaks, False to disable (default)
        tol_MeV: (float): if energy peaks are selected, tolerance in MeV
        kwargs: extra parameters for CoReSi:
            sensor_size (list[float])
            sensor_position (list[float]): must be same coord system as CCevents
            sensor_rotation (3×3 rotation matrix) TODO not implemented yet
    """

    if log:
        global_log.info(f"Offline [reconstruction][{method}]: START")
    stime = time.time()

    if CCevents is None or len(CCevents) == 0:
        global_log.error("Empty input.")
        global_log.info(f"Offline [reconstruction]: {get_stop_string(stime)}")
        return

    if method == "coresi":
        vol = reco_bp_coresi(CCevents, vpitch, vsize, cone_width * 300,
                             energies_MeV=energies_MeV, tol_MeV=tol_MeV, **kwargs)
    else:
        if energies_MeV:
            CCevents = select_CCevents_energies(CCevents, energies_MeV, tol_MeV)
        cones = CCevents2CCcones(CCevents, log=False)
        if method == "cupy":
            vol = reco_bp_cupy(cones, vpitch, vsize, cone_width)
        elif method == "torch":
            vol = reco_bp_torch(cones, vpitch, vsize, cone_width)
        elif method == "numpy":
            vol = reco_bp_numpy(cones, vpitch, vsize, cone_width)
        # elif method == "custom":
        #     from tools.reconstruction_custom import reco_custom
        #     vol = reco_custom(CCevents, vpitch, vsize, cone_width)
        else:
            raise ValueError(f"Unknown method: {method}")

    if log:
        global_log.info(f"Offline [reconstruction][{method}]: {get_stop_string(stime)}")

    return vol


def reco_bp_numpy(cones, vpitch, vsize, cone_width=0.01):
    volume = np.zeros(vsize, dtype=np.float32)
    grid_x = np.linspace(-vsize[0] // 2, vsize[0] // 2, vsize[0]) * vpitch
    grid_y = np.linspace(-vsize[1] // 2, vsize[1] // 2, vsize[1]) * vpitch
    grid_z = np.linspace(-vsize[2] // 2, vsize[2] // 2, vsize[2]) * vpitch
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

    for _, c in cones.iterrows():
        apex = np.array([c[APEX_X], c[APEX_Y], c[APEX_Z]], dtype=np.float32)
        d = np.array([c[DIRECTION_X], c[DIRECTION_Y], c[DIRECTION_Z]], dtype=np.float32)
        cosT = c[COS]

        # Compute distance from apex to each voxel
        voxel_vec = np.stack([X - apex[0], Y - apex[1], Z - apex[2]], axis=-1)
        voxel_distances = np.linalg.norm(voxel_vec, axis=-1)

        # Compute angle with direction vector
        vox_vec_norm = voxel_vec / voxel_distances[..., None]
        dot_products = np.sum(vox_vec_norm * d, axis=-1)

        # Compute mask of voxels satisfying the Compton cone condition
        cone_mask = np.abs(dot_products - cosT) < cone_width

        volume[cone_mask] += 1

    return volume


def reco_bp_cupy(cones, vpitch, vsize, cone_width=0.01):
    try:
        import cupy
    except ImportError:
        global_log.error(f"Please install cupy or use another backend.")
        return np.zeros(vsize, dtype=np.float32)

    volume = cupy.zeros(vsize, dtype=cupy.float32)
    grid_x = cupy.linspace(-vsize[0] // 2, vsize[0] // 2, vsize[0]) * vpitch
    grid_y = cupy.linspace(-vsize[1] // 2, vsize[1] // 2, vsize[1]) * vpitch
    grid_z = cupy.linspace(-vsize[2] // 2, vsize[2] // 2, vsize[2]) * vpitch
    X, Y, Z = cupy.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

    for _, c in cones.iterrows():
        apex = cupy.array([c[APEX_X], c[APEX_Y], c[APEX_Z]])
        d = cupy.array([c[DIRECTION_X], c[DIRECTION_Y], c[DIRECTION_Z]])
        cosT = c[COS]

        # Compute distance from apex to each voxel
        voxel_vec = cupy.stack([X - apex[0], Y - apex[1], Z - apex[2]], axis=-1)
        voxel_distances = cupy.linalg.norm(voxel_vec, axis=-1)

        # Compute angle with direction vector
        vox_vec_norm = voxel_vec / cupy.expand_dims(voxel_distances, axis=-1)
        dot_products = cupy.sum(vox_vec_norm * d, axis=-1)

        # Compute mask of voxels satisfying the Compton cone condition
        cone_mask = cupy.abs(dot_products - cosT) < cone_width

        # Accumulate contribution to the volume
        volume[cone_mask] += 1

    return volume.get()


def reco_bp_torch(cones, vpitch, vsize, cone_width=0.01):
    try:
        import torch
    except ImportError:
        global_log.error(f"Please install torch or use another backend.")
        return np.zeros(vsize, dtype=np.float32)

    dv = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    volume = torch.zeros(vsize, dtype=torch.float32, device=dv)
    grid_x = torch.linspace(-vsize[0] // 2, vsize[0] // 2, vsize[0], device=dv) * vpitch
    grid_y = torch.linspace(-vsize[1] // 2, vsize[1] // 2, vsize[1], device=dv) * vpitch
    grid_z = torch.linspace(-vsize[2] // 2, vsize[2] // 2, vsize[2], device=dv) * vpitch
    X, Y, Z = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

    for _, c in cones.iterrows():
        apex = torch.tensor([c[APEX_X], c[APEX_Y], c[APEX_Z]], device=dv,
                            dtype=torch.float32)
        d = torch.tensor([c[DIRECTION_X], c[DIRECTION_Y], c[DIRECTION_Z]], device=dv,
                         dtype=torch.float32)
        cosT = c[COS]

        # Compute distance from apex to each voxel
        voxel_vec = torch.stack([X - apex[0], Y - apex[1], Z - apex[2]], dim=-1)
        voxel_distances = torch.linalg.norm(voxel_vec, dim=-1)

        # Compute angle with direction vector
        vox_vec_norm = voxel_vec / voxel_distances.unsqueeze(-1)
        dot_products = torch.sum(vox_vec_norm * d, dim=-1)

        # Compute mask of voxels satisfying the Compton cone condition
        cone_mask = torch.abs(dot_products - cosT) < cone_width

        # Accumulate contribution to the volume
        volume[cone_mask] += 1

    return volume.cpu().numpy()


def reco_bp_coresi(CCevents, vpitch, vsize, cone_width, energies_MeV, tol_MeV,
                   sensor_size, sensor_position, sensor_rotation,
                   cone_thickness='parallel'):
    try:
        from coresi.camera import setup_cameras
        from coresi.data import read_data_file
        from coresi.mlem import LM_MLEM
    except ImportError:

        global_log.error(
            f"Please install CoReSi (and torch, if not already done) or use another backend.")
        return np.zeros(vsize, dtype=np.float32)
    fname = 'coresi/coresi_temp.dat'

    if EVENTID in CCevents.columns:
        CCevents = CCevents.drop(columns=[EVENTID])

    # In case there's only one cone, it might be given as a pandas Series...
    if isinstance(CCevents, pandas.Series):
        CCevents = CCevents.to_frame().T

    # Store coresi events since read_data_file requires a file
    CCevents.to_csv(fname, index=False, sep='\t', header=False)

    s = [x / 10 for x in sensor_size]  # convert to cm
    t = [0, 0, 0]  # center of the sensor
    p = [x / 10 for x in sensor_position]  # convert to cm
    e_keV = [x * 1000 for x in energies_MeV] if energies_MeV else [-1]  # convert to keV

    cam = setup_cameras({'n_cameras': 1,
                         'common_attributes': {'n_sca_layers': 1,
                                               'sca_material': 'Si',
                                               'abs_material': 'Si',
                                               'sca_layer_0': {'center': t, 'size': s},
                                               'n_absorbers': 1,
                                               'abs_layer_0': {'center': t, 'size': s}},
                         'position_0': {'frame_origin': p,
                                        'Ox': sensor_rotation[0],
                                        # parallel to scatterer edge
                                        'Oy': sensor_rotation[1],
                                        # parallel to scatterer edge
                                        'Oz': sensor_rotation[2]
                                        # orthogonal to the camera, toward the source
                                        }})

    vol = {'volume_dimensions': [x * vpitch * 0.1 for x in vsize],  # convert to mm
           'n_voxels': vsize,
           'volume_centre': t}

    events = read_data_file(
        Path(fname),
        n_events=-1,
        E0=e_keV,
        cameras=cam,
        remove_out_of_range_energies=False,
        energy_range=None,
        start_position=0,
        tol=tol_MeV,
        volume_config=vol,
    )

    lmmlem = {'cone_thickness': cone_thickness, 'last_iter': 1, 'model': 'cos1rho2',
              'first_iter': 0, 'save_every': 1, 'checkpoint_dir': 'coresi',
              'force_spectral': False, 'n_sigma': 2, 'width_factor': cone_width,
              'sensitivity': False, 'tv': False, 'alpha_tv': 0.0}

    mlem = LM_MLEM(lmmlem, vol, cam, 'test', e_keV, tol_MeV)
    mlem.init_sensitivity(lmmlem, lmmlem['checkpoint_dir'])

    # Run MLEM reconstruction
    chkpt = Path(lmmlem['checkpoint_dir'])
    result = mlem.run(events, 1, 0, 1, chkpt)

    # Convert result to numpy array
    vol = result.values.numpy(force=True)[0]

    return vol


# All cones should intersect at the source point
# Possible reasons for bad cones:
# - rayleigh scattering
# - compton scattering with electron not at rest (doppler broadening)
# - particle-induced X-ray emission (fluorescence, Auger)
# - more than 2 coincident events (nSingles)
# - electron/gamma escape
# - time resolution (pile-up, singles with different eventID, true_coinc)
# - energy/spatial resolution
def valid_psource(CCevents, src_pos, vpitch, vsize, energies_MeV=False, tol_MeV=0.01,
                  cone_width=0.01, plot_seq=False, plot_stk=True, plot_fwhm=False,
                  output_filename=None, colorbar=False, method='numpy', log_scale=False,
                  **kwargs):
    """
    Check that input cones intersect a given point source.

    Units of all inputs must be the same. The default in Gate simulations is mm.

    This function processes a dataframe of cones, reconstructs each cone's contribution
    around a specified source position, and verifies the correctness of the reconstruction
    by checking the intersects. It allows for optional visualization of slices or the sum
    of slices and outputs the processed data in stack form.

    Args:
        CCevents (pandas.DataFrame): DataFrame containing the cone information used for validation.
        src_pos (list[float]): The X,Y,Z coordinates of the point source.
        vpitch (float): The voxel size or pitch of the volume.
        vsize (tuple[int, int, int]): The size of the volume in voxels (X, Y, Z).
        cone_width (float): cone width (the larger the value the thicker the cones).
        plot_seq (bool, optional): If True, displays individual cone slices sequentially. Defaults to False.
        plot_stk (bool, optional): If True, displays the summed stack visualization. Defaults to True.
        method (str): Reconstruction method: "numpy", "cupy", "torch", "coresi"
        energies_MeV: (list[float]): select energy peaks, False to disable (default)
        tol_MeV: (float): if energy peaks are selected, tolerance in MeV
        kwargs: extra parameters for CoReSi:
            sensor_size (list[float])
            sensor_position (list[float]): must be same coord system as CCevents
            sensor_rotation (3×3 rotation matrix): TODO not implemented yet

    Returns:
        numpy.ndarray: The reconstructed slice at the source position.
    """
    stime = time.time()
    global_log.info(f'Offline [validate source]: START')
    if not len(CCevents):
        global_log.error(f"Empty input (no cones in dataframe).")
        global_log.info(
            f"Offline [validate source]: {get_stop_string(stime)}")
        return
    else:
        global_log.debug(f"Input: {len(CCevents)} CCevents")

    # Source position must be in units of voxels in vol
    sp_vox = [int(src_pos[i] / vpitch) + (vsize[i] // 2) for i in range(3)]

    z_slice_stack = np.zeros((vsize[0], vsize[1]), dtype=np.float32)
    CCevents = CCevents.reset_index(drop=True)

    # ##############################################################
    # # Reconstruct and display slices one by one with matplotlib
    # ##############################################################
    if plot_seq:
        for idx, cone in CCevents.iterrows():
            vol = reconstruct(cone.to_frame().T, vpitch, vsize,
                              energies_MeV=energies_MeV, tol_MeV=tol_MeV,
                              cone_width=cone_width, log=False, method=method, **kwargs)
            if np.all(vol == 0):
                global_log.error("Reconstruction returned an empty volume.")
                continue
            z_slice = vol[:, :, sp_vox[2]]
            z_slice_stack += z_slice
            plt.imshow(z_slice, cmap='gray', origin='lower')
            plt.scatter(sp_vox[0], sp_vox[1], c='r', s=10)
            if colorbar: plt.colorbar()
            plt.tight_layout()
            plt.show()

    # ##############################################################
    # # Reconstruct full volume and select slice at source position
    # ##############################################################
    else:
        vol = reconstruct(CCevents, vpitch, vsize,
                          energies_MeV=energies_MeV, tol_MeV=tol_MeV,
                          cone_width=cone_width, log=False, method=method, **kwargs)
        if np.all(vol == 0):
            global_log.error("Reconstruction returned an empty volume.")
            global_log.info('-' * 80)
            return np.zeros((vsize[0], vsize[1]), dtype=np.float32)
        z_slice_stack = vol[:, :, sp_vox[2]]

    # ##############################################################
    # # Display slice at source position
    # ##############################################################
    if plot_stk:
        if plot_fwhm:
            def gauss(x, a, x0, sigma, c):
                return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c

            mid_y = z_slice_stack.shape[0] // 2
            mid_x = z_slice_stack.shape[1] // 2
            profile_v = z_slice_stack[:, mid_x]  # vertical profile: middle column
            profile_h = z_slice_stack[mid_y, :]  # horizontal profile: middle row
            y = np.arange(len(profile_v))
            x = np.arange(len(profile_h))

            p0_v = [profile_v.max(), np.argmax(profile_v), 10, profile_v.min()]
            params_v, _ = curve_fit(gauss, y, profile_v, p0=p0_v)
            fwhm_v = 2.355 * abs(params_v[2])

            p0_h = [profile_h.max(), np.argmax(profile_h), 10, profile_h.min()]
            try:
                params_h, _ = curve_fit(gauss, x, profile_h, p0=p0_h)
                fwhm_h = 2.355 * abs(params_h[2])
            except RuntimeError:
                global_log.warning("Gaussian fit failed.")
                params_h = [np.nan, np.nan, np.nan, np.nan]
                fwhm_h = np.nan

            fig = plt.figure(figsize=(10, 10))
            gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05,
                          hspace=0.05)

            ax_img = fig.add_subplot(gs[1, 0])
            ax_h = fig.add_subplot(gs[0, 0], sharex=ax_img)
            ax_v = fig.add_subplot(gs[1, 1], sharey=ax_img)

            ax_h.plot(x, profile_h)
            ax_h.plot(x, gauss(x, *params_h), 'r--',
                      label=f'Gaussian Fit\nFWHM={fwhm_h:.2f}')
            ax_h.tick_params(axis='x',
                             labelbottom=False)  # Hide x-ticks only for profile
            ax_h.legend(loc='upper right')

            ax_v.plot(profile_v, y)
            ax_v.plot(gauss(y, *params_v), y, 'r--',
                      label=f'Gaussian Fit\nFWHM={fwhm_v:.2f}')
            ax_v.tick_params(axis='y', labelleft=False)  # Hide y-ticks only for profile
            ax_v.legend(loc='lower right')

            fig.text(0.5, 0.98,
                     s=f'{len(CCevents)} cones, {cone_width} width',
                     ha='center', va='top',
                     bbox=dict(facecolor='white', alpha=0.8))
        else:
            fig, ax_img = plt.subplots(figsize=(9, 8))

        # Plot slice and source position on top:
        norm = matplotlib.colors.LogNorm() if log_scale else None  # helps see cone profiles in the background
        im = ax_img.imshow(z_slice_stack, cmap='inferno', origin='lower', norm=norm)
        ax_img.scatter(sp_vox[0], sp_vox[1], c='r', s=10)

        # Extras that only work if FWHM is not plotted:
        if not plot_fwhm:
            # cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
            plt.colorbar(im)
            plt.tight_layout()

            # Add secondary axes with distance units:
            Xmm = ax_img.secondary_xaxis('top')
            Xmm.set_xlabel('X (mm)', color='red')
            Xmm.set_xticks(ax_img.get_xticks())
            Xmm.set_xticklabels(np.round(ax_img.get_xticks() * vpitch, 2), color='red')
            Ymm = ax_img.secondary_yaxis('right', color='red')
            Ymm.set_ylabel('Y (mm)', color='red')
            Ymm.set_yticks(ax_img.get_yticks())
            Ymm.set_yticklabels(np.round(ax_img.get_yticks() * vpitch, 2), color='red')

        if output_filename:
            fig.savefig(output_filename.with_suffix('.png'), dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            global_log.debug(f"Plot saved to {output_filename}")
            base = output_filename.stem
            parent = output_filename.parent
            np.save(parent / f"{base}_2D.npy", z_slice_stack)
            np.save(parent / f"{base}_3D.npy", vol)
        else:
            plt.show()

    global_log.info(f"Offline [validate source]: {get_stop_string(stime)}")

    return z_slice_stack, vsize
