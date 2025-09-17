# Functions to process pixelCoincidences dataframes
# E1 = energy deposited in the Compton scattering, as in CCMod paper

import os
import time
import numpy as np
import pandas
import uproot

from tools.utils import global_log_debug_df, get_stop_string, \
    localFractional2globalCoordinates
from tools.pixelHits import TOA, ENERGY_keV, EVENTID
from tools.pixelClusters import PIX_X_ID, PIX_Y_ID

try:
    from opengate.logger import global_log
except ImportError:
    import logging

    global_log = logging.getLogger("dummy")
    global_log.addHandler(logging.NullHandler())

# Pixel coincidences format definition: same as CoReSi input
coincidences_columns = ['n']
for i in range(2):
    coincidences_columns += [f'evt_{i + 1}', f'PositionX_{i + 1}', f'PositionY_{i + 1}',
                             f'PositionZ_{i + 1}', f'Energy (keV)_{i + 1}']


def gHits2pixelCoincidences_prototype(file_path, source_MeV, tolerance_MeV=0.01,
                                      entry_stop=None):
    """
    Read Gate hits (from DigitizerHitsCollectionActor) and filter Compton/photo-electric coincidences.

    Args:
        file_path (str): Path to the ROOT file containing hit data.
        source_MeV: float or str
            Energy used to select events. If, in the simulation, source.particle was:
            * `gamma` -> use a float
            * `ion xx xxx` -> use a string:
                'DaughterIsotope[excitationEnergy]_gammaEnergy' with energies in keV
                 Isotopes can decay to different daughter states, which can emit different gammas.
                 Only one daughter excitation state can be selected.
                 -> see `get_isotope_data` function in tools.utils_opengate
                 Example:
                    - If source.particle was `ion 71 177`, i.e. a Lu177 source
                      Lu177 can decay to different Hf177 excitation states
                      You can check them with get_isotope_data('ion 71 177')
                      Most probable states are:
                      - Hf177[321.316] This states in turn decays with highest probability
                        to the lower excited state 112.950 keV, emitting a 208.366 keV gamma.
                        To choose those gammas -> Hf177[321.316]_208.366
                      - Hf177[112.950] This state decays to the ground state with a
                        112.950 keV gamma -> Hf177[112.950]_112.950
        tolerance_MeV (float, optional): Energy tolerance for full absorption. Defaults to 10 keV.
        entry_stop (int, optional): Number of entries to read from the ROOT file. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing pixel coincidences.
    """
    stime = time.time()
    global_log.info(f"Offline [coincidences]: START")
    if not os.path.isfile(file_path):
        global_log.error(f"File {file_path} does not exist, probably no hit produced.")
        global_log.info(f"Offline [coincidences]: {get_stop_string(stime)}")
        return pandas.DataFrame()

    stime = time.time()
    hits_df = uproot.open(file_path)['Hits'].arrays(library='pd', entry_stop=entry_stop)
    global_log.debug(f"Input {file_path} ({len(hits_df)} entries)")
    grouped = hits_df.groupby('EventID')
    coincidences = []

    source_is_ion = isinstance(source_MeV, str) and source_MeV[0].isalpha()
    daughter_name = None
    if source_is_ion:
        daughter_name, gamma_energy = source_MeV.split('_')
        global_log.debug(
            f"Filtering {gamma_energy} keV gammas with ParentParticleName={daughter_name}")
        source_MeV = float(gamma_energy) / 1000  # Convert keV to MeV

    def find_descendants(df, part_id):
        descendants = set()
        child = df[df['ParentID'] == part_id]['TrackID'].values
        for child in child:
            descendants.add(child)
            descendants.update(find_descendants(df, child))
        return descendants

    n_events_primary = 0
    n_events_full_edep = 0
    for eventid, grp in grouped:
        compton_pos, photoelec_pos, E1, E2 = False, False, False, False
        if source_is_ion:
            sensor_got_primary = daughter_name in grp['ParentParticleName'].to_numpy()
        else:
            sensor_got_primary = 1 in grp['TrackID'].values
        # Sensor received primary gamma and it interacted
        if sensor_got_primary:
            n_events_primary += 1
            if source_is_ion:
                part_id = \
                grp[grp['ParentParticleName'] == daughter_name]['TrackID'].values[0]
                descendants = find_descendants(grp, part_id)
                totenergy = grp[grp['TrackID'].isin(descendants.union({part_id}))][
                    'TotalEnergyDeposit'].sum()
                full_absorb = abs(totenergy - source_MeV) < tolerance_MeV
                if full_absorb:
                    grp = grp[grp['TrackID'].isin(descendants.union({part_id}))]
                    grp.loc[:, 'TrackID'] -= (part_id - 1)
            else:
                full_absorb = abs(
                    grp['TotalEnergyDeposit'].sum() - source_MeV) < tolerance_MeV
            # All primary energy was deposited
            if full_absorb:
                n_events_full_edep += 1
                grp = grp.sort_values('GlobalTime')  # IMPORTANT !
                h1 = grp.iloc[0]
                # Gamma interacts via Compton, step has dE !=0 and is stored (recoil e- not tracked)
                if h1['TrackID'] == 1 and grp['TrackID'].value_counts()[1] > 1:
                    # if value_counts()[1] == 1, TrackID 1 stopped at 1st step via photoelec (without prior Compton)
                    # TODO: what about rayleigh scattering and pair production?
                    compton_pos = [h1[f'PostPosition_{axis}'] for axis in 'XYZ']
                    E1 = h1['TotalEnergyDeposit']
                    photoelec_pos = [grp.iloc[1][f'PrePosition_{ax}'] for ax in 'XYZ']
                    E2 = source_MeV - E1
                # Gamma interacts via Compton, step has dE = 0 and is not stored, but recoil e- tracked with TrackID=2
                # However I can't use direction of recoil e-... Need to go further
                elif h1['TrackID'] == 2 and h1['TrackCreatorProcess'] == 'compt':
                    compton_pos = [h1[f'PrePosition_{ax}'] for ax in 'XYZ']
                    E1 = h1['KineticEnergy']
                    # Remove TrackID 2 and its descendants from group
                    desc_of_2 = find_descendants(grp, 2)
                    grp = grp[~grp['TrackID'].isin(desc_of_2.union({2}))]
                    h2 = grp.iloc[0]
                    # TODO: check 2 lines below
                    # E2 = h2['TotalEnergyDeposit']
                    E2 = source_MeV - E1
                    photoelec_pos = [h2[f'PrePosition_{ax}'] for ax in 'XYZ']

        if compton_pos:
            coincidences.append(
                [2, 1] + compton_pos + [1000 * E1] + [2] + photoelec_pos + [
                    1000 * E2])

    df = pandas.DataFrame(coincidences, columns=coincidences_columns)
    global_log.debug(f"{n_events_primary} events with primary particle hitting sensor")
    global_log.debug(f"=> {n_events_full_edep} with full energy deposited in sensor")
    global_log.debug(f"  => {len(coincidences)} with at least one Compton interaction")
    global_log.info(f"Offline [coincidences]: {len(coincidences)} coincidences")
    global_log_debug_df(df)
    global_log.info(f"Offline [coincidences]: {get_stop_string(stime)}")
    return df


def gHits2pixelCoincidences(file_path, source_MeV, tolerance_MeV=0.01, entry_stop=None):
    """
    Same as gHits2pixelCoincidences_prototype but faster implementation using NumPy arrays.
    Obtained from GPT-4.1 by feeding it the prototype and asking for optimizations.
    => about 10x faster when used in isotope.py example
    """
    stime = time.time()
    global_log.info(f"Offline [coincidences]: START")
    if not os.path.isfile(file_path):
        global_log.error(f"File {file_path} does not exist, probably no hit produced.")
        global_log.info(f"Offline [coincidences]: {get_stop_string(stime)}")
        return pandas.DataFrame()

    hits_df = uproot.open(file_path)['Hits'].arrays(library='pd', entry_stop=entry_stop)
    global_log.debug(f"Input {file_path} ({len(hits_df)} entries)")

    # Pre-convert columns to NumPy arrays for fast access
    event_ids = hits_df['EventID'].to_numpy()
    parent_names = hits_df['ParentParticleName'].astype(str).to_numpy()
    track_ids = hits_df['TrackID'].to_numpy()
    parent_ids = hits_df['ParentID'].to_numpy()
    total_edep = hits_df['TotalEnergyDeposit'].to_numpy()
    kinetic_energy = hits_df['KineticEnergy'].to_numpy()
    creator_process = hits_df['TrackCreatorProcess'].astype(str).to_numpy()
    particle_names = hits_df['ParticleName'].astype(str).to_numpy()
    global_time = hits_df['GlobalTime'].to_numpy()
    # Pre/Post positions
    pre_pos = np.stack([hits_df[f'PrePosition_{ax}'].to_numpy() for ax in 'XYZ'],
                       axis=1)
    post_pos = np.stack([hits_df[f'PostPosition_{ax}'].to_numpy() for ax in 'XYZ'],
                        axis=1)

    source_is_ion = isinstance(source_MeV, str) and source_MeV[0].isalpha()
    daughter_name = None
    if source_is_ion:
        daughter_name, gamma_energy = source_MeV.split('_')
        global_log.debug(
            f"Filtering {gamma_energy} keV gammas with ParentParticleName={daughter_name}")
        source_MeV = float(gamma_energy) / 1000  # Convert keV to MeV

    # Group indices by EventID for fast access
    event_idx = {}
    for idx, eid in enumerate(event_ids):
        event_idx.setdefault(eid, []).append(idx)

    coincidences = []
    n_events_primary = 0
    n_events_full_edep = 0

    for eid, idxs in event_idx.items():
        idxs = np.array(idxs)
        if source_is_ion:
            has_primary = np.any(parent_names[idxs] == daughter_name)
        else:
            has_primary = np.any(track_ids[idxs] == 1)
        if not has_primary:
            continue
        n_events_primary += 1

        if source_is_ion:
            part_mask = parent_names[idxs] == daughter_name
            part_id = track_ids[idxs][part_mask][0]
            # Find descendants using a set and stack
            descendants = set()
            stack = [part_id]
            while stack:
                pid = stack.pop()
                child_mask = parent_ids[idxs] == pid
                children = track_ids[idxs][child_mask]
                for child in children:
                    if child not in descendants:
                        descendants.add(child)
                        stack.append(child)
            relevant = np.isin(track_ids[idxs], list(descendants) + [part_id])
            totenergy = total_edep[idxs][relevant].sum()
            full_absorb = abs(totenergy - source_MeV) < tolerance_MeV
            if not full_absorb:
                continue
            rel_idxs = idxs[relevant]
            # Renumber TrackID
            track_ids[rel_idxs] -= (part_id - 1)
        else:
            totenergy = total_edep[idxs].sum()
            full_absorb = abs(totenergy - source_MeV) < tolerance_MeV
            if not full_absorb:
                continue
            rel_idxs = idxs

        n_events_full_edep += 1
        # Sort by GlobalTime
        sorted_idx = rel_idxs[np.argsort(global_time[rel_idxs], kind='mergesort')]
        h1 = sorted_idx[0]
        if track_ids[h1] == 1 and np.sum(track_ids[sorted_idx] == 1) > 1:
            compton_pos = post_pos[h1]
            E1 = total_edep[h1]
            photoelec_pos = pre_pos[sorted_idx[1]]
            E2 = source_MeV - E1
        elif track_ids[h1] == 2 and creator_process[h1] == 'compt':
            compton_pos = pre_pos[h1]
            E1 = kinetic_energy[h1]
            # Find descendants of 2
            desc_of_2 = set()
            stack = [2]
            for idx in sorted_idx:
                if track_ids[idx] == 2:
                    stack = [2]
                    break
            while stack:
                pid = stack.pop()
                child_mask = parent_ids[sorted_idx] == pid
                children = track_ids[sorted_idx][child_mask]
                for child in children:
                    if child not in desc_of_2:
                        desc_of_2.add(child)
                        stack.append(child)
            mask2 = ~np.isin(track_ids[sorted_idx], list(desc_of_2) + [2])
            grp2 = sorted_idx[mask2]
            if len(grp2) == 0:
                continue
            h2 = grp2[0]
            E2 = source_MeV - E1
            photoelec_pos = pre_pos[h2]
        else:
            continue

        coincidences.append([2, 1] + compton_pos.tolist() + [1000 * E1] + [2] +
                            photoelec_pos.tolist() + [1000 * E2])

    df = pandas.DataFrame(coincidences, columns=coincidences_columns)
    global_log.debug(f"{n_events_primary} events with primary particle hitting sensor")
    global_log.debug(f"=> {n_events_full_edep} with full energy deposited in sensor")
    global_log.debug(f"  => {len(coincidences)} with at least one Compton interaction")
    global_log.info(f"Offline [coincidences]: {len(coincidences)} coincidences")
    global_log_debug_df(df)
    global_log.info(f"Offline [coincidences]: {get_stop_string(stime)}")
    return df


def pixelClusters2pixelCoincidences(pixelClusters, thickness_mm, charge_speed_mm_ns,
                                    coincidence_window_ns=100):
    stime = time.time()
    global_log.info(f"Offline [coincidences]: START")

    if not len(pixelClusters):
        global_log.error(f"Empty input (no clusters in dataframe).")
        global_log.info(f"Offline [pixelClusters]: {get_stop_string(stime)}")
        return pandas.DataFrame()
    else:
        global_log.debug(f"Input: {len(pixelClusters)} pixel clusters")

    # Group pixelClusters by time
    pixelClusters_copy = pixelClusters.copy()
    pixelClusters_copy = pixelClusters_copy.sort_values(TOA).reset_index(drop=True)
    clustering_ids = np.zeros(len(pixelClusters_copy), dtype=int)
    current_event = 0
    last_time = None

    for i, t in enumerate(pixelClusters_copy[TOA]):
        if last_time is None or (t - last_time) > coincidence_window_ns:
            current_event += 1
        clustering_ids[i] = current_event
        last_time = t

    pixelClusters_copy[EVENTID] = clustering_ids

    grouped = pixelClusters_copy.groupby(EVENTID)
    grouped = [group for group in grouped if len(group[1]) == 2]

    coincidences = []
    pixelCoincidences = pandas.DataFrame()
    for eventid, group in grouped:
        # 0) TODO: Filter fluorescence events? E.g.:
        # if group['Energy (keV)'].between(22, 28).any():
        #     continue

        # 1) Distinguish compton vs photo-electric interactions
        group = group.sort_values(ENERGY_keV)
        Esum_MeV = 0.001 * (group.iloc[0][ENERGY_keV] + group.iloc[1][ENERGY_keV])
        E1max = Esum_MeV ** 2 / (Esum_MeV + 0.511 / 2)
        only_one_possibility = group.iloc[1][ENERGY_keV] > E1max
        # TODO: this limits the number of selected events, propose other ways
        if only_one_possibility:
            cl_photoel = group.iloc[1]
            cl_compton = group.iloc[0]
        else:
            continue

        # 2) Calculate depth difference
        dZ_mm = charge_speed_mm_ns * (cl_compton[TOA] - cl_photoel[TOA])
        dZ_frac = dZ_mm / thickness_mm

        # 3) Calculate absolute depth of Compton interaction
        z_compton = 0  # middle of sensor (in local fractional unit)
        # TODO or use cluster size/energy ?

        # 4) Complete 3D positions
        pos_compton = [cl_compton[PIX_X_ID], cl_compton[PIX_Y_ID], z_compton]
        pos_photoel = [cl_photoel[PIX_X_ID], cl_photoel[PIX_Y_ID], z_compton + dZ_frac]

        # 5) Construct cone
        E1_keV = cl_compton[ENERGY_keV]
        E2_keV = cl_photoel[ENERGY_keV]
        coincidences.append(
            [eventid] + [2, 1] + pos_compton + [E1_keV] + [2] + pos_photoel + [E2_keV])
        pixelCoincidences = pandas.DataFrame(coincidences,
                                             columns=[EVENTID] + coincidences_columns)

    global_log.info(f"Offline [coincidences]: {len(coincidences)} cones")
    global_log_debug_df(pixelCoincidences)
    global_log.info(f"Offline [coincidences]: {get_stop_string(stime)}")
    return pixelCoincidences


def local2global(pixelCoincidences, translation, rotation, npix, pitch, thickness):
    """
    Converts local fractional coordinates in a coresi events DataFrame to global coordinates.
    Assumes three sets of position columns per event.

    Args:
        pixelCoincidences (DataFrame): Input DataFrame with coresi_events_columns.
        translation (list): Translation vector [x, y, z] of the sensor in global coordinates.
        rotation (list of list): Rotation matrix (3x3) of the sensor.
        npix (int): Number of pixels along one dimension (assuming square grid).
        pitch (float): Pixel pitch (size of one pixel).
        thickness (float): Sensor thickness.

    Returns:
        DataFrame: Updated DataFrame with position columns replaced by global coordinates.
    """

    global_log.info(f"Offline [transform coord]: START")
    stime = time.time()

    df_copy = pixelCoincidences.copy()
    if df_copy.empty:
        global_log.error('Input DataFrame is empty. No coordinates to convert.')
    else:
        for i in range(1, 2):  # For each event (1, 2, 3)
            position_cols = [f"PositionX_{i}", f"PositionY_{i}", f"PositionZ_{i}"]

            def convert_row(row):
                c = [row[col] for col in position_cols]
                return localFractional2globalCoordinates(c, translation, rotation, npix,
                                                         pitch, thickness)

            coords = df_copy.apply(convert_row, axis=1, result_type='expand')
            df_copy[position_cols] = coords

        global_log_debug_df(df_copy)
    global_log.info(f"Offline [transform coord]: {get_stop_string(stime)}")
    return df_copy


def filter_pixel_coincidences(pixelCoincidences, energies_MeV, tol_MeV):
    energies_keV = np.array(energies_MeV) * 1000
    tol_keV = tol_MeV * 1000
    energy_sum = pixelCoincidences['Energy (keV)_1'] + pixelCoincidences['Energy (keV)_2']
    mask = np.any([
        np.abs(energy_sum - e) <= tol_keV for e in energies_keV
    ], axis=0)
    return pixelCoincidences[mask]
