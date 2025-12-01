# Functions to process Compton camera events (CCevents)
# A CCevent is defined as two interactions in the sensor with 3D positions:
# 1) a Compton interaction, with the energy of the recoiling electron (E1)
# 2) a 2nd interaction, which might be a photo-electric absorption or another Compton interaction.
#    => in either case, the energy (E2) should be that of the initially scattered gamma (Egamma - E1)

import math
import numpy as np
import pandas
import uproot
from collections import Counter
from tools.utils import localFractional2globalCoordinates, log_offline_process
from tools.pixelHits import TOA, ENERGY_keV, EVENTID
from tools.pixelClusters import PIX_X_ID, PIX_Y_ID

from tools.logging_custom import global_log

# CCevents format definition: same as CoReSi input
CCevents_columns = ['n']
for i in range(2):
    CCevents_columns += [f'evt_{i + 1}', f'PositionX_{i + 1}', f'PositionY_{i + 1}',
                         f'PositionZ_{i + 1}', f'Energy (keV)_{i + 1}']


@log_offline_process('CCevents', input_type='file')
def gHits2CCevents_prototype(file_path, source_MeV, entry_stop=None):
    """
    Read Gate hits (from DigitizerHitsCollectionActor) and filter CCevents.

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
        entry_stop (int, optional): Number of entries to read from the ROOT file. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing CCevents.
    """

    hits_df = uproot.open(file_path)['Hits'].arrays(library='pd', entry_stop=entry_stop)

    grouped = hits_df.groupby('EventID')
    CCevents = []

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
        pos_compton, pos_2nd, E1, E2 = False, False, False, False
        if source_is_ion:
            sensor_got_primary = daughter_name in grp['ParentParticleName'].to_numpy()
        else:
            sensor_got_primary = 1 in grp['TrackID'].values
        # Sensor received primary gamma and it interacted
        if sensor_got_primary:
            n_events_primary += 1
            if source_is_ion:
                part_id = grp[grp['ParentParticleName'] == daughter_name]['TrackID'].values[0]
                descendants = find_descendants(grp, part_id)
                totenergy = grp[grp['TrackID'].isin(descendants.union({part_id}))]['TotalEnergyDeposit'].sum()
                full_absorb = math.isclose(totenergy, source_MeV, rel_tol=0.0, abs_tol=1e-6)
                if full_absorb:
                    grp = grp[grp['TrackID'].isin(descendants.union({part_id}))]
                    grp.loc[:, 'TrackID'] -= (part_id - 1)
            else:
                totenergy = grp['TotalEnergyDeposit'].sum()
                full_absorb = math.isclose(totenergy, source_MeV, rel_tol=0.0, abs_tol=1e-6)

            # All primary energy was deposited
            if full_absorb:
                n_events_full_edep += 1
                # Ideally we would use 'ProcessDefinedStep = compt' but, if sim.keep_zero_edep is not set to True:
                # - The compton step is not stored when the recoil electron is tracked
                # - ProcessDefinedStep might be 'Transportation' instead of 'compt'
                # => we can sort steps by time
                grp = grp.sort_values('GlobalTime')
                h1 = grp.iloc[0]
                # Gamma interacts via Compton, step has dE !=0 and is stored (either because of untracked recoil e- or doppler event)
                if h1['TrackID'] == 1 and grp['TrackID'].value_counts()[1] > 1:  # if value_counts()[1] == 1, TrackID 1 stopped at 1st step via photoelec (without prior Compton)
                    # TODO: what about rayleigh scattering and pair production?
                    pos_compton = [h1[f'PostPosition_{axis}'] for axis in 'XYZ']
                    E1 = h1['PreKineticEnergy'] - h1['PostKineticEnergy'] # Can't use TotalEnergyDeposit since it might just be the atomic e- energy in case of doppler event
                    h2 = grp[grp['TrackID'] == 1].iloc[1] # TODO: what if the 2nd interaction is compton with recoil e-?
                    pos_2nd = [h2[f'PostPosition_{ax}'] for ax in 'XYZ']
                    E2 = source_MeV - E1
                # Gamma interacts via Compton, step has dE = 0 and is not stored, but recoil e- tracked with TrackID=2
                # However I can't use direction of recoil e-... Need to go further
                elif h1['TrackCreatorProcess'] == 'compt':
                    grp = grp.sort_values('PreGlobalTime')
                    h1 = grp.iloc[0]
                    pos_compton = [h1[f'PrePosition_{ax}'] for ax in 'XYZ']
                    E1 = h1['KineticEnergy']
                    # Remove TrackID 2 and its descendants from group
                    desc_of_2 = find_descendants(grp, h1['TrackID'])
                    grp = grp[~grp['TrackID'].isin(desc_of_2.union({h1['TrackID']}))]
                    h2 = grp.iloc[0]
                    E2 = source_MeV - E1
                    pos_2nd = [h2[f"{'Post' if h2['TrackID'] == 1 else 'Pre'}Position_{ax}"] for ax in 'XYZ']

        if pos_compton:
            CCevents.append([eventid] + [2, 1] + pos_compton + [1000 * E1] + [2] + pos_2nd + [1000 * E2])

    df = pandas.DataFrame(CCevents, columns=[EVENTID] + CCevents_columns)
    global_log.debug(f"{n_events_primary} events with primary particle hitting sensor")
    global_log.debug(f"=> {n_events_full_edep} with full energy deposited in sensor")
    global_log.debug(f"  => {len(CCevents)} with at least one Compton interaction")

    return df

@log_offline_process('CCevents', input_type='file')
def gHits2CCevents(file_path, source_MeV, entry_stop=None):
    """
    Same as gHits2CCevents_prototype but faster implementation using NumPy arrays.
    Obtained from GPT-5 mini by feeding it the prototype and asking for optimizations.
    => about 10x faster when used in isotope.py example
    """

    hits_df = uproot.open(file_path)['Hits'].arrays(library='pd', entry_stop=entry_stop)

    # Convert needed columns to NumPy arrays for speed
    event_ids = hits_df['EventID'].to_numpy()
    track_ids = hits_df['TrackID'].to_numpy()
    parent_ids = hits_df['ParentID'].to_numpy()
    parent_names = hits_df['ParentParticleName'].astype(str).to_numpy()
    total_edep = hits_df['TotalEnergyDeposit'].to_numpy()
    pre_kin = hits_df.get('PreKineticEnergy', hits_df.get('KineticEnergy')).to_numpy()
    post_kin = hits_df.get('PostKineticEnergy', np.zeros_like(pre_kin)).to_numpy()
    kinetic_energy = hits_df['KineticEnergy'].to_numpy()
    creator_process = hits_df['TrackCreatorProcess'].astype(str).to_numpy()
    global_time = hits_df['GlobalTime'].to_numpy()
    pre_global_time = hits_df.get('PreGlobalTime', global_time).to_numpy()

    pre_pos = np.stack([hits_df[f'PrePosition_{ax}'].to_numpy() for ax in 'XYZ'], axis=1)
    post_pos = np.stack([hits_df[f'PostPosition_{ax}'].to_numpy() for ax in 'XYZ'], axis=1)

    source_is_ion = isinstance(source_MeV, str) and source_MeV and source_MeV[0].isalpha()
    daughter_name = None
    if source_is_ion:
        daughter_name, gamma_energy = source_MeV.split('_')
        global_log.debug(f"Filtering {gamma_energy} keV gammas with ParentParticleName={daughter_name}")
        source_MeV = float(gamma_energy) / 1000.0  # keV -> MeV

    # Prepare grouping by EventID using a single sort + unique
    order = np.argsort(event_ids, kind='mergesort')
    sorted_eids = event_ids[order]
    unique_eids, starts = np.unique(sorted_eids, return_index=True)
    ends = np.append(starts[1:], len(sorted_eids))

    CCevents = []
    n_events_primary = 0
    n_events_full_edep = 0

    # local helper: find descendants within indices (relative to event index array)
    def find_descendants_in_event(track_arr, parent_arr, target):
        descendants = set()
        stack = [target]
        while stack:
            pid = stack.pop()
            # children where parent == pid
            children = track_arr[parent_arr == pid]
            for c in children:
                if c not in descendants:
                    descendants.add(c)
                    stack.append(c)
        return descendants

    # Iterate events
    for uid_idx, eid in enumerate(unique_eids):
        idx_slice = order[starts[uid_idx]:ends[uid_idx]]  # global indices for this event
        if source_is_ion:
            has_primary = np.any(parent_names[idx_slice] == daughter_name)
        else:
            has_primary = np.any(track_ids[idx_slice] == 1)
        if not has_primary:
            continue
        n_events_primary += 1

        # Work on copies of arrays where we may renumber track ids for ion case
        track_ids_copy = track_ids.copy()

        if source_is_ion:
            # find the primary particle's track id within the event
            mask_primary = parent_names[idx_slice] == daughter_name
            part_ids = track_ids[idx_slice][mask_primary]
            if len(part_ids) == 0:
                continue
            part_id = int(part_ids[0])
            # Build descendants (within this event only)
            # Use local arrays for event
            ev_track = track_ids[idx_slice]
            ev_parent = parent_ids[idx_slice]
            descendants = set()
            stack = [part_id]
            while stack:
                pid = stack.pop()
                children = ev_track[ev_parent == pid]
                for c in children:
                    if c not in descendants:
                        descendants.add(c)
                        stack.append(c)
            relevant_ids = list(descendants) + [part_id]
            relevant_mask = np.isin(track_ids[idx_slice], relevant_ids)
            totenergy = total_edep[idx_slice][relevant_mask].sum()
            full_absorb = math.isclose(totenergy, source_MeV, rel_tol=0.0, abs_tol=1e-6)
            if not full_absorb:
                continue
            # global indices of relevant hits
            rel_idxs = idx_slice[relevant_mask]
            # renumber track ids within the global array copy
            track_ids_copy[rel_idxs] = track_ids_copy[rel_idxs] - (part_id - 1)
        else:
            totenergy = total_edep[idx_slice].sum()
            full_absorb = math.isclose(totenergy, source_MeV, rel_tol=0.0, abs_tol=1e-6)
            if not full_absorb:
                continue
            rel_idxs = idx_slice

        n_events_full_edep += 1

        # Sort by GlobalTime (like prototype)
        sorted_rel = rel_idxs[np.argsort(global_time[rel_idxs], kind='mergesort')]
        h1 = sorted_rel[0]

        # Case A: TrackID 1 has multiple entries (compton step stored)
        if track_ids_copy[h1] == 1 and np.sum(track_ids_copy[sorted_rel] == 1) > 1:
            # pos_compton = PostPosition of first
            pos_compton = post_pos[h1].tolist()
            # match prototype: PreKinetic - PostKinetic (use arrays)
            E1 = float(pre_kin[h1] - post_kin[h1])
            # find second hit with TrackID == 1
            idxs_1 = sorted_rel[track_ids_copy[sorted_rel] == 1]
            if len(idxs_1) < 2:
                continue
            h2 = idxs_1[1]
            pos_2nd = post_pos[h2].tolist()
            E2 = source_MeV - E1

        # Case B: first recorded hit is a 'compt' creator -> handle recoil tracking
        elif creator_process[h1] == 'compt':
            # sort by PreGlobalTime and take first
            sorted_by_pre = rel_idxs[np.argsort(pre_global_time[rel_idxs], kind='mergesort')]
            if len(sorted_by_pre) == 0:
                continue
            h1_pre = sorted_by_pre[0]
            pos_compton = pre_pos[h1_pre].tolist()
            E1 = float(kinetic_energy[h1_pre])
            # remove descendants of h1_pre
            ev_track = track_ids[sorted_by_pre]
            ev_parent = parent_ids[sorted_by_pre]
            desc = find_descendants_in_event(ev_track, ev_parent, int(track_ids[h1_pre]))
            mask_keep = ~np.isin(track_ids[sorted_by_pre], list(desc) + [int(track_ids[h1_pre])])
            grp2 = sorted_by_pre[mask_keep]
            if len(grp2) == 0:
                continue
            h2 = grp2[0]
            E2 = source_MeV - E1
            pos_2nd = (post_pos[h2] if track_ids_copy[h2] == 1 else pre_pos[h2]).tolist()
        else:
            continue

        # avoid degenerate equal positions (as in prototype workaround)
        if not np.allclose(pos_compton, pos_2nd):
            CCevents.append([int(eid)] + [2, 1] + pos_compton + [1000.0 * E1] + [2] + pos_2nd + [1000.0 * E2])

    df = pandas.DataFrame(CCevents, columns=[EVENTID] + CCevents_columns)
    global_log.debug(f"{n_events_primary} events with primary particle hitting sensor")
    global_log.debug(f"=> {n_events_full_edep} with full energy deposited in sensor")
    global_log.debug(f"  => {len(CCevents)} with at least one Compton interaction")

    return df

# @log_offline_process('CCevents', input_type='file')
# def gHits2CCevents_prototype(file_path, source_MeV, tolerance_MeV=0.001,
#                              entry_stop=None):
#     """
#     Read Gate hits (from DigitizerHitsCollectionActor) and filter CCevents.
#     TODO: use track IDs instead of the source energy and tolerance since to avoid losing Doppler-broadened events.
#
#     Args:
#         file_path (str): Path to the ROOT file containing hit data.
#         source_MeV: float or str
#             Energy used to select events. If, in the simulation, source.particle was:
#             * `gamma` -> use a float
#             * `ion xx xxx` -> use a string:
#                 'DaughterIsotope[excitationEnergy]_gammaEnergy' with energies in keV
#                  Isotopes can decay to different daughter states, which can emit different gammas.
#                  Only one daughter excitation state can be selected.
#                  -> see `get_isotope_data` function in tools.utils_opengate
#                  Example:
#                     - If source.particle was `ion 71 177`, i.e. a Lu177 source
#                       Lu177 can decay to different Hf177 excitation states
#                       You can check them with get_isotope_data('ion 71 177')
#                       Most probable states are:
#                       - Hf177[321.316] This states in turn decays with highest probability
#                         to the lower excited state 112.950 keV, emitting a 208.366 keV gamma.
#                         To choose those gammas -> Hf177[321.316]_208.366
#                       - Hf177[112.950] This state decays to the ground state with a
#                         112.950 keV gamma -> Hf177[112.950]_112.950
#         tolerance_MeV (float, optional): Energy tolerance for full absorption. Defaults to 10 keV.
#         entry_stop (int, optional): Number of entries to read from the ROOT file. Defaults to None.
#
#     Returns:
#         pandas.DataFrame: DataFrame containing CCevents.
#     """
#
#     hits_df = uproot.open(file_path)['Hits'].arrays(library='pd', entry_stop=entry_stop)
#
#     grouped = hits_df.groupby('EventID')
#     CCevents = []
#
#     source_is_ion = isinstance(source_MeV, str) and source_MeV[0].isalpha()
#     daughter_name = None
#     if source_is_ion:
#         daughter_name, gamma_energy = source_MeV.split('_')
#         global_log.debug(
#             f"Filtering {gamma_energy} keV gammas with ParentParticleName={daughter_name}")
#         source_MeV = float(gamma_energy) / 1000  # Convert keV to MeV
#
#     def find_descendants(df, part_id):
#         descendants = set()
#         child = df[df['ParentID'] == part_id]['TrackID'].values
#         for child in child:
#             descendants.add(child)
#             descendants.update(find_descendants(df, child))
#         return descendants
#
#     n_events_primary = 0
#     n_events_full_edep = 0
#     for eventid, grp in grouped:
#         pos_compton, pos_2nd, E1, E2 = False, False, False, False
#         if source_is_ion:
#             sensor_got_primary = daughter_name in grp['ParentParticleName'].to_numpy()
#         else:
#             sensor_got_primary = 1 in grp['TrackID'].values
#         # Sensor received primary gamma and it interacted
#         if sensor_got_primary:
#             n_events_primary += 1
#             if source_is_ion:
#                 part_id = \
#                     grp[grp['ParentParticleName'] == daughter_name]['TrackID'].values[0]
#                 descendants = find_descendants(grp, part_id)
#                 totenergy = grp[grp['TrackID'].isin(descendants.union({part_id}))][
#                     'TotalEnergyDeposit'].sum()
#                 full_absorb = abs(totenergy - source_MeV) < tolerance_MeV
#                 if full_absorb:
#                     grp = grp[grp['TrackID'].isin(descendants.union({part_id}))]
#                     grp.loc[:, 'TrackID'] -= (part_id - 1)
#             else:
#                 full_absorb = abs(
#                     grp['TotalEnergyDeposit'].sum() - source_MeV) < tolerance_MeV
#             # All primary energy was deposited
#             if full_absorb:
#                 n_events_full_edep += 1
#                 grp = grp.sort_values('GlobalTime')  # IMPORTANT !
#                 h1 = grp.iloc[0]
#                 # Gamma interacts via Compton, step has dE !=0 and is stored (recoil e- not tracked)
#                 if h1['TrackID'] == 1 and grp['TrackID'].value_counts()[1] > 1:
#                     # if value_counts()[1] == 1, TrackID 1 stopped at 1st step via photoelec (without prior Compton)
#                     # TODO: what about rayleigh scattering and pair production?
#                     pos_compton = [h1[f'PostPosition_{axis}'] for axis in 'XYZ']
#                     E1 = h1['TotalEnergyDeposit']
#                     h2 = grp.iloc[1]
#                     pos_2nd = [h2[f'PostPosition_{ax}'] for ax in 'XYZ']
#                     E2 = source_MeV - E1
#                 # Gamma interacts via Compton, step has dE = 0 and is not stored, but recoil e- tracked with TrackID=2
#                 # However I can't use direction of recoil e-... Need to go further
#                 elif h1['TrackID'] == 2 and h1['TrackCreatorProcess'] == 'compt':
#                     pos_compton = [h1[f'PrePosition_{ax}'] for ax in 'XYZ']
#                     E1 = h1['KineticEnergy']
#                     # Remove TrackID 2 and its descendants from group
#                     desc_of_2 = find_descendants(grp, 2)
#                     grp = grp[~grp['TrackID'].isin(desc_of_2.union({2}))]
#                     h2 = grp.iloc[0]
#                     E2 = source_MeV - E1
#                     pos_2nd = [h2[f"{'Post' if h2['TrackID'] == 1 else 'Pre'}Position_{ax}"] for ax in 'XYZ']
#
#         if pos_compton:
#             CCevents.append([eventid] + [2, 1] + pos_compton + [1000 * E1] + [2] + pos_2nd + [1000 * E2])
#
#     df = pandas.DataFrame(CCevents, columns=[EVENTID] + CCevents_columns)
#     global_log.debug(f"{n_events_primary} events with primary particle hitting sensor")
#     global_log.debug(f"=> {n_events_full_edep} with full energy deposited in sensor")
#     global_log.debug(f"  => {len(CCevents)} with at least one Compton interaction")
#

# @log_offline_process('CCevents', input_type = 'file')
# def gHits2CCevents(file_path, source_MeV, tolerance_MeV=0.01, entry_stop=None):
#     """
#     Same as gHits2CCevents_prototype but faster implementation using NumPy arrays.
#     Obtained from GPT-4.1 by feeding it the prototype and asking for optimizations.
#     => about 10x faster when used in isotope.py example
#     """
#
#     hits_df = uproot.open(file_path)['Hits'].arrays(library='pd', entry_stop=entry_stop)
#
#     # Pre-convert columns to NumPy arrays for fast access
#     event_ids = hits_df['EventID'].to_numpy()
#     parent_names = hits_df['ParentParticleName'].astype(str).to_numpy()
#     track_ids = hits_df['TrackID'].to_numpy()
#     parent_ids = hits_df['ParentID'].to_numpy()
#     total_edep = hits_df['TotalEnergyDeposit'].to_numpy()
#     kinetic_energy = hits_df['KineticEnergy'].to_numpy()
#     creator_process = hits_df['TrackCreatorProcess'].astype(str).to_numpy()
#     global_time = hits_df['GlobalTime'].to_numpy()
#     # Pre/Post positions
#     pre_pos = np.stack([hits_df[f'PrePosition_{ax}'].to_numpy() for ax in 'XYZ'],
#                        axis=1)
#     post_pos = np.stack([hits_df[f'PostPosition_{ax}'].to_numpy() for ax in 'XYZ'],
#                         axis=1)
#
#     source_is_ion = isinstance(source_MeV, str) and source_MeV[0].isalpha()
#     daughter_name = None
#     if source_is_ion:
#         daughter_name, gamma_energy = source_MeV.split('_')
#         global_log.debug(
#             f"Filtering {gamma_energy} keV gammas with ParentParticleName={daughter_name}")
#         source_MeV = float(gamma_energy) / 1000  # Convert keV to MeV
#
#     # Group indices by EventID for fast access
#     event_idx = {}
#     for idx, eid in enumerate(event_ids):
#         event_idx.setdefault(eid, []).append(idx)
#
#     CCevents = []
#     n_events_primary = 0
#     n_events_full_edep = 0
#
#     for eid, idxs in event_idx.items():
#         idxs = np.array(idxs)
#         if source_is_ion:
#             has_primary = np.any(parent_names[idxs] == daughter_name)
#         else:
#             has_primary = np.any(track_ids[idxs] == 1)
#         if not has_primary:
#             continue
#         n_events_primary += 1
#
#         if source_is_ion:
#             part_mask = parent_names[idxs] == daughter_name
#             part_id = track_ids[idxs][part_mask][0]
#             # Find descendants using a set and stack
#             descendants = set()
#             stack = [part_id]
#             while stack:
#                 pid = stack.pop()
#                 child_mask = parent_ids[idxs] == pid
#                 children = track_ids[idxs][child_mask]
#                 for child in children:
#                     if child not in descendants:
#                         descendants.add(child)
#                         stack.append(child)
#             relevant = np.isin(track_ids[idxs], list(descendants) + [part_id])
#             totenergy = total_edep[idxs][relevant].sum()
#             full_absorb = abs(totenergy - source_MeV) < tolerance_MeV
#             if not full_absorb:
#                 continue
#             rel_idxs = idxs[relevant]
#             # Renumber TrackID
#             track_ids[rel_idxs] -= (part_id - 1)
#         else:
#             totenergy = total_edep[idxs].sum()
#             full_absorb = abs(totenergy - source_MeV) < tolerance_MeV
#             if not full_absorb:
#                 continue
#             rel_idxs = idxs
#
#         n_events_full_edep += 1
#         # Sort by GlobalTime
#         sorted_idx = rel_idxs[np.argsort(global_time[rel_idxs], kind='mergesort')]
#         h1 = sorted_idx[0]
#         if track_ids[h1] == 1 and np.sum(track_ids[sorted_idx] == 1) > 1:
#             pos_compton = post_pos[h1]
#             E1 = total_edep[h1]
#             pos_2nd = post_pos[sorted_idx[1]]
#             E2 = source_MeV - E1
#         elif track_ids[h1] == 2 and creator_process[h1] == 'compt':
#             pos_compton = pre_pos[h1]
#             E1 = kinetic_energy[h1]
#             # Find descendants of 2
#             desc_of_2 = set()
#             stack = [2]
#             for idx in sorted_idx:
#                 if track_ids[idx] == 2:
#                     stack = [2]
#                     break
#             while stack:
#                 pid = stack.pop()
#                 child_mask = parent_ids[sorted_idx] == pid
#                 children = track_ids[sorted_idx][child_mask]
#                 for child in children:
#                     if child not in desc_of_2:
#                         desc_of_2.add(child)
#                         stack.append(child)
#             mask2 = ~np.isin(track_ids[sorted_idx], list(desc_of_2) + [2])
#             grp2 = sorted_idx[mask2]
#             if len(grp2) == 0:
#                 continue
#             h2 = grp2[0]
#             E2 = source_MeV - E1
#             pos_2nd = pre_pos[h2]
#         else:
#             continue
#
#         if not np.array_equal(pos_compton, pos_2nd): # TODO that's a workaround -> solve bug
#             CCevents.append([eid] + [2, 1] + pos_compton.tolist() + [1000 * E1] + [2] +
#                             pos_2nd.tolist() + [1000 * E2])
#
#     df = pandas.DataFrame(CCevents, columns=[EVENTID] + CCevents_columns)
#     global_log.debug(f"{n_events_primary} events with primary particle hitting sensor")
#     global_log.debug(f"=> {n_events_full_edep} with full energy deposited in sensor")
#     global_log.debug(f"  => {len(CCevents)} with at least one Compton interaction")
#
#     return df

@log_offline_process('CCevents', input_type = 'dataframe')
def pixelClusters2CCevents(pixelClusters, thick, speed, twindow):
    """
    Converts a DataFrame of pixel clusters into Compton camera events (CCevents) by grouping clusters
    based on their time-of-arrival and reconstructing the positions and energies of Compton and photoelectric interactions.
    Make sure that units are consistent in the inputs.

    Args:
        pixelClusters (pandas.DataFrame): DataFrame containing pixelClusters.
        thick (float): Sensor thickness (unit must be consistent with charge_speed).
        speed (float): Speed at which charge propagates through the sensor (unit must be consistent with thickness).
        twindow (float, optional): time window for coincident clusters (unit must be consistent with the TOA column of pixelClusters, i.e. ns).

    Returns:
        pandas.DataFrame: DataFrame of CCevents, each corresponding to a pair of clusters interpreted as a Compton and a photoelectric interaction.
        The output includes event ID, interaction types, 3D positions, and energies for both interactions.

    Notes:
        - Only events with exactly two clusters within the time window are considered.
        - The z-coordinate is reconstructed using the time difference and charge propagation speed.
    """

    # Group pixelClusters by time
    pixelClusters_copy = pixelClusters.copy()
    pixelClusters_copy = pixelClusters_copy.sort_values(TOA).reset_index(drop=True)
    clustering_ids = np.zeros(len(pixelClusters_copy), dtype=int)
    current_event = 0
    last_time = None

    for i, t in enumerate(pixelClusters_copy[TOA]):
        if last_time is None or (t - last_time) > twindow:
            current_event += 1
        clustering_ids[i] = current_event
        last_time = t

    pixelClusters_copy[EVENTID] = clustering_ids

    grouped = pixelClusters_copy.groupby(EVENTID)
    grouped = [group for group in grouped if len(group[1]) == 2]

    CCevents = []
    CCevents_df = pandas.DataFrame()
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
        dZ_mm = speed * (cl_compton[TOA] - cl_photoel[TOA])
        dZ_frac = dZ_mm / thick

        # 3) Calculate absolute depth of Compton interaction
        z_compton = 0  # middle of sensor (in local fractional unit)
        # TODO or use cluster size/energy ?

        # 4) Complete 3D positions
        pos_compton = [cl_compton[PIX_X_ID], cl_compton[PIX_Y_ID], z_compton]
        pos_photoel = [cl_photoel[PIX_X_ID], cl_photoel[PIX_Y_ID], z_compton + dZ_frac]

        # 5) Construct cone
        E1_keV = cl_compton[ENERGY_keV]
        E2_keV = cl_photoel[ENERGY_keV]
        CCevents.append(
            [eventid] + [2, 1] + pos_compton + [E1_keV] + [2] + pos_photoel + [E2_keV])
        CCevents_df = pandas.DataFrame(CCevents,
                                       columns=[EVENTID] + CCevents_columns)

    return CCevents_df


def local2global(CCevents, translation, rotation, npix, pitch, thickness):
    """
    Converts local fractional coordinates in a coresi events DataFrame to global coordinates.
    Assumes three sets of position columns per event.

    Args:
        CCevents (DataFrame): Input DataFrame with coresi_events_columns.
        translation (list): Translation vector [x, y, z] of the sensor in global coordinates.
        rotation (list of list): Rotation matrix (3x3) of the sensor.
        npix (int): Number of pixels along one dimension (assuming square grid).
        pitch (float): Pixel pitch (size of one pixel).
        thickness (float): Sensor thickness.

    Returns:
        DataFrame: Updated DataFrame with position columns replaced by global coordinates.
    """

    df_copy = CCevents.copy()
    for i in range(1, 3):  # For each event (1, 2)
        position_cols = [f"PositionX_{i}", f"PositionY_{i}", f"PositionZ_{i}"]

        def convert_row(row):
            c = [row[col] for col in position_cols]
            return localFractional2globalCoordinates(c, translation, rotation, npix,
                                                     pitch, thickness)

        coords = df_copy.apply(convert_row, axis=1, result_type='expand')
        df_copy[position_cols] = coords

    return df_copy


def filter_bad_CCevents(CCevents, energies_MeV, tol_MeV):
    """
    Filters events based on the sum of energies of the two interactions.
    Similar to filter_bad_events in coresi.data
    """
    energies_keV = np.array(energies_MeV) * 1000
    tol_keV = tol_MeV * 1000
    energy_sum = CCevents['Energy (keV)_1'] + CCevents['Energy (keV)_2']
    mask = np.any([
        np.abs(energy_sum - e) <= tol_keV for e in energies_keV
    ], axis=0)
    CCevents = CCevents[mask]
    global_log.debug(f"{len(CCevents)} CCevents after energy selection:\n{CCevents.head().to_string(index=False)}")
    return CCevents

def compare_simulated_CCevents(df1, df2):
    """
    Read 2 dataframes of CCevents and return the list of EventIDs that are different.
    If one has EventIDs that the other does not have, those are included too.
    """

    cols = ['EventID'] + CCevents_columns
    MISSING = object()
    def row_key(row):
        return tuple(MISSING if pandas.isna(x) else x for x in row)
    keys1 = df1.loc[:, cols].apply(row_key, axis=1)
    keys2 = df2.loc[:, cols].apply(row_key, axis=1)
    c1, c2 = Counter(keys1), Counter(keys2)
    eid_idx = cols.index("EventID") if "EventID" in cols else 0
    ids = []
    for k in set(c1) | set(c2):
        if c1[k] != c2[k]:
            eid = k[eid_idx]
            if eid is not MISSING:
                ids.append(int(eid))
    return sorted(set(ids))
