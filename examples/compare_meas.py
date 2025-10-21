# Show how to compare measurement with simulation
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from opengate import g4_units, Simulation
from pandas import read_csv
from tools.CCevents import local2global, pixelClusters2CCevents, gHits2CCevents
from tools.allpix import gHits2allpix2pixelHits
from tools.utils import charge_speed_mm_ns, get_pixID_2D
from tools.pixelHits import ENERGY_keV, remove_edge_pixels, pixet2pixelHit
from tools.pixelClusters_custom import pixelHits2pixelClusters
from tools.reconstruction import valid_psource, reconstruct
from tools.utils_opengate import setup_pixels, setup_hits, set_fluorescence
from tools.utils_plot import compare_pixelClusters, plot_energies, compare_recos, \
    plot_reco

path = Path('/media/billoud/029A94FF9A94F101/2nd_DRIVE/Advapix_TPX3_2mm_CdTe/In/')
um, mm, keV, Bq, s = g4_units.um, g4_units.mm, g4_units.keV, g4_units.Bq, g4_units.s

def run(sensor_translation, time_sec, meas_file):
    ## ============================
    ## ==  GATE SIMULATION       ==
    ## ============================
    sim = Simulation()
    sim.world.size = [300 * mm, 300 * mm, 300 * mm]
    sim.world.material = "G4_AIR"
    npix, pitch, t = 256, 55 * um, 2 * mm
    sensor = sim.add_volume("Box", "sensor")
    sensor.material = 'G4_CADMIUM_TELLURIDE'
    sensor.size = [npix * pitch, npix * pitch, t]
    sensor.translation = sensor_translation # [0, -11.33, 128.5]
    setup_pixels(sim, npix, sensor, pitch, t)
    hits = setup_hits(sim, sensor_name=sensor.name)
    source = sim.add_source("GenericSource", "source") # in the center if translation not specified
    # source.particle, source.energy.mono = "gamma", 245 * keV
    source.particle, source.half_life = 'ion 49 111', 2.81 * g4_units.day # In111
    source.activity = 4.5e6 * Bq
    sim.physics_manager.enable_decay = True
    sim.physics_manager.physics_list_name = 'G4EmLivermorePhysics'  # Doppler effect
    set_fluorescence(sim)
    hits.output_filename = Path(meas_file).stem + "_gateHits.root"
    sim.output_dir = path / f'simulation/{sim.physics_manager.physics_list_name}_{time_sec}s'
    sim.run_timing_intervals = [[0, time_sec * s]]
    sim.run(start_new_process=True)

    ## ============================
    ## ==  OFFLINE PROCESSING    ==
    ## ============================
    bias = -500  # in V
    mobility_e = 1000  # main charge carrier mobility in cm^2/Vs
    spd = charge_speed_mm_ns(mobility_cm2_Vs=1000, bias_V=bias, thick_mm=t)

    # ########################## HITS  ##################################
    hit_allp = gHits2allpix2pixelHits(sim,
                                      npix=npix,
                                      config='precise',
                                      skip_hitless_events=True,
                                      bias_V=bias,
                                      mobility_electron_cm2_Vs=mobility_e,
                                      mobility_hole_cm2_Vs=500,
                                      threshold_smearing=30,
                                      electronics_noise=110,
                                      charge_per_step=100,
                                      )
    hit_allp.to_csv(sim.output_dir / (Path(meas_file).stem + "_pixelHits.root"), index=False)
    hit_meas = pixet2pixelHit(path / meas_file, path.parent / 'CALIBRATION', nrows=len(hit_allp))

    # TOA CUTS
    hit_allp = hit_allp[hit_allp['ToA (ns)'] <= 2.1e10]  # outliers in Gate global time

    # ENERGY CUTS
    emin, emax = 10, 280
    hit_allp = hit_allp[(hit_allp[ENERGY_keV] > emin) & (hit_allp[ENERGY_keV] < emax)]
    hit_meas = hit_meas[(hit_meas[ENERGY_keV] > emin) & (hit_meas[ENERGY_keV] < emax)]

    # BORDER PIXEL CUTS
    cut_edge = 5
    hit_allp = remove_edge_pixels(hit_allp, npix, edge_thickness=cut_edge)
    hit_meas = remove_edge_pixels(hit_meas, npix, edge_thickness=cut_edge)

    # ######################### CLUSTERS  ###############################
    clust_allp = pixelHits2pixelClusters(hit_allp, window_ns=100, npix=256)
    clust_meas = pixelHits2pixelClusters(hit_meas, window_ns=100, npix=256)

    # ENERGY CUTS
    eClstrMax = 260
    clust_allp = clust_allp[clust_allp[ENERGY_keV] < eClstrMax]
    clust_meas = clust_meas[clust_meas[ENERGY_keV] < eClstrMax]

    compare_pixelClusters(clust_meas, clust_allp, name_a='Measurement', name_b='Allpix',
                          energy_bins=eClstrMax)

    # ######################## CCevents  ################################
    ev_allp = pixelClusters2CCevents(clust_allp, thick=t, speed=spd, twindow=100)
    ev_meas = pixelClusters2CCevents(clust_meas, thick=t, speed=spd, twindow=100)
    ev_refc = gHits2CCevents(sim.output_dir / hits.output_filename, 'Cd111[245.390]_245.390')
    # 'Cd111[245.390]_245.390' if isotope source

    plot_energies([hit_meas, hit_allp], [clust_meas, clust_allp], [ev_meas, ev_allp],
                  max_keV=260, names=['Measurement', 'Allpix'])

    ev_meas = local2global(ev_meas, sensor.translation, sensor.rotation, npix, pitch, t)
    ev_allp = local2global(ev_allp, sensor.translation, sensor.rotation, npix, pitch, t)

    ev_allp.to_csv(sim.output_dir / (Path(meas_file).stem + "_CCevents_allpix.root"), index=False)
    ev_meas.to_csv(sim.output_dir / (Path(meas_file).stem + "_CCevents_meas.root"), index=False)
    ev_refc.to_csv(sim.output_dir / (Path(meas_file).stem + "_CCevents_ref.root"), index=False)

    return ev_allp, ev_meas, ev_refc,

if __name__ == "__main__":

    measurements = [
        [[0 * mm, 0 * mm, 128 * mm], 100, 'In_5deg.t3pa'],
        [[-90 * mm, 0 * mm, 90 * mm], 100, 'In_45deg.t3pa'],
        [[-128 * mm, 0 * mm, 0 * mm], 100, 'In_90deg.t3pa'],
    ]

    ev_refc, ev_meas, ev_allp =  pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for sensor_translation, time, meas in measurements:
        ev_allp_i, ev_meas_i, ev_refc_i = run(sensor_translation, time, meas)
        ev_allp = pd.concat([ev_allp, ev_allp_i], ignore_index=True)
        ev_meas = pd.concat([ev_meas, ev_meas_i], ignore_index=True)
        ev_refc = pd.concat([ev_refc, ev_refc_i], ignore_index=True)

    reco_params = {'method': 'torch',
                   'vpitch': 1,
                   'vsize': [256, 256, 256],
                   'cone_width': 0.01,
                   'energies_MeV': [0.245], 'tol_MeV': 0.05}
    vm, va, vr = (reconstruct(ev, **reco_params) for ev in (ev_meas, ev_allp, ev_refc))

    # # ############################ DISPLAY  ###############################
    import matplotlib.pyplot as plt
    plt.imshow(va[:,127,:],origin='lower'), plt.show()
    plt.imshow(vm[:,127,:],origin='lower'), plt.show()
    plt.imshow(vr[:,127,:],origin='lower'), plt.show()
    # plot_reco(va, reco_params['vpitch'],[14.08, 14.08, 2.0],[0 * mm, 0 * mm, 128 * mm])
    #compare_recos([vm, va, vr], ['Measurement', 'Allpix', 'Reference'], 0)
