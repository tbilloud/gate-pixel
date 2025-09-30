# Show how to compare measurement with simulation

from pathlib import Path
from pandas import read_csv
from tools.CCevents import local2global, pixelClusters2CCevents, gHits2CCevents
from tools.utils import charge_speed_mm_ns, create_sensor_object
from tools.pixelHits import ENERGY_keV, remove_edge_pixels, singles2pixelHits
from tools.pixelClusters_custom import pixelHits2pixelClusters
from tools.reconstruction import valid_psource, reconstruct
from tools.utils_plot import compare_pixelClusters, plot_energies, compare_recos

path = Path('/Users/thomas/DATA')
path_meas = path / 'TIMEPIX3/Adavapix_TPX3_2mm_CdTe/In_5deg_pixelHits.csv'
path_sim = path / 'SIMULATION/advapix_In111/ion_49_111_detX11.33mm_detZ-129.5mm/4500kBq_20s/'
sensor_pos = [11.33, 0., -128.5]
source_pos = [0., 0., 0.]
npix, pitch, thick = 256, 0.055, 2
source_MeV = 0.245
s = create_sensor_object(size=[14.08, 14.08, 2], translation=sensor_pos)
spd = charge_speed_mm_ns(mobility_cm2_Vs=1000, bias_V=450, thick_mm=thick)
single_file, single_name = path_sim / 'gateSingles_blur.root', 'Single_b'

# ########################## HITS  ##################################
nhits = 10_000
hits_meas = read_csv(path_meas, nrows=nhits)
hits_allp = read_csv(path_sim / 'pixelHits_allpix.csv', nrows=nhits)

# TOA CUTS
hits_allp = hits_allp[hits_allp['ToA (ns)'] <= 2.1e10]  # outliers in Gate global time

# ENERGY CUTS
emin, emax = 10, source_MeV * 1000
hits_meas = hits_meas[(hits_meas[ENERGY_keV] > emin) & (hits_meas[ENERGY_keV] < emax)]
hits_allp = hits_allp[(hits_allp[ENERGY_keV] > emin) & (hits_allp[ENERGY_keV] < emax)]

# BORDER PIXEL CUTS
cut_edge = 5
hits_meas = remove_edge_pixels(hits_meas, npix, edge_thickness=cut_edge)
hits_allp = remove_edge_pixels(hits_allp, npix, edge_thickness=cut_edge)

# ######################### CLUSTERS  ###############################
clust_meas = pixelHits2pixelClusters(hits_meas, window_ns=100, npix=256)
clust_allp = pixelHits2pixelClusters(hits_allp, window_ns=100, npix=256)

# ENERGY CUTS
eClstrMax = 260
clust_meas = clust_meas[clust_meas[ENERGY_keV] < eClstrMax]
clust_allp = clust_allp[clust_allp[ENERGY_keV] < eClstrMax]
# compare_pixelClusters(clust_meas, clust_allp, name_a='Measurement', name_b='Allpix', energy_bins=eClstrMax)

# ######################## CCevents  ################################
ev_meas = pixelClusters2CCevents(clust_meas, thick=thick, speed=spd, twindow=100)
ev_allp = pixelClusters2CCevents(clust_allp, thick=thick, speed=spd, twindow=100)
plot_energies(min_keV=0, max_keV=260,
              names=['Measurement', 'Allpix'],
              hits_list=[hits_meas, hits_allp],
              clusters_list=[clust_meas, clust_allp],
              CCevents_list=[ev_meas, ev_allp],
              ylog=False)

# ############################## RECO  ###############################
ev_meas = local2global(ev_meas, s.translation, s.rotation, npix, pitch, thick)
ev_allp = local2global(ev_allp, s.translation, s.rotation, npix, pitch, thick)
ev_refc = gHits2CCevents(path_sim / 'gateHits.root', source_MeV='Cd111[245.390]_245.390', entry_stop=nhits * 2)
reco_params = {'method': 'torch',
               'vpitch': 1,
               'vsize': tuple(int(x * 4) for x in [s.size[0], s.size[1], abs(sensor_pos[2])]),
               'cone_width': 0.002,
               'energies_MeV': [source_MeV], 'tol_MeV': 0.03}
v_meas = reconstruct(ev_meas, **reco_params)
v_allp = reconstruct(ev_allp, **reco_params)
v_refc = reconstruct(ev_refc, **reco_params)
# v_meas = np.flip(v_meas, axis=2)

# ############################ DISPLAY  ###############################
compare_recos([v_meas, v_allp, v_refc], names=['Measurement', 'Allpix', 'Reference'])