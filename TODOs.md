# RepeatParametrisedVolume() triggers 'WARNING Could not check overlap ...'

# Check tools.utils_opengate.set_fluorescence()

=> deexcitation_ignore_cut impacts number of hits, and depends on cuts

# Use 'hits.keep_zero_edep = True' to simplify and speed-up gHits2cones?

# Adapt Allpix2 interface to Opengate 10.0.2

With this version, a prefix to repeated parametrized volumes was added. I.e. values in
branch HitUniqueVolumeID values are 'pixel_param-0_0_xxxxx' instead of '0_0_xxxxx' (by
default the volume name see to be 0_0). This
prevents the Allpix2 module 'DepositionReader' from reading the volume name properly...
Solutions:

- possible to remove prefix using gate?
- rewrite ROOT file without prefix?
- update detector_name_chars in DepositionReader and change detector name in
  geometry_conf_content?
  => I tried, looks like there's a problem with character `-` in `pixel_parm-0_0`.

# Allpix2 loses global time precision with TextWriter

=> when using global time as ToA, as I do for now, this is problematic. Above 1ms, ToA
precision is 10ns, which might cause artefacts in cone reconstruction
=> I proposed a fix, pull request was merged, should be in next Allpix version (3.2.1).
See https://github.com/allpix-squared/allpix-squared/pull/54

# Gate hits sometimes have anomalously large global times

=> e.g. 1e12 ns for a 20s simulation with isotope (see main_In111_advapix.py)

# CoReSi:
- avoid the need to copy-paste 'constants.yaml' in the project's root directory
- Allow dataframe input instead of saving `coresi_temp.dat' and using read_data_file()
- Bad events lead to NANs in the whole volume, detect them in advance
- Sensor rotation only works if it was around the z-axis in the Gate simulation
- Add all CoReSi parameters to reco_bp_coresi() parameters
- Allow simple back-projection without the need for sensor geometry as input
- Allow reconstructing volumes with sensor inside it