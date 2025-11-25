# DigitizerBlurringActor incompatible with ion sources

If sim.physics_manager.enable_decay = True, DigitizerBlurringActor gives duplicated
'singles'. This is especially problematic for some clustering algorithms.

# napari/opengate conflict

Napari and OpenGate use the same QT backends (PyQt5). Using both in the same script
leads to warnings and crashes.

## Ubuntu

```
WARNING: QObject::moveToThread: Current thread (0x57ad941535d0) is not the object's thread (0x57ad94c1ef50).
Cannot move to target thread (0x57ad941535d0)
```

Solution:

```
mv venv/lib/python3.11/site-packages/opengate_core/plugins venv/lib/python3.11/site-packages/opengate_core/plugins.bak
```

-> replace venv and 3.11 if needed.

## Macos

```
objc[16117]: Class QT_... is implemented in both .../opengate_core/.dylibs/QtCore and .../QtCore (0x16c7d1278) ... One of the duplicates must be removed or renamed.
objc[16117]: Class KeyV... is implemented in both .../opengate_core/.dylibs/QtCore and .../QtCore (0x16c7d12a0) ... One of the duplicates must be removed or renamed.
objc[16117]: Class RunL... is implemented in both .../opengate_core/.dylibs/QtCore and .../QtCore (0x16c7d12f0) ... One of the duplicates must be removed or renamed.
```

Solution: TODO !

# RepeatParametrisedVolume() triggers 'WARNING Could not check overlap ...'

# Use 'hits.keep_zero_edep = True' to simplify and speed-up gHits2CCevents?

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
- Sensor rotation only works if it was around the z-axis in the Gate simulation
- Add all CoReSi parameters to reco_bp_coresi() parameters
- Allow simple back-projection without the need for sensor geometry as input
- Allow reconstructing volumes with sensor inside it
- Add constants for CdTe, in particular for Doppler broadening