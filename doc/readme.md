This framework combines simulation tools (Gate10, Allpix2), functions to process pixel
hit data (either simulated or measured), and reconstruction tools (custom and CoReSi).
Gate10: https://opengate-python.readthedocs.io/en/master/user_guide/user_guide_intro.html
Allpix2: https://allpix-squared.docs.cern.ch/docs/
CoReSi: https://github.com/CoReSi-SPECT/coresi

# Basics

This framework is designed for easy prototyping and is not optimized for speed. It
primarily uses pandas dataframes, which are more intuitive than numpy arrays for tasks
such as clustering pixel hits. Users can add their own functions, and some extra
functions are available for faster processing (see [below](#Extra-functions));
additional functions may be added in the future.

Functions in the `tools` sub-directory are custom-developed for this framework and can
be imported in the main scripts. You can test two basic scripts immediately after
installation; additional examples are available in the `examples` sub-directory.

Using `source.n` to generate cones from Gate hits is suitable, but it should be avoided
for pixel hit and cluster processing functions, as it assigns the same global time to
all events, simulating extreme pile-up. If using it, add individual time stamps to
events in the pixel hit dataframe.

Several reconstruction functions are available. Basic ones are intended for
understanding reconstruction and prototyping. They use back-projection. Their logic is
the same but based on different libraries (reco_bp_xxx): the default is numpy, Pytorch
and Cupy are faster but might lead to memory error in case of large volumes.
For advanced reconstruction, use CoReSi. In this case the reconstruction function,
reconstruct(), needs coresi events as input, not cones.
The method can be chosen by the user.

# Offline processing

In Open Gate, functions that are not executed during the simulation are called offline.
In this framework, that means the functions that:

- run Allpix2
- produce and process pixel hits, clusters, cones
- reconstruct images of the source

All those functions convert data from an input format to an output format (e.g.
pixelHis -> pixelClusters).
They can be replaced by custom functions if the formats stay the same.

# How to speed-up

## Gate

Use acceptance angle to focus particles to sensor, e.g.:

```
source.direction.acceptance_angle.volumes = ["sensor"]
source.direction.acceptance_angle.intersection_flag = True
```

-> does not seem to work with isotropic sources, though.

## Allpix

Depending on the simulation parameters (geometry, gam mebergy, etc...) Allpix2
simulation can be much slower than the Gate simulation.
Allpix2 can be sped up using, in gHits2allpix2pixelHits():

- `config='fast'`, if using a silicon sensor
- `charge_per_step=x`, with x>>10 (default value). E.g. 100 or 1000 (but check
  precision)
- `skip_hitless_events=True`: considerably faster if many events without hits in the
  sensor (e.g. if gamma energy is high or if the source is not focused to the sensor, as
  with isotop sources)

## Data processing

Some functions have faster (experimental) versions. For example:

- gHits2cones_numpy
- pixelHits2pixelClusters_numpy

## Reconstruction

Performance varies depending on the simulation and platform/GPU, but typically, from
fastest to slowest:

- CoReSi
- PyTorch
- Cupy
- Numpy

=> See examples/main_compare_recos.py

# Utility functions

Several utility functions are written in the `tools` subdirectory.

## utils.setup_pixels()

Save code lines to define pixels:

```
  from tools.utils_opengate import setup_pixels
  
  # ===========================
  # ==   GEOMETRY            ==
  # ===========================
  npix, pitch, thickness = 256, 55 * um, 1 * mm
  sensor = sim.add_volume("Box", "sensor")
  sensor.material = "cadmium_telluride"  # or 'Silicon'
  sensor.size = [npix * pitch, npix * pitch, thickness]
  sensor.translation = [0 * mm, 0 * mm, (thickness + al + gap) / 2 - thickness / 2]
  setup_pixels(sim, npix, sensor, pitch, thickness)
```

## utils.subdir_output()

Create a sub-directory for simulation output and avoid overwriting:

```
  from tools.utils_opengate import subdir_output

  ## ============================
  ## ==  RUN                   ==
  ## ============================
  sim.output_dir = subdir_output(base_dir='output', sim=sim, geo_name='minipix_closed_noFluo_noDoppler')
  sim.run()
  shutil.copy2(os.path.abspath(sys.argv[0]), sim.output_dir)
```

Here shutil is used to copy the main script to the output directory, which is useful for
later analysis.
Indeed, the opengate Simulation() object is need for some offline analysis functions.

## utils_opengate.theta_phi()

In case of an isotropic source, calculate the theta/phi angles that just fits the sensor
area, to avoid shooting particles everywhere in space. This saves time.
WARNING: DO NOT USE THIS IF USING source.activity = .. * Bq
It is similar to using the acceptance angle, e.g.:

```
source.direction.acceptance_angle.volumes = ["sensor"]
source.direction.acceptance_angle.intersection_flag = True
```

(
see https://opengate-python.readthedocs.io/en/master/user_guide/user_guide_reference_sources_generic_source.html#acceptance-angle)
except that it shoots exactly the number of particles set with source.n
Example usage:

```
  from tools.utils_opengate import theta_phi
  
  ## ============================
  ## == SOURCE                 ==
  ## ============================
  source = sim.add_source("GenericSource", "source")
  source.n = 100
  source.particle = "gamma"
  source.energy.mono = 140 * keV
  source.position.translation = [0 * mm, 0 * mm, -5 * mm]
  source.direction.theta, source.direction.phi = theta_phi(sensor, source)
```

## utils_opengate.get_global_translation()

Get the global translation of a volume. This is useful if the volume has mother volumes.
It was written for plot_reco() in case the parameter `detector` is used
and the sensor is in a mother volume.

# [Allpix²](#allpix2)

Allpix² is a C++ software for precise simulation of semiconductor pixel detectors.
It simulates the transport of charge carriers in semiconductor sensors and their signal
induction.
It is used primarily for detector R&D in particle physics.  
https://cern.ch/allpix-squared

Allpix² can read the hits root file from Gate.
Combined with Gate10, the entire simulation can be done with a single python file, using
the function
gHits2allpix2pixelHits() after the sim.run() in the main.py script. It does the
following:

1) run Gate10 and creates the hits root file
2) generate the three .conf files needed by Allpix²
3) run Allpix² and creates the output files data.txt and modules.root in the
   sub-folder 'allpix'
4) read data.txt and return a pandas dataframe with the pixel hits

An Allpix² simulation needs 3 configuration (.conf) files:

- detector geometry
- detector model
- simulation parameters

The main configuration file (simulation parameters) is a 'simulation chain' made of
several components:

- global parameters
- electric field
- charge deposition
- charge propagation
- charge transfer
- digitization
  https://allpix-squared.docs.cern.ch/docs/03_getting_started/06_simulation_chain/

For each component, several modules are available.

- Charge propagation:
    - ProjectionPropagation: fast but only silicon sensors, linear electric field, one
      carrier type at a time
    - GenericPropagation
    - TransientPropagation
- Charge transfer:
    - SimpleTransfer: no ToA (as of v3.1.0, 2025-01-08)
    - CapacitiveTransfer
    - PulseTransfer
    - InducedTransfer
- Digitization:
    - DefaultDigitizer
    - CSADigitizer

# Analysing hits

Gate hits (sometime called gHits here) allows to reconstruct ideal cones, as if the
Compton camera was perfect.

Pixel hits from Allpix2 allow to reconstruct cones as would be done with a Timepix3
detector.

Since pixel hit formats differ in Allpix2 and acquisition software (e.g.
Advacam's Pixet), a new format is defined here, in tools/pixelHits.py.

## Analysing Geant4 steps / Gate hits

Gate hits are similar, but not identical, to Geant4 steps. 

Geant4 steps can be logged in terminal with 
`sim.g4_verbose, sim.g4_verbose_level_tracking = True, 1`
=> EventIDs are not logged, hence better do that with small number of events

Gate hits are saved in ROOT files when the 'DigitizerHitsCollectionActor' actor is used.
They can be read as pandas dataframe with
`uproot.open(file_path)['Hits'].arrays(library='pd', entry_stop=nrows)`
To print them with same parameters as Geant4 steps, select columns:
```
'EventID', 'PostPosition_X', 'PostPosition_Y', 'PostPosition_Z', 'KineticEnergy', 
'TotalEnergyDeposit', 'StepLength', 'TrackLength', 'HitUniqueVolumeID',
'ProcessDefinedStep', 'ParticleName', 'TrackID', 'ParentID', 'ParentParticleName',
'TrackCreatorProcess', 'TrackCreatorModelName'
```

Notes:
- ProcessDefinedStep (Gate) is pre-step, ProcName (G4) is post-step
  => thus when new tracks are generated in sensor, their ProcessDefinedStep is none.
- KineticEnergy (Gate) is pre-step, KinE (G4) is post-step
- StepLength (Gate) / StepLeng (G4) can be used to match hits (Gate) / steps (G4)

## Analysing Allpix2 pixel hits

Timepix3 can measure particle interactions in frame-based or data driven mode. When used
in data driven mode, data is stored as a list of pixel hits, where hits contain energy
and time.

Allpix2 simulation produces PixelHits objects which resemble pixel hits measured with
Timepix3 detectors.
They can be stored as:

- text files using TextWriter module (use `include="PixelHit"` to avoid other data)
- ROOT files using ROOTObjectWriter module

### Pixel hit format

When using TextWriter, Allpix2 stores pixel hits per event with format:
PixelHit pixelID_x, pixelID_y, TOT, local_time, global_time, pixelGlobalCoordX,
pixelGlobalCoordY, pixelGlobalCoordZ

### Energy

Timepix3 measures the energy deposited in individual pixels via TOT (
Time-Over-Threshold).
When a detector is calibrated with a per-pixel energy calibration procedure, TOT can be
converted to energy.

### Time (TOA)

Time-of-Arrival (TOA) is measured with Timepix3 with 1.6 ns granularity.

A so-called time-walk correction can be applied to improve precision, since higher
energy deposits induce faster pulses on pixel pre-amplifiers. Precision???

Even though Compton, photo-electric and fluorescent events occur almost simultaneously (
within few ps?), the time it takes for the charge carriers to drift to the pixel
electrode can be long (depending on semiconductor and bias voltage) and it depends on
the depth of interaction.

Drift time = distance / drift speed = distance / (mobility * electric field)
Drift time = (distance * thickness) / (mobility * voltage)

Drift time depends on the electric field profile in the sensor.
CdTe sensor can have ohmic or Schottky-diode contacts.
In ohmic contacts, the electric field is constant across the sensor.

Mobility in Silicon (wikipedia):

- electrons: ~1000 cm^2/Vs
- holes: ~450 cm^2/Vs

Mobility in CdTe (wikipedia):

- electrons: ~1100 cm^2/Vs
- holes: ~100 cm^2/Vs

Supposing an ohmic sensor and constant mobility:

Drift time for 1mm in CdTe @ 1000V:
e-: ~10 ns
holes: ~100 ns

Drift speed in CdTe @ 1000V:
e-: ~100 um/ns
holes: ~10 um/ns

Drift distance during 1.6ns:
e-: ~160 um
hole: ~16 um

### Position

With a single layer Timepix3 camera:

- X/Y coordinates can be calculated from cluster shapes
- Depth of interactions (Z) is more difficult to determine

Delta_Z between two interactions can be calculated from their delta_TOA. See, for
example:
10.1088/1748-0221/15/01/C01014
