# Simulation and data processing tools for single-layer Timepix3 Compton cameras

A single python script to:

- Simulate gamma sources and interactions via Gate 10 / Geant4
- Simulate the Timepix3 detector response via Allpix2
- Process Timepix3 data, either simulated or measured
- Generate coincident events & cones
- Reconstruct the source via basic back-projection or CoReSi
- Validate cone intersections in case of point sources
- Visualize 3D images

Cones can be constructed from:

- Geant4/Gate 'hits'
- Gate 'singles'
- Allpix² 'pixel hits'
- measured data

Requires:

- Linux, MacOS, or Windows + WSL
- 20 GB of disk space
- Python 3.11 or 3.12 and pip
- Allpix²
- Optional: CUDA, napari

## [Quick installation](#quick-install)

Assuming python 3.11 is installed:

```
git clone git@github.com:tbilloud/gate-pixel.git && cd gate-pixel && python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && export PYTHONPATH=. && export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000 && mv venv/lib/python3.11/site-packages/opengate_core/plugins venv/lib/python3.11/site-packages/opengate_core/plugins.bak 
```

For Allpix², assuming that dependencies are installed (see below) and ROOT is
configured (with `source thisroot.sh`):

```
mkdir allpix && cd allpix && git clone --depth 1 git@github.com:allpix-squared/allpix-squared.git && cd allpix-squared && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../install-noG4 -DBUILD_GeometryBuilderGeant4=OFF -DBUILD_DepositionCosmics=OFF -DBUILD_DepositionGeant4=OFF -DBUILD_DepositionGenerator=OFF -DBUILD_GDMLOutputWriter=OFF -DBUILD_VisualizationGeant4=OFF .. && make -j4 && make install && cd .. && rm -rf .git* && cd ../..
```

-> replace 3.11 with 3.12 if needed.
-> if it fails, or for installing optional tools, follow the steps below.

## [Detailed installation](#install)

### 0) Install python

Install python 3.11 or 3.12 and pip (e.g. use pyenv on MacOS)

### 1) Clone repository:

```
git clone https://github.com/tbilloud/gate-pixel
```

### 2) Create a virtual environment

```
cd ComptonCamera
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

`pip install -r requirements.txt`

=> that might take a while

### 4) Set up environment

```
export PYTHONPATH=. 
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000
```

If environment is not set properly, you might get errors when running scripts, like:  
`ImportError: .../libG4geometry-cf4c216c.so: cannot allocate memory in static TLS block`
`ERROR: ld.so: object ... from LD_PRELOAD cannot be preloaded (...): ignored.'`

For PyCharm:

- Right-click on the script you want to run (e.g. example.py) and select `Run ...`
- You should get messages starting with `The opengate_core library cannot be loaded.`
- It might work anyway, but to remove the warnings:
    - Create a .env file, replacing `path` with your absolute path to the repository:
      ```
      LD_LIBRARY_PATH=path/venv/lib/python3.11/site-packages/opengate_core.libs:$LD_LIBRARY_PATH
      LD_PRELOAD=path/venv/lib/python3.11/site-packages/opengate_core.libs/libG4processes-d7125d28.so:path/venv/lib/python3.11/site-packages/opengate_core.libs/libG4geometry-cf4c216c.so
      GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000
      PYTHONPATH=path/coresi/coresi:$PYTHONPATH
      ```
    - Expand the box on top with the name of the script you want to run (near ▶)
    - In `Edit Configurations`, with `path` being your absolute path to the repository,
      set:
        - In `Working directory`, put `/path/Compton-Camera-TPX3`
        - In `Paths to ".env" files`, put `/path/Compton-Camera-TPX3/.env`

### 5) Optional: Install Allpix²

#### Prerequisites Ubuntu

```
sudo apt-get install libboost-all-dev # installs BOOST
sudo apt-get install libeigen3-dev # installs Eigen3
wget https://root.cern/download/root_v6.32.10.Linux-ubuntu22.04-x86_64-gcc11.4.tar.gz
tar -xzvf root_v6.32.10.Linux-ubuntu22.04-x86_64-gcc11.4.tar.gz
source root/bin/thisroot.sh # installs ROOT6
```

Note: adapt lines 3-4 to your system (https://root.cern/install/all_releases/)

#### Prerequisites MacOS

```
xcode-select --install # install Command Line Developer Tools, if not already there
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" # installs brew
brew install cmake # installs CMake
brew install boost # installs BOOST
brew install eigen # installs Eigen3
brew install root # installs ROOT6
```

Notes:

- Only tested with brew. Other ways might work too (macports, or from source).
- As of April 2025, brew installs ROOT 6.34.08 built with C++17 (needed by Allpix²)

#### Then install Allpix² without Geant4:

https://allpix-squared.docs.cern.ch/docs/02_installation/

```
mkdir allpix
cd allpix
git clone --depth 1 git@github.com:allpix-squared/allpix-squared.git
cd allpix-squared
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install-noG4 -DBUILD_GeometryBuilderGeant4=OFF -DBUILD_DepositionCosmics=OFF -DBUILD_DepositionGeant4=OFF -DBUILD_DepositionGenerator=OFF -DBUILD_GDMLOutputWriter=OFF -DBUILD_VisualizationGeant4=OFF ..
make -j4
make install
cd ..
rm -rf .git* # remove the git folder to avoid conflicts
cd ../..
```

This creates an `allpix` sub-directory, which will contain:

- an `allpix-squared` sub-directory: allpix source code in allpix-squared sub-folder, if
  installed as described in the readme.md file.
- configuration files generated for the simulation:
    - `detector_model.conf`: details about the sensor, electrodes, etc.
    - `geometry.conf`: position and orientation of the detector.
    - `main.conf`: main configuration file.
- output files generated by Allpix2:
    - `modules.root`: some results specific to Allpix modules.
    - `data.txt`: pixel hits. Format:
      pixID_x, pixID_y, TOT, local_time, global_time, pixGlobalX, pixGlobalY, pixGlobalZ
- weighting potential files, if using `gHits2allpix2pixelHits(config='precise')`

### 6) Optional: Install PyTorch

To use the PyTorch-based functions (point source validation, reconstruction):
`pip install torch`
-> Fastest basic reconstruction (i.e. apart from CoReSi), and cross-platform.

### 7) Optional: Install CuPy (Linux, Windows)

To use the CuPy-based functions (point source validation, reconstruction):
a) Install CUDA (https://developer.nvidia.com/cuda-downloads)
b) Install the Cupy package suited to your CUDA version, e.g.  
`pip install cupy-cuda115`

### 8) Optional: Install CoReSi (needs python >= 3.11)

```
mkdir coresi
cd coresi
git clone --depth 1 https://github.com/CoReSi-SPECT/coresi
rm -rf coresi/.git* # remove the git folder to avoid conflicts
cd ..
export PYTHONPATH=./coresi/coresi:$PYTHONPATH
cp coresi/coresi/constants.yaml .
pip install torch
```

This will create a `coresi` sub-directory, which will contain:

- a `coresi` sub-directory with the CoReSi source code
- files containing coincident events, which are used as input when running CoReSi (see
  TODOs)

## [Installation without simulation packages](#install-offline)

Use cases:

- For analyzing measured data, avoiding opengate and Allpix² saves time and disk space.
- For visualising with napari on macOS, a second virtual environment is needed (
  see [napari/opengate conflict](#napariopengate-conflict) below). No need to install
  opengate/Alllpix again.

```
git clone https://github.com/tbilloud/gate-pixel
cd ComptonCamera
python3 -m venv venv-offline
source venv-offline/bin/activate
pip install -r requirements-offline.txt
```

In opengate, running code without simulation is called 'offline'. See examples/offline.

## [Getting started](#getting-started)

From the root directory (if using Pycharm, set the working directory to the root), run:

`python3 example.py`

Or, if Allpix² is not needed:

`python3 examples/gateHits2pixelCoincidences.py`

If you get the error:
`QObject::moveToThread: Current thread (0x5dfc4786c040) is not the object's thread...`  
=> see section [napari/opengate conflict](#napariopengate-conflict) below.

The 1st time you run a simulation, Gate10 will install Geant4 datasets, which can take a
while. This is done only once.

TIP: if re-installing the virtual environment, save time buy copy-pasting the datasets
folder, located in `venv/lib/python3.11/site-packages/opengate_core/geant4_data`

If using optional reconstruction functions (i.e. you installed cupy, torch, CoReSi,
and/or custom functions), also try:

```
python3 examples/compare_recos.py
```

In main scripts, the first part (code until block 'OFFLINE PROCESSING') is the Gate 10
simulation.
Then, several functions are available. Their parameters are documented in the code (
`tools` sub-directory).
Step-by-step:

1) Simulate pixel hits:

- from Gate singles with gSingles2pixelHits()
- from Allpix² output with gHits2allpix2pixelHits()

2) Convert pixel hits to clusters:
   pixelHits2pixelClusters()
   There are different ways to cluster hits. One can add a custom clustering function:
    - add a pixelClusters_custom.py in the `tools` sub-directory
    - Input dataframes are described in pixelHits.py
    - Output dataframes should have the same columns as in pixelClusters.py.

3) Identify coincidences:

- from Gate4 hits with gHits2pixelCoincidences()
- from pixel hits

4) Generate cones from coincidences

- with pixelCoincidences2cones()

5) Check cone intersections from a point source:

- validate_psource() plots cone projections at the source position.

6) Reconstruct 3D image with:

- simple backprojection (cpu or gpu, if available)
- advanced techniques via CoReSi (not fully implemented yet)

For more information, see the [documentation](doc/readme.md).

### Output

By default, output data goes to different folders. Allpix data goes to the 'allpix'
sub-folder, Gate data to the folder indicared in sim.output_dir (usually just `output`).

Using the function subfolder_output(sim) from tools.utils_opengate, a sub-folder is
created containing all data, except the weighting potential file that allways go to
`allpix`.
That is because the weighting potential file can be reused, and this only happens in
case the `precise` Allpix configuration is used.
The subfolder uses the source name and activity

### Plotting

Plotting functions using napari:

- scroll between cones with scroll_cones_napari()
- show reconstructed source and detector geometry in 3D with plot_reco(). Does not work
  on MacOS yet, see [napari/opengate conflict](#napariopengate-conflict) below.

## [napari/opengate conflict]

Napari and OpenGate use the same QT backends (PyQt5), which causes conflicts.
Using Qt-based code (e.g. napari) and gate in the same script leads to warnings and
crashes.

### Ubuntu

```
WARNING: QObject::moveToThread: Current thread (0x57ad941535d0) is not the object's thread (0x57ad94c1ef50).
Cannot move to target thread (0x57ad941535d0)
```

Solution:

```
mv venv/lib/python3.11/site-packages/opengate_core/plugins venv/lib/python3.11/site-packages/opengate_core/plugins.bak
```

-> replace venv and 3.11 if needed.

### Macos

```
objc[16117]: Class QT_... is implemented in both .../opengate_core/.dylibs/QtCore and .../QtCore (0x16c7d1278) ... One of the duplicates must be removed or renamed.
objc[16117]: Class KeyV... is implemented in both .../opengate_core/.dylibs/QtCore and .../QtCore (0x16c7d12a0) ... One of the duplicates must be removed or renamed.
objc[16117]: Class RunL... is implemented in both .../opengate_core/.dylibs/QtCore and .../QtCore (0x16c7d12f0) ... One of the duplicates must be removed or renamed.
```

Solution: TODO !
