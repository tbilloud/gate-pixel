# Simulation and data processing tools for single-layer Timepix3 Compton cameras

A single python script to:

- Simulate gamma sources and interactions via Gate 10 / Geant4
- Simulate the Timepix3 detector response (data-driven mode) via Allpix²
- Process Timepix3 data, either simulated or measured
- Generate Compton camera events & cones
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

## [Quick installation](#quick-install)

Assuming Python 3.11 is installed (replace 3.11 with 3.12 if needed):

```
git clone https://github.com/tbilloud/gate-pixel.git && cd gate-pixel && python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && export PYTHONPATH=. && export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000 && mv venv/lib/python3.11/site-packages/opengate_core/plugins venv/lib/python3.11/site-packages/opengate_core/plugins.bak 
```

For Allpix², assuming dependencies are installed (
see [5) Optional: Install Allpix2](#5-optional-install-allpix2)) and ROOT is
configured (with `source thisroot.sh`):

```
mkdir allpix && cd allpix && git clone https://github.com/allpix-squared/allpix-squared.git && cd allpix-squared && git reset --hard f542ff9 && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../install-noG4 -DBUILD_GeometryBuilderGeant4=OFF -DBUILD_DepositionCosmics=OFF -DBUILD_DepositionGeant4=OFF -DBUILD_DepositionGenerator=OFF -DBUILD_GDMLOutputWriter=OFF -DBUILD_VisualizationGeant4=OFF .. && make -j4 && make install && cd .. && rm -rf .git* && cd ../..
```

For CoReSi:

```
mkdir coresi && cd coresi && git clone --depth 1 https://github.com/CoReSi-SPECT/coresi && rm -rf coresi/.git* && cd .. && export PYTHONPATH=./coresi/coresi:$PYTHONPATH && cp coresi/coresi/constants.yaml . && pip install torch
```

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

=> might take a while

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
git clone git@github.com:allpix-squared/allpix-squared.git
cd allpix-squared
git reset --hard f542ff9
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
- output files generated by Allpix²:
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
- files containing Compton camera events, which are used as input when running CoReSi (
  see
  TODOs)

## [Getting started](#getting-started)

From the root directory (if using Pycharm, set the working directory to the root), run:

`python3 examples/main.py`

Or, if Allpix² is not needed:

`python3 examples/without_allpix.py`

The 1st time you run a simulation, Gate10 will install Geant4 datasets, which can take a
while. This is done only once.

If using optional reconstruction functions (i.e. you installed cupy, torch, CoReSi,
and/or custom functions), also try:

```
python3 examples/compare_recos.py
```

A script is composed of a Gate simulation and/or 'offline' processing.
Offline processing include the Allpix² simulation, pixel hit/cluster/event/cone
processing, and image reconstruction.
Offline processing with Allpix² requires the `sim` object from Gate
-> use `shutil.copy2(os.path.abspath(sys.argv[0]), sim.output_dir)` after `sim.run()`

Data from different tools go to different folders.

- Gate -> `sim.output_dir` (ROOT files with hits, etc).
- Allpix -> `allpix` sub-folder (configuration files, etc).
- CoReSi -> `coresi` sub-folder (event files, etc).

After running the Gate simulation, one can:

1) Simulate pixel hits:

- from Gate hits with gHits2allpix2pixelHits()
- from Gate singles with gSingles2pixelHits()

2) Cluster pixel hits with pixelHits2pixelClusters()

3) Identify Compton camera events (CCevents):

- from Gate hits with gHits2CCevents()
- from clusters with pixelClusters2CCevents()

4) Generate cones from CCevents with CCevents2CCcones()

5) Check cone intersections from a point source with validate_psource()

The function needs the source position as input. It draws a point at its location on top
of a reconstruction slice.

6) Reconstruct 3D image with:

- simple back-projection (cpu or gpu, if available)
- advanced techniques via CoReSi (WIP)

7) Display a 3D image with plot_reco()

Or multiple images side-by-side with plot_recos().
On macOS, these functions cannot yet be used in the same script as Gate (
see [napari/opengate conflict](#napariopengate-conflict)).

See the [documentation](doc/readme.md) and function definitions in the code for more
details.

## Customise

Functions in this tool box are prototypes, easy to understand but not fast. 
Also, there are different ways to process pixel hits or reconstruct images.
Any step can be easily customised.

For example, to add a custom clustering function:

1.Add a file named e.g. `pixelClusters_custom.py` in the `tools` sub-directory
2.Write a function named e.g. `pixelHits2pixelClusters_custom()` whose input/output dataframes have the same columns as the original function (check this using `sim.verbose_level = 'DEBUG' in one of the provided example).
3.Import the function in your main script

=> an example is provided: `tools/pixelClusters_custom.py`

