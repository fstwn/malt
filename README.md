# HopsLauncher and Research Environment

## Prerequisites

This requires you to have the following:
- Windows (unfortunately Hops and rhino.inside won't work on OSX for now)
- Anaconda / Miniconda
- Rhino 7

## Installation

There are two ways to install:

### 1. Set up Environment manually using Conda

First add conda-forge and open3d-admin to channels
```
conda config --add channels conda-forge
conda config --append channels open3d-admin
```

Next create a new virtual environment, you can choose a different name, here
we use `ddu_ias_research`:
```
conda create -n ddu_ias_research python=3.7 numpy scipy scikit-learn compas compas_view2 compas_fab compas_cgal igl tensorflow h5py flask flake8 open3d
```

Now we activate our newly created conda environment:
```
conda activate ddu_ias_research
```

Now we use `pip` to install `ghhops-server` and `rhinoinside` since they are
not available through conda channels:
```
pip install ghhops-server rhinoinside
```

Finally, we install `compas_rhino` and the other compas extensions by running:
```
python -m compas_rhino.install -v 7.0 -p compas compas_rhino compas_ghpython compas_fab compas_cgal
```

### 2. Set up Environment automatically using conda environment file

Set up the environment `ddu_ias_research` by running:
```
conda env create -f "ddu_ias_research.yml"
```

Now we activate our newly created conda environment:
```
conda activate ddu_ias_research
```

Finally, we install `compas_rhino` and the other compas extensions by running:
```
python -m compas_rhino.install -v 7.0 -p compas compas_rhino compas_ghpython compas_fab compas_cgal
```
