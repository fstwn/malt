# Malt and DDU/IAS Research Virtual Environment

## Prerequisites

This requires you to have the following:
- Windows (unfortunately Hops and rhino.inside won't work on OSX for now)
- [Anaconda / Miniconda](https://www.anaconda.com/products/individual)
- Rhino 7.4 or newer
- [Hops](https://developer.rhino3d.com/guides/grasshopper/hops-component/) ([Install using Rhino package manager](rhino://package/search?name=hops))

## Installation

There are four steps for getting the provided Hops components up an running:
1. Cloning this repository into a working directory of your choice
2. Setting up the `ddu_ias_research` virtual environment using `conda` (either manually or automatically)
3. Running `malt.py` in the `ddu_ias_research` environment to start the Hops Server
4. Using a Hops Component in Grasshopper to query one of the provided Hops Routes

### 1.a Set up the Virtual Environment *manually* using Conda

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

### 1.b Set up the Virtual Environment *automatically* using conda environment file

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

### 2. Running `malt.py` in the Virtual Environment

Make sure your current working directory is the directory where `malt.py` 
is located. Otherwise browse to this directory using `cd`. Make sure the
`ddu_ias_research` conda environment is active, otherwise run:
```
conda activate ddu_ias_research
```

Now you can start the Hops Server by running:
```
python malt.py
```

