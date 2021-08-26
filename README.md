# Malt and DDU/IAS Research Virtual Environment

## About Malt

Malt is a collection of Hops components for Rhino Grasshopper.

- The Hops components run using a local [ghhops-server](https://github.com/mcneel/compute.rhino3d/tree/master/src/ghhops-server-py).
- Rhino functionality is provided using [Rhino.Inside.Cpython](https://github.com/mcneel/rhino.inside-cpython).

## About the DDU/IAS Research Virtual Environment

....

## Prerequisites

This requires you to have the following:
- Windows (unfortunately Hops and Rhino.Inside.Cpython won't work under OSX for now)
- [Anaconda / Miniconda](https://www.anaconda.com/products/individual)
- Rhino 7.4 or newer
- [Hops](https://developer.rhino3d.com/guides/grasshopper/hops-component/) ([Install using Rhino package manager](rhino://package/search?name=hops))

## Installation

There are four steps for getting the provided Hops components up an running:
1. Cloning this repository into a working directory of your choice
2. Setting up the `ddu_ias_research` virtual environment using `conda`
3. Running `malt.py` in the `ddu_ias_research` environment to start the Hops Server
4. Using a Hops Component in Grasshopper to query one of the provided Hops Routes

### 2. Set up the Virtual Environment using conda environment file

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

### 3. Running `malt.py` in the Virtual Environment

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

### 4. Using one of the provided Hops Components in Grasshopper
