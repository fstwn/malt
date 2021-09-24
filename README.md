# Malt and DDU/IAS Research Virtual Environment

## About Malt

Malt is a collection of Hops components for Rhino Grasshopper.

- The Hops components run using a local [ghhops-server](https://github.com/mcneel/compute.rhino3d/tree/master/src/ghhops-server-py).
- Rhino functionality is provided using [Rhino.Inside.Cpython](https://github.com/mcneel/rhino.inside-cpython).

## About the DDU/IAS Research Virtual Environment

The provided environment file `ddu_ias_research_env.yml` tries to unify several
tools used at DDU and IAS into a single conda virtual environment for better
interaction and collaboration in research.

## Prerequisites

If you want to use the provided Hops Components, you need the following:
- Windows (unfortunately Hops and Rhino.Inside.Cpython won't work under OSX for now)
- [Anaconda / Miniconda](https://www.anaconda.com/products/individual)
- Rhino 7.4 or newer
- [Hops](https://developer.rhino3d.com/guides/grasshopper/hops-component/) ([Install using Rhino package manager](rhino://package/search?name=hops))

# Installation

## 1. Clone the repository into a directory of your choice

First off, clone or download this repository and unzip it (if needed) into a
working directory of your choice. For the purpose of this guide, I will use
`C:\source\repos\malt`.

## 2. Set up the Virtual Environment using conda environment file

***NOTE: If you have not installed 
[Anaconda / Miniconda](https://www.anaconda.com/products/individual) yet, NOW
is the time to do it.***

Set up a new conda virtual environment with the name `ddu_ias_research` using
the provided environment file by running:
```
conda env create -f "ddu_ias_research_env.yml"
```
*NOTE: This step can take quite some time and will take up a good amount of
space on your disk. You can expect at least ~5GB!*

Now we activate our newly created conda environment:
```
conda activate ddu_ias_research
```

## 3. Installing the malt python package

Using a Windows Terminal or Powershell, `cd` into the directory where you have 
cloned/unpacked the `malt` repository. For me that's running:
```
cd "C:\source\repos\malt"
```

With the virtual environment activated and while in the root directory of the
malt repository (where `setup.py` is located!), run the following command:
```
pip install -e .
```

*NOTE: This will install the `malt` package and its submodules in development
mode, in case you want to extend and/or modify it. If you simply want to use
the provided functions and components, you can also simply call
`pip install .`*

## 4. Installing COMPAS and its extensions for use in Rhino

Finally, we install `compas_rhino` and the other compas extensions by running:
```
python -m compas_rhino.install -v 7.0 -p compas compas_rhino compas_ghpython compas_fab compas_cgal malt
```

*NOTE: we are also registering the `malt` package with compas. We do
this so that we can call the functions not just by using Hops but also by using
[COMPAS Remote Procedure Calls](https://compas.dev/compas/latest/tutorial/rpc.html).*

## 5. Running the Hops Server in the Virtual Environment

Make sure your current working directory is the directory where `componentserver.py` 
is located. Otherwise browse to this directory using `cd` (as we did in step 3).
 Make sure the `ddu_ias_research` conda environment is active, otherwise run:
```
conda activate ddu_ias_research
```

Now you can start the Hops Server by running:
```
python componentserver.py
```

## 6. Using one of the provided Hops Components in Grasshopper

Once the server is running, you can query it at different endpoints. When you
start the server, all available endpoints are printed to the console:

![Available Endpoints](/resources/readme/readme_01.png)

For a demo you can open the latest example file available in the `gh_dev`
folder. But you can of course also start from scratch:

Open Rhino and Grasshopper and start by placing a Hops Component on the canvas:

![Placing a new Hops component](/resources/readme/readme_02.png)

Doubleclick the Hops Component and set it to one of the available endpoints.
Note that the Hops Server is running under `http://localhost:5000/`.

![Setting an Endpoint](/resources/readme/readme_03.png)

The respective component that is available at this Endpoint will then be loaded:

![Setting an Endpoint](/resources/readme/readme_04.png)

I recommend to run the available Hops Components asynchronously because this
will add a lot of responsiveness to your Grasshopper definition. I did not test
the caching functionality extensively, so feel free to experiment with that.
For more info on the available settings, please see [here](https://developer.rhino3d.com/guides/grasshopper/hops-component/#component-settings).

![Asynchronous execution](/resources/readme/readme_05.png)

You can now use the loaded Hops Component like any other Grasshopper component.
In this example, I first computed geodesic distances on a mesh from a source
vertex using the Heat Method available at the `/intri.HeatMethodDistance`
endpoint. Then I use the resulting values at each vertex to draw isocurves
on the mesh using the `/igl.MeshIsoCurves` endpoint.

![Geodesic Heat Isocurves](/resources/readme/readme_06.png)

## To-Do & Extension Ideas

### Possible Future Integrations

- [Google Mediapipe](https://github.com/google/mediapipe)