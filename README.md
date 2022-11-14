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
- [Hops](https://developer.rhino3d.com/guides/grasshopper/hops-component/) ([Install using Rhino package manager by using the `_PackageManager` command](rhino://package/search?name=hops))

# Installation

## 1. Clone the repository into a directory of your choice

First off, clone or download this repository and unzip it (if needed) into a
working directory of your choice. For the purpose of this guide, I will use
`C:\source\repos` as my directory for all repositories.

If you want to clone using the Command line, `cd` into your repo directory, i.e.:
```
cd "C:\source\repos"
```

You can then clone the repository into your current working directory:
```
git clone https://github.com/fstwn/malt.git
```

You should now end up with a new folder `malt` inside your working directory,
containing all the files of the repository, so that the full path is
`C:\source\repos\malt`

## 2. Set up the Virtual Environment using conda environment file

***NOTE: If you have not installed 
[Anaconda / Miniconda](https://www.anaconda.com/products/individual) yet, NOW
is the time to do it.***

Using a Windows Terminal or Powershell, `cd` into the directory where **you** have 
cloned/unpacked the `malt` repository. For me that's running:
```
cd "C:\source\repos\malt"
```

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

With the virtual environment activated and while in the root directory of the
malt repository (where `setup.py` is located!), run the following command:
```
pip install -e .
```

*NOTE: This will install the `malt` package and its submodules in development
mode, in case you want to extend and/or modify it. If you simply want to use
the provided functions and components, you can also simply call
`pip install .`*

## 4. Running the Hops Server in the Virtual Environment

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

Note that you can also run the componentserver using different command line options:
- `python componentserver.py -d` will run the server in debug mode.
- `python componentserver.py -f` will run the server without using Flask.
- `python componentserver.py -n` will run the server in network access mode. **WARNING: THIS IS POTENTIALLY *VERY* DANGEROUS!**

## 5. Using one of the provided Hops Components in Grasshopper

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

The component that is available at this endpoint will then be loaded:

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

## 6. Development

### 6.1 Components

If you want to contribute to malt development, the easiest way is to develop
own components and add them to `componentserver.py`. For now, all components
have to be defined in this file, since Hops does not support it in any other
way yet.

### 6.2 Submodules

If you want to contribute by adding submodules, please add your modules in the
`/malt` directory. Make sure that you have installed the malt package in
development mode using `pip install -e .`.

### 6.3 Tests

Malt uses `pytest` for testing. It is included in the `ddu_ias_research.yml`
conda environment file and does not need to be installed separately.
Tests go in the `/tests` directory. They are organized the same way as the
structure of the malt package and its submodules.

To run all available test, call
```
invoke test
```

### 6.4 Linting

Please use the `flake8` linter when contributing code to malt. It is included
in the `ddu_ias_research.yml` conda environment file and does not need to be
installed separately.

To lint all code, call
```
invoke lint
```

## Licensing & References

- Original code is licensed under the MIT License.
- The `malt.ipc` module is based on code by Yijiang Huang. This code is licensed under the MIT License found under `licenses/compas_rpc_example`.
- The `malt.intri` module is based on the [intrinsic-triangulations-tutorial](https://github.com/nmwsharp/intrinsic-triangulations-tutorial) code by Nicholas Sharp and Mark Gillespie & Keenan Crane. Unfortunately, no license is provided by its authors for this public open-source code. The code is based on the paper *"Navigating Intrinsic Triangulations" by Nicholas Sharp, Yousuf Soliman & Keenan Crane, ACM Transactions on Graphics, 2019*.
- The redistributed executables of [ShapeSPH](https://github.com/mkazhdan/ShapeSPH) are licensed under the MIT License found under `licenses/ShapeSPH`.
- The [FFTW](http://www.fftw.org/) .dlls that are redistributed with the ShapeSPH executables are licensed under the GNU General Public License founde under `licenses/FFTW`.
- The `malt.sshd` module is based on [ShapeDescriptor](https://github.com/ReNicole/ShapeDescriptor) by GitHub user [ReNicole](https://github.com/ReNicole). Unfortunately, no license is provided by its author for this public open-source code. The code is based on the paper *"Description of 3D-shape using a complex function on the sphere" by D.V. Vranic, D. Saupe, 2002*.

## To-Do & Extension Ideas

### Possible Future Integrations

- [Google Mediapipe](https://github.com/google/mediapipe)
