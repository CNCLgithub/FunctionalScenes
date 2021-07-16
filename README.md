# sconfig

A utility to organize complicated project dependencies centered around Sylab's singularity containers.

## Goals

Containers such as Singularity offer an excellent refuge for most academic projects providing a general, reproducable platform to organize dependencies.
However, the garauntee of these features often arrises from immutability that can hinder project development.

A potential remedy lies with external environment blobs (such as `conda` caches) yet these mutable structures usually revive concerns originally quelled by containers such as volatility and host dependencies. 

The goal of this project is to organize immutable and volatile blobs within a project environment such that the harm of volatility is constrained to a small subset of blobs with the majority of system level dependencies still preserved in containers. 
Thus, a developer should be able to easily install, remove, or modify user-level dependencies in real-time without the use of time-consuming build processes. 
At the same time, a `setup` utility can rely on a project manifest to offer weak reproducability (ie, as long as mirrors are still valid) to automate the relatively fast build process those volative blobs providing some sense of consistency across project instances. 

## Usage

This project will generate a series of files that can serve as a template to:

1. Design a project manifest that provides the details of a reproducable container as well as volatile blobs
2. Automate the build process of all blobs
3. Automate the integration of blobs during runtime

Although not necessary, this utility is designed around having a subfolder within the primary project that segregates environment data similar to:

``` text
.
├── env.d # <- environment folder
└── script.sh # <- some code
. 
. 
. # other project material
```

### Project initialization

First add `sconfig` to your project, preferable inside `env.d`. There are several ways to go about this.
You can use either `git subtree` (preferred) or `git clone` to place this repo in the desired location.

Next run the `init.sh` script optionally providing the destination of where the project manifest will be located. 

``` bash
bash env.d/sconfig/init.sh "env.d"
```

This will generate a series of template files that can be tailored to fit project specifics. These files (described in more detail below) can then be commited to your project for reproducability.

The result should be something like:
``` text
.
├── env.d
│   ├── default.conf # <- new
│   ├── load_config.sh # <- new
│   ├── run.sh # <- new
│   ├── sconfig
│   └── setup.sh # <- new
└── script.sh
```

### Defining the project environment

The first file to tailor to your project will be `default.conf`.
This file defines a series of associative arrays that collectively define the project. 

- `SENV` array defines environment critical components such as the location and manifest of the singularity container, the path to the singularity binary, and the destination of volatile blobs

- `SPATHS` array defines host-level paths to be bound to specific local directories inside the container at run time

- `SVARS` array defines any project specific environmental variables that might be used at runtime

> NOTE: While you can certainly add more fields to existing arrays or even add entire new arrays, it is advised that you do not remove or rename the original set as they are referenced throughout the other scripts generated but `sconfig/init.sh`.

Once configured, `load_config.sh` will source these arrays and perform several preprocessing steps
If additional features are required, this file will most likely need to be modified. 

`load_config.sh` also supports an alternative config file `user.conf` (in the same directory as `default.conf`) that overrides defaults. 
This is useful when creating an instance of your project on a new host that has a unique topology (such as an HPC), where you would like to modify your config without commiting those changes to your repo. 

### Defining project build

`setup.sh`

`Singularity`

### Defining runtime

`run.sh`

## Design

If there are now immutable (interal, within the container) and volatile (external, host-side) blobs, an immediate question arises: Where do I put dependecy `x`?

From my perspective there are two kinds of dependencies:

1. Hard constraints: system level dependencies that cannot change without an overhaul of most other packages (ie `CUDA`, drivers, compilers/interpreters...)
2. Soft constraints: libraries or packages can be easily added / interchanged at the user level with the use of a package manager

Given that reproducability must be transiently weakend for ease of development, the aim of this subdivision is to put all hard constraints in the manifest of immutable containers and organize soft constraints using a project manifest
