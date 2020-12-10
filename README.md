# FieldFunctions


## Setup and running

1. Run `./setup.sh cont_build python julia` to build the container and setup enviroment
2. Enter `./run.sh julia` to get into Julia REPL

This project has automatic configuration!! This configuration is defined in `default.conf`.
You should always prepend `./run.sh` before any command (including running programs like `julia`) to ensure consistency. 
If you wish to have different values than `default.conf`, simply:

``` sh
cp default.conf user.conf
vi user.conf # edit to your liking without adding new elements
```

## Mac and Window users

In order to use singularity you must have a virtual machine running. 
Assuming you have vagrant (and something like virtualbox) setup on your host, you can follow these steps

### Using `setup.sh`


### Using `run.sh`

Provision the virtual machine defined in `Vagrantfile` with:

``` sh
vagrant up
```




Create a `user.conf` as described above

> Note: git will not track `user.conf`

Modify `user.conf` such that `path` is set to route through vagrant

``` yaml
[ENV]
path:vagrant ssh -c singularity
```


## Contributing

### Contributing Commandments

Thou ...
1. Shalt place all re-used code in packages (`src` or `functional_scenes`)
2. Shalt place all interactive code in `scripts`
3. Shalt not use "hard" paths. Instead update `PATHS` in the config.
4. Shalt add contributions to branches derived from `master` or `dev`
4. Shalt not use `git add *`
5. Shalt not commit large files (checkpoints, datasets, etc). Update `setup.sh` accordingly.


### Project layout

The python package environment is managed by poetry, located under `functional_scenes` and can be imported using `import functional_scenes`

Likewise, the Julia package is described under `src` and `test`

All scripts are located under `scripts` and data/output is under `output` as specific in the project config (`default.conf` or `user.conf`)



### Changing the enviroment

To add new python or julia packages use the provided package managers (`poetry add` or `Pkg.add ` for python and julia respectively.)

For julia you can also use `] add ` in the REPL

> for more info checkout [poetry](https://python-poetry.org/docs/cli/) and [Pkg](https://julialang.github.io/Pkg.jl/v1/managing-packages/)
