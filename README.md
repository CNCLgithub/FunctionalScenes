# FieldFunctions


## Setup and running
2. Run `./setup.sh cont_build python julia` to build the container and setup enviroment
3. Enter `./run.sh julia` to get into Julia REPL


## Contributing

### Contributing Commandments

Thou ...
1. Shalt place all re-used code in packages (`src` or `field_functions`)
2. Shalt place all interactive code in `scripts`
3. Shalt not use "hard" paths. Instead update `PATHS` in the config.
4. Shalt add contributions to branches derived from `master` or `dev`
4. Shalt not use `git add *`
5. Shalt not commit large files (checkpoints, datasets, etc). Update `setup.sh` accordingly.


### Project layout

The python package environment is managed by poetry, located under `field_functions` and can be imported using `import field_functions`

Likewise, the Julia package is described under `src` and `test`

All scripts are located under `scripts` and data/output is under `output` as specific in the project config (`default.conf` or `user.conf`)



### Changing the enviroment

To add new python or julia packages use the provided package managers (`poetry add` or `Pkg.add ` for python and julia respectively.)

> for more info checkout [poetry](https://python-poetry.org/docs/cli/) and [Pkg](https://julialang.github.io/Pkg.jl/v1/managing-packages/)
