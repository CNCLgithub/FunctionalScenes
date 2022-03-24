#!/bin/bash

#################################################################################
# Environment definition
#################################################################################
sconfig_dir=$(realpath "$0" | xargs dirname)
. "$sconfig_dir/load_config.sh"

#################################################################################
# Usage
#################################################################################
usage="$(basename "$0") [targets...] -- setup project according to [default|local].conf
supported targets:
    cont_[pull|build] : either pull the singularity container or build from ENV => build
    all : all non container blobs
    python : build the python environment
    julia : build julia environment

examples:
    # pull container and setup all external blobs
    ./env.d/setup.sh cont_pull all
    # build and setup python speficially (if supported)
    ./env.d/setup.sh cont_build python
"
[ $# -eq 0 ] || [[ "${@}" =~ "help" ]] && echo "$usage"

#################################################################################
# Variable declarations
#################################################################################
cont_pull_url="https://yale.box.com/shared/static/934sa3kkcl63aj0nwzmqwxtw25efgpj8.sif"
SING="${SENV[sing]}"
BUILD="${SENV[envd]}/${SENV[def]}"
cont_dest="${SENV[envd]}/${SENV[cont]}"


#################################################################################
# Container setup
#################################################################################
[[ "${@}" =~ "cont_build" ]] || [[ "${@}" =~ "cont_pull" ]] || \
    echo "Not touching container"

[[ "${@}" =~ "cont_pull" ]] && [[ -z "${cont_pull_url}" ]] || \
    [[ "${cont_pull_url}" == " " ]] && \
    echo "Tried to pull but no link provided in \$cont_pull_url"
[[ "${@}" =~ "cont_pull" ]] && [[ -n "${cont_pull_url}" ]] && \
    [[ "${cont_pull_url}" != " " ]] && \
    echo "pulling container" && \
    wget "$cont_pull_url" -O "${cont_dest}"

[[ "${@}" =~ "cont_build" ]] && echo "building ${BUILD} -> ${cont_dest}" && \
    SINGULARITY_TMPDIR="${SPATHS[tmp]}" sudo -E $SING build \
    "$cont_dest" "$BUILD"

#################################################################################
# Python setup
#################################################################################
[[ "${@}" =~ "python" ]] || echo "Not touching python"
[[ "${@}" =~ "all" ]] || [[ "${@}" =~ "python" ]] && \
    echo "building python env at ${SENV[pyenv]}" && \
    $SING exec "${cont_dest}" bash -c "virtualenv ${SENV[pyenv]} && \
    source ${SENV[pyenv]}/bin/activate && \
    python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install -r requirements.txt"
    # ./env.d/run.sh curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz && \
    # ./env.d/run.sh tar xzf 1.10.0.tar.gz -C "${SENV[pyenv]}" && \
    # ./env.d/run.sh python -m pip install -v git+https://github.com/facebookresearch/pytorch3d.git@stable

#################################################################################
# Julia setup
#################################################################################
[[ "${@}" =~ "julia" ]] || echo "Not touching julia"
[[ "${@}" =~ "all" ]] || [[ "${@}" =~ "julia" ]] && \
    echo "building julia env" && \
    ./env.d/run.sh julia -e '"using Pkg; Pkg.instantiate();"'

#################################################################################
# Project data
# (ie datasets and checkpoints)
#################################################################################
[[ "${@}" =~ "datasets" ]] || [[ "${@}" =~ "datasets" ]] || \
    echo "Not touching datasets"
[[ "${@}" =~ "all" ]] || [[ "${@}" =~ "datasets" ]] && \
    echo "pulling datasets" && \
    wget "https://yale.box.com/shared/static/6nfzj0i9wbhseduiy9n9kq53hiwdfd50.tar" \
    -O "${SPATHS[datasets]}/alexnet_places365.pth.tar"

[[ "${@}" =~ "checkpoints" ]] || [[ "${@}" =~ "checkpoints" ]] || \
    echo "Not touching checkpoints"
[[ "${@}" =~ "all" ]] || [[ "${@}" =~ "checkpoints" ]] && \
echo "pulling checkpoints" && \
    wget "https://yale.box.com/shared/static/uxhtsdqfec28cu2tpbf8rtvkph16dhqz" \
    -O "${SPATHS[checkpoints]}/ddp" && \
    wget "https://yale.box.com/shared/static/cuwcgxk0b0o7h5jck35xbhmw80h8wm6v" \
    -O "${SPATHS[checkpoints]}/vae"
