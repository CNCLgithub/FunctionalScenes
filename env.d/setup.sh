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
cont_pull_url="https://yale.box.com/shared/static/ycvqzzp57v54dog2pompyitj8d7c9e93.sif"
min_pull_url="https://yale.box.com/shared/static/piau0ilrc23ki62ohgpo34oh4sq7mkrz.sif"
SING="${SENV[sing]}"
BUILD="${SENV[envd]}/${SENV[def]}"
cont_dest="${SENV[envd]}/${SENV[cont]}"
min_dest="${SENV[envd]}/min.sif"
rstudio_dest="${SENV[envd]}/rstudio"


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
    APPTAINER_TMPDIR="${SPATHS[tmp]}" sudo -E $SING build \
    "$cont_dest" "$BUILD"

[[ "${@}" =~ "min_pull" ]] && \
    echo "pulling container" && \
    wget "$min_pull_url" -O "${min_dest}"

[[ "${@}" =~ "min_build" ]] && echo "building ${BUILD} -> ${min_dest}" && \
    APPTAINER_TMPDIR="${SPATHS[tmp]}" sudo -E $SING build \
    "$min_dest" "${SENV[envd]}/Singularity.minimal"

[[ "${@}" =~ "rstudio_build" ]] && echo "building ${BUILD} -> ${rstudio_dest}" && \
    APPTAINER_TMPDIR="${SPATHS[tmp]}" sudo -E $SING build \
    "$rstudio_dest" "${SENV[envd]}/Singularity.rstudio"

#################################################################################
# Python setup
#################################################################################
[[ "${@}" =~ "python" ]] || echo "Not touching python"
[[ "${@}" =~ "all" ]] || [[ "${@}" =~ "python" ]] && \
    echo "building python env at ${SENV[pyenv]}" && \
    $SING exec "${cont_dest}" bash -c "virtualenv ${SENV[pyenv]} && \
    source ${SENV[pyenv]}/bin/activate && \
    python -m pip install --upgrade pip" && \
    ./env.d/run.sh python -m pip install --no-cache-dir -r /project/env.d/requirements.txt


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
    wget "https://yale.box.com/shared/static/2fif6cgs7by5kt35n5zskze4m5b8sxjk.gz" \
    -O "${SPATHS[datasets]}/ccn_2023_exp.tar.gz" && \
    tar -xzf "${SPATHS[datasets]}/ccn_2023_exp.tar.gz" -C "${SPATHS[datasets]}/"

[[ "${@}" =~ "checkpoints" ]] || [[ "${@}" =~ "checkpoints" ]] || \
    echo "Not touching checkpoints"
[[ "${@}" =~ "all" ]] || [[ "${@}" =~ "checkpoints" ]] && \
echo "pulling checkpoints" && \
    wget "https://yale.box.com/shared/static/bhxpl16t015fmics1qxdmes3wndsv2yu.ckpt" \
    -O "${SPATHS[checkpoints]}/scene_vae.ckpt" && \
    wget "https://yale.box.com/shared/static/4myglrftnixyce8u9l41c0zhuz11yv9g.ckpt" \
    -O "${SPATHS[checkpoints]}/og_decoder.ckpt"
