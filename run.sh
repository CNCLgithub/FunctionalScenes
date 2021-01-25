#!/bin/bash

cd "$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

. load_config.sh

# Define the path to the container and conda env
CONT="${ENV['cont']}"

# Parse the incoming command
COMMAND="$@"

# Enter the container and run the command
SING="${ENV['path']} exec --nv"
mounts=(${ENV[mounts]})
BS=""
for i in "${mounts[@]}";do
    if [[ $i ]]; then
        BS="${BS} -B $i:$i"
    fi
done

# expose these paths under `/${path}`
for i in "${!PATHS[@]}"
do
  BS="${BS} -B ${PATHS[$i]}:/$i"
done

# add the repo path to "/project"
BS="${BS} -B ${PWD}:/project"


export CUB_HOME="${PWD}/${ENV[pyenv]}/cub-1.10.0"
echo $CUB_HOME
export LD_PRELOAD="/lib/x86_64-linux-gnu/libstdc++.so.6"
$SING $BS $CONT bash -c "source /project/${ENV[pyenv]}/bin/activate \
        && export JULIA_PROJECT=${PWD} \
        && export TMPDIR=${PATHS[tmp]} \
        && export PYTHON=/project/${ENV[pyenv]}/bin/python3 \
        && export JULIA_DEPOT_PATH=/project/${ENV[julia_depot]} \
        && export PYCALL_JL_RUNTIME_PYTHON=/project/${ENV[pyenv]}/bin/python3 \
        && cd $PWD \
        && exec $COMMAND \
        && deactivate"
