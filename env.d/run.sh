#!/bin/bash

#################################################################################
# Environment definition
#################################################################################
sconfig_dir=$(realpath "$0" | xargs dirname)
. "$sconfig_dir/load_config.sh"

#################################################################################
# Usage
#################################################################################
usage="$(basename "$0") CMD -- Execute command within environment and return

examples:
    # Python version inside environment (if installed)
    ./env.d/run.sh python --version

    # Enter a bash session inside the environment
    ./env.d/run.sh bash

    # Execute a script and exit
    ./env.d/run.sh scripts/myscript.sh

    # Enter the Julia repl (if installed)
    ./env.d/run.sh julia
"
[ $# -eq 0 ] && echo "$usage" && exit 0

#################################################################################
# Variable declarations
#################################################################################

# Define the path to the container and conda env
CONT="${SENV[envd]}/${SENV[cont]}"

# Parse the incoming command
COMMAND="$@"

#################################################################################
# Mount additional file systems
#################################################################################
SING="${SENV[sing]} exec --nv"
mounts=(${SENV[mounts]})
BS=""
echo "( ) Adding mount points"
for i in "${mounts[@]}";do
    if [[ $i ]]; then
        printf "\t%s \u2190 %s\n" "$i" "$i"
        BS="${BS} -B $i:$i"
    fi
done
printf "(\xE2\x9C\x94) Adding mount points\n"

#################################################################################
# Bind project SPATHS
#################################################################################
base_path="${SENV[spath]}"
BS="${BS} -B ${PWD}:/project"
echo "( ) Binding project paths"
printf "\t%s \u2190 %s\n" "/project" "${PWD}"
for i in "${!SPATHS[@]}"
do
    printf "\t%s \u2190 %s\n" "${base_path}/$i" "${SPATHS[$i]}"
    BS="${BS},${SPATHS[$i]}:${base_path}/$i"
done
printf "(\xE2\x9C\x94) Binding project paths\n"

#################################################################################
# Execution
#################################################################################
echo "( ) Executing ${COMMAND}"
printf "=%.0s"  $(seq 1 63)
printf "\n"
$SING $BS $CONT bash -c "source /project/${SENV[pyenv]}/bin/activate \
    cd /project && \
    exec $COMMAND && \
    cd - && \
    deactivate"
printf "=%.0s"  $(seq 1 63)
printf "\n"
printf "(\xE2\x9C\x94) Executing %s\n" "${COMMAND}"
