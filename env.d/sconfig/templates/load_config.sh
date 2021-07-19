#!/bin/bash

#################################################################################
# Loads config
#################################################################################
echo "( ) Loading project config ..."
sconfig_dir=$(realpath "$0" | xargs dirname)
if [ -f "${sconfig_dir}/user.conf" ]; then
    CFGFILE="${sconfig_dir}/user.conf"
    printf "\tFound user config at %s\n" "$CFGFILE"
else
    CFGFILE="${sconfig_dir}/default.conf"
    printf "\tNo user config found, using default (%s)\n" "$CFGFILE"
fi

. "$CFGFILE"

printf "(\xE2\x9C\x94) Loading project config \n"

#################################################################################
# ensure that all the paths exist on the host
#################################################################################
echo "( ) Assessing project paths ..."
for i in "${!SPATHS[@]}"
do
    if [ ! -d "${SPATHS[$i]}" ]; then
        mkdir -p "${SPATHS[$i]}"
        printf "\tMade missing path: ${SPATHS[$i]}\n"
    fi
done
printf "(\xE2\x9C\x94) Assessing project paths \n"

#################################################################################
# Export VARIABLES
#################################################################################

echo "( ) Exporting project variables ..."
for i in "${!SVARS[@]}"
do
    printf "\t%s \u2190 %s\n" "${i}" "${SVARS[$i]}"
    export "${i}=${SVARS[$i]}"
done
printf "(\xE2\x9C\x94) Exporting project variables \n"
