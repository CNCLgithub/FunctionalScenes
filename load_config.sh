#!/bin/bash

# Loads config

if [ -f "user.conf" ]; then
    echo "Found user config, overriding default..."
    CFGFILE="user.conf"
else
    echo "No user config found, using default"
    CFGFILE="default.conf"
fi

while read line; do
    if [[ $line =~ ^"["(.+)"]"$ ]]; then
        arrname=${BASH_REMATCH[1]}
        declare -A $arrname
    elif [[ $line =~ ^([_[:alpha:]][_[:alnum:]]*)":"(.*) ]]; then
        declare ${arrname}[${BASH_REMATCH[1]}]="${BASH_REMATCH[2]}"
    fi
done < $CFGFILE

# ensure that all the paths exist on the host

for i in "${!PATHS[@]}"
do
    if [ ! -d "${PATHS[$i]}" ]; then
        echo "Make missing path: ${PATHS[$i]}"
        mkdir -p "${PATHS[$i]}"
    fi
done
