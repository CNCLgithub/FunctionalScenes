#!/bin/bash

ENVD=$(dirname "$0")            # relative
ENVD=$(cd "$ENVD" && pwd)    # absolutized and normalized
WDIR="$ENVD/rstudio"
# echo "$ENVD"
# echo "$WDIR"
apptainer exec --bind "$WDIR/run":"/run" \
    --bind "$WDIR/var":"/var/lib/rstudio-server",\
    --bind "$WDIR/database.conf":"/etc/rstudio/database.conf" \
    --bind "$WDIR":"/home/rstudio" \
    --bind "$PWD/scripts":"/home/rstudio/project/scripts" \
    --bind "$ENVD/spaths":"/home/rstudio/project/spaths" \
    env.d/verse_4.0.4.sif rserver --www-address=127.0.0.1
    # env.d/rstudio_4.0.4.sif rserver --www-address=127.0.0.1
